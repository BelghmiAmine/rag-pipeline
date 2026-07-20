"""
RAG evaluation using LLM-as-a-judge.

Inputs: a retrieve.py-style results JSON containing queries, references, and
retrieved_docs (already produced by your existing retrieval pipeline).

Steps (RAG mode):
  1. Generate an answer using the retrieved docs.
  2. Judge the answer on a 1-5 rubric vs the reference (Answer score).
  3. For each retrieved doc, judge whether it helps answer the question
     (Context Hit@k and Context Precision@k).

Steps (closed-book mode):
  1. Generate an answer without retrieved docs.
  2. Judge the answer on a 1-5 rubric vs the reference (Answer score).
  (Retrieval relevance is skipped — irrelevant when docs aren't used.)
"""

import os
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from llm_client import make_generator_client, make_judge_client, LLM_MODEL_NAME, SYSTEM_PROMPT

SEED = 42
MAX_WORKERS = 16
# Independent multilingual judge, deliberately NOT Apertus: the evaluated answers come
# from Apertus-8B/70B, so judging with Apertus-70B would introduce self-preference bias.
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

CLOSED_BOOK_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the question concisely and factually based on your own knowledge.
Always respond in the same language as the question.
Give only the answer — no explanations, no hedging, no citations."""

EVAL_PROMPT = """###Task Description:
You will receive a question, a generated response, and a reference answer that gets a score of 5.

1. Write a brief feedback (1-2 sentences) assessing the quality of the response strictly based on the score rubric.
2. After the feedback, output the score as an integer between 1 and 5.
3. The output format MUST be exactly: "Feedback: <your feedback> [RESULT] <integer 1-5>"
4. Do NOT generate any other opening, closing, or explanations.

###Question:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubric:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, or hallucinated.
Score 2: The response is mostly incorrect.
Score 3: The response is partially correct but missing key information or contains inaccuracies.
Score 4: The response is mostly correct and factual, with minor issues.
Score 5: The response is fully correct, accurate, and factual.

###Feedback:"""

RELEVANCE_PROMPT = """###Task Description:
You will receive a question and a retrieved document. Decide whether the document contains information that helps answer the question.

Output format MUST be exactly: "[RESULT] YES" or "[RESULT] NO" — nothing else.

###Question:
{question}

###Retrieved document:
{doc}

###Result:"""


def parse_judge_output(text: str) -> tuple[int | None, str]:
    if not text or "[RESULT]" not in text:
        return None, text or ""
    try:
        feedback_part, score_part = text.split("[RESULT]", 1)
        feedback = feedback_part.replace("Feedback:", "").strip()
        score_str = score_part.strip().split()[0].rstrip(".")
        score = int(score_str)
        if score < 1 or score > 5:
            return None, feedback
        return score, feedback
    except Exception:
        return None, text


def parse_relevance_output(text: str) -> bool | None:
    if not text or "[RESULT]" not in text:
        return None
    try:
        verdict = text.split("[RESULT]", 1)[1].strip().split()[0].upper()
        if verdict.startswith("YES"):
            return True
        if verdict.startswith("NO"):
            return False
        return None
    except Exception:
        return None


def run_llm_judge_eval(
    retrieval_results_path: str,
    llm_model_name: str = LLM_MODEL_NAME,
    judge_model_name: str = DEFAULT_JUDGE_MODEL,
    num_queries: int | None = None,
    closed_book: bool = False,
    inference: str = "online",
    print_answers: int = 0,
):
    print(f"Loading retrieval results from '{retrieval_results_path}'...")
    with open(retrieval_results_path, encoding="utf-8") as f:
        data = json.load(f)

    queries: list[str] = data["queries"]
    references: list[str] = data["references"]
    all_docs: list[list[str]] = data["retrieved_docs"]

    if num_queries is not None:
        queries = queries[:num_queries]
        references = references[:num_queries]
        all_docs = all_docs[:num_queries]

    mode = "CLOSED-BOOK" if closed_book else "RAG"
    k = len(all_docs[0]) if all_docs else 0
    print(f"Evaluating {len(queries)} queries in {mode} mode.")
    print(f"  Generator: {llm_model_name}")
    print(f"  Judge:     {judge_model_name}")
    if judge_model_name == llm_model_name:
        print("  WARNING: judge == generator -> self-preference bias. Use an independent judge.")

    # Generator may be local (vLLM) or online (EPFL); judge is ALWAYS the EPFL API model.
    gen_client = make_generator_client(inference, timeout=180.0)
    judge_client = make_judge_client(timeout=180.0)

    # --- Step 1: generate answers (RAG or closed-book) ---
    def generate_one(idx: int, query: str, docs: list[str]) -> tuple[int, str]:
        if closed_book:
            messages = [
                {"role": "system", "content": CLOSED_BOOK_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"},
            ]
        else:
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n---\nQuestion: {query}",
                },
            ]
        try:
            completion = gen_client.chat.completions.create(
                model=llm_model_name,
                messages=messages,
                temperature=0,
                seed=SEED,
            )
            return idx, completion.choices[0].message.content
        except Exception as e:
            print(f"\n[gen {idx}] failed: {e}")
            return idx, ""

    responses = [""] * len(queries)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(generate_one, i, q, d): i
            for i, (q, d) in enumerate(zip(queries, all_docs))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating ({mode})"):
            try:
                idx, response = future.result(timeout=180)
            except Exception:
                idx = futures[future]
                response = ""
            responses[idx] = response

    # --- Step 2: judge each answer (1-5) ---
    def judge_one(idx: int, question: str, response: str, reference: str) -> tuple[int, int | None, str]:
        prompt = EVAL_PROMPT.format(
            question=question, response=response, reference_answer=reference,
        )
        try:
            completion = judge_client.chat.completions.create(
                model=judge_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=SEED,
                max_tokens=512,
            )
            text = completion.choices[0].message.content
            score, feedback = parse_judge_output(text)
            return idx, score, feedback
        except Exception as e:
            return idx, None, f"JUDGE_ERROR: {e}"

    scores: list[int | None] = [None] * len(queries)
    feedbacks: list[str] = [""] * len(queries)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(judge_one, i, q, r, ref): i
            for i, (q, r, ref) in enumerate(zip(queries, responses, references))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging answers"):
            try:
                idx, score, feedback = future.result(timeout=180)
            except Exception as e:
                idx = futures[future]
                score, feedback = None, f"FUTURE_ERROR: {e}"
            scores[idx] = score
            feedbacks[idx] = feedback

    valid = [s for s in scores if s is not None]
    mean_score = sum(valid) / len(valid) if valid else 0.0
    normalized = (mean_score - 1) / 4 if valid else 0.0

    # --- Step 3: judge retrieval relevance (RAG mode only) ---
    context_hit_at_k: float | None = None
    context_precision_at_k: float | None = None

    if not closed_book:
        def judge_doc(qid: int, did: int, question: str, doc: str) -> tuple[int, int, bool | None]:
            prompt = RELEVANCE_PROMPT.format(question=question, doc=doc)
            try:
                completion = judge_client.chat.completions.create(
                    model=judge_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    seed=SEED,
                    max_tokens=16,
                )
                return qid, did, parse_relevance_output(completion.choices[0].message.content)
            except Exception:
                return qid, did, None

        # relevance[i] is a list of (bool | None) of length k for query i
        relevance: list[list[bool | None]] = [[None] * k for _ in range(len(queries))]
        total_calls = len(queries) * k
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = []
            for i, (q, docs) in enumerate(zip(queries, all_docs)):
                for j, d in enumerate(docs):
                    futures.append(pool.submit(judge_doc, i, j, q, d))
            for future in tqdm(as_completed(futures), total=total_calls, desc="Judging retrieval"):
                try:
                    qid, did, verdict = future.result(timeout=180)
                except Exception:
                    continue
                relevance[qid][did] = verdict

        # Context Hit@k: fraction of queries with at least one True
        hits = [any(v is True for v in row) for row in relevance]
        context_hit_at_k = sum(hits) / len(hits) if hits else 0.0

        # Context Precision@k: average fraction of True per query
        precisions = []
        for row in relevance:
            decided = [v for v in row if v is not None]
            if decided:
                precisions.append(sum(1 for v in decided if v) / len(decided))
        context_precision_at_k = sum(precisions) / len(precisions) if precisions else 0.0

    # --- Report ---
    print(f"\nResults ({len(valid)}/{len(scores)} valid answer judgements):")
    print(f"  Answer score (1-5):    {mean_score:.3f}")
    print(f"  Answer normalized:     {normalized:.3f}")
    if not closed_book:
        print(f"  Context Hit@{k}:         {context_hit_at_k:.3f}")
        print(f"  Context Precision@{k}:   {context_precision_at_k:.3f}")

    # Per-question records: generated answer + judge score/feedback, for inspection + the report.
    records = []
    for i, (q, ref, resp, sc, fb) in enumerate(zip(queries, references, responses, scores, feedbacks)):
        records.append({
            "idx": i,
            "question": q,
            "reference": ref,
            "response": resp,
            "score_1_5": sc,
            "judge_feedback": fb,
        })

    if print_answers > 0:
        print(f"\n--- First {min(print_answers, len(records))} answers ---")
        for r in records[:print_answers]:
            print(f"\n[{r['idx']:3d}] score={r['score_1_5']}")
            print(f"      Q:   {r['question'][:200]}")
            print(f"      REF: {r['reference'][:200]}")
            print(f"      GEN: {r['response'][:300]}")

    out = {
        "mode": "closed-book" if closed_book else "rag",
        "generator_model": llm_model_name,
        "judge_model": judge_model_name,
        "n_total": len(scores),
        "n_valid_answer_judgements": len(valid),
        "answer_score_mean_1_5": mean_score,
        "answer_score_normalized_0_1": normalized,
    }
    if not closed_book:
        out["k"] = k
        out["context_hit_at_k"] = context_hit_at_k
        out["context_precision_at_k"] = context_precision_at_k
    out["records"] = records
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate RAG using LLM-as-a-judge."
    )
    parser.add_argument(
        "--retrieval-results", type=str, required=True,
        help="Path to retrieval results JSON (from retrieve.py).",
    )
    parser.add_argument(
        "--llm", type=str, default=LLM_MODEL_NAME,
        help="Generator model name (EPFL model id, or vLLM served-model-name when --inference local).",
    )
    parser.add_argument(
        "--inference", type=str, default="online", choices=["online", "local"],
        help="Generator endpoint. online: EPFL API; local: vLLM on localhost. Judge always uses EPFL API.",
    )
    parser.add_argument(
        "--judge", type=str, default=DEFAULT_JUDGE_MODEL,
        help="Judge LLM model name (EPFL API).",
    )
    parser.add_argument(
        "--num-queries", type=int, default=None,
        help="Evaluate only the first N queries (default: all).",
    )
    parser.add_argument(
        "--closed-book", action="store_true",
        help="Closed-book mode: ignore retrieved docs, ask the LLM directly.",
    )
    parser.add_argument(
        "--print-answers", type=int, default=0, metavar="N",
        help="Print the first N generated answers (question/reference/response) to stdout. 0 = none.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results as JSON.",
    )
    args = parser.parse_args()

    results = run_llm_judge_eval(
        retrieval_results_path=args.retrieval_results,
        llm_model_name=args.llm,
        judge_model_name=args.judge,
        num_queries=args.num_queries,
        closed_book=args.closed_book,
        inference=args.inference,
        print_answers=args.print_answers,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved to '{args.output}'")
