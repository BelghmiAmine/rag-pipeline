"""
MCQ evaluation on the INCLUDE benchmark (CohereLabs/include-base-44).

Inputs: a retrieve.py-style results JSON where:
  - queries          = INCLUDE question strings
  - references       = gold answer indices (int 0..3)
  - extra.options    = the 4 answer options per question
  - retrieved_docs   = top-k Wikipedia chunks per question (used only in RAG mode)

Steps:
  1. For each question, build an MCQ prompt:
       - closed-book mode: question + options + "Answer:"
       - RAG mode:         retrieved docs + question + options + "Answer:"
  2. Ask the model. Force a 4-token completion. Parse the first A/B/C/D.
  3. Score deterministically against the gold index.

Output: a small JSON with accuracy + parse_rate (no per-query dump by default).
"""

import os
import re
import json
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from llm_client import make_generator_client, LLM_MODEL_NAME

SEED = 42
MAX_WORKERS = 16
LETTERS = ["A", "B", "C", "D"]

CB_PROMPT = """Answer the following multiple-choice question.
Respond with ONLY the letter of the correct option (A, B, C, or D). No explanations, no punctuation.

Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""

RAG_PROMPT = """You are given context documents and a multiple-choice question.
Use the context if it helps; otherwise rely on general knowledge.
Respond with ONLY the letter of the correct option (A, B, C, or D). No explanations, no punctuation.

Context:
{context}

---
Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""


def extract_letter(text: str) -> int | None:
    """Return 0..3 if the response contains an unambiguous A/B/C/D, else None."""
    if not text:
        return None
    upper = text.upper()
    # Prefer the first standalone letter token
    m = re.search(r"\b([ABCD])\b", upper)
    if m:
        return LETTERS.index(m.group(1))
    # Fallback: first occurrence of any of the letters anywhere
    for ch in upper:
        if ch in LETTERS:
            return LETTERS.index(ch)
    return None


def run_include_eval(
    retrieval_results_path: str,
    llm_model_name: str = LLM_MODEL_NAME,
    num_queries: int | None = None,
    closed_book: bool = False,
    inference: str = "online",
    print_answers: int = 0,
):
    print(f"Loading retrieval results from '{retrieval_results_path}'...")
    with open(retrieval_results_path, encoding="utf-8") as f:
        data = json.load(f)

    queries: list[str] = data["queries"]
    references: list[int] = data["references"]  # gold index 0..3
    all_docs: list[list[str]] = data["retrieved_docs"]
    extra = data.get("extra") or {}
    options: list[list[str]] = extra.get("options", [])
    subjects: list[str] = extra.get("subjects", [""] * len(queries))
    domains: list[str] = extra.get("domains", [""] * len(queries))

    if not options:
        raise ValueError(
            "No 'extra.options' in retrieval results. Did you run retrieve.py with "
            "--dataset-type include?"
        )

    if num_queries is not None:
        queries = queries[:num_queries]
        references = references[:num_queries]
        all_docs = all_docs[:num_queries]
        options = options[:num_queries]
        subjects = subjects[:num_queries]
        domains = domains[:num_queries]

    mode = "CLOSED-BOOK" if closed_book else "RAG"
    print(f"Evaluating {len(queries)} INCLUDE questions in {mode} mode.")
    print(f"  Model: {llm_model_name}")

    client = make_generator_client(inference, timeout=120.0)

    def answer_one(idx: int, q: str, opts: list[str], docs: list[str]) -> tuple[int, int | None, str]:
        if closed_book:
            prompt = CB_PROMPT.format(
                question=q, a=opts[0], b=opts[1], c=opts[2], d=opts[3],
            )
        else:
            context = "\n\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(docs))
            prompt = RAG_PROMPT.format(
                context=context, question=q,
                a=opts[0], b=opts[1], c=opts[2], d=opts[3],
            )
        try:
            completion = client.chat.completions.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                seed=SEED,
                max_tokens=8,
            )
            raw = completion.choices[0].message.content or ""
            return idx, extract_letter(raw), raw
        except Exception as e:
            print(f"\n[{idx}] failed: {e}")
            return idx, None, ""

    preds: list[int | None] = [None] * len(queries)
    raws: list[str] = [""] * len(queries)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(answer_one, i, q, o, d): i
            for i, (q, o, d) in enumerate(zip(queries, options, all_docs))
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Answering ({mode})"):
            try:
                idx, pred, raw = future.result(timeout=120)
            except Exception:
                idx = futures[future]
                pred, raw = None, ""
            preds[idx] = pred
            raws[idx] = raw

    correct = sum(1 for p, r in zip(preds, references) if p is not None and p == r)
    parsed = sum(1 for p in preds if p is not None)
    n = len(preds)
    accuracy = correct / n if n else 0.0
    parse_rate = parsed / n if n else 0.0
    # Conditional accuracy on questions the model parsed correctly
    cond_accuracy = correct / parsed if parsed else 0.0

    # Per-domain breakdown
    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)
    for p, r, d in zip(preds, references, domains):
        domain_total[d] += 1
        if p is not None and p == r:
            domain_correct[d] += 1
    per_domain = {
        d: {
            "n": domain_total[d],
            "accuracy": domain_correct[d] / domain_total[d] if domain_total[d] else 0.0,
        }
        for d in sorted(domain_total)
    }

    print(f"\nResults:")
    print(f"  Accuracy:        {correct}/{n}  = {accuracy:.4f}")
    print(f"  Parse rate:      {parsed}/{n}  = {parse_rate:.4f}")
    print(f"  Cond. accuracy:  {correct}/{parsed} = {cond_accuracy:.4f}")
    if per_domain:
        print(f"  Per-domain accuracy:")
        for d, v in per_domain.items():
            print(f"    {d:30s}  n={v['n']:4d}  acc={v['accuracy']:.4f}")

    # Per-question records — go into the saved JSON; useful for inspection/debugging.
    records = []
    for i, (q, opts, gold, pred, raw) in enumerate(zip(queries, options, references, preds, raws)):
        records.append({
            "idx": i,
            "question": q,
            "options": opts,
            "gold_letter": LETTERS[gold] if 0 <= gold < 4 else None,
            "pred_letter": LETTERS[pred] if pred is not None else None,
            "pred_raw": raw,
            "correct": pred is not None and pred == gold,
        })

    if print_answers > 0:
        print(f"\n--- First {min(print_answers, len(records))} answers ---")
        for r in records[:print_answers]:
            mark = "OK " if r["correct"] else ("--" if r["pred_letter"] is None else "X ")
            print(f"\n[{r['idx']:3d}] {mark} gold={r['gold_letter']} pred={r['pred_letter']!r} raw={r['pred_raw']!r}")
            print(f"      Q: {r['question'][:200]}")
            for letter, opt in zip(LETTERS, r["options"]):
                marker = " <-- gold" if letter == r["gold_letter"] else ""
                print(f"      {letter}) {opt[:120]}{marker}")

    return {
        "mode": "closed-book" if closed_book else "rag",
        "model": llm_model_name,
        "n_total": n,
        "n_parsed": parsed,
        "n_correct": correct,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "conditional_accuracy": cond_accuracy,
        "per_domain": per_domain,
        "records": records,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MCQ evaluation on INCLUDE (closed-book or RAG)."
    )
    parser.add_argument(
        "--retrieval-results", type=str, required=True,
        help="Path to retrieval results JSON produced by retrieve.py --dataset-type include.",
    )
    parser.add_argument(
        "--llm", type=str, default=LLM_MODEL_NAME,
        help="Generator model name (EPFL model id, or vLLM served-model-name when --inference local).",
    )
    parser.add_argument(
        "--inference", type=str, default="online", choices=["online", "local"],
        help="online: EPFL API; local: vLLM server on localhost.",
    )
    parser.add_argument(
        "--num-queries", type=int, default=None,
        help="Evaluate only the first N (default: all).",
    )
    parser.add_argument(
        "--closed-book", action="store_true",
        help="Closed-book mode: ignore retrieved docs.",
    )
    parser.add_argument(
        "--print-answers", type=int, default=0, metavar="N",
        help="Print the first N answers (question + options + gold + pred + raw) to stdout. 0 = none.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save aggregate results as JSON.",
    )
    args = parser.parse_args()

    results = run_include_eval(
        retrieval_results_path=args.retrieval_results,
        llm_model_name=args.llm,
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
