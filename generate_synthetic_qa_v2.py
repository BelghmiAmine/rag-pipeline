"""
Generate an ENHANCED synthetic Question-Answer set from Wikipedia, per language.

This is the v2 ("hard") variant of generate_synthetic_qa.py. It keeps the same
in-distribution, single-passage design (so the gold source is guaranteed to be in
the FAISS index and retrieval stays clean), but changes two things:

  1. HARDER questions. The generator is instructed to write questions that require
     combining several facts, comparison, aggregation, temporal ordering, or an
     inference supported by the passage -- NOT a single-sentence lookup.
  2. RICHER source articles + difficulty metadata. We sample longer, denser
     articles and attach the generator's self-declared question_type / difficulty
     label (emitted for free in the same generation call), so downstream analysis
     can still break the CPT effect down by question type.

Pipeline per language:
  1. Load wikimedia/wikipedia (20231101 snapshot), sample N substantial articles.
  2. For each article: GENERATOR LLM writes one hard (question, answer) pair
     + a self-declared question_type and difficulty.
  3. An INDEPENDENT CRITIC LLM scores each pair on 3 axes:
     groundedness, relevance, standalone.
  4. Keep only well-grounded, self-contained pairs, then rank by total critic
     score and keep exactly TARGET_QA of them (uniform size across languages).
  5. Save per-language JSON.
"""

import os
import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Import the EPFL endpoint from llm_client (not wikipedia_RAG) so this script does NOT
# pull in langchain/sentence-transformers, which are absent from the lean vLLM eval image.
from llm_client import EPFL_BASE_URL

# --- Config ---
SEED = 42
MAX_WORKERS = 16
# NOTE: matched-pair design => a same-family generator bias cancels in the
# cpt-vs-base delta, so Apertus-70B is acceptable here. Keep the generator OUT of
# the Apertus-8B family only matters for absolute cross-model claims (not made).
DEFAULT_GENERATOR_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
# Independent multilingual critic, different family from the generator and from the
# 8B models being evaluated downstream, to avoid self-evaluation bias.
DEFAULT_CRITIC_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "tr", "zh"]
WIKIPEDIA_SNAPSHOT = "20231101"
# Sample more, and longer, articles: harder questions need denser source material,
# and the groundedness/standalone filter below rejects more, so we over-sample.
DEFAULT_ARTICLES_PER_LANG = 3000
MIN_ARTICLE_CHARS = 1200          # denser articles than the v1 500-char floor
MAX_ARTICLE_CHARS = 4000          # give the generator more material to combine
# Keep exactly this many of the hardest well-grounded pairs per language.
DEFAULT_TARGET_QA = 1000
# A pair must clear these critic thresholds before it can be kept, so that "hard"
# never means "ambiguous / unanswerable".
MIN_GROUNDEDNESS = 4
MIN_STANDALONE = 4

DEFAULT_OUTPUT_DIR = "/mnt/nlp/scratch/home/belghmi/data/lang_synthetic_qa_pairs_v2"

LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "it": "Italian", "ja": "Japanese", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish",
    "zh": "Chinese",
}

# Note: wikimedia/wikipedia uses "zh" not "zh_cn" for Chinese
WIKI_CONFIG_OVERRIDE = {}

QA_GENERATION_PROMPT = """Your task is to write ONE challenging factoid question and its concise answer, based on a Wikipedia article excerpt.

The question must be HARDER than a simple lookup. It must require AT LEAST ONE of:
- combining two or more distinct facts stated in the excerpt (multi-hop),
- a comparison, aggregation, or count over information in the excerpt,
- temporal or numerical reasoning (ordering events, computing a duration or difference),
- an inference that is clearly supported by the excerpt but not stated word-for-word.

Rules:
- The answer MUST be fully derivable from the excerpt alone (no outside knowledge).
- Do NOT write a question that can be answered by copying a single sentence verbatim.
- Phrase it like a question a real person would type into a search engine (do NOT say "according to the passage", "in the context", or "in this article").
- Both the question and the answer MUST be written in {language}.
- The answer should be a short phrase or a single sentence, not a paragraph.
- Also label the question's type and difficulty.

Respond with a single JSON object and nothing else:
{{"question": "...", "answer": "...", "question_type": "<multi_hop|comparison|aggregation|temporal|inference>", "difficulty": "<medium|hard>"}}

Context:
{context}
"""

GROUNDEDNESS_PROMPT = """Given the question and context below, rate from 1 to 5 how clearly the question is answerable using ONLY the provided context.

- 1: The context does not contain the answer.
- 5: The context clearly and unambiguously contains the answer.

Respond with a single JSON object and nothing else:
{{"score": <integer 1-5>}}

Question: {question}

Context: {context}
"""

RELEVANCE_PROMPT = """Rate from 1 to 5 how useful the following question is as a general-knowledge factoid question a real person might ask.

- 1: Trivial, malformed, or unlikely to be asked.
- 5: Well-formed, clear, and useful as a general-knowledge question.

Respond with a single JSON object and nothing else:
{{"score": <integer 1-5>}}

Question: {question}
"""

STANDALONE_PROMPT = """Rate from 1 to 5 how context-independent the following question is.

- 1: The question refers to "the passage", "the context", "this article", or otherwise cannot be understood without additional context.
- 5: The question is fully self-contained and could be answered by anyone with appropriate knowledge.

Respond with a single JSON object and nothing else:
{{"score": <integer 1-5>}}

Question: {question}
"""


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 512) -> str | None:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=SEED,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception:
        return None


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def process_article(
    client: OpenAI,
    generator_model: str,
    critic_model: str,
    article_text: str,
    lang_name: str,
) -> dict | None:
    excerpt = article_text[:MAX_ARTICLE_CHARS]

    gen_response = call_llm(
        client, generator_model,
        QA_GENERATION_PROMPT.format(language=lang_name, context=excerpt),
        max_tokens=512,
    )
    qa = parse_json(gen_response)
    if not qa or "question" not in qa or "answer" not in qa:
        return None
    question = str(qa["question"]).strip()
    answer = str(qa["answer"]).strip()
    if not question or not answer:
        return None

    # Critique with an INDEPENDENT model (not the generator) to avoid self-grading.
    ground = parse_json(call_llm(
        client, critic_model,
        GROUNDEDNESS_PROMPT.format(question=question, context=excerpt),
        max_tokens=64,
    ))
    rel = parse_json(call_llm(
        client, critic_model,
        RELEVANCE_PROMPT.format(question=question),
        max_tokens=64,
    ))
    sa = parse_json(call_llm(
        client, critic_model,
        STANDALONE_PROMPT.format(question=question),
        max_tokens=64,
    ))

    return {
        "question": question,
        "answer": answer,
        "source_doc": excerpt,
        "question_type": str(qa.get("question_type", "")).strip() or None,
        "generator_difficulty": str(qa.get("difficulty", "")).strip() or None,
        "groundedness_score": ground.get("score") if ground else None,
        "relevance_score": rel.get("score") if rel else None,
        "standalone_score": sa.get("score") if sa else None,
    }


def sample_articles(lang: str, n_articles: int) -> list[str]:
    config = WIKI_CONFIG_OVERRIDE.get(lang, f"{WIKIPEDIA_SNAPSHOT}.{lang}")
    print(f"  Loading wikimedia/wikipedia config '{config}'...")
    ds = load_dataset("wikimedia/wikipedia", config, split="train")

    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    sampled = []
    for i in indices:
        text = ds[i]["text"]
        if isinstance(text, str) and len(text) >= MIN_ARTICLE_CHARS:
            sampled.append(text)
        if len(sampled) >= n_articles:
            break
    return sampled


def select_best_qa(results: list[dict], target_qa: int) -> list[dict]:
    """
    Keep the best well-grounded, self-contained questions.

    Two-stage selection:
      1. Filter: a pair must have all three critic scores, groundedness >= MIN_GROUNDEDNESS,
         and standalone >= MIN_STANDALONE. This keeps only questions that are clearly
         answerable from the passage and self-contained (hardness itself comes from the
         generation prompt, which asks for multi-hop / reasoning questions).
      2. Rank by total critic score (groundedness + relevance + standalone), desc,
         breaking ties by a deterministic seeded shuffle so the selection among
         equally-scored pairs is reproducible rather than dependent on completion order.
    """
    valid = [
        r for r in results
        if r["groundedness_score"] is not None
        and r["relevance_score"] is not None
        and r["standalone_score"] is not None
        and r["groundedness_score"] >= MIN_GROUNDEDNESS
        and r["standalone_score"] >= MIN_STANDALONE
    ]

    rng = random.Random(SEED)
    rng.shuffle(valid)  # deterministic tiebreak before the stable sort
    valid.sort(
        key=lambda r: r["groundedness_score"] + r["relevance_score"] + r["standalone_score"],
        reverse=True,
    )
    return valid[:target_qa]


def generate_for_language(
    client: OpenAI,
    generator_model: str,
    critic_model: str,
    lang: str,
    n_articles: int,
    target_qa: int,
    output_dir: str,
):
    print(f"\n=== {lang} ({LANG_NAMES.get(lang, lang)}) ===")
    articles = sample_articles(lang, n_articles)
    print(f"  Sampled {len(articles)} articles. Generating hard QA pairs in parallel...")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                process_article, client, generator_model, critic_model, text,
                LANG_NAMES.get(lang, lang),
            ): i
            for i, text in enumerate(articles)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {lang}"):
            try:
                result = future.result(timeout=240)
                if result is not None:
                    results.append(result)
            except Exception:
                pass

    selected = select_best_qa(results, target_qa)
    if len(selected) < target_qa:
        print(
            f"  WARNING: only {len(selected)} valid hard QA pairs available for '{lang}' "
            f"(< target {target_qa}). Keeping all of them. "
            f"Consider increasing --n-articles."
        )
    if selected:
        print(f"  Generated: {len(results)} | Kept best: {len(selected)} (target {target_qa})")
        # Print a few kept pairs so `runai logs` lets you eyeball quality on smoke runs.
        for r in selected[:8]:
            print(f"    [type={r.get('question_type')} diff={r.get('generator_difficulty')}] "
                  f"Q: {r['question']}  ||  A: {r['answer']}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "language": lang,
            "language_name": LANG_NAMES.get(lang, lang),
            "variant": "v2_hard",
            "generator_model": generator_model,
            "critic_model": critic_model,
            "wikipedia_snapshot": WIKIPEDIA_SNAPSHOT,
            "min_groundedness": MIN_GROUNDEDNESS,
            "min_standalone": MIN_STANDALONE,
            "n_articles_sampled": len(articles),
            "n_qa_generated": len(results),
            "target_qa": target_qa,
            "n_qa_kept": len(selected),
            "qa_pairs": selected,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved to '{out_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate an enhanced (harder) synthetic QA set from Wikipedia per language."
    )
    parser.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGUAGES,
        help="Languages to process (default: 13 study languages)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_GENERATOR_MODEL,
        help="Generator LLM model name (EPFL API)",
    )
    parser.add_argument(
        "--critic-model", default=DEFAULT_CRITIC_MODEL,
        help="Independent critic LLM for grading QA pairs (must differ from --model)",
    )
    parser.add_argument(
        "--n-articles", type=int, default=DEFAULT_ARTICLES_PER_LANG,
        help="Number of articles to sample per language",
    )
    parser.add_argument(
        "--target-qa", type=int, default=DEFAULT_TARGET_QA,
        help="Keep exactly this many hardest well-grounded QA pairs per language",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-language JSON files",
    )
    args = parser.parse_args()

    if args.critic_model == args.model:
        print(
            f"WARNING: critic model == generator model ('{args.model}'). "
            f"This reintroduces self-evaluation bias; pass a different --critic-model."
        )

    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(base_url=EPFL_BASE_URL, api_key=api_key, timeout=240.0)

    for lang in args.languages:
        try:
            generate_for_language(
                client=client,
                generator_model=args.model,
                critic_model=args.critic_model,
                lang=lang,
                n_articles=args.n_articles,
                target_qa=args.target_qa,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"FAILED for language '{lang}': {e}")
