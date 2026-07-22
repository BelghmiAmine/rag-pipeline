"""
Generate synthetic Question-Answer pairs from Wikipedia, per language.

Following the methodology from:
https://huggingface.co/learn/cookbook/rag_evaluation

For each language:
  1. Load wikimedia/wikipedia dataset (20231101 snapshot).
  2. Sample N articles at random.
  3. For each article: generate a (question, answer) pair using a GENERATOR LLM.
  4. Critique each QA pair on 3 axes (groundedness, relevance, standalone) using a
     SEPARATE, independent CRITIC LLM (not the generator, to avoid self-evaluation bias).
  5. Rank all generated pairs by total critique score and keep exactly the best
     TARGET_QA pairs (default 500) so every language has the same dataset size.
  6. Save per-language JSON file.
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

from wikipedia_RAG import EPFL_BASE_URL

# --- Config ---
SEED = 42
MAX_WORKERS = 16
DEFAULT_GENERATOR_MODEL = "deepseek-ai/DeepSeek-V3.2"
# Independent multilingual critic, different from the generator and from the models
# being evaluated downstream (Apertus 8B/70B), to avoid self-evaluation bias.
DEFAULT_CRITIC_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "tr", "zh"]
WIKIPEDIA_SNAPSHOT = "20231101"
DEFAULT_ARTICLES_PER_LANG = 1500
MIN_ARTICLE_CHARS = 500
MAX_ARTICLE_CHARS = 3000
# Keep exactly this many of the best-scored QA pairs per language (uniform dataset size).
DEFAULT_TARGET_QA = 500

DEFAULT_OUTPUT_DIR = "/mnt/nlp/scratch/home/belghmi/data/lang_synthetic_qa_pairs"

LANG_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "it": "Italian", "ja": "Japanese", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish",
    "zh": "Chinese",
}

# Note: wikimedia/wikipedia uses "zh" not "zh_cn" for Chinese
WIKI_CONFIG_OVERRIDE = {}

QA_GENERATION_PROMPT = """Your task is to write a single factoid question and its concise answer based on a Wikipedia article excerpt.

Rules:
- The question must be answerable from the context with a specific, concise piece of factual information.
- The question must be phrased like one a user would type into a search engine (do NOT say "according to the passage" or "in the context").
- Both the question and the answer MUST be written in {language}.
- The answer should be a short noun phrase or sentence, not a paragraph.

Respond with a single JSON object and nothing else:
{{"question": "...", "answer": "..."}}

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
    Rank all critiqued QA pairs and keep exactly `target_qa` of the best ones, so
    every language ends up with the same dataset size.

    Ranking: by total critique score (groundedness + relevance + standalone),
    descending. Ties (very common at 5/5/5) are broken by a deterministic shuffle
    seeded with SEED, so the selection among equally-scored pairs is reproducible
    and unbiased rather than dependent on completion order.
    """
    scored = [
        r for r in results
        if r["groundedness_score"] is not None
        and r["relevance_score"] is not None
        and r["standalone_score"] is not None
    ]

    rng = random.Random(SEED)
    rng.shuffle(scored)  # deterministic tiebreak before the stable sort
    scored.sort(
        key=lambda r: r["groundedness_score"] + r["relevance_score"] + r["standalone_score"],
        reverse=True,
    )
    return scored[:target_qa]


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
    print(f"  Sampled {len(articles)} articles. Generating QA pairs in parallel...")

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
            f"  WARNING: only {len(selected)} valid QA pairs available for '{lang}' "
            f"(< target {target_qa}). Keeping all of them. "
            f"Consider increasing --n-articles."
        )
    print(f"  Generated: {len(results)} | Kept best: {len(selected)} (target {target_qa})")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "language": lang,
            "language_name": LANG_NAMES.get(lang, lang),
            "generator_model": generator_model,
            "critic_model": critic_model,
            "wikipedia_snapshot": WIKIPEDIA_SNAPSHOT,
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
        description="Generate synthetic QA pairs from Wikipedia per language."
    )
    parser.add_argument(
        "--languages", nargs="+", default=DEFAULT_LANGUAGES,
        help="Languages to process (default: 14 MKQA-style langs)",
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
        help="Keep exactly this many best-scored QA pairs per language",
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
