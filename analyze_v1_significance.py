"""
Re-analyze the v1 synthetic LLM-as-judge results with more appropriate statistics.

The thesis currently tests the CPT effect with a paired t-test across the 13 language
means (n=13), which gives p~0.15 for the +0.014 synthetic-RAG delta. That test is
conservative: it collapses each language to a single number, so its power is limited by
between-language heterogeneity, not by how many questions we asked.

Because the design is matched-pair, every question was answered by BOTH arms
(base-SFT and cpt-SFT) on the SAME retrieval file, in the SAME order. The judge output
stores per-question `records` with an `idx` and a 1-5 `score_1_5`, so we can pair the two
arms question-by-question (by idx) and recover the per-question paired difference of the
normalized (0-1) score. From those we compute three tests, weakest-power to strongest:

  1. Across-language paired t-test (n = #languages)   -- reproduces the thesis number.
  2. DerSimonian-Laird random-effects meta-analysis    -- the honest middle ground:
     pools the per-language effects while accounting for both within-language sampling
     error AND between-language heterogeneity (reports tau^2 and I^2).
  3. Pooled question-level paired t-test (n = #questions) -- maximum power, but
     anti-conservative (treats questions as independent, ignores language clustering).
     Reported only as an upper bound on significance.

Nothing is re-generated; this only reads the existing *-judge.json files.

Usage (where the synthetic_results JSONs live, e.g. a small runai job or the login node):
  python3 analyze_v1_significance.py \
    --results-dir /mnt/nlp/scratch/home/belghmi/synthetic_results
"""

import os
import json
import argparse
import numpy as np
from scipy import stats

LANGS = ["ar", "de", "en", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "tr", "zh"]


def load_norm_scores(path: str) -> dict[int, float]:
    """idx -> normalized (0-1) answer score, for questions with a valid judge score."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d.get("records", []):
        s = r.get("score_1_5")
        if s is not None:
            out[r["idx"]] = (s - 1) / 4.0
    return out


def paired_diffs(results_dir: str, lang: str, mode: str) -> np.ndarray:
    """Per-question (cpt - base) normalized-score differences for one language/mode."""
    base = load_norm_scores(os.path.join(results_dir, f"basesft-{lang}-{mode}-judge.json"))
    cpt = load_norm_scores(os.path.join(results_dir, f"cptsft-{lang}-{mode}-judge.json"))
    common = sorted(set(base) & set(cpt))
    return np.array([cpt[i] - base[i] for i in common], dtype=float)


def dersimonian_laird(yi: np.ndarray, vi: np.ndarray) -> dict:
    """Random-effects meta-analysis (DerSimonian-Laird). yi: per-study effect, vi: its variance."""
    k = len(yi)
    w = 1.0 / vi
    y_fixed = np.sum(w * yi) / np.sum(w)
    Q = float(np.sum(w * (yi - y_fixed) ** 2))
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_star = 1.0 / (vi + tau2)
    y_re = float(np.sum(w_star * yi) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    z = y_re / se_re
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    return {
        "estimate": y_re, "se": se_re, "z": z, "p": p,
        "ci_lo": y_re - 1.96 * se_re, "ci_hi": y_re + 1.96 * se_re,
        "tau2": tau2, "Q": Q, "df": df, "I2": I2,
    }


def analyze_mode(results_dir: str, mode: str):
    print(f"\n{'='*70}\nMODE: {mode.upper()}\n{'='*70}")

    lang_means, lang_vars, lang_ns, all_diffs = [], [], [], []
    print(f"{'lang':6}{'n':>6}{'mean_diff':>12}{'se':>10}")
    for lang in LANGS:
        d = paired_diffs(results_dir, lang, mode)
        if len(d) < 2:
            print(f"{lang:6}{len(d):>6}   (skipped: too few paired questions)")
            continue
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        lang_means.append(m)
        lang_vars.append(se ** 2)              # variance of the language-mean estimate
        lang_ns.append(len(d))
        all_diffs.extend(d.tolist())
        print(f"{lang:6}{len(d):>6}{m:>+12.4f}{se:>10.4f}")

    if len(all_diffs) == 0:
        print("\n  NO PAIRED PER-QUESTION DATA FOUND.")
        print("  The judge JSONs in this directory have no `records` field (they store only the")
        print("  aggregate mean). Per-question scores were never saved, so this analysis cannot run.")
        print("  Re-run llm_as_judge_eval.py (current version saves `records`) to regenerate them.")
        return

    yi = np.array(lang_means)
    vi = np.array(lang_vars)
    all_diffs = np.array(all_diffs)
    k = len(yi)

    # 1) Across-language paired t-test (the thesis's current test).
    t1, p1 = stats.ttest_1samp(yi, 0.0)
    print(f"\n[1] Across-language t-test (n={k} languages)  <- current thesis method")
    print(f"    mean of language deltas = {yi.mean():+.4f}   sd = {yi.std(ddof=1):.4f}")
    print(f"    t = {t1:+.3f}   p = {p1:.4f}")

    # 2) Random-effects meta-analysis (the recommended, honest test).
    re = dersimonian_laird(yi, vi)
    print(f"\n[2] Random-effects meta-analysis (DerSimonian-Laird)  <- recommended")
    print(f"    pooled delta = {re['estimate']:+.4f}   95% CI [{re['ci_lo']:+.4f}, {re['ci_hi']:+.4f}]")
    print(f"    z = {re['z']:+.3f}   p = {re['p']:.4f}")
    print(f"    heterogeneity: tau^2 = {re['tau2']:.5f}   I^2 = {re['I2']:.1f}%   (Q={re['Q']:.2f}, df={re['df']})")

    # 3) Pooled question-level paired t-test (upper bound on significance).
    t3, p3 = stats.ttest_1samp(all_diffs, 0.0)
    print(f"\n[3] Pooled question-level t-test (n={len(all_diffs)} questions)  <- anti-conservative upper bound")
    print(f"    mean diff = {all_diffs.mean():+.4f}   sd = {all_diffs.std(ddof=1):.4f}")
    print(f"    t = {t3:+.3f}   p = {p3:.2e}")

    # Optional: a proper linear mixed model (question-level, random intercept per language),
    # if statsmodels is available. This is the formally-correct version of test [2].
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        rows = []
        for lang in LANGS:
            d = paired_diffs(results_dir, lang, mode)
            for v in d:
                rows.append({"diff": float(v), "lang": lang})
        df = pd.DataFrame(rows)
        md = smf.mixedlm("diff ~ 1", df, groups=df["lang"])
        mf = md.fit(reml=True, method="lbfgs")
        b = mf.params["Intercept"]; se = mf.bse["Intercept"]; p = mf.pvalues["Intercept"]
        print(f"\n[4] Linear mixed model  diff ~ 1 + (1|lang)  (statsmodels)")
        print(f"    intercept (pooled delta) = {b:+.4f}   se = {se:.4f}   p = {p:.4f}")
    except Exception as e:
        print(f"\n[4] (linear mixed model skipped: {type(e).__name__} — statsmodels not available)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-analyze v1 synthetic judge results with proper statistics.")
    ap.add_argument("--results-dir", default="/mnt/nlp/scratch/home/belghmi/synthetic_results",
                    help="Directory holding {basesft,cptsft}-{lang}-{rag,cb}-judge.json files.")
    ap.add_argument("--modes", nargs="+", default=["rag", "cb"], choices=["rag", "cb"])
    args = ap.parse_args()

    for mode in args.modes:
        analyze_mode(args.results_dir, mode)
