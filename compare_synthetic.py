import json, glob, os

SYN = "/mnt/nlp/scratch/home/belghmi/synthetic_results"

# rows[(lang, mode)] = {"basesft": score, "cptsft": score}
rows = {}
extra = {}  # (lang, mode) -> {tag: (n_valid, context_hit, context_prec)}
for f in glob.glob(os.path.join(SYN, "*sft-*-judge.json")):
    d = json.load(open(f))
    parts = os.path.basename(f)[:-len("-judge.json")].split("-")  # basesft, en, rag
    tag, lang, mode = parts[0], parts[1], parts[2]
    rows.setdefault((lang, mode), {})[tag] = d.get("answer_score_normalized_0_1")
    extra.setdefault((lang, mode), {})[tag] = (
        d.get("n_valid_answer_judgements"),
        d.get("context_hit_at_k"),
        d.get("context_precision_at_k"),
    )

print(f"{'lang':6}{'mode':5}{'base':>8}{'cpt':>8}{'delta':>8}")
sums = {}  # mode -> [base_sum, cpt_sum, n]
for (lang, mode), v in sorted(rows.items()):
    b, c = v.get("basesft"), v.get("cptsft")
    if b is not None and c is not None:
        print(f"{lang:6}{mode:5}{b:8.3f}{c:8.3f}{c-b:+8.3f}")
        s = sums.setdefault(mode, [0.0, 0.0, 0])
        s[0] += b; s[1] += c; s[2] += 1

print("-" * 35)
for mode in ("cb", "rag"):
    if mode in sums:
        b, c, n = sums[mode]
        print(f"{'MEAN':6}{mode:5}{b/n:8.3f}{c/n:8.3f}{(c-b)/n:+8.3f}")

# overall mean across all completed (lang, mode) cells
allb = sum(s[0] for s in sums.values()); allc = sum(s[1] for s in sums.values())
alln = sum(s[2] for s in sums.values())
if alln:
    print(f"{'MEAN':6}{'all':5}{allb/alln:8.3f}{allc/alln:8.3f}{(allc-allb)/alln:+8.3f}")

# flag any cell where the two arms have very different valid-judgement counts
print("\nCells with mismatched n_valid (>25 diff) — interpret their delta with caution:")
flagged = False
for (lang, mode), e in sorted(extra.items()):
    nb = e.get("basesft", (None,))[0]; nc = e.get("cptsft", (None,))[0]
    if nb is not None and nc is not None and abs(nb - nc) > 25:
        print(f"  {lang}-{mode}: base n_valid={nb}, cpt n_valid={nc}")
        flagged = True
if not flagged:
    print("  none")
