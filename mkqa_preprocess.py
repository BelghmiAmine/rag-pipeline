import json
from collections import defaultdict

file_path = "mkqa.jsonl"
output_path = "mkqa_pairs.json"

langs = ["ar", "da", "de", "en", "es", "fi", "fr", "he", "hu", "it", "ja", "ko", "km", "ms", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr", "vi", "zh_cn", "zh_hk", "zh_tw"]

# Stored as: pairs[lang] = list of {"query": ..., "answer": ...} dicts
pairs = defaultdict(list)

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)

            # Skip unanswerable questions (based on English label)
            answer_type = data.get('answers', {}).get('en', [{}])[0].get('type')
            if answer_type == 'unanswerable':
                continue

            for lang in langs:
                query = data.get('queries', {}).get(lang)
                answers = data.get('answers', {}).get(lang, [])
                answer = answers[0].get('text') if answers else None

                # Skip if query or answer is missing for this language
                if query and answer:
                    pairs[lang].append({"query": query, "answer": answer})

        except json.JSONDecodeError:
            continue

# Summary
print("--- Dataset summary ---")
for lang in langs:
    print(f"  {lang}: {len(pairs[lang])} pairs")

# Inspect a sample
print("\n--- Sample for French ---")
for item in pairs["fr"][:3]:
    print(f"  Q: {item['query']}")
    print(f"  A: {item['answer']}")
    print()

# Save to JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"\nSaved {sum(len(v) for v in pairs.values())} total pairs across {len(langs)} languages to '{output_path}'")