"""Step 2a: how much of the fixed answer text survives on-device OCR.

Input: results/ocr_pixel.json (DemoOcrBench on the phone) + data/facts.json + data/questions.json.
Per image: character error rate (CER) of OCR vs rendered lines, both whitespace-stripped — this also counts
reading-order differences (ML Kit returns a menu's name column and price column as separate blocks) — and
word recall, which ignores order: share of rendered words that appear as an OCR word — and line pairing:
share of rendered lines (e.g. "김치찌개 Kimchi Jjigae $16.99") found whole inside one OCR line, for ML Kit's raw
text lines vs the geometry-rebuilt `rows` (DemoOcrBench.rows). Pairing is what keeps a price with its item.
Per question: is an accepted answer string still findable in the OCR text of a gold image
(same normalization as validate_dataset.py: whitespace removed, lowercased)?
Writes results/ocr_score.json.
"""

import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
facts = {f["id"]: f for f in json.loads((HERE / "data/facts.json").read_text(encoding="utf-8"))}
qs = json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))
ocr_file = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results/ocr_pixel.json")
ocr = {x["id"]: x for x in json.loads(ocr_file.read_text(encoding="utf-8"))["items"]}
norm = lambda s: re.sub(r"\s+", "", s).lower()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_recall(lines, text):
    ref = [w.lower() for w in " ".join(lines).split()]
    hyp = {w.lower() for w in text.split()}
    return sum(w in hyp for w in ref) / len(ref) if ref else 1.0


def pairing(lines, ocr_lines):
    hyp = [norm(x) for x in ocr_lines]
    multi = [norm(l) for l in lines if len(l.split()) >= 2]
    return sum(any(l in h for h in hyp) for l in multi), len(multi)


per_image = {}
for i, f in facts.items():
    ref, hyp = norm("".join(f["lines"])), norm(ocr[i]["text"])
    per_image[i] = {"kind": f["kind"], "style": f["style"], "photo": f["photo"], "ref_chars": len(ref),
                    "pair_raw": pairing(f["lines"], ocr[i]["text"].splitlines()),
                    "pair_rows": pairing(f["lines"], ocr[i]["rows"]),
                    "word_recall": round(word_recall(f["lines"], ocr[i]["text"]), 4),
                    "cer": round(lev(ref, hyp) / len(ref), 4) if ref else None, "ms": round(ocr[i]["ms"], 1)}

per_q = []
for q in qs:
    if not q["gold"]:
        continue
    found = [g for g in q["gold"] if any(norm(a) in norm(ocr[g]["text"]) for a in q["accept"])]
    per_q.append({"id": q["id"], "type": q["type"], "gold": q["gold"], "answer_in_ocr": bool(found)})

by_kind = {}
for v in per_image.values():
    if v["cer"] is not None:
        by_kind.setdefault(v["style"], []).append(v["cer"])
lost = [p for p in per_q if not p["answer_in_ocr"]]
summary = {
    "images": len(per_image),
    "mean_cer_by_style": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())},
    "mean_word_recall_by_style": {k: round(sum(v["word_recall"] for v in per_image.values() if v["style"] == k) / len(c), 4)
                                  for k, c in sorted(by_kind.items())},
    "line_pairing_by_style": {k: {m: f'{sum(v[m][0] for v in per_image.values() if v["style"] == k)}/'
                                     f'{sum(v[m][1] for v in per_image.values() if v["style"] == k)}' for m in ("pair_raw", "pair_rows")}
                              for k in sorted(by_kind)},
    "line_pairing_all": {m: f'{sum(v[m][0] for v in per_image.values())}/{sum(v[m][1] for v in per_image.values())}'
                         for m in ("pair_raw", "pair_rows")},
    "mean_cer_all": round(sum(c for v in by_kind.values() for c in v) / sum(len(v) for v in by_kind.values()), 4),
    "answer_in_ocr": f"{len(per_q) - len(lost)}/{len(per_q)}",
    "answer_lost": [{"id": p["id"], "type": p["type"], "gold": p["gold"],
                     "accept": next(q["accept"] for q in qs if q["id"] == p["id"])} for p in lost],
}
print(json.dumps(summary, ensure_ascii=False, indent=1))
for p in lost:
    for g in p["gold"]:
        print(f"--- {p['id']} {g} OCR:", ocr[g]["text"].replace("\n", " | ")[:400])
(HERE / "results/ocr_score.json").write_text(json.dumps({"summary": summary, "images": per_image, "questions": per_q},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")
