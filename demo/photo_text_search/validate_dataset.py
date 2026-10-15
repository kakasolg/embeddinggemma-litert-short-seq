"""Check the fixed answers before any model runs.

1. Every gold image exists and its rendered text contains at least one accepted answer string.
2. "deep" questions: the target item starts past token 64 of the document text (prompt + BOS included),
   so plain 64-token truncation really would cut it.
3. "none" questions: no image contains the key terms together (the answer is really absent).
Needs the Gemma tokenizer (google/embeddinggemma-300m, already cached in the eg2 venv).
"""

import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
facts = {f["id"]: f for f in json.loads((HERE / "data/facts.json").read_text(encoding="utf-8"))}
qs = json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))
norm = lambda s: re.sub(r"\s+", "", s).lower()
text_of = lambda i: " ".join(facts[i]["lines"])

DEEP_TARGET = {"D01": "Black Sesame Affogato", "D02": "굴 순두부", "D03": "Duck Confit Hash", "D04": "차돌박이 추가",
               "D05": "삼선짬뽕", "D06": "Yuzu Shio Ramen", "D07": "Khao Soi Chicken", "D08": "오미자차"}
NONE_TERMS = {"A01": [["hanbit", "한빛"], ["wi-fi", "password", "와이파이"]], "A02": [["corner slice"], ["validat"]],
              "A03": [["nami"], ["corkage"]], "A04": [["서울분식"], ["예약", "confirmation"]],
              "A05": [["moonrise"], ["happy hour"]]}

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

bad = 0
for q in qs:
    for g in q["gold"]:
        if g not in facts:
            print("MISSING gold", q["id"], g); bad += 1
    if q["gold"]:
        hit = [g for g in q["gold"] if any(norm(a) in norm(text_of(g)) for a in q["accept"])]
        if not hit:
            print("ANSWER NOT IN GOLD TEXT", q["id"], q["accept"], q["gold"]); bad += 1
    if q["type"] == "deep":
        g = q["gold"][0]
        doc = "title: none | text: " + text_of(g)
        target = DEEP_TARGET[q["id"]]
        pos_char = doc.index(target)
        n_before = len(tok(doc[:pos_char])["input_ids"]) - 1  # minus the EOS the tokenizer appends
        total = len(tok(doc)["input_ids"])
        ok = n_before > 64
        print(f"deep {q['id']} {g}: target starts at token {n_before} of {total} {'OK' if ok else 'TOO EARLY'}")
        bad += not ok
    if q["type"] == "none":
        groups = NONE_TERMS[q["id"]]
        clash = [i for i in facts if all(any(t in text_of(i).lower() for t in grp) for grp in groups)]
        print(f"none {q['id']}: images with all key terms: {clash or 'none'}")
        bad += bool(clash)
print("types:", {t: sum(q['type'] == t for q in qs) for t in sorted({q['type'] for q in qs})}, "total", len(qs))
print("problems:", bad)
sys.exit(1 if bad else 0)
