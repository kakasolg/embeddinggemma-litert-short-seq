"""Step 2b: PC baseline — original EmbeddingGemma-300M (PyTorch, sentence-transformers) over the phone's OCR text.

The 60 fixed questions (data/questions.json, locked at 1ca40a5) against the 80 images, each image indexed as one
document "title: none | text: <OCR>". Conditions:
  full        – whole text at up to 2,048 tokens (the model's best case; not what fits on the phone)
  first64     – plain truncation to 64 tokens
  chunk64     – the wrapper's prefix-preserving 64-token chunks, image score = max over its chunks
                (same rule as DocumentChunker.kt / chunking.py plan())
  chunk64_raw – chunk64 on ML Kit's raw text instead of the geometry-rebuilt rows (Field Lab parity)
  chunk64_gold– chunk64 on the rendered text (a perfect-OCR ceiling)
Answerable questions (55): recall@1/@3 and MRR of the first gold image. "none" questions (5): top-1 score,
compared with the top-1 score of answerable questions, since abstention is a threshold on that score.
Run from the repo root (needs requirements-bench.txt):  python demo/photo_text_search/pc_baseline.py
Writes demo/photo_text_search/results/pc_baseline.json.
"""

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunking import PREFIX_QUERY, embed_pt, plan, tok_  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
facts = {f["id"]: f for f in json.loads((HERE / "data/facts.json").read_text(encoding="utf-8"))}
qs = json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))
ocr = {x["id"]: x for x in json.loads((HERE / "results/ocr_pixel.json").read_text(encoding="utf-8"))["items"]}
ids_ = sorted(facts)
tok = tok_()

texts = {
    "rows": {i: " ".join(ocr[i]["rows"]) for i in ids_},
    "raw": {i: " ".join(ocr[i]["text"].split()) for i in ids_},
    "gold": {i: " ".join(facts[i]["lines"]) for i in ids_},
}


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


Q = unit(embed_pt("full", [tok(PREFIX_QUERY + q["q"])["input_ids"] for q in qs]))


def whole(src, n):
    full = [tok(f"title: none | text: {texts[src][i]}")["input_ids"] for i in ids_]
    D = unit(embed_pt(n, full))
    return Q @ D.T, [len(x) for x in full]


def chunked(src):
    owner, chunks = [], []
    for k, i in enumerate(ids_):
        body = texts[src][i] or " "
        for c in plan(tok, None, body):
            owner.append(k); chunks.append(c)
    C = unit(embed_pt(64, chunks))
    S = Q @ C.T
    out = np.full((len(qs), len(ids_)), -1.0, np.float32)
    for j, k in enumerate(owner):
        out[:, k] = np.maximum(out[:, k], S[:, j])
    return out, np.bincount(owner, minlength=len(ids_))


def evaluate(S):
    per_q, rank_of = [], {}
    for qi, q in enumerate(qs):
        order = np.argsort(-S[qi])
        top = [ids_[k] for k in order[:3]]
        rec = {"id": q["id"], "type": q["type"], "top3": top, "top1_score": round(float(S[qi, order[0]]), 4)}
        if q["gold"]:
            r = min(int(np.where(order == ids_.index(g))[0][0]) + 1 for g in q["gold"])
            rec["rank"] = r
        per_q.append(rec)
    ans = [p for p in per_q if "rank" in p]

    def agg(ps):
        return {"n": len(ps), "r@1": round(np.mean([p["rank"] <= 1 for p in ps]), 4),
                "r@3": round(np.mean([p["rank"] <= 3 for p in ps]), 4),
                "mrr": round(np.mean([1 / p["rank"] for p in ps]), 4)}
    by_type = {t: agg([p for p in ans if p["type"] == t]) for t in sorted({p["type"] for p in ans})}
    none = [p for p in per_q if "rank" not in p]
    hit1 = sorted(p["top1_score"] for p in ans if p["rank"] == 1)
    return {"all": agg(ans), "by_type": by_type,
            "none_top1": {p["id"]: p["top1_score"] for p in none},
            "answerable_hit_top1_score": {"min": hit1[0], "p10": hit1[len(hit1) // 10], "median": hit1[len(hit1) // 2]},
            "none_above_min_hit": sum(p["top1_score"] >= hit1[0] for p in none),
            "questions": per_q}


results, meta = {}, {}
S, lens = whole("rows", "full"); results["full"] = evaluate(S)
meta["doc_tokens_rows"] = {"max": max(lens), "median": int(np.median(lens)), "over64": sum(x > 64 for x in lens)}
S, _ = whole("rows", 64); results["first64"] = evaluate(S)
for name, src in (("chunk64", "rows"), ("chunk64_raw", "raw"), ("chunk64_gold", "gold")):
    S, n = chunked(src); results[name] = evaluate(S)
    meta[f"chunks_{src}"] = {"total": int(n.sum()), "max_per_image": int(n.max())}

print(json.dumps(meta, ensure_ascii=False))
print(f"{'condition':13} {'r@1':>6} {'r@3':>6} {'mrr':>6}  deep r@1  none≥minhit  minhit")
for k, v in results.items():
    a = v["all"]
    print(f"{k:13} {a['r@1']:6.3f} {a['r@3']:6.3f} {a['mrr']:6.3f}  {v['by_type']['deep']['r@1']:8.3f}  "
          f"{v['none_above_min_hit']}/5  {v['answerable_hit_top1_score']['min']:.3f}")
for t in results["chunk64"]["by_type"]:
    print(t, {k: results[k]["by_type"][t]["r@1"] for k in results})
for p in results["chunk64"]["questions"]:
    if p.get("rank", 1) > 1:
        print("miss chunk64", p["id"], p["type"], "rank", p["rank"], "top3", p["top3"])
(HERE / "results/pc_baseline.json").write_text(json.dumps({"model": "google/embeddinggemma-300m (PyTorch)",
                                                           "ocr": "results/ocr_pixel.json", "meta": meta,
                                                           "results": results}, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
