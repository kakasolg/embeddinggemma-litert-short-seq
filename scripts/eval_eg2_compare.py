"""EmbeddingGemma 300M (this repo's base model) vs EmbeddingGemma 2 on the same four public tasks, PC PyTorch.

Same texts, same prompts (the model-card strings, which the EmbeddingGemma 2 paper describes as its training format),
same truncation lengths and the same scoring as eval_public_bench.py. Only the model and its tokenizer change.
The 300M vectors come from that script's cache (pt64 / pt256 / ptfull, run its `embed` step first); EmbeddingGemma 2
vectors are cached in $EGSS_CACHE/eg2/. Needs sentence-transformers and access to google/embeddinggemma-2.

Usage:
    python eval_eg2_compare.py            # writes results/eg2_compare_pc.json
"""


import json
import sys
from pathlib import Path

import numpy as np

LENS = {"64": 64, "256": 256, "full": 2048}
DIMS = [768, 256, 128]


def embed_eg2(texts, L):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("google/embeddinggemma-2", device=dev)
    tok = st.tokenizer
    ids = [tok(t)["input_ids"][:L] for t in texts]
    V = np.zeros((len(ids), st.get_sentence_embedding_dimension()), np.float32)
    order = np.argsort([len(x) for x in ids])
    # Sorted by length, so size each batch by a token budget (32 x 2048 tokens did not fit in 16 GB).
    batches, s = [], 0
    while s < len(order):
        m = len(ids[order[min(s + 31, len(order) - 1)]])
        n = max(1, min(32, 4096 // max(m, 1)))
        batches.append(order[s:s + n])
        s += n
    with torch.no_grad():
        for idx in batches:
            xs = [ids[i] for i in idx]
            m = max(len(x) for x in xs)
            inp = torch.full((len(xs), m), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros_like(inp)
            for k, x in enumerate(xs):
                inp[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            o = st({"input_ids": inp.to(dev), "attention_mask": att.to(dev)})["sentence_embedding"]
            V[idx] = torch.nn.functional.normalize(o.float(), dim=-1).cpu().numpy()
    lens = np.array([len(x) for x in ids])
    return V, lens


def cut(V, d):
    W = V[:, :d]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_public_bench as eb
    out = Path(__file__).resolve().parent.parent / "results" / "eg2_compare_pc.json"

    data = eb.build()
    d2 = eb.CACHE / "eg2"
    d2.mkdir(exist_ok=True)
    v1, v2, toklen = {}, {}, {}
    for k, L in LENS.items():
        f = d2 / f"eg2_{k}.npy"
        if not f.exists():
            V, lens = embed_eg2(data["texts"], L)
            np.save(f, V)
            np.save(d2 / f"eg2_{k}_len.npy", lens)
            print("embedded EG2", k, flush=True)
        v2[k] = np.load(f)
        toklen[k] = np.load(d2 / f"eg2_{k}_len.npy")
        v1[k] = np.load(eb.CACHE / f"pt{k}.npy")

    v1len = np.array([len(x) for x in data["ids"]])
    v2len = toklen["full"]
    res = {"token_length_median": {"v1": float(np.median(v1len)), "eg2": float(np.median(v2len))},
           "token_length_over64_pct": {"v1": float(np.mean(v1len > 64) * 100), "eg2": float(np.mean(v2len > 64) * 100)},
           "scores": {}}
    print(f"tokens median v1 {np.median(v1len):.0f} eg2 {np.median(v2len):.0f}; >64: v1 {np.mean(v1len > 64) * 100:.1f}% "
          f"eg2 {np.mean(v2len > 64) * 100:.1f}%")
    print("\n## EG2 - v1 at the same length and dimension (STS Spearman / retrieval nDCG@10, 95% paired bootstrap)")
    for k in LENS:
        for dim in DIMS:
            A, B = cut(v1[k], dim), cut(v2[k], dim)
            cells, row = [], {}
            for n, t in data["tasks"].items():
                sa, pa = eb.score_task(A, t)
                sb, pb = eb.score_task(B, t)
                x, y = list(sb.values())[0], list(sa.values())[0]
                lo, hi = eb.boot_delta(t["kind"], pb, pa)
                row[n] = {"v1": y, "eg2": x, "delta": [x - y, lo, hi]}
                cells.append(f"{n} {y:.4f}→{x:.4f} ({x - y:+.4f} [{lo:+.4f}, {hi:+.4f}])")
            res["scores"][f"{k}/d{dim}"] = row
            print(f"  len {k} d{dim}: " + " | ".join(cells), flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
