"""Matryoshka check: keep the first d of the 768 output dimensions, renormalize, and rescore.

Uses the vectors cached by eval_public_bench.py (run its `embed` step first) and the same task loaders and scoring.
EmbeddingGemma was trained with Matryoshka representation learning for 768/512/256/128.

For every cached model and every d it reports the score and its difference from the same model at 768
(95% paired bootstrap). For the .tflite files it also reports the difference from the original PyTorch model
at the same truncation length and the same d, so conversion error and dimension loss are kept apart.

Usage:
    python eval_matryoshka.py            # writes results/matryoshka_pc.json
"""

import json
from pathlib import Path

import numpy as np

import eval_public_bench as eb

DIMS = [768, 512, 256, 128]


def cut(V, d):
    W = V[:, :d]
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def main():
    data = eb.build()
    vecs = {f.stem: np.load(f) for f in sorted(eb.CACHE.glob("*.npy"))}
    out, per = {}, {}
    for name, V in vecs.items():
        out[name], per[name] = {}, {}
        for d in DIMS:
            W = cut(V, d)
            sc = {n: eb.score_task(W, t) for n, t in data["tasks"].items()}
            out[name][d] = {n: s[0] for n, s in sc.items()}
            per[name][d] = {n: s[1] for n, s in sc.items()}

    print("## Score by dimension (STS = Spearman, retrieval = nDCG@10); [95% CI of difference from 768]")
    for name in vecs:
        for d in DIMS:
            cells = []
            for n, t in data["tasks"].items():
                s = list(out[name][d][n].values())[0]
                if d == 768:
                    cells.append(f"{n} {s:.4f}")
                    continue
                base = list(out[name][768][n].values())[0]
                lo, hi = eb.boot_delta(t["kind"], per[name][d][n], per[name][768][n])
                out[name][d][n]["delta_vs_768"] = [s - base, lo, hi]
                cells.append(f"{n} {s:.4f} ({s - base:+.4f} [{lo:+.4f}, {hi:+.4f}])")
            print(f"  {name} d={d}: " + " | ".join(cells))

    print("\n## .tflite minus original at the same length and dimension (95% paired bootstrap)")
    for name in vecs:
        if name.startswith("pt"):
            continue
        ref = f"pt{eb.seq_of(name)}"
        if ref not in vecs:
            continue
        for d in DIMS:
            cells = []
            for n, t in data["tasks"].items():
                x = list(out[name][d][n].values())[0]
                y = list(out[ref][d][n].values())[0]
                lo, hi = eb.boot_delta(t["kind"], per[name][d][n], per[ref][d][n])
                out[name][d][n]["delta_vs_" + ref] = [x - y, lo, hi]
                cells.append(f"{n} {x - y:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(f"  {name} - {ref} d={d}: " + " | ".join(cells))

    sizes = {d: d * 4 for d in DIMS}
    res = eb.RESULTS / "matryoshka_pc.json"
    res.write_text(json.dumps({"dims": DIMS, "bytes_per_vector_fp32": sizes,
                               "results": {m: {str(d): v for d, v in r.items()} for m, r in out.items()}},
                              ensure_ascii=False, indent=1))
    print("wrote", res)


if __name__ == "__main__":
    main()
