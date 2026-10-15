"""On-device benchmark on a fixed public sample: latency and cosine to the original model.

The sample is drawn with a fixed seed from the texts built by eval_public_bench.py: 50 pairs each from KorSTS and
STSBenchmark (200 sentences), and 25 queries + 25 documents each from Ko-StrategyQA and SciFact. 300 texts total.
The same token IDs go to the device and to the original PyTorch model.

Steps:
    python device_public_bench.py prep
        -> $EGSS_CACHE/device_bench_sample.ids (one line per text, comma-separated token IDs)
        -> results/device_bench_sample.json   (row numbers, groups and token counts only; no dataset text)
    Run the .ids file through your device runner (see docs/BENCHMARKS.md). For each run, save
        <dir>/device_bench_sample__<model>.r<run>.f32     768 float32 little-endian per text, in .ids order
        <dir>/device_bench_sample__<model>.r<run>.ms.txt  one latency in ms per line
    and append one line per run to <dir>/thermal.jsonl:
        {"run": 1, "model": "<model>", "thermal_before": 0, "thermal_after": 0}
    python device_public_bench.py score <dir> <dir>/thermal.jsonl
        -> results/device_public_bench.json
"""

import json, os, sys
from pathlib import Path
import numpy as np

CACHE = Path(os.environ.get("EGSS_CACHE", Path.home() / ".cache" / "egss-bench"))
RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 20261014
IDS = CACHE / "device_bench_sample.ids"  # kept out of the repo: it is derived from the dataset text
META = RESULTS / "device_bench_sample.json"


def prep():
    data = json.loads((CACHE / "texts.json").read_text())
    rng = np.random.default_rng(SEED)
    rows, groups = [], []
    for n, t in data["tasks"].items():
        if t["kind"] == "sts":
            pick = rng.choice(len(t["rows"]), 50, replace=False)
            for k in sorted(pick):
                a, b, _ = t["rows"][k]
                rows += [a, b]; groups += [f"{n}:sent", f"{n}:sent"]
        else:
            qs = sorted(t["queries"].values()); ds = sorted(t["docs"].values())
            for r in sorted(rng.choice(qs, 25, replace=False)): rows.append(int(r)); groups.append(f"{n}:query")
            for r in sorted(rng.choice(ds, 25, replace=False)): rows.append(int(r)); groups.append(f"{n}:doc")
    ids = [data["ids"][r] for r in rows]
    IDS.write_text("\n".join(",".join(map(str, x)) for x in ids) + "\n")
    lens = [len(x) for x in ids]
    RESULTS.mkdir(exist_ok=True)
    META.write_text(json.dumps({"seed": SEED, "source": "eval_public_bench.py texts (MTEB 2.24.14 loaders)",
                                "rows": rows, "groups": groups, "n_tokens": lens}, indent=0))
    print(f"{len(ids)} texts -> {IDS}; median {np.median(lens):.0f} tokens, {sum(l > 64 for l in lens)} over 64")


def pt_vectors(ids, L):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("google/embeddinggemma-300m", device=dev)
    out = []
    with torch.no_grad():
        for x in ids:
            inp = torch.tensor([x[:L]], device=dev); att = torch.ones_like(inp)
            o = st({"input_ids": inp, "attention_mask": att})["sentence_embedding"]
            out.append(torch.nn.functional.normalize(o.float(), dim=-1)[0].cpu().numpy())
    return np.array(out)


def score(run_dir, thermal_path):
    meta = json.loads(META.read_text())
    ids = [list(map(int, l.split(","))) for l in IDS.read_text().split("\n") if l.strip()]
    groups = np.array(meta["groups"]); lens = np.array(meta["n_tokens"])
    ref = {L: pt_vectors(ids, L) for L in (64, 256)}
    thermal = [json.loads(l) for l in Path(thermal_path).read_text().splitlines() if l.strip()]
    runs = []
    for pos, th in enumerate(thermal, 1):
        stem = f"device_bench_sample__{th['model']}"
        V = np.fromfile(Path(run_dir) / f"{stem}.r{th['run']}.f32", "<f4").reshape(-1, 768)
        ms = np.loadtxt(Path(run_dir) / f"{stem}.r{th['run']}.ms.txt")
        seq = 64 if "seq64" in th["model"] else 256
        cos = (V * ref[seq]).sum(1)
        steady = ms[3:]  # first 3 items of each run are warm-up
        r = {**th, "position": pos, "seq": seq, "n": len(ms), "ms_mean": float(steady.mean()),
             "ms_p50": float(np.median(steady)), "ms_p95": float(np.percentile(steady, 95)), "ms_first": float(ms[0]),
             "cos_vs_pt": {"ref": f"pt{seq}", "mean": float(cos.mean()), "min": float(cos.min()),
                           "p01": float(np.percentile(cos, 1)),
                           "by_group": {g: float(cos[groups == g].mean()) for g in sorted(set(groups))}},
             "truncated_items": int((lens > seq).sum())}
        runs.append(r)
        print(f"{th['model']} r{th['run']}: mean {r['ms_mean']:.1f}ms p95 {r['ms_p95']:.1f} · "
              f"cosine vs pt{seq} mean {cos.mean():.4f} min {cos.min():.4f} · thermal {th['thermal_before']}->{th['thermal_after']}")
    out = RESULTS / "device_public_bench.json"
    out.write_text(json.dumps({"sample": {"seed": SEED, "n": len(ids),
                                          "groups": {g: int((groups == g).sum()) for g in sorted(set(groups))},
                                          "tokens_median": float(np.median(lens)), "over64": int((lens > 64).sum()),
                                          "over256": int((lens > 256).sum())},
                               "runs": runs}, ensure_ascii=False, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        score(sys.argv[2], sys.argv[3])
