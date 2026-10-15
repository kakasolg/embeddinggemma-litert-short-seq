"""Compare EmbeddingGemma-300M LiteRT builds against the original PyTorch model on public benchmarks.

Tasks (loaded through MTEB): KorSTS (test), STSBenchmark (test), Ko-StrategyQA (dev), SciFact (test).
Scores follow the MTEB definitions (STS = Spearman of cosine, retrieval = nDCG@10) but are computed here
directly, with EmbeddingGemma's prompts and an explicit truncation length, so they are comparable within this
script only, not with leaderboard numbers.

Every model receives exactly the same token IDs. "pt:N" means the original sentence-transformers model fed the
first N token IDs (the same head truncation the fixed-length .tflite files apply), so tflite vs pt:N isolates
conversion/quantization error from truncation error.

Model arguments:
  pt:N      original PyTorch model, inputs cut to the first N tokens
  pt:full   original PyTorch model, up to 2048 tokens
  <path>    a .tflite file

Usage:
    python eval_public_bench.py embed pt:64 pt:256 pt:full embeddinggemma-300m_seq64_q8.tflite ...
    python eval_public_bench.py score          # writes results/public_bench_pc.json
    python eval_public_bench.py chunk 64       # chunked retrieval (max over 64-token chunks)

Vectors are cached in $EGSS_CACHE (default ~/.cache/egss-bench).
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

CACHE = Path(os.environ.get("EGSS_CACHE", Path.home() / ".cache" / "egss-bench"))
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = Path(__file__).resolve().parent.parent / "results"
PREFIX_QUERY = "task: search result | query: "
PREFIX_SIMILARITY = "task: sentence similarity | query: "
STS = ["KorSTS", "STSBenchmark"]
RET = ["Ko-StrategyQA", "SciFact"]


def doc_prefix(title):
    return f"title: {title or 'none'} | text: "


def build():
    """All task texts in one list. The list order is the row order of every cached vector file."""
    p = CACHE / "texts.json"
    if p.exists():
        return json.loads(p.read_text())
    import mteb
    texts, tasks = [], {}
    for n in STS:
        t = mteb.get_task(n); t.load_data()
        d = t.dataset["test"]
        rows = []
        for r in d:
            a = len(texts); texts.append(PREFIX_SIMILARITY + r["sentence1"])
            b = len(texts); texts.append(PREFIX_SIMILARITY + r["sentence2"])
            rows.append([a, b, float(r["score"])])
        tasks[n] = {"kind": "sts", "rows": rows}
    for n in RET:
        t = mteb.get_task(n); t.load_data()
        split = t.metadata.eval_splits[0]
        s = t.dataset["default"][split]
        docs = {}
        for r in s["corpus"]:
            docs[r["id"]] = len(texts); texts.append(doc_prefix(r.get("title")) + r["text"])
        qs = {}
        for r in s["queries"]:
            if r["id"] in s["relevant_docs"]:
                qs[r["id"]] = len(texts); texts.append(PREFIX_QUERY + r["text"])
        rel = {q: {d: int(v) for d, v in ds.items() if d in docs} for q, ds in s["relevant_docs"].items() if q in qs}
        tasks[n] = {"kind": "ret", "docs": docs, "queries": qs, "rel": rel}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
    ids = [tok(t)["input_ids"] for t in texts]
    out = {"texts": texts, "tasks": tasks, "ids": ids}
    p.write_text(json.dumps(out, ensure_ascii=False))
    return out


def embed_tflite(path, ids, threads=4):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    ins = it.get_input_details(); out = it.get_output_details()
    seq = ins[0]["shape"][-1]
    mask_i = next((i for i, d in enumerate(ins) if "mask" in d["name"].lower()), 1 if len(ins) > 1 else None)
    V = np.zeros((len(ids), 768), np.float32)
    for j, x in enumerate(ids):
        x = x[:seq]
        for i, d in enumerate(ins):
            a = np.zeros((1, seq), dtype=d["dtype"])
            a[0, :len(x)] = 1 if i == mask_i else x
            it.set_tensor(d["index"], a)
        it.invoke()
        o = it.get_tensor(out[0]["index"]).astype(np.float32)
        if o.ndim == 3:
            o = o[0, :len(x)].mean(0)
        v = o.reshape(-1)[:768]
        V[j] = v / np.linalg.norm(v)
        if j % 2000 == 0:
            print(f"  {Path(path).name} {j}/{len(ids)}", flush=True)
    return V


def embed_pt(n, ids):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("google/embeddinggemma-300m", device=dev)
    L = 2048 if n == "full" else int(n)
    V = np.zeros((len(ids), 768), np.float32)
    order = np.argsort([len(x) for x in ids])
    bs = 32
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            xs = [ids[i][:L] for i in idx]
            m = max(len(x) for x in xs)
            inp = torch.zeros((len(xs), m), dtype=torch.long); att = torch.zeros_like(inp)
            for k, x in enumerate(xs):
                inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
            o = st({"input_ids": inp.to(dev), "attention_mask": att.to(dev)})["sentence_embedding"]
            V[idx] = torch.nn.functional.normalize(o.float(), dim=-1).cpu().numpy()
    return V


def name_of(m):
    return m.replace(":", "") if m.startswith("pt:") else Path(m).stem


def cmd_embed(models):
    data = build()
    for m in models:
        f = CACHE / (name_of(m) + ".npy")
        if f.exists():
            print("skip (cached)", f.name); continue
        t0 = time.time()
        V = embed_pt(m[3:], data["ids"]) if m.startswith("pt:") else embed_tflite(m, data["ids"])
        np.save(f, V)
        print(f"done {name_of(m)} in {time.time() - t0:.0f}s", flush=True)


def score_task(V, t):
    from scipy.stats import spearmanr
    if t["kind"] == "sts":
        r = np.array(t["rows"])
        a, b = r[:, 0].astype(int), r[:, 1].astype(int)
        sims = (V[a] * V[b]).sum(1)
        return {"spearman": float(spearmanr(sims, r[:, 2]).correlation)}, (sims, r[:, 2])
    dids = list(t["docs"]); D = V[[t["docs"][d] for d in dids]]
    nd, r10 = [], []
    for q, row in t["queries"].items():
        rel = t["rel"].get(q) or {}
        if not rel: continue
        top = np.argsort(-(D @ V[row]))[:10]
        dcg = sum((2 ** rel.get(dids[k], 0) - 1) / np.log2(i + 2) for i, k in enumerate(top))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
        nd.append(dcg / idcg if idcg else 0)
        hit = sum(1 for k in top if rel.get(dids[k], 0) > 0)
        r10.append(hit / sum(1 for g in rel.values() if g > 0))
    return {"ndcg@10": float(np.mean(nd)), "recall@10": float(np.mean(r10)), "n_q": len(nd)}, np.array(nd)


def boot_delta(kind, x, y, n=1000, seed=0):
    """95% paired-bootstrap interval of x - y, resampling pairs (STS) or queries (retrieval)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    m = len(x[0]) if kind == "sts" else len(x)
    ds = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        if kind == "sts":
            ds.append(spearmanr(x[0][i], x[1][i]).correlation - spearmanr(y[0][i], y[1][i]).correlation)
        else:
            ds.append(x[i].mean() - y[i].mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def seq_of(name):
    return 64 if "seq64" in name else 128 if "seq128" in name else 256


def cmd_score():
    data = build(); ids = data["ids"]
    lens = np.array([len(x) for x in ids])
    print("## Token lengths (prompt, BOS and EOS included)")
    for n, t in data["tasks"].items():
        if t["kind"] == "sts":
            groups = {"sentences": np.array(t["rows"])[:, :2].astype(int).ravel()}
        else:
            groups = {"documents": np.array(list(t["docs"].values())), "queries": np.array(list(t["queries"].values()))}
        for g, rows in groups.items():
            L = lens[rows]
            print(f"  {n} {g} n={len(L)}: median {np.median(L):.0f} p95 {np.percentile(L, 95):.0f} max {L.max()} · "
                  f">64 {np.mean(L > 64) * 100:.1f}% >128 {np.mean(L > 128) * 100:.1f}% >256 {np.mean(L > 256) * 100:.1f}%")
    vecs = {f.stem: np.load(f) for f in sorted(CACHE.glob("*.npy"))}
    results, per = {}, {}
    print("\n## Quality (STS = Spearman, retrieval = nDCG@10)")
    for name, V in vecs.items():
        sc = {n: score_task(V, t) for n, t in data["tasks"].items()}
        results[name] = {n: s[0] for n, s in sc.items()}; per[name] = {n: s[1] for n, s in sc.items()}
        print(f"  {name}: " + " | ".join(f"{n} {list(r.values())[0]:.4f}" for n, r in results[name].items()))
    print("\n## Difference from the original at the same truncation length (95% paired bootstrap)")
    for name in vecs:
        if name.startswith("pt"): continue
        ref = f"pt{seq_of(name)}"
        if ref not in per: continue
        cells = []
        for n, t in data["tasks"].items():
            d = list(results[name][n].values())[0] - list(results[ref][n].values())[0]
            lo, hi = boot_delta(t["kind"], per[name][n], per[ref][n])
            results[name][n]["delta_vs_" + ref] = [d, lo, hi]
            cells.append(f"{n} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {name} - {ref}: " + " | ".join(cells))
    print("\n## Cosine to the original at the same truncation length")
    short = lens <= 64
    cos = {}
    for name, V in vecs.items():
        if name.startswith("pt"): continue
        ref = vecs.get(f"pt{seq_of(name)}")
        if ref is None: continue
        c = (V * ref).sum(1); cs = c[short]
        cos[name] = {"ref": f"pt{seq_of(name)}", "all_mean": float(c.mean()), "all_min": float(c.min()),
                     "short_mean": float(cs.mean()), "short_min": float(cs.min()), "short_p01": float(np.percentile(cs, 1))}
        print(f"  {name}: mean {c.mean():.4f} min {c.min():.4f} · <=64 tokens mean {cs.mean():.4f} p1 {np.percentile(cs, 1):.4f}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "public_bench_pc.json"
    out.write_text(json.dumps({"results": results, "cos_vs_original": cos,
                               "models": {k: k for k in vecs}}, ensure_ascii=False, indent=1))
    print("wrote", out)


def cmd_chunk(L):
    """Split documents into L-token chunks (the "title: ... | text: " prefix repeated on each chunk), embed the
    chunks with the original model, and score a document by its best chunk. The seq64 .tflite matches pt:64 at
    cosine 0.9985, so the original model stands in for it here."""
    from transformers import AutoTokenizer
    data = build(); texts = data["texts"]
    tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
    chunks, owner = [], []
    for n, t in data["tasks"].items():
        if t["kind"] != "ret": continue
        for row in t["docs"].values():
            pre, body = texts[row].split(" | text: ", 1)
            pids = tok(pre + " | text: ")["input_ids"]
            if pids[-1] == tok.eos_token_id: pids = pids[:-1]
            bids = tok(body, add_special_tokens=False)["input_ids"]
            w = L - len(pids)
            for s in range(0, max(len(bids), 1), w):
                chunks.append(pids + bids[s:s + w]); owner.append(row)
    print(f"{len(chunks)} chunks for {len(set(owner))} documents")
    CV = embed_pt(str(L), chunks)
    Q = np.load(CACHE / "ptfull.npy")  # queries are all under 64 tokens, so any pt:N gives the same vectors
    owner = np.array(owner)
    for n, t in data["tasks"].items():
        if t["kind"] != "ret": continue
        dids = list(t["docs"]); rowidx = {t["docs"][d]: i for i, d in enumerate(dids)}
        sel = np.isin(owner, list(rowidx)); cv = CV[sel]; ow = np.array([rowidx[o] for o in owner[sel]])
        nd = []
        for q, qrow in t["queries"].items():
            rel = t["rel"].get(q) or {}
            if not rel: continue
            s = np.full(len(dids), -9.0); np.maximum.at(s, ow, cv @ Q[qrow])
            top = np.argsort(-s)[:10]
            dcg = sum((2 ** rel.get(dids[k], 0) - 1) / np.log2(i + 2) for i, k in enumerate(top))
            idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
            nd.append(dcg / idcg if idcg else 0)
        print(f"  {n} chunk{L} max: nDCG@10 {np.mean(nd):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        cmd_embed(sys.argv[2:])
    elif sys.argv[1] == "chunk":
        cmd_chunk(int(sys.argv[2]))
    else:
        cmd_score()
