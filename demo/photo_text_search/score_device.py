"""Step 3: score a phone run of the offline demo (PhotoTextSearchDemoViewModel.runBatch → run_<tag>.json).

Answerable questions (55):
  retrieval  – gold photo rank in the phone's top 3 (compared with the PC baseline's chunk64 top 3)
  correct    – the model cited a gold photo AND its answer line contains an accepted string
               (validate_dataset.py normalization: whitespace removed, lowercased; also with "$" and "," dropped)
  wrong_cite – cited only non-gold photos;  abstained – gate said no or "근거: 없음";  unparsed – no 근거 line
None questions (5): abstained = success.
Also: network bytes during the run, airplane mode, latency percentiles, load times, memory, and battery energy
from the unplugged samples only (USB-powered samples are reported but not used).
Usage: python score_device.py results/run_<tag>.json
"""

import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
run_path = Path(sys.argv[1])
run = json.loads(run_path.read_text(encoding="utf-8"))
qs = {q["id"]: q for q in json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))}
pc = {p["id"]: p for p in json.loads((HERE / "results/pc_baseline.json").read_text(encoding="utf-8"))
      ["results"]["chunk64"]["questions"]}
norm = lambda s: re.sub(r"\s+", "", s).lower()
loose = lambda s: norm(s).replace("$", "").replace(",", "")


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1) if xs else None


rows, out = [], {"run": run_path.name, "device": run["device"]}
for it in run["items"]:
    q = qs[it["id"]]
    gold = set(q["gold"])
    cited = None if it["cited"] is None else set(it["cited"])
    abstained = (not it["gate_yes"]) or cited == set()
    r = {"id": it["id"], "type": q["type"], "abstained": abstained,
         "same_top3_as_pc": it["top3"] == pc[it["id"]]["top3"],
         "total_ms": it["query_embed_ms"] + it["score_ms"] + it["gate_ms"] + it["answer_ms"] + it.get("verify_ms", 0.0)}
    if gold:
        ranks = [it["top3"].index(g) + 1 for g in gold if g in it["top3"]]
        r["rank"] = min(ranks) if ranks else None
        has = any(norm(a) in norm(it["answer"]) or loose(a) in loose(it["answer"]) for a in q["accept"])
        if abstained:
            r["outcome"] = "abstained"
        elif cited is None:
            r["outcome"] = "unparsed"
        elif not (cited & gold):
            r["outcome"] = "wrong_cite"
        else:
            r["outcome"] = "correct" if has else "cited_gold_answer_mismatch"
        r["answer_has_accept"] = has
    else:
        r["outcome"] = "abstained_ok" if abstained else "false_answer"
    r["answer"] = it["answer"] if not abstained else "(기권)"
    r["accept"] = q["accept"]
    rows.append(r)

ans = [r for r in rows if r["type"] != "none"]
none = [r for r in rows if r["type"] == "none"]
count = lambda rs, o: sum(r["outcome"] == o for r in rs)
out["retrieval"] = {"r@1": f'{sum(r["rank"] == 1 for r in ans)}/{len(ans)}',
                    "r@3": f'{sum(r["rank"] is not None for r in ans)}/{len(ans)}',
                    "top3_identical_to_pc": f'{sum(r["same_top3_as_pc"] for r in rows)}/{len(rows)}'}
out["answers"] = {o: count(ans, o) for o in ("correct", "cited_gold_answer_mismatch", "wrong_cite", "abstained", "unparsed")}
out["answers"]["n"] = len(ans)
out["answers_by_type"] = {t: f'{sum(r["outcome"] == "correct" for r in ans if r["type"] == t)}/'
                             f'{sum(r["type"] == t for r in ans)}' for t in sorted({r["type"] for r in ans})}
out["none"] = {"abstained_ok": count(none, "abstained_ok"), "false_answer": count(none, "false_answer"), "n": len(none)}
it = run["items"]
out["latency_ms"] = {
    "query_embed": {"p50": pct([x["query_embed_ms"] for x in it], .5), "p90": pct([x["query_embed_ms"] for x in it], .9)},
    "score_80_photos": {"p50": pct([x["score_ms"] for x in it], .5)},
    "gate": {"p50": pct([x["gate_ms"] for x in it], .5), "p90": pct([x["gate_ms"] for x in it], .9)},
    "answer": {"p50": pct([x["answer_ms"] for x in it if x["gate_yes"]], .5),
               "p90": pct([x["answer_ms"] for x in it if x["gate_yes"]], .9)},
    "store_check": {"p50": pct([x["verify_ms"] for x in it if x.get("verify_ms")], .5), "n": sum(1 for x in it if x.get("verify_ms"))},
    "first_token": {"p50": pct([x["first_token_ms"] for x in it if x["first_token_ms"]], .5)},
    "question_total": {"p50": pct([r["total_ms"] for r in rows], .5), "p90": pct([r["total_ms"] for r in rows], .9),
                       "max": pct([r["total_ms"] for r in rows], 1.0)},
}
if "index" in run:
    idx = run["index"]
    out["index"] = {"photos": len(idx), "ocr_ms_sum": round(sum(x["ocr_ms"] for x in idx)),
                    "embed_ms_sum": round(sum(x["embed_ms"] for x in idx)), "chunks": sum(x["chunks"] for x in idx),
                    "ocr_ms_p50": pct([x["ocr_ms"] for x in idx], .5), "embed_ms_p50": pct([x["embed_ms"] for x in idx], .5)}
out["loads_ms"] = {"embedder": round(run["embedder_load_ms"]), "llm": round(run["llm_load_ms"]),
                   "pss_mb_both_loaded": round(run["pss_kb_both_loaded"] / 1024)}
s, e = run["offline_start"], run["offline_end"]
out["network"] = {"airplane_mode_start": s["airplane_mode"], "airplane_mode_end": e["airplane_mode"],
                  "active_network_start": s["active_network"], "active_network_end": e["active_network"],
                  "app_tx_bytes": e["uid_tx_bytes"] - s["uid_tx_bytes"], "app_rx_bytes": e["uid_rx_bytes"] - s["uid_rx_bytes"]}
b = run["battery"]
unplugged = [x for x in b if x["plugged"] == 0]
energy = {"samples": len(b), "unplugged_samples": len(unplugged), "thermal_max": max(x["thermal"] for x in b),
          "level_start": b[0]["level"], "level_end": b[-1]["level"], "total_s": round(run["total_ms"] / 1000)}
if len(unplugged) == len(b) and len(b) > 2:
    j = sum(abs(x["current_ua"]) / 1e6 * x["voltage_mv"] / 1e3 * 1.0 for x in b)
    energy.update({"energy_j_approx": round(j), "mwh_approx": round(j / 3.6, 1),
                   "charge_used_mah": round((b[0]["charge_uah"] - b[-1]["charge_uah"]) / 1000, 1),
                   "mean_power_w": round(j / len(b), 2)})
else:
    energy["note"] = "USB connected for (part of) the run — battery numbers not valid"
out["battery"] = energy
print(json.dumps(out, ensure_ascii=False, indent=1))
for r in rows:
    if r["outcome"] not in ("correct", "abstained_ok"):
        print(f'{r["id"]:4} {r["type"]:12} {r["outcome"]:27} rank={r.get("rank")} ans={r["answer"][:90]!r} accept={r["accept"]}')
(HERE / "results" / f"score_{run_path.stem}.json").write_text(json.dumps({"summary": out, "questions": rows},
                                                                          ensure_ascii=False, indent=1), encoding="utf-8")
