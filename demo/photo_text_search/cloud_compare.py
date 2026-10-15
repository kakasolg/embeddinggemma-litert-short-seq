"""Step 4: the same 80 photos and 60 questions on Gemini Flash, next to the on-device v5 runs (README "4단계 설계").

Arms:
  A  images  – the 80 JPEGs uploaded once with the Files API; every question sends all 80 + the question   (Flash)
  B  text    – the phone's OCR rows (results/ocr_pixel.json) for all 80 photos + the question             (Flash)
  C  images  – same as A on the cheapest model                                                             (Flash-Lite)
Only synthetic demo data is sent. The key comes from the environment variable GEMINI_API_KEY and is sent as a header,
never in a URL or printed.

Usage (from the repo root):
  python demo/photo_text_search/cloud_compare.py models          # list flash models
  python demo/photo_text_search/cloud_compare.py run A|B|C [--model NAME] [--limit N]
  python demo/photo_text_search/cloud_compare.py score A|B|C
Writes demo/photo_text_search/results/cloud_<arm>.json and score_cloud_<arm>.json.
"""

import json, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
API = "https://generativelanguage.googleapis.com"
RES = HERE / "results"
SEED = 20261015

facts = {f["id"]: f for f in json.loads((HERE / "data/facts.json").read_text(encoding="utf-8"))}
qs = json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))
ocr = {x["id"]: x for x in json.loads((RES / "ocr_pixel.json").read_text(encoding="utf-8"))["items"]}
ids = sorted(facts)
order = ids[:]
random.Random(SEED).shuffle(order)
label_of = {pid: f"사진{i + 1:02d}" for i, pid in enumerate(order)}
id_of = {v: k for k, v in label_of.items()}


def key():
    import os
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        sys.exit("Set GEMINI_API_KEY to your own Gemini API key.")
    return k


def call(method, url, body=None, headers=None, raw=False, tries=6):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    h = {"x-goog-api-key": key(), **({"Content-Type": "application/json"} if not raw and data else {}), **(headers or {})}
    for t in range(tries):
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = r.read()
                return out, dict(r.headers), len(data or b"")
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if e.code in (429, 500, 503) and t < tries - 1:
                time.sleep(min(60, 5 * 2 ** t))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg!r}") from None


def list_models():
    out, _, _ = call("GET", f"{API}/v1beta/models?pageSize=200")
    return [m for m in json.loads(out)["models"] if "flash" in m["name"]
            and "generateContent" in m.get("supportedGenerationMethods", [])]


def upload_all():
    """Files API, resumable upload. Reuses uploads younger than 40 h (files expire after 48 h)."""
    cache = RES / "cloud_files.json"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if time.time() - c["t"] < 40 * 3600 and set(c["files"]) == set(ids):
            return c
    files, sent, t0 = {}, 0, time.time()
    for pid in ids:
        blob = (HERE / "data/images" / f"{pid}.jpg").read_bytes()
        _, hdr, n1 = call("POST", f"{API}/upload/v1beta/files", {"file": {"display_name": f"demo-{pid}"}}, {
            "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(blob)), "X-Goog-Upload-Header-Content-Type": "image/jpeg"})
        up = next(v for k, v in hdr.items() if k.lower() == "x-goog-upload-url")
        out, _, n2 = call("POST", up, blob, {"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize",
                                             "Content-Length": str(len(blob))}, raw=True)
        files[pid] = json.loads(out)["file"]["uri"]
        sent += n1 + n2
    c = {"t": time.time(), "files": files, "upload_bytes": sent, "upload_s": round(time.time() - t0, 1),
         "image_bytes": sum((HERE / "data/images" / f"{p}.jpg").stat().st_size for p in ids)}
    cache.write_text(json.dumps(c, indent=1), encoding="utf-8")
    return c


RULES = """위 사진의 글자만 근거로 답해라. 사진에 없는 내용은 지어내지 마라.
가게나 장소는 질문이 어떻게 부르는지에 따라 판단한다.
- 질문에 가게 이름이 있으면 그 이름의 가게 사진만 친다. 영어 표기와 한글 표기가 같은 이름이면 같은 가게다
  (예: 'Haneul' = '하늘'). 이름이 비슷해도 다른 가게는 다른 가게다(예: '미래제과'와 '미래분식').
- 질문이 이름 대신 종류나 특징으로 부르면(예: '아이스크림 가게', 'the burger place') 그 종류·특징에 맞는 가게의 사진을 친다.
이 기준에 맞는 사진만 근거로 써라.
맞는 사진이 하나도 없으면 근거를 '없음'으로 적어라.
답이 '안 된다', '불가', 'No', 'closed'처럼 부정이어도 사진에 그 내용이 있으면 근거에 그 사진을 적어라.
가격·시간·번호는 사진에 적힌 그대로 옮겨라.

아래 두 줄 형식으로만 답해라.
근거: 사진NN (여러 장이면 쉼표로)
답변: <질문과 같은 언어로 한 문장>"""


def request_body(arm, question, files):
    parts = [{"text": "아래는 내 폰에 있는 사진들이다." + (" 사진의 글자는 폰에서 읽은 것이라 일부 틀린 글자가 있을 수 있다." if arm == "B" else "")}]
    for pid in order:
        if arm == "B":
            parts.append({"text": f"[{label_of[pid]}]\n" + "\n".join(ocr[pid]["rows"])})
        else:
            parts.append({"text": f"[{label_of[pid]}]"})
            parts.append({"file_data": {"mime_type": "image/jpeg", "file_uri": files[pid]}})
    parts.append({"text": f"질문: {question}\n\n{RULES}"})
    return {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.0}}


def parse(raw):
    cite = re.search(r"근거\s*[:：]\s*(.+)", raw)
    ans = re.search(r"답변\s*[:：]\s*(.+)", raw, re.S)
    if not cite:
        cited = None
    elif "없음" in cite[1] or "없다" in cite[1]:
        cited = []
    else:
        cited = [id_of[f"사진{n}"] for n in re.findall(r"사진\s*(\d{2})", cite[1]) if f"사진{n}" in id_of] or None
    return cited, (ans[1].strip() if ans else raw.strip())


def run(arm, model, limit):
    files = upload_all() if arm in "AC" else None
    out_path = RES / f"cloud_{arm}.json"
    done = {x["id"]: x for x in json.loads(out_path.read_text(encoding="utf-8"))["items"]} if out_path.exists() else {}
    items = []
    for q in qs[:limit]:
        if q["id"] in done:
            items.append(done[q["id"]]); continue
        body = request_body(arm, q["q"], files["files"] if files else None)
        t0 = time.time()
        out, _, sent = call("POST", f"{API}/v1beta/{model}:generateContent", body)
        ms = (time.time() - t0) * 1000
        r = json.loads(out)
        raw = "".join(p.get("text", "") for c in r.get("candidates", [])[:1] for p in c.get("content", {}).get("parts", []))
        cited, answer = parse(raw)
        items.append({"id": q["id"], "q": q["q"], "raw": raw, "answer": answer, "cited": cited, "latency_ms": round(ms, 1),
                      "req_bytes": sent, "resp_bytes": len(out), "usage": r.get("usageMetadata", {}),
                      "model_version": r.get("modelVersion")})
        print(q["id"], round(ms), cited, answer[:60])
        out_path.write_text(json.dumps({"arm": arm, "model": model, "files": {k: v for k, v in (files or {}).items() if k != "files"},
                                        "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")


def score(arm):
    d = json.loads((RES / f"cloud_{arm}.json").read_text(encoding="utf-8"))
    norm = lambda s: re.sub(r"\s+", "", s).lower()
    loose = lambda s: norm(s).replace("$", "").replace(",", "")
    qd = {q["id"]: q for q in qs}
    rows = []
    for it in d["items"]:
        q = qd[it["id"]]; gold = set(q["gold"]); cited = None if it["cited"] is None else set(it["cited"])
        abst = cited == set()
        if gold:
            has = any(norm(a) in norm(it["answer"]) or loose(a) in loose(it["answer"]) for a in q["accept"])
            o = ("abstained" if abst else "unparsed" if cited is None else "wrong_cite" if not cited & gold
                 else "correct" if has else "cited_gold_answer_mismatch")
        else:
            o = "abstained_ok" if abst else "false_answer"
        rows.append({"id": it["id"], "type": q["type"], "outcome": o, "answer": it["answer"], "accept": q["accept"],
                     "latency_ms": it["latency_ms"], "usage": it["usage"]})
    ans = [r for r in rows if r["type"] != "none"]; none = [r for r in rows if r["type"] == "none"]
    lat = sorted(r["latency_ms"] for r in rows)
    pct = lambda xs, p: round(xs[min(len(xs) - 1, int(len(xs) * p))], 1)
    u = lambda k: sum(r["usage"].get(k, 0) for r in rows)
    it = d["items"]
    s = {"arm": arm, "model": d["model"], "model_version": it[0].get("model_version") if it else None, "n": len(rows),
         "answers": {o: sum(r["outcome"] == o for r in ans) for o in
                     ("correct", "cited_gold_answer_mismatch", "wrong_cite", "abstained", "unparsed")} | {"n": len(ans)},
         "answers_by_type": {t: f'{sum(r["outcome"] == "correct" for r in ans if r["type"] == t)}/{sum(r["type"] == t for r in ans)}'
                             for t in sorted({r["type"] for r in ans})},
         "none": {"abstained_ok": sum(r["outcome"] == "abstained_ok" for r in none), "n": len(none)},
         "latency_ms": {"p50": pct(lat, .5), "p90": pct(lat, .9), "max": lat[-1]} if lat else {},
         "tokens": {"prompt": u("promptTokenCount"), "output": u("candidatesTokenCount"), "thoughts": u("thoughtsTokenCount"),
                    "per_question_prompt": round(u("promptTokenCount") / max(1, len(rows)))},
         "bytes": {"upload_once": d["files"].get("upload_bytes", 0), "upload_s": d["files"].get("upload_s"),
                   "per_question_request_p50": pct(sorted(x["req_bytes"] for x in it), .5) if it else 0,
                   "request_total": sum(x["req_bytes"] for x in it), "response_total": sum(x["resp_bytes"] for x in it)}}
    print(json.dumps(s, ensure_ascii=False, indent=1))
    for r in rows:
        if r["outcome"] not in ("correct", "abstained_ok"):
            print(f'{r["id"]:4} {r["type"]:12} {r["outcome"]:27} ans={r["answer"][:90]!r} accept={r["accept"]}')
    (RES / f"score_cloud_{arm}.json").write_text(json.dumps({"summary": s, "questions": rows}, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "models":
        for m in list_models():
            print(m["name"], m.get("inputTokenLimit"), m.get("displayName"))
    elif cmd == "run":
        arm = sys.argv[2]
        model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else \
            ("models/gemini-flash-lite-latest" if arm == "C" else "models/gemini-flash-latest")
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(qs)
        run(arm, model, limit)
    elif cmd == "score":
        score(sys.argv[2])
