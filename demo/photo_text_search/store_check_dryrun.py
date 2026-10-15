"""v5 dry run: apply the code store check (same rules as PhotoTextSearchDemo.StoreNames) to saved phone runs.

Usage: python store_check_dryrun.py results/run_tap_v3.json [...]
Prints the store groups found in the 80 photos' OCR rows and, per run, every citation the check would drop.
"""

import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ocr = {x["id"]: x for x in json.loads((HERE / "results/ocr_pixel.json").read_text(encoding="utf-8"))["items"]}
qs = {q["id"]: q for q in json.loads((HERE / "data/questions.json").read_text(encoding="utf-8"))}
STATUS = re.compile(r"^\s*\d{1,2}\s*:\s*\d{2}\b|\b5G\b")
NAV = re.compile(r"^[«<]?\s*Details$")
norm = lambda s: re.sub(r"[^a-z0-9가-힣]", "", s.lower())


def name_row(rows):
    """First content row; on an app screen (a "Details" nav row) the row right after the nav is the screen title, so skip it."""
    content = [r for r in rows if not STATUS.search(r) and re.search(r"[A-Za-z가-힣]", r)]
    nav = next((i for i, r in enumerate(content) if NAV.match(r.strip())), None)
    if nav is not None:
        content = content[nav + 2:]
    return content[0] if content else None


def aliases(row):
    latin = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", row)]
    hangul = re.findall(r"[가-힣]+", row)
    out = set()
    if len(latin) >= 2:
        out.add(latin[0] + latin[1])
    elif len(latin) == 1 and len(latin[0]) >= 5 and re.search(r"[a-z]", latin[0]):
        out.add(latin[0])
    if hangul and len(hangul[0]) >= 2:
        out.add(hangul[0])
    return out


# union-find over aliases: a bilingual name row links its Latin and Hangul keys into one store
parent = {}
find = lambda a: a if parent.setdefault(a, a) == a else find(parent[a])
own = {}
for pid, it in ocr.items():
    row = name_row(it["rows"])
    al = aliases(row) if row else set()
    own[pid] = al
    al = list(al)
    for a in al:
        find(a)
    for a in al[1:]:
        parent[find(a)] = find(al[0])
groups = {}
for a in parent:
    groups.setdefault(find(a), set()).add(a)
group_of = {a: find(a) for a in parent}


def keep(question, pid):
    q = norm(question)
    mentioned = {g for g, al in groups.items() if any(a in q for a in al)}
    if not mentioned:
        return True, "질문에 아는 가게 없음"
    text = norm(" ".join(ocr[pid]["rows"]))
    if any(a in text for g in mentioned for a in groups[g]):
        return True, "사진에 그 가게 이름 있음"
    photo_groups = {group_of[a] for a in own[pid]}
    if not photo_groups or photo_groups & mentioned:
        return True, "사진 가게 모름"
    return False, f"질문 {sorted(a for g in mentioned for a in groups[g])} / 사진 {sorted(own[pid])}"


if __name__ == "__main__":
    print(len(groups), "store groups:", sorted(sorted(v) for v in groups.values()))
    for f in sys.argv[1:]:
        items = json.loads(Path(f).read_text(encoding="utf-8"))["items"]
        print("==", f)
        for it in items:
            for pid in it["cited"] or []:
                ok, why = keep(it["q"], pid)
                if not ok:
                    q = qs[it["id"]]
                    tag = "정답 사진" if pid in q["gold"] else ("답 없음 질문" if not q["gold"] else "오답 사진")
                    print(f"  drop {it['id']} {pid} ({tag}) — {why}")
