"""Render the demo dataset: ~80 fictional photos / screenshots + facts.json + questions.json (+ contact sheets).

No model runs here. Deterministic: every image uses a seed derived from its id, so re-running gives the same files.
Fonts: Noto Sans KR (OFL) for everything that may contain Hangul, Consolas for receipts (English only),
Georgia for one bistro menu. Set FONT_DIR to a folder holding NotoSansKR-VF.ttf, consola(b).ttf and georgia(b).ttf
(default: /mnt/c/Windows/Fonts, i.e. WSL). Consolas and Georgia ship with Windows, so on other systems the images will
not come out identical — the rendered images are committed in data/images/ for that reason.

    python gen_dataset.py            # renders data/images/*.jpg, data/facts.json, data/questions.json, data/questions.md
    python gen_dataset.py --sheets   # also data/contact_sheet_*.jpg for review
"""

import argparse, hashlib, json, math, os, random, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec  # noqa: E402
import questions as qmod  # noqa: E402

OUT = HERE / "data"
IMG = OUT / "images"
FONTS = Path(os.environ.get("FONT_DIR", "/mnt/c/Windows/Fonts"))
NOTO = FONTS / "NotoSansKR-VF.ttf"
MONO = FONTS / "consola.ttf"
MONO_B = FONTS / "consolab.ttf"
SERIF = FONTS / "georgia.ttf"
SERIF_B = FONTS / "georgiab.ttf"


def font(size, weight=400, path=NOTO):
    f = ImageFont.truetype(str(path), size)
    if path == NOTO:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def seed_of(item_id):
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def money(d):
    return f"${d:.2f}"


def receipt_math(r):
    sub = sum(Decimal(p) for _, p in r["items"])
    tax = (sub * Decimal(r["tax_rate"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    tip = Decimal(r["tip"])
    return dict(subtotal=sub, tax=tax, tip=tip, total=sub + tax + tip)


# ── drawing helpers ─────────────────────────────────────────────────────────

def wrap(draw, text, f, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def status_bar(d, W, rng, dark=False):
    c = (240, 240, 240) if dark else (30, 30, 30)
    d.text((40, 22), f"{rng.randint(9, 11)}:{rng.randint(10, 59):02d}", font=font(30, 600), fill=c)
    d.text((W - 170, 22), "5G  " + str(rng.randint(41, 96)) + "%", font=font(28, 500), fill=c)


def phone_canvas(H, rng, bg=(250, 250, 250), dark=False):
    W = 900
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    status_bar(d, W, rng, dark)
    return im, d, W


# ── renderers: each returns (PIL image, text lines in reading order) ─────────

def r_menu_screen(it, rng):
    rows = it["items"]
    H = max(1700, 300 + 78 * len(rows) + 120)
    im, d, W = phone_canvas(H, rng)
    accent = rng.choice([(176, 58, 46), (35, 92, 153), (40, 120, 80), (120, 70, 30)])
    d.rectangle([0, 80, W, 250], fill=accent)
    d.text((44, 105), it["title"], font=font(46, 700), fill=(255, 255, 255))
    d.text((44, 178), it["sub"], font=font(28, 400), fill=(235, 235, 235))
    y = 280
    lines = [it["title"], it["sub"]]
    fn, fp = font(34, 500), font(34, 600)
    for name, price in rows:
        p = price if price in ("포함",) or "/" in price else "$" + price
        d.text((44, y), name, font=fn, fill=(25, 25, 25))
        d.text((W - 44 - d.textlength(p, font=fp), y), p, font=fp, fill=(25, 25, 25))
        d.line([44, y + 62, W - 44, y + 62], fill=(225, 225, 225), width=2)
        lines.append(f"{name} {p}")
        y += 78
    return im, lines


def r_menu_board(it, rng):
    rows = it["items"]
    W, H = 1400, max(900, 330 + 92 * len(rows) + 80)
    bg = rng.choice([(36, 46, 40), (28, 28, 30), (92, 62, 40)])
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    serif = it["title"].startswith("Blue Harbor")
    tf = font(70, 700) if not serif else ImageFont.truetype(str(SERIF_B), 70)
    d.text(((W - d.textlength(it["title"], font=tf)) / 2, 60), it["title"], font=tf, fill=(245, 240, 225))
    sf = font(36, 400)
    d.text(((W - d.textlength(it["sub"], font=sf)) / 2, 160), it["sub"], font=sf, fill=(210, 205, 190))
    y = 260
    lines = [it["title"], it["sub"]]
    fn, fp = font(46, 500), font(46, 700)
    for name, price in rows:
        p = "$" + price if "/" not in price else price
        d.text((120, y), name, font=fn, fill=(240, 236, 220))
        d.text((W - 120 - d.textlength(p, font=fp), y), p, font=fp, fill=(250, 220, 120))
        lines.append(f"{name} {p}")
        y += 92
    return im, lines


SIGN_COLORS = {"blue": ((20, 70, 160), (255, 255, 255)), "green": ((20, 110, 60), (255, 255, 255)),
               "white": ((245, 245, 240), (20, 20, 20)), "black": ((20, 20, 22), (255, 255, 255)),
               "yellow": ((245, 205, 40), (20, 20, 20)), "red": ((180, 25, 30), (255, 255, 255))}


def r_sign(it, rng):
    bgc, fg = SIGN_COLORS[it["color"]]
    W = 1300
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    blocks = []
    for i, t in enumerate(it["lines"]):
        f = font(74 if i == 0 else 52, 800 if i == 0 else 600)
        for ln in wrap(probe, t, f, W - 200):
            blocks.append((ln, f))
    H = 160 + sum(f.size + 34 for _, f in blocks)
    sign = Image.new("RGB", (W, H), bgc)
    d = ImageDraw.Draw(sign)
    d.rectangle([18, 18, W - 18, H - 18], outline=fg, width=8)
    y = 80
    for ln, f in blocks:
        d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=fg)
        y += f.size + 34
    wall = Image.new("RGB", (W + 300, H + 300), tuple(rng.randint(120, 175) for _ in range(3)))
    wall.paste(sign, (150, 150))
    return wall, list(it["lines"])


def r_receipt(it, rng):
    m = receipt_math(it)
    f, fb = ImageFont.truetype(str(MONO), 30), ImageFont.truetype(str(MONO_B), 34)
    rows = [(it["shop"], fb, "c"), ("123 Fictional St · Demo City", f, "c"), (it["date"], f, "c"),
            (f"Server: {it['server']}", f, "c"), ("-" * 30, f, "c")]
    for name, price in it["items"]:
        rows.append((f"{name:<20}{money(Decimal(price)):>10}", f, "l"))
    rows += [("-" * 30, f, "c"), (f"{'Subtotal':<20}{money(m['subtotal']):>10}", f, "l"),
             (f"{'Tax':<20}{money(m['tax']):>10}", f, "l"), (f"{'Tip':<20}{money(m['tip']):>10}", f, "l"),
             (f"{'TOTAL':<20}{money(m['total']):>10}", fb, "l"), ("", f, "c"), ("Thank you!", f, "c")]
    W, H = 640, 120 + 48 * len(rows)
    paper = Image.new("RGB", (W, H), (250, 249, 244))
    d = ImageDraw.Draw(paper)
    y = 60
    for t, ff, al in rows:
        x = (W - d.textlength(t, font=ff)) / 2 if al == "c" else 40
        d.text((x, y), t, font=ff, fill=(35, 35, 35))
        y += 48
    table = Image.new("RGB", (W + 420, H + 260), rng.choice([(120, 82, 52), (60, 60, 64), (170, 150, 120)]))
    table.paste(paper, (210, 130))
    lines = [t for t, _, _ in rows if t and not t.startswith("-")]
    return table, lines


def r_notice_paper(it, rng):
    W, H = 1100, 1350
    paper = Image.new("RGB", (W, H), rng.choice([(252, 251, 246), (253, 246, 205)]))
    d = ImageDraw.Draw(paper)
    y = 170
    for i, t in enumerate(it["lines"]):
        f = font(78 if i == 0 else 56, 800 if i == 0 else 500)
        for ln in wrap(d, t, f, W - 160):
            d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=(30, 30, 30))
            y += f.size + 40
        if i == 0:
            y += 40
    wall = Image.new("RGB", (W + 360, H + 300), tuple(rng.randint(150, 205) for _ in range(3)))
    wall.paste(paper, (180, 150))
    dd = ImageDraw.Draw(wall)
    for x, yy in [(180, 150), (180 + W, 150)]:
        dd.rectangle([x - 50, yy - 18, x + 50, yy + 18], fill=(230, 225, 200))
    return wall, list(it["lines"])


def r_notice_screen(it, rng):
    im, d, W = phone_canvas(1600, rng, bg=(245, 245, 247))
    d.rounded_rectangle([40, 140, W - 40, 1400], radius=36, fill=(255, 255, 255))
    d.ellipse([80, 180, 160, 260], fill=rng.choice([(230, 120, 90), (90, 140, 220), (120, 180, 120)]))
    d.text((180, 195), it["lines"][0], font=font(36, 700), fill=(20, 20, 20))
    y = 330
    for t in it["lines"][1:]:
        f = font(48, 600)
        for ln in wrap(d, t, f, W - 200):
            d.text((90, y), ln, font=f, fill=(30, 30, 30))
            y += 74
        y += 16
    return im, list(it["lines"])


def r_booking(it, rng):
    im, d, W = phone_canvas(1700, rng)
    d.rectangle([0, 80, W, 180], fill=(255, 255, 255))
    d.text((40, 105), "‹  Details", font=font(34, 600), fill=(40, 90, 200))
    d.ellipse([W / 2 - 60, 240, W / 2 + 60, 360], fill=(40, 160, 90))
    d.line([W / 2 - 30, 300, W / 2 - 8, 325, W / 2 + 35, 272], fill=(255, 255, 255), width=12)
    head = it["lines"][0]
    hf = font(48, 700)
    d.text(((W - d.textlength(head, font=hf)) / 2, 400), head, font=hf, fill=(20, 20, 20))
    f0 = font(42, 500)
    rows = [ln for t in it["lines"][1:] for ln in wrap(d, t, f0, W - 200)]
    d.rounded_rectangle([50, 520, W - 50, 560 + 100 * len(rows) + 40], radius=30, fill=(255, 255, 255),
                        outline=(225, 225, 230), width=3)
    y = 560
    for t in rows:
        f = font(42, 600 if y == 560 else 500)
        d.text((100, y), t, font=f, fill=(30, 30, 30))
        y += 100
    return im, list(it["lines"])


def r_widget(it, rng):
    im, d, W = phone_canvas(1500, rng, bg=(30, 40, 60), dark=True)
    d.rounded_rectangle([60, 220, W - 60, 300 + 90 * len(it["lines"])], radius=40, fill=(245, 245, 250))
    y = 260
    for i, t in enumerate(it["lines"]):
        d.text((110, y), t, font=font(48 if i == 0 else 40, 700 if i == 0 else 500), fill=(25, 25, 35))
        y += 90
    return im, list(it["lines"])


def r_chat(it, rng):
    im, d, W = phone_canvas(1500, rng, bg=(178, 199, 218))
    y = 200
    for t in it["lines"]:
        me = t.startswith("수아")
        f = font(40, 500)
        tw = d.textlength(t, font=f)
        x0 = W - 60 - tw - 50 if me else 60
        d.rounded_rectangle([x0, y, x0 + tw + 50, y + 90], radius=28, fill=(255, 235, 60) if me else (255, 255, 255))
        d.text((x0 + 25, y + 20), t, font=f, fill=(20, 20, 20))
        y += 130
    return im, list(it["lines"])


def r_no_text(it, rng):
    W, H = 1400, 1050
    im = Image.new("RGB", (W, H), tuple(rng.randint(60, 200) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(14):
        cx, cy, r = rng.randint(0, W), rng.randint(0, H), rng.randint(60, 320)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(rng.randint(40, 240) for _ in range(3)))
    return im.filter(ImageFilter.GaussianBlur(18)), []


RENDER = {"menu_screen": r_menu_screen, "menu_board": r_menu_board, "sign": r_sign, "receipt": r_receipt,
          "notice_paper": r_notice_paper, "notice_screen": r_notice_screen, "booking": r_booking,
          "widget": r_widget, "chat": r_chat, "no_text": r_no_text}


# ── camera-like distortion ──────────────────────────────────────────────────

def _coeffs(src, dst):
    import numpy as np
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def photoify(im, rng):
    import numpy as np
    W, H = im.size
    j = lambda s: rng.uniform(0.02, 0.07) * s
    dst = [(j(W), j(H)), (W - j(W), j(H)), (W - j(W), H - j(H)), (j(W), H - j(H))]
    im = im.transform((W, H), Image.PERSPECTIVE, _coeffs([(0, 0), (W, 0), (W, H), (0, H)], dst),
                      Image.BICUBIC, fillcolor=tuple(rng.randint(40, 90) for _ in range(3)))
    im = im.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False, fillcolor=(60, 60, 60))
    a = np.asarray(im).astype(np.float32)
    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.95, 1.12), W)[None, :, None]
    gy = np.linspace(rng.uniform(0.85, 1.05), rng.uniform(0.8, 1.05), H)[:, None, None]
    a = a * gx * gy * rng.uniform(0.92, 1.05)
    a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.3)))
    if max(im.size) > 1600:
        im.thumbnail((1600, 1600), Image.LANCZOS)
    return im


# ── main ────────────────────────────────────────────────────────────────────

def kind_of(item_id):
    return {"m": "menu", "p": "parking", "r": "receipt", "n": "notice", "b": "booking", "x": "distractor"}[item_id[0]]


def build():
    IMG.mkdir(parents=True, exist_ok=True)
    facts, receipts = [], {}
    for it in spec.ALL:
        rng = random.Random(seed_of(it["id"]))
        im, lines = RENDER[it["style"]](it, rng)
        if it["photo"]:
            im = photoify(im, rng)
        q = 82 if it["photo"] else 88
        im.save(IMG / f"{it['id']}.jpg", "JPEG", quality=q, optimize=True)
        row = dict(id=it["id"], kind=kind_of(it["id"]), style=it["style"], photo=it["photo"], lang=it["lang"],
                   size=list(im.size), lines=lines)
        if it["style"] == "receipt":
            m = receipt_math(it)
            row["math"] = {k: money(v) for k, v in m.items()}
            receipts[it["id"]] = row["math"]
        facts.append(row)
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")

    qs = []
    for q in qmod.Q:
        q = dict(q)
        if q["answer"].startswith("@"):
            rid, field = q["answer"][1:].split(".")
            q["answer"] = receipts[rid][field]
            q["accept"] = [receipts[rid][field].lstrip("$")]
        qs.append(q)
    (OUT / "questions.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| id | type | q | gold | answer |", "|---|---|---|---|---|"]
    md += [f"| {q['id']} | {q['type']} | {q['q']} | {', '.join(q['gold']) or '—'} | {q['answer']} |" for q in qs]
    (OUT / "questions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return facts, qs


def sheets(facts, per=20, cols=5, tw=300, th=420):
    for s in range(0, len(facts), per):
        chunk = facts[s:s + per]
        rows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * (tw + 20) + 20, rows * (th + 70) + 20), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            im = Image.open(IMG / f"{f['id']}.jpg")
            im = ImageOps.contain(im, (tw, th))
            x, y = 20 + (i % cols) * (tw + 20), 20 + (i // cols) * (th + 70)
            sheet.paste(im, (x + (tw - im.width) // 2, y))
            d.text((x, y + th + 8), f"{f['id']} · {f['kind']} · {'photo' if f['photo'] else 'screen'}",
                   font=font(22, 600), fill=(20, 20, 20))
        sheet.save(OUT / f"contact_sheet_{s // per + 1}.jpg", "JPEG", quality=85)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sheets", action="store_true")
    a = ap.parse_args()
    facts, qs = build()
    print(f"{len(facts)} images, {len(qs)} questions → {OUT}")
    if a.sheets:
        sheets(facts)
        print("contact sheets written")
