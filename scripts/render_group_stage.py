"""Render the WC2026 group stage (12 group cards, 3-column grid) as a dark-themed PNG.

Reads the /api/wc2026/schedule payload and draws a supersampled image, then downscales
with LANCZOS for crisp anti-aliasing. Mirrors the web Schedule group-card design.
"""
import json
import sys
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1] if len(sys.argv) > 1 else "sched.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"web\public\images\group-stage-wc26.png"

data = json.load(open(SRC, encoding="utf-8"))
groups = data["group_stage"]
n_matches = sum(len(g["matches"]) for g in groups)


def fix(txt):
    """Undo the double-encoding seen in the 'window' field (latin-1 over utf-8)."""
    try:
        return txt.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return txt


# --- layout (base units; multiplied by SS for supersampling) -------------------
SS = 2
COLS = 3
CARD_W = 600
CARD_H = 300
CARD_GAP = 26
MARGIN = 30
TOP = 74
BOTTOM = 28

ROWS = (len(groups) + COLS - 1) // COLS
W = MARGIN * 2 + COLS * CARD_W + (COLS - 1) * CARD_GAP
H = TOP + ROWS * CARD_H + (ROWS - 1) * CARD_GAP + BOTTOM

# --- colours -------------------------------------------------------------------
BG = (8, 12, 11)
CARD_BG = (15, 21, 19)
CARD_BORDER = (32, 42, 39)
BADGE = (26, 160, 106)
GREEN = (52, 227, 155)
WHITE = (226, 232, 230)
GRAY = (120, 134, 130)
GRAY_DIM = (90, 102, 98)
BLACK = (6, 10, 9)


def s(v):
    return int(round(v * SS))


img = Image.new("RGB", (s(W), s(H)), BG)
d = ImageDraw.Draw(img)

FONTS = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(rf"{FONTS}\{name}", s(size))


f_title = font("arialbd.ttf", 20)
f_group = font("arialbd.ttf", 16)
f_badge = font("arialbd.ttf", 15)
f_md = font("arialbd.ttf", 9)
f_team = font("arial.ttf", 13)
f_teamb = font("arialbd.ttf", 13)
f_score = font("arialbd.ttf", 12)
f_win = font("arial.ttf", 10)


def tw(txt, fnt):
    return d.textlength(txt, font=fnt)


def truncate(txt, fnt, max_px):
    if tw(txt, fnt) <= max_px:
        return txt
    while txt and tw(txt + "…", fnt) > max_px:
        txt = txt[:-1]
    return txt + "…"


def rounded(x0, y0, x1, y1, rad, fill, outline=None, ow=1):
    d.rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], radius=s(rad), fill=fill,
                        outline=outline, width=s(ow))


# --- title ---------------------------------------------------------------------
d.text((s(MARGIN), s(22)), f"GROUP STAGE  ·  {n_matches} MATCHES (PREDICTED)",
       font=f_title, fill=WHITE)


def winner_label(m):
    hw, dr, aw = m["home_win"], m["draw"], m["away_win"]
    if hw > max(dr, aw):
        return m["home"], hw
    if aw > dr:
        return m["away"], aw
    return "Draw", dr


def draw_group(g, gx, gy):
    rounded(gx, gy, gx + CARD_W, gy + CARD_H, 12, CARD_BG, CARD_BORDER, 1)
    pad = 18
    left = gx + pad
    right = gx + CARD_W - pad

    # header: badge + "Group X"
    bs = 30
    rounded(left, gy + pad, left + bs, gy + pad + bs, 8, BADGE)
    lt = g["group"]
    d.text((s(left + bs / 2) - tw(lt, f_badge) / 2, s(gy + pad + 6)), lt, font=f_badge, fill=BLACK)
    d.text((s(left + bs + 12), s(gy + pad + 6)), f"Group {g['group']}", font=f_group, fill=WHITE)

    y = gy + pad + bs + 16
    for md in (1, 2, 3):
        mds = [x for x in g["matches"] if x["matchday"] == md]
        window = fix(mds[0]["window"]) if mds else ""
        d.text((s(left), s(y)), f"MATCHDAY {md}  ·  {window}".upper(), font=f_md, fill=GRAY_DIM)
        y += 17
        for m in mds:
            # right cluster: score (green) + winner label (gray), right-aligned
            wname, wp = winner_label(m)
            wlabel = f"{wname} {round(wp * 100)}%"
            wlabel = truncate(wlabel, f_win, s(150))
            score = f"{m['top_score']['home']}–{m['top_score']['away']}" if m.get("top_score") else ""
            wx = s(right) - tw(wlabel, f_win)
            d.text((wx, s(y + 2)), wlabel, font=f_win, fill=GRAY)
            if score:
                d.text((wx - tw(score, f_score) - s(12), s(y + 1)), score, font=f_score, fill=GREEN)

            # left: "Home  v  Away"
            max_match = (wx - s(left)) - s(14)
            home = m["home"]
            away = m["away"]
            hw = tw(home, f_teamb)
            vw = tw(" v ", f_team)
            ax = s(left) + hw + vw
            away = truncate(away, f_team, max_match - hw - vw)
            d.text((s(left), s(y)), home, font=f_teamb, fill=WHITE)
            d.text((s(left) + hw, s(y + 1)), " v ", font=f_team, fill=GRAY_DIM)
            d.text((ax, s(y)), away, font=f_team, fill=WHITE)
            y += 23
        y += 9


for i, g in enumerate(groups):
    r, c = divmod(i, COLS)
    gx = MARGIN + c * (CARD_W + CARD_GAP)
    gy = TOP + r * (CARD_H + CARD_GAP)
    draw_group(g, gx, gy)

img = img.resize((W, H), Image.LANCZOS)
img.save(OUT)
print(f"saved {OUT}  ({W}x{H})")
