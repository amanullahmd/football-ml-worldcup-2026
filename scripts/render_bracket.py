"""Render the WC2026 knockout bracket (mirrored, Final centred) as a dark-themed PNG.

Reads bracket JSON (the /api/wc2026/bracket payload) and draws a supersampled image,
then downscales with LANCZOS for crisp anti-aliasing. Mirrors the web component layout.
"""
import json
import sys
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1] if len(sys.argv) > 1 else "bracket_25k.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"web\public\images\knockout-bracket-wc26.png"

data = json.load(open(SRC, encoding="utf-8"))
bracket = data["bracket"]
n_sims = data["n_simulations"]
byid = {m["id"]: m for m in bracket}


def feeders(i):
    return [s for s in (byid[i]["home_src"], byid[i]["away_src"]) if s is not None]


# --- split tree at the Final ---------------------------------------------------
sf = feeders(104)  # [left SF, right SF]


def half(sf_id):
    SF = [sf_id]
    QF = [f for s in SF for f in feeders(s)]
    R16 = [f for s in QF for f in feeders(s)]
    R32 = [f for s in R16 for f in feeders(s)]
    return {"R32": R32, "R16": R16, "QF": QF, "SF": SF}


L = half(sf[0])
R = half(sf[1])

# --- geometry (base units; everything multiplied by SS for supersampling) ------
SS = 2
CARD_W = 212
CARD_H = 116
COL_GAP = 74
COL_PITCH = CARD_W + COL_GAP
ROW_H = 150
MARGIN_X = 36
TOP = 116            # space for title + round headers
BOTTOM = 40

NCOLS = 9
colL = {"R32": 0, "R16": 1, "QF": 2, "SF": 3}
colR = {"R32": 8, "R16": 7, "QF": 6, "SF": 5}
colof = {104: 4}
yof = {}


def place(side, col):
    for i, mid in enumerate(side["R32"]):
        yof[mid] = i * ROW_H + ROW_H / 2
        colof[mid] = col["R32"]
    for r in ("R16", "QF", "SF"):
        for mid in side[r]:
            fs = feeders(mid)
            yof[mid] = sum(yof[f] for f in fs) / len(fs)
            colof[mid] = col[r]


place(L, colL)
place(R, colR)
yof[104] = (yof[sf[0]] + yof[sf[1]]) / 2

bracket_h = len(L["R32"]) * ROW_H
W = MARGIN_X * 2 + (NCOLS - 1) * COL_PITCH + CARD_W
H = TOP + bracket_h + BOTTOM

# --- colours -------------------------------------------------------------------
BG = (8, 12, 11)
CARD_BG = (15, 21, 19)
CARD_BORDER = (32, 42, 39)
GOLD = (251, 191, 36)
GOLD_DIM = (120, 92, 24)
GREEN = (52, 227, 155)
LINE = (46, 110, 84)
WHITE = (226, 232, 230)
GRAY = (120, 134, 130)
GRAY_DIM = (78, 90, 86)


def col_left_px(c):
    return MARGIN_X + c * COL_PITCH


# --- supersampled canvas -------------------------------------------------------
def s(v):
    return int(round(v * SS))


img = Image.new("RGB", (s(W), s(H)), BG)
d = ImageDraw.Draw(img)

FONTS = r"C:\Windows\Fonts"


def font(name, size):
    return ImageFont.truetype(rf"{FONTS}\{name}", s(size))


f_title = font("arialbd.ttf", 22)
f_hdr = font("arialbd.ttf", 13)
f_id = font("arialbd.ttf", 11)
f_meta = font("arial.ttf", 10)
f_slot = font("arial.ttf", 9)
f_team = font("arialbd.ttf", 14)
f_pct = font("arial.ttf", 11)


def text_w(txt, fnt):
    return d.textlength(txt, font=fnt)


def truncate(txt, fnt, max_px):
    if text_w(txt, fnt) <= max_px:
        return txt
    while txt and text_w(txt + "…", fnt) > max_px:
        txt = txt[:-1]
    return txt + "…"


# --- title + round headers -----------------------------------------------------
d.text((s(MARGIN_X), s(20)), f"KNOCKOUT BRACKET  ·  R32 → FINAL  ({n_sims:,} SIMS)",
       font=f_title, fill=WHITE)

headers = [("ROUND OF 32", 0), ("ROUND OF 16", 1), ("QUARTER-FINALS", 2), ("SEMI-FINALS", 3),
           ("FINAL", 4), ("SEMI-FINALS", 5), ("QUARTER-FINALS", 6), ("ROUND OF 16", 7), ("ROUND OF 32", 8)]
for label, c in headers:
    x = col_left_px(c)
    tw = text_w(label, f_hdr)
    d.text((s(x + CARD_W / 2) - tw / 2, s(72)), label, font=f_hdr, fill=GREEN)


# --- connectors (drawn first, behind cards) ------------------------------------
def champ_edge(parent_id, src_id):
    w = byid[parent_id].get("winner")
    sw = byid[src_id].get("winner")
    return bool(w and sw and w["team"] == sw["team"])


def hline(x1, x2, y, color, width):
    d.line([(s(x1), s(y)), (s(x2), s(y))], fill=color, width=s(width))


def vline(x, y1, y2, color, width):
    d.line([(s(x), s(y1)), (s(x), s(y2))], fill=color, width=s(width))


for m in bracket:
    if m["round"] == "3P":
        continue
    pid = m["id"]
    pc = colof.get(pid)
    if pc is None:
        continue
    for src in (m["home_src"], m["away_src"]):
        if src is None or src not in colof:
            continue
        cc = colof[src]
        left_feed = cc < pc
        x1 = col_left_px(cc) + (CARD_W if left_feed else 0)
        x2 = col_left_px(pc) + (0 if left_feed else CARD_W)
        y1 = TOP + yof[src]
        y2 = TOP + yof[pid]
        mx = (x1 + x2) / 2
        champ = champ_edge(pid, src)
        color = GOLD if champ else LINE
        wd = 2.4 if champ else 1.6
        hline(x1, mx, y1, color, wd)
        vline(mx, y1, y2, color, wd)
        hline(mx, x2, y2, color, wd)


# --- cards ---------------------------------------------------------------------
def rounded(x0, y0, x1, y1, rad, fill, outline, ow=1):
    d.rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], radius=s(rad), fill=fill,
                        outline=outline, width=s(ow))


def draw_card(m, mirror=False, col=None):
    c = colof.get(m["id"], 0) if col is None else col
    x = col_left_px(c)
    cy = TOP + yof.get(m["id"], ROW_H / 2)
    y = cy - CARD_H / 2
    is_final = m["round"] == "F"
    rounded(x, y, x + CARD_W, y + CARD_H, 10, CARD_BG,
            GOLD if is_final else CARD_BORDER, 2 if is_final else 1)

    pad = 12
    left = x + pad
    right = x + CARD_W - pad

    # header row: id + venue·date
    mid_label = "FINAL" if is_final else f"M{m['id']}"
    meta = f"{m['venue']} · {m['date'][5:]}"
    meta = truncate(meta, f_meta, s(CARD_W - pad * 2 - 8) - text_w(mid_label, f_id))
    hy = y + 9
    if mirror:
        d.text((s(right) - text_w(mid_label, f_id), s(hy)), mid_label, font=f_id, fill=GREEN)
        d.text((s(left), s(hy + 1)), meta, font=f_meta, fill=GRAY_DIM)
    else:
        d.text((s(left), s(hy)), mid_label, font=f_id, fill=(GOLD if is_final else GREEN))
        d.text((s(right) - text_w(meta, f_meta), s(hy + 1)), meta, font=f_meta, fill=GRAY_DIM)

    rows = [("home_slot", "home", y + 30), ("away_slot", "away", y + 70)]
    win = m.get("winner")
    for slot_key, side_key, ry in rows:
        slot = m.get(slot_key, "")
        team = m.get(side_key)
        # slot label
        if mirror:
            d.text((s(right) - text_w(slot, f_slot), s(ry)), slot, font=f_slot, fill=GRAY_DIM)
        else:
            d.text((s(left), s(ry)), slot, font=f_slot, fill=GRAY_DIM)
        ty = ry + 13
        if not team:
            name = "—"
            d.text((s(left), s(ty)), name, font=f_team, fill=GRAY)
            continue
        is_win = bool(win and win["team"] == team["team"])
        name = team["team"]
        prefix = "› " if is_win else ""  # › chevron
        pct = f"{round(team['prob'] * 100)}%"
        name_col = GOLD if is_win else WHITE
        max_name = s(CARD_W - pad * 2 - 10) - text_w(pct, f_pct) - text_w(prefix, f_team)
        disp = prefix + truncate(name, f_team, max_name)
        if mirror:
            d.text((s(right) - text_w(disp, f_team), s(ty)), disp, font=f_team, fill=name_col)
            d.text((s(left), s(ty + 1)), pct, font=f_pct, fill=GRAY)
        else:
            d.text((s(left), s(ty)), disp, font=f_team, fill=name_col)
            d.text((s(right) - text_w(pct, f_pct), s(ty + 1)), pct, font=f_pct, fill=GRAY)
    # divider
    dy = y + 64
    d.line([(s(left), s(dy)), (s(right), s(dy))], fill=CARD_BORDER, width=s(1))


for mid in L["R32"] + L["R16"] + L["QF"] + L["SF"]:
    draw_card(byid[mid], mirror=False)
draw_card(byid[104])
for mid in R["R32"] + R["R16"] + R["QF"] + R["SF"]:
    draw_card(byid[mid], mirror=True)

# --- downscale for anti-aliasing ----------------------------------------------
img = img.resize((W, H), Image.LANCZOS)
img.save(OUT)
print(f"saved {OUT}  ({W}x{H})")
