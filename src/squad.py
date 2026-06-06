"""Squad-strength layer — real WC2026 26-man rosters + FIFA-24 ability overlay.

Roster source : data/raw/players/wc2026_squads.csv  (the actual 26-man squads
                announced for the 2026 World Cup — scripts/fetch_wc2026_squads.py).
Ability source: data/raw/players/fifa24_players.csv  (per-player attribute ratings
                used only to estimate each squad player's overall ability).

This layer powers the 2026 FORECAST (squad strength, projected XI, and the
"who's missing" what-if). It is NOT used to retrain the historical W/D/L model
(that would leak future info into past matches). Only the 48 World Cup teams
appear here — by construction of the roster file.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.wc2026_config import normalize

ROOT = Path(__file__).resolve().parents[1]
FIFA_CSV = ROOT / "data" / "raw" / "players" / "fifa24_players.csv"
SQUADS_CSV = ROOT / "data" / "raw" / "players" / "wc2026_squads.csv"

# FIFA country spellings -> canonical team names (kept for player_perf compat).
FIFA_COUNTRY_ALIASES = {
    "Korea Republic": "South Korea", "Korea DPR": "North Korea", "IR Iran": "Iran",
    "China PR": "China", "USA": "United States", "Republic of Ireland": "Ireland",
    "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde", "Curaçao": "Curacao", "Congo DR": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

GK_COLS = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
DEF_COLS = ["marking", "stand_tackle", "slide_tackle", "interceptions", "heading", "strength", "reactions", "short_pass"]
MID_COLS = ["short_pass", "long_pass", "vision", "ball_control", "dribbling", "stamina", "reactions", "composure"]
ATT_COLS = ["finishing", "shot_power", "dribbling", "ball_control", "att_position", "composure", "acceleration", "sprint_speed"]

POS_TO_ROLE = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "ATT"}


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _norm_country(c: str) -> str:
    c = str(c).strip()
    return normalize(FIFA_COUNTRY_ALIASES.get(c, c))


# ---------------------------------------------------------------------------
# FIFA-24 ability ratings (overlay) — also imported by src/player_perf.py
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    """FIFA-24 players with a position-appropriate `overall` rating (cached)."""
    try:
        df = pd.read_csv(FIFA_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(FIFA_CSV, encoding="latin-1")
    if "player" in df.columns:
        df["player"] = (df["player"].astype(str).str.replace("�", "", regex=False)
                        .str.replace(r"\s+", " ", regex=True).str.strip())
    skills = list(set(GK_COLS + DEF_COLS + MID_COLS + ATT_COLS))
    for c in skills:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in skills if c in df.columns], how="all")
    df["gk_score"] = df[GK_COLS].mean(axis=1)
    df["def_score"] = df[DEF_COLS].mean(axis=1)
    df["mid_score"] = df[MID_COLS].mean(axis=1)
    df["att_score"] = df[ATT_COLS].mean(axis=1)

    def overall(r):
        if r.gk_score >= 50 and r.gk_score > max(r.def_score, r.mid_score, r.att_score):
            return r.gk_score
        return max(r.def_score, r.mid_score, r.att_score)

    df["overall"] = df.apply(overall, axis=1)
    df["key"] = df["player"].map(_ascii)
    df["team"] = df["country"].map(_norm_country)
    return df


@lru_cache(maxsize=1)
def _fifa_by_team() -> dict:
    """team -> list of (key, overall, age) from FIFA-24, for country-scoped matching."""
    df = load_players()
    out: dict[str, list] = {}
    for r in df.itertuples():
        age = float(r.age) if not pd.isna(getattr(r, "age", np.nan)) else None
        out.setdefault(r.team, []).append((r.key, float(r.overall), age))
    return out


def _match_rating(team: str, key: str, candidates: list) -> tuple[float, float] | None:
    """Match a squad player (ascii `key`) to a FIFA-24 rating within their country."""
    # 1) exact full-name match
    for k, ov, age in candidates:
        if k == key:
            return ov, age
    # 2) containment either way (e.g. 'alisson' ⊂ 'alisson becker')
    best = None
    for k, ov, age in candidates:
        if key and (key in k or k in key):
            if best is None or ov > best[0]:
                best = (ov, age)
    if best:
        return best
    # 3) surname match
    sur = key.split()[-1] if key.split() else key
    for k, ov, age in candidates:
        toks = k.split()
        if sur and toks and sur == toks[-1]:
            return ov, age
    return None


# ---------------------------------------------------------------------------
# Real WC2026 squads (roster) + ability overlay
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_squads() -> pd.DataFrame:
    """Real 26-man WC2026 rosters with an estimated `overall` per player."""
    df = pd.read_csv(SQUADS_CSV, encoding="utf-8")
    df["team"] = df["team"].map(normalize)
    df["role"] = df["pos"].map(POS_TO_ROLE).fillna("MID")
    df["key"] = df["player"].map(_ascii)

    by_team = _fifa_by_team()
    overalls, ages, matched = [], [], []
    for r in df.itertuples():
        hit = _match_rating(r.team, r.key, by_team.get(r.team, []))
        if hit:
            overalls.append(hit[0]); ages.append(hit[1]); matched.append(True)
        else:
            overalls.append(np.nan); ages.append(np.nan); matched.append(False)
    df["overall"] = overalls
    df["age"] = ages
    df["matched"] = matched

    # Fallback overall for unmatched players: team's matched median (else 71),
    # with a small position-neutral discount so unknowns don't outrank stars.
    out = []
    for team, g in df.groupby("team"):
        base = g.loc[g.matched, "overall"].median()
        base = float(base) if not pd.isna(base) else 71.0
        g = g.copy()
        g["overall"] = g["overall"].fillna(max(base - 3.0, 60.0))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _apply_exclude(sq: pd.DataFrame, exclude: list[str] | None) -> pd.DataFrame:
    if not exclude:
        return sq
    ex = [_ascii(e) for e in exclude if e.strip()]
    if not ex:
        return sq
    mask = sq.player.apply(lambda p: any(e in _ascii(p) for e in ex))
    return sq[~mask]


def team_strength(team: str, exclude: list[str] | None = None) -> dict | None:
    df = load_squads()
    sq = _apply_exclude(df[df.team == team].copy(), exclude)
    if sq.empty:
        return None
    sq = sq.sort_values("overall", ascending=False)
    top23 = sq.head(23)
    top11 = sq.head(11)

    def role_top(role, n):
        r = sq[sq.role == role].head(n)
        return float(r.overall.mean()) if len(r) else float(sq.overall.head(n).mean())

    ages = sq["age"].dropna()
    return {
        "team": team,
        "n_players": int(len(sq)),
        "squad_overall": float(top23.overall.mean()),
        "top11_overall": float(top11.overall.mean()),
        "attack_rating": role_top("ATT", 3),
        "midfield_rating": role_top("MID", 3),
        "defense_rating": role_top("DEF", 4),
        "gk_rating": role_top("GK", 1),
        "avg_age": float(ages.mean()) if len(ages) else None,
        "squad_value_m": None,
        "depth": float(sq.iloc[11:23].overall.mean()) if len(sq) > 11 else float(top23.overall.mean()),
        "excluded": exclude or [],
    }


def all_team_strengths(teams: list[str] | None = None) -> pd.DataFrame:
    df = load_squads()
    teams = teams or sorted(df.team.unique())
    rows = [s for t in teams if (s := team_strength(t))]
    return pd.DataFrame(rows).sort_values("squad_overall", ascending=False).reset_index(drop=True)


def top_players(team: str, n: int = 11, exclude: list[str] | None = None) -> list[dict]:
    """Projected starting XI as a realistic 4-3-3 (best player per role)."""
    df = load_squads()
    sq = _apply_exclude(df[df.team == team].copy(), exclude).sort_values("overall", ascending=False)
    if sq.empty:
        return []

    def fmt(r):
        return {"player": str(r.player), "position": r.role, "overall": round(float(r.overall), 1),
                "club": str(r.club).strip(), "age": int(r.age) if not pd.isna(r.age) else None}

    if n == 11:
        quota = {"GK": 1, "DEF": 4, "MID": 3, "ATT": 3}
        picked, used = [], set()
        for role, k in quota.items():
            for r in sq[sq.role == role].head(k).itertuples():
                picked.append(fmt(r)); used.add(r.Index)
        # backfill if a role was short (e.g. squad with few of a position)
        if len(picked) < 11:
            for r in sq[~sq.index.isin(used)].itertuples():
                picked.append(fmt(r))
                if len(picked) == 11:
                    break
        order = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}
        return sorted(picked, key=lambda p: (order[p["position"]], -p["overall"]))
    return [fmt(r) for r in sq.head(n).itertuples()]
