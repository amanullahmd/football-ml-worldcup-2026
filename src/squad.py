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

# Club-strength tiers (≈ ability scale). Derived from the player's CURRENT club in
# the real 26-man squad — a reliable proxy that doesn't depend on the incomplete
# FIFA-24 set (e.g. Vinicius@Real Madrid rates elite even if absent from FIFA-24).
CLUB_TIER = {
    # elite
    "Real Madrid": 90, "Manchester City": 90,
    "Barcelona": 88, "Bayern Munich": 88, "Liverpool": 88, "Arsenal": 88,
    "Paris Saint-Germain": 88, "Inter Milan": 87, "Inter": 87,
    # very strong
    "Atlético Madrid": 86, "Bayer Leverkusen": 86, "Chelsea": 86, "Napoli": 86,
    "Juventus": 85, "AC Milan": 85, "Milan": 85, "Tottenham Hotspur": 85, "Tottenham": 85,
    "Borussia Dortmund": 85, "Atalanta": 85, "Aston Villa": 85, "Newcastle United": 85,
    "Manchester United": 85, "RB Leipzig": 85,
    # strong
    "Benfica": 83, "Sporting CP": 83, "Porto": 83, "PSV Eindhoven": 83, "Feyenoord": 83,
    "Ajax": 83, "Marseille": 83, "Monaco": 83, "Villarreal": 83, "Real Sociedad": 83,
    "Athletic Bilbao": 83, "Real Betis": 82, "Brighton": 82, "West Ham United": 82,
    "Crystal Palace": 82, "Bologna": 82, "Roma": 83, "Lazio": 83, "Fiorentina": 82,
    "Eintracht Frankfurt": 82, "VfB Stuttgart": 82, "Lille": 82, "Lyon": 82, "Nice": 82,
    "Galatasaray": 82, "Fenerbahçe": 82, "Brentford": 81, "Fulham": 81, "Wolverhampton Wanderers": 81,
    "Nottingham Forest": 81, "Everton": 80, "Sevilla": 81, "Valencia": 80,
    # Saudi (star-laden), big South American, MLS-with-stars, other notable
    "Al-Hilal": 82, "Al-Nassr": 81, "Al-Ittihad": 81, "Al-Ahli": 81, "Al-Qadsiah": 79,
    "Flamengo": 81, "Palmeiras": 81, "Botafogo": 80, "Fluminense": 79, "Santos": 78,
    "River Plate": 80, "Boca Juniors": 80, "Inter Miami CF": 80, "Inter Miami": 80,
    "Los Angeles FC": 78, "Celtic": 79, "Rangers": 78, "Mainz 05": 79, "SC Freiburg": 79,
    "Werder Bremen": 78, "Borussia Mönchengladbach": 79, "TSG Hoffenheim": 78, "FC Augsburg": 77,
    "Torino": 78, "Udinese": 78, "Genoa": 77, "Lens": 80, "Rennes": 80, "Strasbourg": 79,
    "Sunderland": 78, "Leeds United": 78, "Burnley": 76, "Stoke City": 74, "Hull City": 73,
    "Birmingham City": 73, "Swansea City": 73, "Middlesbrough": 74, "Norwich City": 74,
    "Club Brugge": 79, "Anderlecht": 78, "Union Saint-Gilloise": 78, "Genk": 77,
    "Red Bull Salzburg": 79, "Shakhtar Donetsk": 78, "Dynamo Kyiv": 76, "Zenit Saint Petersburg": 78,
    "Dinamo Zagreb": 77, "Red Star Belgrade": 76, "Olympiacos": 77, "PAOK": 76, "AEK Athens": 75,
    "Slavia Prague": 77, "Sparta Prague": 76, "Viktoria Plzeň": 74,
    "América": 76, "Guadalajara": 75, "Monterrey": 76, "Tigres UANL": 76, "Cruz Azul": 75,
    "Toluca": 75, "Pumas": 74, "Mamelodi Sundowns": 74, "Orlando Pirates": 73,
}
CLUB_DEFAULT = 70.0  # smaller clubs / domestic leagues


def _club_alias(c: str) -> str:
    c = str(c).strip()
    repl = {"Spurs": "Tottenham", "Man City": "Manchester City", "Man Utd": "Manchester United",
            "PSG": "Paris Saint-Germain", "Atletico Madrid": "Atlético Madrid"}
    return repl.get(c, c)


def club_rating(club: str) -> float:
    c = _club_alias(club)
    if c in CLUB_TIER:
        return float(CLUB_TIER[c])
    # partial match (handles 'AC Milan' vs 'Milan', 'FC Barcelona' vs 'Barcelona', etc.)
    cl = c.lower()
    for k, v in CLUB_TIER.items():
        kl = k.lower()
        if kl in cl or cl in kl:
            return float(v)
    return CLUB_DEFAULT


# --- Current-season (2025/26) performance signals -------------------------
CLUB_FORM_CSV = ROOT / "data" / "raw" / "players" / "club_form_2526.csv"
GOALSCORERS_CSV = ROOT / "data" / "raw" / "goalscorers.csv"


@lru_cache(maxsize=1)
def _club_form_map() -> dict:
    """ascii player name -> current club stats (real, from FPL 2025/26)."""
    if not CLUB_FORM_CSV.exists():
        return {}
    df = pd.read_csv(CLUB_FORM_CSV)
    m: dict[str, dict] = {}
    for r in df.itertuples():
        rec = {"goals": int(r.goals), "assists": int(r.assists), "xg": float(r.xg),
               "minutes": int(r.minutes), "club": str(r.club)}
        m[str(r.key)] = rec
        m.setdefault(str(getattr(r, "web_key", "")), rec)
    return m


@lru_cache(maxsize=1)
def _intl_goals_map(years: int = 4) -> dict:
    """(team, ascii name / surname) -> recent international goals (real)."""
    if not GOALSCORERS_CSV.exists():
        return {}
    g = pd.read_csv(GOALSCORERS_CSV, parse_dates=["date"])
    cut = g["date"].max() - pd.Timedelta(days=365 * years)
    g = g[(g["date"] >= cut) & (~g["own_goal"].fillna(False))].copy()
    g["team"] = g["team"].map(lambda t: normalize(FIFA_COUNTRY_ALIASES.get(str(t).strip(), str(t).strip())))
    full, sur = {}, {}
    for (team, scorer), grp in g.groupby(["team", "scorer"]):
        k = _ascii(scorer)
        full[(team, k)] = int(len(grp))
        parts = k.split()
        if parts:
            sur[(team, parts[-1])] = max(sur.get((team, parts[-1]), 0), int(len(grp)))
    return {"full": full, "sur": sur}


def _player_perf(team: str, name: str, club: str):
    """Return (club_form_dict_or_None, intl_goals_int) for one player."""
    key = _ascii(name)
    cf = _club_form_map().get(key)
    ig_map = _intl_goals_map()
    ig = ig_map.get("full", {}).get((team, key))
    if ig is None:
        parts = key.split()
        ig = ig_map.get("sur", {}).get((team, parts[-1])) if parts else None
    return cf, int(ig or 0)


def _form_boost(cf: dict | None, intl_goals: int) -> float:
    """Bounded ability boost from current form (club G/A + intl goals)."""
    boost = 0.0
    if cf and cf["minutes"] >= 200:
        boost += min((cf["goals"] + 0.5 * cf["assists"]) * 0.35, 7.0)
    boost += min(intl_goals * 0.2, 4.0) * 0.6
    return min(boost, 8.0)


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
    # tidy display names: strip soft-hyphen / zero-width / replacement chars
    df["player"] = (df["player"].astype(str)
                    .str.replace(r"[­​�]", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    df["role"] = df["pos"].map(POS_TO_ROLE).fillna("MID")
    df["key"] = df["player"].map(_ascii)

    by_team = _fifa_by_team()
    # Ability per player = club-tier (damped by actual minutes for EPL players so
    # fringe elite-club squad members aren't over-rated), nudged toward FIFA-24
    # overall when found, plus a bounded current-form boost (club G/A + intl goals).
    club_r, fifa_ov, ages, matched = [], [], [], []
    cg, ca, cxg, igs, boosts = [], [], [], [], []
    for r in df.itertuples():
        cf, ig = _player_perf(r.team, r.player, r.club)
        cr = club_rating(r.club)
        if cf is not None:  # EPL player — damp club rating by playing time
            pw = max(0.55, min(cf["minutes"] / 1500.0, 1.0))
            cr = CLUB_DEFAULT + (cr - CLUB_DEFAULT) * pw
        club_r.append(cr)
        hit = _match_rating(r.team, r.key, by_team.get(r.team, []))
        if hit:
            fifa_ov.append(hit[0]); ages.append(hit[1]); matched.append(True)
        else:
            fifa_ov.append(np.nan); ages.append(np.nan); matched.append(False)
        cg.append(cf["goals"] if cf else None)
        ca.append(cf["assists"] if cf else None)
        cxg.append(cf["xg"] if cf else None)
        igs.append(ig)
        boosts.append(_form_boost(cf, ig))
    df["club_rating"] = club_r
    df["age"] = ages
    df["matched"] = matched
    fifa = np.array(fifa_ov, dtype=float)
    club = np.array(club_r, dtype=float)
    blended = np.where(np.isnan(fifa), club, 0.65 * club + 0.35 * fifa)
    df["club_goals"] = cg
    df["club_assists"] = ca
    df["club_xg"] = cxg
    df["intl_goals"] = igs
    df["overall"] = np.round(np.minimum(blended + np.array(boosts), 95.0), 1)
    return df


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
                "club": str(r.club).strip(), "age": int(r.age) if not pd.isna(r.age) else None,
                "intl_goals": int(getattr(r, "intl_goals", 0) or 0),
                "club_goals": (None if pd.isna(getattr(r, "club_goals", np.nan)) else int(r.club_goals)),
                "club_assists": (None if pd.isna(getattr(r, "club_assists", np.nan)) else int(r.club_assists)),
                "club_xg": (None if pd.isna(getattr(r, "club_xg", np.nan)) else round(float(r.club_xg), 1))}

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
