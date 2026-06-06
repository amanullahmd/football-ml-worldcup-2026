"""Squad-strength layer — turns per-player ratings into national-team strength.

Data source: FIFA-24 player dataset (skill attributes per player + country).
Because video-game ratings are a single recent snapshot, this layer is used for
the 2026 FORECAST (and a "who's missing" what-if), NOT to retrain the historical
W/D/L model (that would leak future info into past matches).

Pipeline:
  load_players() -> per-player overall + inferred position (GK/DEF/MID/ATT)
  team_strength(team, exclude=[]) -> dict of aggregated features
  all_team_strengths() -> DataFrame for every team

The `exclude` argument implements the dev-docs requirement: e.g. France without
Mbappe should have a lower attack rating and therefore weaker predictions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.wc2026_config import normalize

ROOT = Path(__file__).resolve().parents[1]
PLAYERS_CSV = ROOT / "data" / "raw" / "players" / "fifa24_players.csv"

# FIFA country spellings -> our canonical team names (extends wc2026 aliases).
FIFA_COUNTRY_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "China PR": "China",
    "United States": "United States",
    "USA": "United States",
    "Republic of Ireland": "Ireland",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

GK_COLS = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
DEF_COLS = ["marking", "stand_tackle", "slide_tackle", "interceptions", "heading", "strength", "reactions", "short_pass"]
MID_COLS = ["short_pass", "long_pass", "vision", "ball_control", "dribbling", "stamina", "reactions", "composure"]
ATT_COLS = ["finishing", "shot_power", "dribbling", "ball_control", "att_position", "composure", "acceleration", "sprint_speed"]


def _norm_country(c: str) -> str:
    c = str(c).strip()
    c = FIFA_COUNTRY_ALIASES.get(c, c)
    return normalize(c)


@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    """Load + enrich the player table (cached)."""
    # Source CSV is UTF-8 but has some lossy accent bytes (stored as U+FFFD).
    try:
        df = pd.read_csv(PLAYERS_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(PLAYERS_CSV, encoding="latin-1")
    # Tidy display names: drop replacement chars and squeeze double spaces.
    if "player" in df.columns:
        df["player"] = (df["player"].astype(str)
                        .str.replace("�", "", regex=False)
                        .str.replace(r"\s+", " ", regex=True).str.strip())

    skill_cols = list(set(GK_COLS + DEF_COLS + MID_COLS + ATT_COLS))
    for c in skill_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in skill_cols if c in df.columns], how="all")

    df["gk_score"] = df[GK_COLS].mean(axis=1)
    df["def_score"] = df[DEF_COLS].mean(axis=1)
    df["mid_score"] = df[MID_COLS].mean(axis=1)
    df["att_score"] = df[ATT_COLS].mean(axis=1)

    # Position inference
    def infer_pos(r):
        if r.gk_score >= 50 and r.gk_score > max(r.def_score, r.mid_score, r.att_score):
            return "GK"
        scores = {"DEF": r.def_score, "MID": r.mid_score, "ATT": r.att_score}
        return max(scores, key=scores.get)

    df["position"] = df.apply(infer_pos, axis=1)

    # Position-appropriate overall (0-100ish)
    def overall(r):
        if r.position == "GK":
            return r.gk_score
        if r.position == "DEF":
            return r.def_score
        if r.position == "MID":
            return r.mid_score
        return r.att_score

    df["overall"] = df.apply(overall, axis=1)
    df["team"] = df["country"].map(_norm_country)

    # Parse value column (e.g. "€12.5M" or numbers) -> millions float (best-effort)
    if "value" in df.columns:
        df["value_m"] = df["value"].apply(_parse_value)
    else:
        df["value_m"] = np.nan
    return df


def _parse_value(v) -> float:
    s = str(v).strip().replace("€", "").replace("$", "").replace(",", "")
    if not s or s.lower() == "nan":
        return np.nan
    mult = 1.0
    if s.endswith("M"):
        mult = 1.0; s = s[:-1]
    elif s.endswith("K"):
        mult = 0.001; s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def team_strength(team: str, exclude: list[str] | None = None) -> dict | None:
    """Aggregate squad-strength features for one national team.

    `exclude` is a list of player-name substrings to drop (injuries/suspensions).
    Returns None if the team has no players in the dataset.
    """
    df = load_players()
    sq = df[df.team == team].copy()
    if exclude:
        ex = [e.strip().lower() for e in exclude if e.strip()]
        if ex:
            mask = sq.player.str.lower().apply(lambda p: any(e in p for e in ex))
            sq = sq[~mask]
    if sq.empty:
        return None

    sq = sq.sort_values("overall", ascending=False)
    top23 = sq.head(23)
    top11 = sq.head(11)

    def role_top(role, n):
        r = sq[sq.position == role].head(n)
        return float(r.overall.mean()) if len(r) else float(sq.overall.head(n).mean())

    return {
        "team": team,
        "n_players": int(len(sq)),
        "squad_overall": float(top23.overall.mean()),
        "top11_overall": float(top11.overall.mean()),
        "attack_rating": role_top("ATT", 3),
        "midfield_rating": role_top("MID", 3),
        "defense_rating": role_top("DEF", 4),
        "gk_rating": role_top("GK", 1),
        "avg_age": float(top23["age"].mean()) if "age" in top23 else None,
        "squad_value_m": float(top23["value_m"].sum(skipna=True)) if "value_m" in top23 else None,
        "depth": float(sq.iloc[11:23].overall.mean()) if len(sq) > 11 else float(top23.overall.mean()),
        "excluded": exclude or [],
    }


def all_team_strengths(teams: list[str] | None = None) -> pd.DataFrame:
    df = load_players()
    teams = teams or sorted(df.team.unique())
    rows = []
    for t in teams:
        s = team_strength(t)
        if s:
            rows.append(s)
    out = pd.DataFrame(rows).sort_values("squad_overall", ascending=False).reset_index(drop=True)
    return out


def top_players(team: str, n: int = 11, exclude: list[str] | None = None) -> list[dict]:
    """Projected starting XI / key players for display."""
    df = load_players()
    sq = df[df.team == team].copy()
    if exclude:
        ex = [e.strip().lower() for e in exclude if e.strip()]
        if ex:
            sq = sq[~sq.player.str.lower().apply(lambda p: any(e in p for e in ex))]
    sq = sq.sort_values("overall", ascending=False).head(n)
    return [
        {"player": str(r.player), "position": r.position, "overall": round(float(r.overall), 1),
         "club": str(r.club).strip() if "club" in sq else "", "age": int(r.age) if not pd.isna(r.age) else None}
        for r in sq.itertuples()
    ]
