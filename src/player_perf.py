"""Player-performance layer.

Combines THREE signals into a per-player performance score for the projected XI:
  1. FIFA-24 attribute rating       (src/squad.py)            — always available
  2. Real international goal form    (data/raw/goalscorers.csv) — always available
  3. Club xG / goals this season     (best-effort web fetch)    — optional, cached

Signal 3 (live club stats) is fetched if reachable; many providers (Understat,
FBref) rate-limit/Cloudflare-block automated requests, so the layer degrades
gracefully to signals 1+2 when the fetch fails. Nothing is fabricated — if club
data is missing it is simply absent, never mocked.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.wc2026_config import normalize as norm_team
from src.squad import _norm_country, load_players

ROOT = Path(__file__).resolve().parents[1]
GOALSCORERS = ROOT / "data" / "raw" / "goalscorers.csv"
CLUB_XG_CACHE = ROOT / "data" / "raw" / "players" / "club_xg.csv"


def _ascii(s: str) -> str:
    """Lower-case, strip accents — for fuzzy name matching across datasets."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


@lru_cache(maxsize=1)
def intl_goal_form(years: int = 4) -> pd.DataFrame:
    """Per (team, scorer) international goals over the last `years`, from real data.

    Returns columns: team, scorer, key (ascii), goals, npg (non-penalty goals).
    """
    g = pd.read_csv(GOALSCORERS, parse_dates=["date"])
    cutoff = g["date"].max() - pd.Timedelta(days=365 * years)
    g = g[(g["date"] >= cutoff) & (~g["own_goal"].fillna(False))]
    g["team"] = g["team"].map(lambda t: norm_team(_norm_country(t)))
    grp = g.groupby(["team", "scorer"], dropna=True)
    out = grp.agg(goals=("scorer", "size"),
                  npg=("penalty", lambda s: int((~s.fillna(False)).sum()))).reset_index()
    out["key"] = out["scorer"].map(_ascii)
    return out


def team_intl_form(team: str, years: int = 4) -> dict:
    """Aggregate recent international scoring for a team (real data)."""
    f = intl_goal_form(years)
    sub = f[f.team == team]
    return {
        "total_intl_goals": int(sub["goals"].sum()),
        "distinct_scorers": int((sub["goals"] > 0).sum()),
        "top_scorers": sub.sort_values("goals", ascending=False)
                          .head(5)[["scorer", "goals"]].to_dict("records"),
        "years": years,
    }


def player_intl_goals(team: str, player_name: str, years: int = 4) -> int:
    """Best-effort lookup of a player's recent international goals by fuzzy name."""
    f = intl_goal_form(years)
    sub = f[f.team == team]
    if sub.empty:
        return 0
    key = _ascii(player_name)
    # exact ascii match, else surname containment
    exact = sub[sub.key == key]
    if not exact.empty:
        return int(exact["goals"].iloc[0])
    surname = key.split()[-1] if key.split() else key
    cont = sub[sub.key.str.contains(re.escape(surname), na=False)] if surname else sub.iloc[0:0]
    return int(cont["goals"].max()) if not cont.empty else 0


# ---------------------------------------------------------------------------
# Optional: live club stats (best-effort, cached, graceful fallback)
# ---------------------------------------------------------------------------
def load_club_xg() -> pd.DataFrame | None:
    """Return cached club xG table if present, else None."""
    if CLUB_XG_CACHE.exists() and CLUB_XG_CACHE.stat().st_size > 0:
        try:
            return pd.read_csv(CLUB_XG_CACHE)
        except Exception:
            return None
    return None


def fetch_club_xg(save: bool = True) -> pd.DataFrame | None:
    """Attempt to fetch current-season club player xG from Understat (top-5 leagues).

    Returns a DataFrame [player, key, team, games, goals, assists, xG, xA] or None
    if all sources are unreachable (then the system uses signals 1+2 only).
    """
    import requests
    leagues = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
    season = "2024"
    rows = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; football-ml/1.0)"}
    for lg in leagues:
        try:
            r = requests.get(f"https://understat.com/league/{lg}/{season}",
                             timeout=25, headers=headers)
            m = re.search(r"playersData\s*=\s*JSON.parse\('([^']+)'\)", r.text)
            if not m:
                continue
            data = json.loads(m.group(1).encode().decode("unicode_escape"))
            for p in data:
                rows.append({
                    "player": p.get("player_name"),
                    "key": _ascii(p.get("player_name", "")),
                    "club": p.get("team_title"),
                    "games": int(p.get("games", 0) or 0),
                    "goals": int(p.get("goals", 0) or 0),
                    "assists": int(p.get("assists", 0) or 0),
                    "xG": round(float(p.get("xG", 0) or 0), 2),
                    "xA": round(float(p.get("xA", 0) or 0), 2),
                })
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if save:
        CLUB_XG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLUB_XG_CACHE, index=False)
    return df


def player_club_xg(player_name: str, club_df: pd.DataFrame | None) -> dict | None:
    if club_df is None or club_df.empty:
        return None
    key = _ascii(player_name)
    hit = club_df[club_df.key == key]
    if hit.empty:
        surname = key.split()[-1] if key.split() else key
        hit = club_df[club_df.key.str.contains(re.escape(surname), na=False)] if surname else hit
    if hit.empty:
        return None
    r = hit.sort_values("xG", ascending=False).iloc[0]
    return {"club": r["club"], "games": int(r["games"]), "goals": int(r["goals"]),
            "assists": int(r["assists"]), "xG": float(r["xG"]), "xA": float(r["xA"])}
