"""Download current-season (2025/26) player performance data.

Reliable, reachable source: the official Fantasy Premier League API (real
goals / assists / xG / xA / minutes for every Premier League player). FBref —
the only complete cross-league source — blocks automated requests (HTTP 403),
so full-world current club stats can't be fetched here; we combine this EPL data
with real international goals (data/raw/goalscorers.csv) and club tier for
cross-league coverage.

Run:  python scripts/download_player_perf.py
Output: data/raw/players/club_form_2526.csv
"""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "players" / "club_form_2526.csv"
FPL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("downloading Fantasy Premier League player data (2025/26) ...")
    d = requests.get(FPL, timeout=40, headers={"User-Agent": "football-ml/1.0"}).json()
    clubs = {t["id"]: t["name"] for t in d["teams"]}
    rows = []
    for e in d["elements"]:
        name = f"{e['first_name']} {e['second_name']}".strip()
        rows.append({
            "player": name,
            "key": _ascii(name),
            "web_key": _ascii(e.get("web_name", "")),
            "club": clubs.get(e["team"], ""),
            "league": "EPL",
            "minutes": int(e.get("minutes", 0) or 0),
            "goals": int(e.get("goals_scored", 0) or 0),
            "assists": int(e.get("assists", 0) or 0),
            "xg": float(e.get("expected_goals", 0) or 0),
            "xa": float(e.get("expected_assists", 0) or 0),
            "form": float(e.get("form", 0) or 0),
        })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    played = [r for r in rows if r["minutes"] > 0]
    print(f"wrote {len(rows)} EPL players ({len(played)} with minutes) -> {OUT}")
    top = sorted(rows, key=lambda r: -r["goals"])[:3]
    for r in top:
        print(f"  {r['player']:24s} {r['club']:16s} G{r['goals']} A{r['assists']} xG{r['xg']:.1f}")


if __name__ == "__main__":
    main()
