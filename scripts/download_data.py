"""Download all source datasets for the football ML system.

Free, no-API-key sources:
  1. martj42/international_results  — 49k international matches since 1872
  2. football-data.co.uk             — historical betting odds + match stats for 22 leagues
                                       (acts as a market-calibrated supplementary signal)

Run:  python data/download_all.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLUB = RAW / "club"
RAW.mkdir(parents=True, exist_ok=True)
CLUB.mkdir(parents=True, exist_ok=True)


def fetch(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False
    print(f"  GET {url}")
    r = requests.get(url, timeout=120, headers={"User-Agent": "football-ml/1.0"})
    if not r.ok:
        print(f"    ! {r.status_code}")
        return False
    dest.write_bytes(r.content)
    return True


# ---------------------------------------------------------------------------
# 1. International matches (martj42)
# ---------------------------------------------------------------------------
def download_international():
    print("[1] International matches (martj42/international_results)")
    base = "https://raw.githubusercontent.com/martj42/international_results/master/"
    for f in ("results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"):
        fetched = fetch(base + f, RAW / f)
        print(f"    {f}: {'fetched' if fetched else 'cached'}  {(RAW/f).stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# 2. football-data.co.uk — top European leagues, recent seasons
# ---------------------------------------------------------------------------
# Each league has a 3-letter code + filename "<code>.csv" per season.
# We grab the last ~10 seasons of the Big Five for supplementary club-form features.
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
}


def season_codes(n: int = 10) -> list[str]:
    """Return last `n` season codes like '2425', '2324', ..."""
    out = []
    # Current European season as of 2026 → 25/26.
    start_yy = 25  # 2025/26 season
    for k in range(n):
        a = (start_yy - k) % 100
        b = (a + 1) % 100
        out.append(f"{a:02d}{b:02d}")
    return out


def download_club_leagues(n_seasons: int = 10):
    print(f"[2] football-data.co.uk — {len(LEAGUES)} leagues × {n_seasons} seasons")
    for code, name in LEAGUES.items():
        league_dir = CLUB / code
        league_dir.mkdir(exist_ok=True)
        for season in season_codes(n_seasons):
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            dest = league_dir / f"{season}.csv"
            try:
                fetch(url, dest)
                time.sleep(0.3)  # be polite
            except Exception as e:
                print(f"    ! {code} {season}: {e}")
        files = list(league_dir.glob("*.csv"))
        sz = sum(f.stat().st_size for f in files)
        print(f"    {code} ({name}): {len(files)} seasons, {sz:,} bytes")


# ---------------------------------------------------------------------------
# 3. (optional) StatsBomb open data — xG + event-level data
# ---------------------------------------------------------------------------
def download_statsbomb_index():
    """Just grab the competitions index so train.py can discover what's available.
    Full event data is huge (~5GB); we leave it to the training script to pull
    only the competitions/seasons we need (World Cups, Euros, Copa America)."""
    print("[3] StatsBomb open data — competitions index")
    url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    fetch(url, RAW / "statsbomb_competitions.json")


def download_player_ratings():
    """FIFA-24 player attribute ratings (used as the squad-ability overlay)."""
    print("[4] FIFA-24 player ratings")
    url = "https://raw.githubusercontent.com/reh1548/FIFA-24-Player-Dataset/main/player_stats.csv"
    dest = RAW / "players" / "fifa24_players.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetched = fetch(url, dest)
    print(f"    fifa24_players.csv: {'fetched' if fetched else 'cached'}")


def main():
    download_international()
    download_club_leagues(n_seasons=10)
    download_statsbomb_index()
    download_player_ratings()
    print("\nDone. Datasets are in data/raw/.")
    print("Next: python scripts/fetch_wc2026_squads.py   (real 26-man WC2026 squads)")


if __name__ == "__main__":
    main()
