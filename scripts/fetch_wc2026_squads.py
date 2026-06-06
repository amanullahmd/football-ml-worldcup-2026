"""Fetch the real 26-man FIFA World Cup 2026 squads from Wikipedia.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads (announced 2 Jun 2026).
Uses the MediaWiki action API per team section (short → reliable), parses the
{{nat fs ...}} player templates, and writes data/raw/players/wc2026_squads.csv
with columns: team,player,pos,club.

Run:  python scripts/fetch_wc2026_squads.py
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wc2026_config import normalize, all_teams

OUT = ROOT / "data" / "raw" / "players" / "wc2026_squads.csv"
API = "https://en.wikipedia.org/w/api.php"
PAGE = "2026 FIFA World Cup squads"
HEADERS = {"User-Agent": "football-ml/1.0 (educational project)"}

# Wikipedia section names → our canonical team names
TEAM_ALIASES = {
    "Curaçao": "Curacao",
    "Ivory Coast": "Ivory Coast",
    "United States": "United States",
    "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
    "South Korea": "South Korea",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
}
SKIP = {"Statistics", "References", "Notes", "See also"}


_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def api(params: dict) -> dict:
    params["format"] = "json"
    params["maxlag"] = 5
    for attempt in range(6):
        r = _SESSION.get(API, params=params, timeout=30)
        if r.status_code == 429 or (r.status_code == 200 and "maxlag" in r.text[:200]):
            wait = 2 ** attempt
            print(f"    rate-limited, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikipedia API rate limit — giving up after retries")


def split_team_chunks(wikitext: str) -> list[tuple[str, str]]:
    """Split full page wikitext into (team_header, body) by level-3 headers (=== Team ===)."""
    parts = re.split(r"\n===\s*([^=].*?)\s*===\s*\n", wikitext)
    out = []
    for i in range(1, len(parts), 2):
        header = re.sub(r"\[\[|\]\]", "", parts[i]).strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((header, body))
    return out


def _clean_link(val: str) -> str:
    """'[[Real Madrid CF|Real Madrid]]' -> 'Real Madrid'; '[[Flamengo]]' -> 'Flamengo'."""
    val = val.strip()
    m = re.search(r"\[\[([^\]]+)\]\]", val)
    inner = m.group(1) if m else val
    if "|" in inner:
        inner = inner.split("|")[-1]
    # strip residual wiki markup / refs / disambiguation parens
    inner = re.sub(r"\{\{.*?\}\}", "", inner)
    inner = re.sub(r"<ref.*?</ref>", "", inner, flags=re.S)
    inner = re.sub(r"\s*\([^)]*footballer[^)]*\)", "", inner, flags=re.I)
    return inner.strip()


def parse_players(wikitext: str) -> list[dict]:
    """Parse {{nat fs ... player ...}} templates into {pos,player,club}."""
    players = []
    # each player template starts with 'nat fs' ... 'player'
    for block in re.split(r"\{\{\s*nat fs", wikitext):
        if "name" not in block or "pos" not in block:
            continue
        pos = re.search(r"\|\s*pos\s*=\s*([A-Za-z]{2})", block)
        # capture up to the next |param= or }} so internal wikilink pipes survive
        name = re.search(r"\|\s*name\s*=\s*(.+?)\s*(?=\|\s*[a-z0-9]+\s*=|\}\})", block, re.S)
        club = re.search(r"\|\s*club\s*=\s*(.+?)\s*(?=\|\s*[a-z0-9]+\s*=|\}\})", block, re.S)
        if not (pos and name):
            continue
        p = pos.group(1).upper()
        if p not in ("GK", "DF", "MF", "FW"):
            continue
        nm = _clean_link(name.group(1))
        cl = _clean_link(club.group(1)) if club else ""
        if nm:
            players.append({"pos": p, "player": nm, "club": cl})
    return players


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wc = set(all_teams())
    # ONE request for the full page wikitext (avoids per-section rate limiting).
    print("fetching full squads page wikitext ...")
    full = api({"action": "parse", "page": PAGE, "prop": "wikitext"})["parse"]["wikitext"]["*"]

    rows = []
    for header, body in split_team_chunks(full):
        canon = normalize(TEAM_ALIASES.get(header, header))
        if canon not in wc:
            continue
        players = parse_players(body)
        for pl in players:
            rows.append({"team": canon, **pl})
        print(f"  {canon:28s} {len(players)} players")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "player", "pos", "club"])
        w.writeheader()
        w.writerows(rows)

    teams = sorted({r["team"] for r in rows})
    wc = set(all_teams())
    missing = sorted(wc - set(teams))
    print(f"\nwrote {len(rows)} players for {len(teams)} teams -> {OUT}")
    if missing:
        print(f"WARNING: WC2026 teams with no squad parsed: {missing}")


if __name__ == "__main__":
    main()
