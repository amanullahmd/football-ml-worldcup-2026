"""Official FIFA World Cup 2026 knockout bracket (as drawn / published).

Encodes the real Round-of-32 -> Final structure, including each match's venue,
date, and slot specification, plus the third-place-team assignment rule.

Slot spec is a 2-tuple:
    ("1", "E")            -> winner of group E
    ("2", "B")            -> runner-up of group B
    ("3", "ABCDF")        -> a 3rd-placed team from one of those groups
    ("M", 74)             -> winner of match 74

The 8 "best third-placed teams" are assigned to the eight 3rd-place slots
respecting each slot's allowed group-set (bipartite matching), mirroring FIFA's
official combination table. Any valid matching yields statistically equivalent
advancement probabilities for simulation purposes.

Sources:
  - Official FIFA 2026 knockout bracket (R32-QF venues/dates/slots)
  - en.wikipedia.org/wiki/2026_FIFA_World_Cup (SF/Final: M101 Arlington 14 Jul,
    M102 Atlanta 15 Jul, M104 final MetLife NJ 19 Jul)
"""
from __future__ import annotations

# Each entry: id, round, venue, date, home slot, away slot
BRACKET = [
    # ---- Round of 32 ----
    {"id": 73, "round": "R32", "venue": "Los Angeles",  "date": "2026-06-28", "home": ("2", "A"), "away": ("2", "B")},
    {"id": 74, "round": "R32", "venue": "Boston",       "date": "2026-06-29", "home": ("1", "E"), "away": ("3", "ABCDF")},
    {"id": 75, "round": "R32", "venue": "Monterrey",    "date": "2026-06-29", "home": ("1", "F"), "away": ("2", "C")},
    {"id": 76, "round": "R32", "venue": "Houston",      "date": "2026-06-29", "home": ("1", "C"), "away": ("2", "F")},
    {"id": 77, "round": "R32", "venue": "New York NJ",  "date": "2026-06-30", "home": ("1", "I"), "away": ("3", "CDFGH")},
    {"id": 78, "round": "R32", "venue": "Dallas",       "date": "2026-06-30", "home": ("2", "E"), "away": ("2", "I")},
    {"id": 79, "round": "R32", "venue": "Mexico City",  "date": "2026-06-30", "home": ("1", "A"), "away": ("3", "CEFHI")},
    {"id": 80, "round": "R32", "venue": "Atlanta",      "date": "2026-07-01", "home": ("1", "L"), "away": ("3", "EHIJK")},
    {"id": 81, "round": "R32", "venue": "San Francisco","date": "2026-07-01", "home": ("1", "D"), "away": ("3", "BEFIJ")},
    {"id": 82, "round": "R32", "venue": "Seattle",      "date": "2026-07-01", "home": ("1", "G"), "away": ("3", "AEHIJ")},
    {"id": 83, "round": "R32", "venue": "Toronto",      "date": "2026-07-02", "home": ("2", "K"), "away": ("2", "L")},
    {"id": 84, "round": "R32", "venue": "Los Angeles",  "date": "2026-07-02", "home": ("1", "H"), "away": ("2", "J")},
    {"id": 85, "round": "R32", "venue": "Vancouver",    "date": "2026-07-02", "home": ("1", "B"), "away": ("3", "EFGIJ")},
    {"id": 86, "round": "R32", "venue": "Miami",        "date": "2026-07-03", "home": ("1", "J"), "away": ("2", "H")},
    {"id": 87, "round": "R32", "venue": "Kansas City",  "date": "2026-07-03", "home": ("1", "K"), "away": ("3", "DEIJL")},
    {"id": 88, "round": "R32", "venue": "Dallas",       "date": "2026-07-03", "home": ("2", "D"), "away": ("2", "G")},

    # ---- Round of 16 ----
    {"id": 89, "round": "R16", "venue": "Philadelphia", "date": "2026-07-04", "home": ("M", 74), "away": ("M", 77)},
    {"id": 90, "round": "R16", "venue": "Houston",      "date": "2026-07-04", "home": ("M", 73), "away": ("M", 75)},
    {"id": 91, "round": "R16", "venue": "New York NJ",  "date": "2026-07-05", "home": ("M", 76), "away": ("M", 78)},
    {"id": 92, "round": "R16", "venue": "Mexico City",  "date": "2026-07-05", "home": ("M", 79), "away": ("M", 80)},
    {"id": 93, "round": "R16", "venue": "Dallas",       "date": "2026-07-06", "home": ("M", 83), "away": ("M", 84)},
    {"id": 94, "round": "R16", "venue": "Seattle",      "date": "2026-07-06", "home": ("M", 81), "away": ("M", 82)},
    {"id": 95, "round": "R16", "venue": "Atlanta",      "date": "2026-07-07", "home": ("M", 86), "away": ("M", 88)},
    {"id": 96, "round": "R16", "venue": "Vancouver",    "date": "2026-07-07", "home": ("M", 85), "away": ("M", 87)},

    # ---- Quarter-finals ----
    {"id": 97,  "round": "QF", "venue": "Boston",       "date": "2026-07-09", "home": ("M", 89), "away": ("M", 90)},
    {"id": 98,  "round": "QF", "venue": "Los Angeles",  "date": "2026-07-10", "home": ("M", 93), "away": ("M", 94)},
    {"id": 99,  "round": "QF", "venue": "Miami",        "date": "2026-07-11", "home": ("M", 91), "away": ("M", 92)},
    {"id": 100, "round": "QF", "venue": "Kansas City",  "date": "2026-07-11", "home": ("M", 95), "away": ("M", 96)},

    # ---- Semi-finals ----
    {"id": 101, "round": "SF", "venue": "Dallas (Arlington)", "date": "2026-07-14", "home": ("M", 97), "away": ("M", 98)},
    {"id": 102, "round": "SF", "venue": "Atlanta",      "date": "2026-07-15", "home": ("M", 99), "away": ("M", 100)},

    # ---- Third-place play-off ----
    {"id": 103, "round": "3P", "venue": "Miami",       "date": "2026-07-18", "home": ("L", 101), "away": ("L", 102)},

    # ---- Final ----
    {"id": 104, "round": "F",  "venue": "New York NJ (MetLife)", "date": "2026-07-19", "home": ("M", 101), "away": ("M", 102)},
]

BY_ID = {m["id"]: m for m in BRACKET}
ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]

# Eight 3rd-place slots and the groups each may draw from (from the official bracket).
THIRD_SLOTS = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}


def assign_thirds(third_groups: list[str]) -> dict[int, str] | None:
    """Bipartite-match 8 qualifying third-place groups to the 8 third slots.

    `third_groups` = the (<=8) group letters whose 3rd-placed team qualified.
    Returns {match_id: group_letter} or None if no valid assignment (shouldn't
    happen for FIFA-valid combinations).
    """
    slots = sorted(THIRD_SLOTS.items(), key=lambda kv: len(kv[1]))  # most-constrained first
    assign: dict[int, str] = {}
    used: set[str] = set()

    def bt(i: int) -> bool:
        if i == len(slots):
            return True
        mid, allowed = slots[i]
        for g in third_groups:
            if g not in used and g in allowed:
                used.add(g); assign[mid] = g
                if bt(i + 1):
                    return True
                used.remove(g); del assign[mid]
        return False

    return dict(assign) if bt(0) else None


def resolve_r32_concrete(winners: dict[str, str], runners: dict[str, str],
                         third_team_by_group: dict[str, str],
                         third_assignment: dict[int, str]) -> dict[int, tuple[str, str]]:
    """Turn R32 slot specs into concrete (home, away) team pairs."""
    def team(slot, mid):
        typ, val = slot
        if typ == "1":
            return winners[val]
        if typ == "2":
            return runners[val]
        if typ == "3":
            grp = third_assignment.get(mid)
            return third_team_by_group.get(grp) if grp else None
        return None

    out = {}
    for m in BRACKET:
        if m["round"] != "R32":
            continue
        out[m["id"]] = (team(m["home"], m["id"]), team(m["away"], m["id"]))
    return out
