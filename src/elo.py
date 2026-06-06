"""Simple Elo rating engine for international football.

K-factor scaled by tournament importance and goal difference, following the
World Football Elo Ratings convention (eloratings.net).
"""
from __future__ import annotations

import math
from collections import defaultdict

INITIAL_ELO = 1500.0
HOME_ADV = 100.0  # Elo points added to the home side (neutral=0)

# K-factor weights by competition type — best-effort string matching.
K_WEIGHTS = {
    "FIFA World Cup": 60,
    "Copa America": 50,
    "UEFA Euro": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Confederations Cup": 40,
    "FIFA World Cup qualification": 40,
    "UEFA Nations League": 35,
    "Friendly": 20,
}


def k_for(tournament: str) -> float:
    for key, k in K_WEIGHTS.items():
        if key.lower() in (tournament or "").lower():
            return k
    return 30.0  # default for other competitive matches


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def goal_diff_multiplier(gd: int) -> float:
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


class EloEngine:
    def __init__(self, initial: float = INITIAL_ELO, home_adv: float = HOME_ADV):
        self.initial = initial
        self.home_adv = home_adv
        self.ratings: dict[str, float] = defaultdict(lambda: initial)

    def get(self, team: str) -> float:
        return self.ratings[team]

    def update(self, home: str, away: str, home_goals: int, away_goals: int,
               tournament: str = "Friendly", neutral: bool = False) -> tuple[float, float, float, float]:
        """Apply Elo update. Returns (home_elo_before, away_elo_before, home_elo_after, away_elo_after)."""
        rh = self.ratings[home]
        ra = self.ratings[away]

        eff_rh = rh + (0.0 if neutral else self.home_adv)
        exp_h = expected_score(eff_rh, ra)

        if home_goals > away_goals:
            score_h = 1.0
        elif home_goals < away_goals:
            score_h = 0.0
        else:
            score_h = 0.5

        k = k_for(tournament)
        gd = abs(home_goals - away_goals)
        mult = goal_diff_multiplier(gd)

        delta = k * mult * (score_h - exp_h)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return rh, ra, self.ratings[home], self.ratings[away]
