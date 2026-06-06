"""Dixon-Coles bivariate Poisson model for football goal prediction.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the UK Football Betting Market". The de-facto standard for
football score modelling (used by bookmakers).

Each team gets:
    attack_strength  (higher = scores more)
    defense_strength (higher = concedes more, i.e. weaker defense)

For a match (home, away) on neutral / non-neutral ground:
    lambda_home = exp(attack[home] - defense[away] + home_adv * (not neutral))
    lambda_away = exp(attack[away] - defense[home])

Goals are sampled as ~Poisson(lambda_*), with a low-score correlation
correction tau(i, j, lambda_h, lambda_a, rho) that fixes 0-0/1-0/0-1/1-1
which a pure independent-Poisson model underestimates.

We fit by weighted MLE with exponential time decay: matches `t` days before
the reference date get weight exp(-xi * t / 365). xi ~ 0.0065-0.018 (Dixon-Coles).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float = 0.0065) -> np.ndarray:
    """Exponential decay weights. xi=0.0065 ≈ half-life of ~3 years.

    Half-life formula: half_life_days = ln(2) / (xi/365) ~= 365*ln(2)/xi
    """
    days = (ref_date - pd.to_datetime(dates)).dt.days.values.astype(float)
    days = np.clip(days, 0, None)
    return np.exp(-xi * days / 365.0)


class DixonColesModel:
    def __init__(self, xi: float = 0.0065, rho_init: float = -0.05):
        self.xi = xi
        self.rho_init = rho_init
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defense_: np.ndarray | None = None
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0

    # ---------------- fit -----------------------------------------------
    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            min_matches: int = 8, verbose: bool = True) -> "DixonColesModel":
        """`df` must have columns: date, home_team, away_team, home_score, away_score, neutral."""
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # Drop teams with fewer than `min_matches` games — they add noise.
        counts: dict[str, int] = {}
        for col in ("home_team", "away_team"):
            for t, c in df[col].value_counts().items():
                counts[t] = counts.get(t, 0) + int(c)
        keep = {t for t, c in counts.items() if c >= min_matches}
        df = df[df.home_team.isin(keep) & df.away_team.isin(keep)].reset_index(drop=True)

        teams = sorted(set(df.home_team).union(df.away_team))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        if ref_date is None:
            ref_date = pd.to_datetime(df["date"]).max()
        w = time_weights(df["date"], ref_date, self.xi)

        h_idx = df.home_team.map(idx).values
        a_idx = df.away_team.map(idx).values
        h_goals = df.home_score.values
        a_goals = df.away_score.values
        neutral = df.neutral.astype(int).values if "neutral" in df.columns else np.zeros(len(df))

        # Parameter vector: [attack (n), defense (n-1), home_adv, rho]
        # constraint: sum(attack) = 0 (identifiability)  --> last defense computed from constraint? simpler: fix sum(attack)=0 by reparam
        # We'll use n attack + n defense free, but add penalty for sum(attack) deviating from 0.
        rng = np.random.default_rng(0)
        x0 = np.concatenate([
            rng.normal(0, 0.05, n),    # attack
            rng.normal(0, 0.05, n),    # defense
            np.array([0.25, self.rho_init]),  # home_adv, rho
        ])

        def unpack(x):
            return x[:n], x[n:2 * n], x[-2], x[-1]

        def neg_log_lik(x):
            atk, dfn, ha, rho = unpack(x)
            lam_h = np.exp(atk[h_idx] - dfn[a_idx] + ha * (1 - neutral))
            lam_a = np.exp(atk[a_idx] - dfn[h_idx])
            lam_h = np.clip(lam_h, 1e-6, 50.0)
            lam_a = np.clip(lam_a, 1e-6, 50.0)
            log_p = (h_goals * np.log(lam_h) - lam_h - _logfact(h_goals)
                     + a_goals * np.log(lam_a) - lam_a - _logfact(a_goals))
            # Dixon-Coles tau correction (only affects rows with i,j <= 1)
            mask = (h_goals <= 1) & (a_goals <= 1)
            if mask.any():
                tau = np.ones_like(log_p)
                hi = h_goals[mask]; ai = a_goals[mask]
                lh = lam_h[mask]; la = lam_a[mask]
                t = np.where((hi == 0) & (ai == 0), 1 - lh * la * rho,
                    np.where((hi == 0) & (ai == 1), 1 + lh * rho,
                    np.where((hi == 1) & (ai == 0), 1 + la * rho,
                                                    1 - rho)))
                t = np.clip(t, 1e-6, None)
                tau[mask] = t
                log_p = log_p + np.log(tau)
            # Identifiability penalty: encourage sum(attack)=0
            penalty = 1.0 * (atk.sum() ** 2 + dfn.sum() ** 2)
            return -(w * log_p).sum() + penalty

        bounds = [(-3, 3)] * (2 * n) + [(0.0, 1.0), (-0.4, 0.4)]
        if verbose:
            print(f"Dixon-Coles fit: {n} teams, {len(df)} matches, ref_date={ref_date.date()}")
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=400, ftol=1e-7))
        if verbose:
            print(f"  success={res.success}  neg_log_lik={res.fun:.1f}  iters={res.nit}")
        atk, dfn, ha, rho = unpack(res.x)
        self.teams_ = teams
        self.team_idx_ = idx
        self.attack_ = atk
        self.defense_ = dfn
        self.home_adv_ = float(ha)
        self.rho_ = float(rho)
        return self

    # ---------------- predict -------------------------------------------
    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        if home not in self.team_idx_ or away not in self.team_idx_:
            raise KeyError(f"Unknown team: {home if home not in self.team_idx_ else away}")
        ih = self.team_idx_[home]
        ia = self.team_idx_[away]
        lam_h = math.exp(self.attack_[ih] - self.defense_[ia] + (0.0 if neutral else self.home_adv_))
        lam_a = math.exp(self.attack_[ia] - self.defense_[ih])
        return lam_h, lam_a

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     max_goals: int = 10) -> np.ndarray:
        """Probability matrix P[i, j] = P(home_goals=i, away_goals=j)."""
        lam_h, lam_a = self.lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
        pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
        m = np.outer(ph, pa)
        # Apply Dixon-Coles correction to low scores
        for i in range(2):
            for j in range(2):
                m[i, j] *= dc_tau(i, j, lam_h, lam_a, self.rho_)
        m = m / m.sum()
        return m

    def predict(self, home: str, away: str, neutral: bool = True,
                max_goals: int = 10) -> dict:
        m = self.score_matrix(home, away, neutral, max_goals)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        eh = float((m.sum(axis=1) * np.arange(max_goals + 1)).sum())
        ea = float((m.sum(axis=0) * np.arange(max_goals + 1)).sum())

        # Common derived markets
        btts = float(m[1:, 1:].sum())
        over_25 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 2))
        over_15 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 1))
        over_35 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 3))

        # Top 5 most likely exact scores
        flat = [(i, j, float(m[i, j])) for i in range(max_goals + 1) for j in range(max_goals + 1)]
        flat.sort(key=lambda x: -x[2])
        top_scores = [{"home": i, "away": j, "prob": p} for i, j, p in flat[:5]]

        return {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "expected_home_goals": eh,
            "expected_away_goals": ea,
            "btts": btts,
            "over_1_5": over_15,
            "over_2_5": over_25,
            "over_3_5": over_35,
            "top_scores": top_scores,
            "score_matrix": m.tolist(),
        }

    # ---------------- introspection -------------------------------------
    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams_,
            "attack": self.attack_,
            "defense": self.defense_,
            "overall": self.attack_ - self.defense_,
        }).sort_values("overall", ascending=False).reset_index(drop=True)


# helper -----------------------------------------------------------------
_LOGFACT_CACHE = {0: 0.0}


def _logfact(n):
    arr = np.asarray(n)
    out = np.zeros_like(arr, dtype=float)
    flat = arr.flatten()
    res = np.empty_like(flat, dtype=float)
    for k, v in enumerate(flat):
        v = int(v)
        if v not in _LOGFACT_CACHE:
            _LOGFACT_CACHE[v] = math.lgamma(v + 1)
        res[k] = _LOGFACT_CACHE[v]
    return res.reshape(arr.shape)
