"""FastAPI server — match prediction + World Cup 2026 dashboard.

Run:
    uvicorn app.main:app --reload --port 8000

UI:
    http://127.0.0.1:8000/             landing
    http://127.0.0.1:8000/predict      any-match predictor
    http://127.0.0.1:8000/worldcup     WC2026 dashboard

API:
    GET  /api/teams
    GET  /api/metrics
    POST /api/predict           {"home": "...", "away": "...", "neutral": true}
    GET  /api/wc2026/groups
    GET  /api/wc2026/predictions
    GET  /api/wc2026/simulation
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "models"
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Load artifacts once at startup.
# Prefer the "pro" stacked ensemble (train_pro.py); fall back to the simple
# calibrated CatBoost model (train.py) if the pro artifacts aren't present.
# ---------------------------------------------------------------------------
print("loading models...")
DC = joblib.load(MOD / "dixon_coles.joblib")
FEATURES = joblib.load(MOD / "feature_list.joblib")
ELO = pd.read_csv(PROC / "final_elo.csv").set_index("team")["elo"].to_dict()
MATCHES = pd.read_parquet(PROC / "matches_features.parquet")

PRO = (MOD / "stack_meta.joblib").exists()
if PRO:
    STACK_META = joblib.load(MOD / "stack_meta.joblib")
    BASE_NAMES = joblib.load(MOD / "base_names.joblib") if (MOD / "base_names.joblib").exists() \
        else ["catboost", "lgbm"]
    BASES = {name: joblib.load(MOD / f"base_{name}.joblib") for name in BASE_NAMES}
    CALIBRATORS = joblib.load(MOD / "calibrator.joblib")
    REG_H = joblib.load(MOD / "goals_home.joblib")
    REG_A = joblib.load(MOD / "goals_away.joblib")
    METRICS = json.loads((MOD / "metrics_pro.json").read_text())
    print("  using PRO stacked ensemble (CatBoost+XGBoost+LightGBM -> LogReg, isotonic-calibrated)")
else:
    CLS = joblib.load(MOD / "catboost_result_calibrated.joblib")
    REG_H = joblib.load(MOD / "catboost_goals_home.joblib")
    REG_A = joblib.load(MOD / "catboost_goals_away.joblib")
    METRICS = json.loads((MOD / "metrics.json").read_text())
    print("  using simple calibrated CatBoost model")

print(f"ready. {len(DC.teams_)} DC teams, {len(ELO)} Elo teams, {len(MATCHES):,} historical matches")


def _ml_proba(X: pd.DataFrame) -> np.ndarray:
    """Return calibrated [home, draw, away] probabilities from whichever model is loaded."""
    if PRO:
        import numpy as _np
        feat = _np.zeros((len(X), len(BASE_NAMES) * 3))
        for bi, name in enumerate(BASE_NAMES):
            feat[:, bi*3:(bi+1)*3] = BASES[name].predict_proba(X.values)
        raw = STACK_META.predict_proba(feat)
        cal = _np.column_stack([CALIBRATORS[c].predict(raw[:, c]) for c in range(3)])
        cal = _np.clip(cal, 1e-6, None)
        return cal / cal.sum(axis=1, keepdims=True)
    return CLS.predict_proba(X)

# WC2026 config
from src.wc2026_config import GROUPS, HOST_COUNTRIES, all_teams as wc_teams
from src.wc2026_bracket import (BRACKET, BY_ID, ROUND_ORDER, THIRD_SLOTS,
                                assign_thirds, resolve_r32_concrete)
from src.elo import k_for as elo_k_for


# Build a quick lookup of latest form per team
def latest_form(team: str) -> dict | None:
    h = MATCHES[MATCHES.home_team == team].tail(1)
    a = MATCHES[MATCHES.away_team == team].tail(1)
    cand = pd.concat([h.assign(side="h"), a.assign(side="a")])
    if cand.empty:
        return None
    row = cand.sort_values("date").tail(1).iloc[0]
    if row.side == "h":
        return dict(
            f5=row.home_form5_pts, f5gf=row.home_form5_gf, f5ga=row.home_form5_ga,
            f10=row.home_form10_pts, f10gf=row.home_form10_gf, f10ga=row.home_form10_ga,
        )
    return dict(
        f5=row.away_form5_pts, f5gf=row.away_form5_gf, f5ga=row.away_form5_ga,
        f10=row.away_form10_pts, f10gf=row.away_form10_gf, f10ga=row.away_form10_ga,
    )


ALL_TEAMS = sorted(set(MATCHES.home_team).union(MATCHES.away_team) | set(DC.teams_) | set(ELO.keys()))
FORM_CACHE = {t: latest_form(t) for t in ALL_TEAMS}

# Real World Cup match cadence (teams play every ~4 days during the tournament).
# Data check: the 25th percentile of rest-days across all international matches is
# exactly 4 days, i.e. dense tournament scheduling. Applied EQUALLY to both teams,
# so it shifts no probability toward either side — it is not a bias term.
WC_REST_DAYS = 4


def build_feature_row(home: str, away: str, neutral: bool) -> pd.DataFrame:
    fh = FORM_CACHE.get(home) or dict(f5=1, f5gf=1, f5ga=1, f10=1, f10gf=1, f10ga=1)
    fa = FORM_CACHE.get(away) or dict(f5=1, f5gf=1, f5ga=1, f10=1, f10gf=1, f10ga=1)
    h_elo = ELO.get(home, 1500.0)
    a_elo = ELO.get(away, 1500.0)
    h2h = MATCHES[
        ((MATCHES.home_team == home) & (MATCHES.away_team == away)) |
        ((MATCHES.home_team == away) & (MATCHES.away_team == home))
    ].tail(5)
    hw = dr = aw = 0
    gf = ga = 0.0
    for _, r in h2h.iterrows():
        if r.home_team == home:
            hg, ag = r.home_score, r.away_score
        else:
            hg, ag = r.away_score, r.home_score
        if hg > ag: hw += 1
        elif hg < ag: aw += 1
        else: dr += 1
        gf += hg; ga += ag
    n = max(len(h2h), 1)

    # Dixon-Coles team strengths (zero if team not in DC training set)
    dc_idx = DC.team_idx_
    dc_ha = DC.attack_[dc_idx[home]]  if home in dc_idx else 0.0
    dc_hd = DC.defense_[dc_idx[home]] if home in dc_idx else 0.0
    dc_aa = DC.attack_[dc_idx[away]]  if away in dc_idx else 0.0
    dc_ad = DC.defense_[dc_idx[away]] if away in dc_idx else 0.0
    dc_lh = float(np.exp(dc_ha - dc_ad + (0.0 if neutral else DC.home_adv_)))
    dc_la = float(np.exp(dc_aa - dc_hd))

    row = {
        "home_elo": h_elo, "away_elo": a_elo, "elo_diff": h_elo - a_elo,
        "dc_home_attack": dc_ha, "dc_home_defense": dc_hd,
        "dc_away_attack": dc_aa, "dc_away_defense": dc_ad,
        "dc_lambda_home": dc_lh, "dc_lambda_away": dc_la,
        "home_form5_pts": fh["f5"], "home_form5_gf": fh["f5gf"], "home_form5_ga": fh["f5ga"],
        "away_form5_pts": fa["f5"], "away_form5_gf": fa["f5gf"], "away_form5_ga": fa["f5ga"],
        "home_form10_pts": fh["f10"], "home_form10_gf": fh["f10gf"], "home_form10_ga": fh["f10ga"],
        "away_form10_pts": fa["f10"], "away_form10_gf": fa["f10gf"], "away_form10_ga": fa["f10ga"],
        "h2h_home_wins": hw, "h2h_draws": dr, "h2h_away_wins": aw,
        "h2h_home_gf": gf / n, "h2h_home_ga": ga / n,
        "tournament_k": elo_k_for("FIFA World Cup"),
        "neutral_int": int(neutral),
        "home_rest": WC_REST_DAYS, "away_rest": WC_REST_DAYS,
    }
    return pd.DataFrame([row])[FEATURES]


# ---------------------------------------------------------------------------
# Squad-strength layer (FIFA-24 player ratings -> national-team strength)
# ---------------------------------------------------------------------------
from scipy.stats import poisson as _poisson
from src.dixon_coles import dc_tau as _dc_tau
try:
    from src.squad import team_strength as _team_strength, all_team_strengths as _all_squads, \
        top_players as _top_players
    # Every national team present in the player dataset (all countries).
    SQUADS_DF = _all_squads().reset_index(drop=True)
    SQUAD_FULL = {r["team"]: r for r in SQUADS_DF.to_dict("records")}
    SQUAD_OK = True
    print(f"  squad layer: {len(SQUAD_FULL)} national-team squads loaded")
except Exception as e:  # pragma: no cover
    SQUAD_OK = False
    SQUAD_FULL = {}
    print(f"  squad layer unavailable: {e}")

# Player-performance layer (real international goals + optional cached club xG)
try:
    from src.player_perf import (team_intl_form as _team_intl_form,
                                 player_intl_goals as _player_intl_goals,
                                 load_club_xg as _load_club_xg,
                                 player_club_xg as _player_club_xg)
    CLUB_XG = _load_club_xg()
    PERF_OK = True
    print(f"  player-performance layer: intl goals OK, club xG {'OK' if CLUB_XG is not None else 'not cached'}")
except Exception as e:  # pragma: no cover
    PERF_OK = False
    CLUB_XG = None
    print(f"  player-performance layer unavailable: {e}")


def squad_attack_mult(team: str, exclude: list[str] | None) -> float:
    """How much a team's attack changes when `exclude` players are unavailable."""
    if not SQUAD_OK or not exclude:
        return 1.0
    base = SQUAD_FULL.get(team) or _team_strength(team)
    if not base:
        return 1.0
    adj = _team_strength(team, exclude=exclude)
    if not adj or not base.get("attack_rating"):
        return 1.0
    return max(0.6, min(1.2, adj["attack_rating"] / base["attack_rating"]))


def _dc_metrics_from_lambdas(lh: float, la: float, rho: float, max_goals: int = 8) -> dict:
    """Dixon-Coles score-matrix metrics from explicit lambdas (for squad-adjusted goals)."""
    ph = _poisson.pmf(np.arange(max_goals + 1), max(lh, 0.05))
    pa = _poisson.pmf(np.arange(max_goals + 1), max(la, 0.05))
    m = np.outer(ph, pa)
    for i in range(2):
        for j in range(2):
            m[i, j] *= _dc_tau(i, j, lh, la, rho)
    m = m / m.sum()
    p_home = float(np.tril(m, -1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    eh = float((m.sum(axis=1) * np.arange(max_goals + 1)).sum())
    ea = float((m.sum(axis=0) * np.arange(max_goals + 1)).sum())
    btts = float(m[1:, 1:].sum())
    o15 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 1))
    o25 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 2))
    o35 = float(sum(m[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 3))
    flat = sorted(((i, j, float(m[i, j])) for i in range(max_goals + 1) for j in range(max_goals + 1)),
                  key=lambda x: -x[2])
    top = [{"home": i, "away": j, "prob": p} for i, j, p in flat[:5]]
    return {"home_win": p_home, "draw": p_draw, "away_win": p_away,
            "expected_home_goals": eh, "expected_away_goals": ea, "btts": btts,
            "over_1_5": o15, "over_2_5": o25, "over_3_5": o35,
            "top_scores": top, "score_matrix": m.tolist()}


def predict_match(home: str, away: str, neutral: bool = True,
                  home_out: list[str] | None = None, away_out: list[str] | None = None) -> dict:
    X = build_feature_row(home, away, neutral)

    # ML probabilities + expected goals (squad-agnostic by design)
    proba = _ml_proba(X)[0]
    ml = {"home_win": float(proba[0]), "draw": float(proba[1]), "away_win": float(proba[2])}
    eg_h = float(REG_H.predict(X)[0])
    eg_a = float(REG_A.predict(X)[0])

    # Squad-availability multipliers (1.0 if nobody excluded / no squad data)
    mult_h = squad_attack_mult(home, home_out)
    mult_a = squad_attack_mult(away, away_out)

    # Dixon-Coles — adjusted by squad availability when players are missing
    try:
        lh, la = DC.lambdas(home, away, neutral=neutral)
        lh *= mult_h
        la *= mult_a
        dc = _dc_metrics_from_lambdas(lh, la, DC.rho_, max_goals=8)
    except KeyError:
        dc = None

    if dc is not None:
        final = {
            "home_win": 0.5 * ml["home_win"] + 0.5 * dc["home_win"],
            "draw":     0.5 * ml["draw"]     + 0.5 * dc["draw"],
            "away_win": 0.5 * ml["away_win"] + 0.5 * dc["away_win"],
        }
        s = sum(final.values())
        final = {k: v / s for k, v in final.items()}
        exp_h = 0.5 * (eg_h * mult_h) + 0.5 * dc["expected_home_goals"]
        exp_a = 0.5 * (eg_a * mult_a) + 0.5 * dc["expected_away_goals"]
        score_matrix = dc["score_matrix"]
        top_scores = dc["top_scores"]
        btts, over_15, over_25, over_35 = dc["btts"], dc["over_1_5"], dc["over_2_5"], dc["over_3_5"]
    else:
        final = ml
        exp_h, exp_a = eg_h * mult_h, eg_a * mult_a
        score_matrix, top_scores = None, []
        btts = over_15 = over_25 = over_35 = None

    sq_h = SQUAD_FULL.get(home) if SQUAD_OK else None
    sq_a = SQUAD_FULL.get(away) if SQUAD_OK else None

    return {
        "home": home, "away": away, "neutral": neutral,
        "probabilities": final,
        "ml_probabilities": ml,
        "dc_probabilities": {k: dc[k] for k in ("home_win", "draw", "away_win")} if dc else None,
        "expected_goals": {"home": exp_h, "away": exp_a},
        "elo": {"home": ELO.get(home, 1500.0), "away": ELO.get(away, 1500.0)},
        "squad": {
            "home": {"overall": sq_h["squad_overall"], "attack": sq_h["attack_rating"]} if sq_h else None,
            "away": {"overall": sq_a["squad_overall"], "attack": sq_a["attack_rating"]} if sq_a else None,
            "home_attack_mult": mult_h, "away_attack_mult": mult_a,
            "home_out": home_out or [], "away_out": away_out or [],
        },
        "score_matrix": score_matrix,
        "top_scores": top_scores,
        "markets": {
            "btts": btts, "over_1_5": over_15, "over_2_5": over_25, "over_3_5": over_35,
        },
    }


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Football ML — WC2026", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True
    home_out: list[str] = []   # players unavailable for home (injury/suspension)
    away_out: list[str] = []   # players unavailable for away


@app.get("/api/teams")
def api_teams():
    out = []
    for t in ALL_TEAMS:
        out.append({"team": t, "elo": round(ELO.get(t, 1500.0), 1)})
    out.sort(key=lambda x: -x["elo"])
    return out


@app.get("/api/metrics")
def api_metrics():
    return METRICS


@app.get("/api/calibration")
def api_calibration():
    """Reliability-diagram data (pro model only): predicted confidence vs actual accuracy."""
    return {
        "is_pro": PRO,
        "bins": METRICS.get("calibration_bins", []),
        "rps": METRICS.get("calibrated", {}).get("rps"),
        "brier": METRICS.get("calibrated", {}).get("brier"),
        "log_loss": METRICS.get("calibrated", {}).get("log_loss"),
        "accuracy": METRICS.get("calibrated", {}).get("accuracy"),
    }


@app.get("/api/markets")
def api_markets():
    """Backtested binary-market accuracy (pro model only) — where 85-97% genuinely lives."""
    return {"is_pro": PRO, "markets": METRICS.get("binary_markets", {})}


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    if req.home == req.away:
        raise HTTPException(400, "home and away must differ")
    return predict_match(req.home, req.away, req.neutral,
                         home_out=req.home_out, away_out=req.away_out)


@app.get("/api/squads")
def api_squads():
    """Squad-strength ranking for all national teams with FIFA-24 player data."""
    if not SQUAD_OK:
        return {"available": False, "teams": []}
    import math
    cols = ["team", "squad_overall", "top11_overall", "attack_rating", "midfield_rating",
            "defense_rating", "gk_rating", "avg_age", "squad_value_m", "n_players"]

    def clean(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    teams = [{k: clean(r.get(k)) for k in cols} for r in SQUADS_DF.to_dict("records")]
    return {"available": True, "teams": teams}


@app.get("/api/squad/{team}")
def api_squad(team: str, exclude: str = ""):
    """One team's squad strength + projected (fixed) XI, each player enriched with
    REAL international goal form and club xG (when available). `exclude` = players out."""
    if not SQUAD_OK:
        raise HTTPException(404, "squad layer unavailable")
    ex = [e for e in exclude.split(",") if e.strip()]
    strength = _team_strength(team, exclude=ex)
    if not strength:
        raise HTTPException(404, f"no squad data for {team}")

    xi = _top_players(team, n=11, exclude=ex)
    if PERF_OK:
        for p in xi:
            p["intl_goals"] = _player_intl_goals(team, p["player"])
            cx = _player_club_xg(p["player"], CLUB_XG)
            if cx:
                p["club_xg"] = cx
    form = _team_intl_form(team) if PERF_OK else None
    return {"strength": strength, "xi": xi, "full": SQUAD_FULL.get(team), "intl_form": form}


@app.get("/api/wc2026/groups")
def api_groups():
    return {"groups": GROUPS, "hosts": HOST_COUNTRIES}


# Standard FIFA 4-team round-robin matchday pattern (positions 1-4 as drawn).
# Covers all 6 pairings exactly once across 3 matchdays.
_MATCHDAY_PATTERN = {
    1: [(0, 1), (2, 3)],
    2: [(0, 2), (3, 1)],
    3: [(3, 0), (1, 2)],
}
# Group-stage matchday date windows (FIFA: 11-27 June 2026). Per-group exact
# dates/venues are set by FIFA's matchday blocks; we label by matchday + window.
_MATCHDAY_WINDOW = {1: "11–17 Jun", 2: "18–23 Jun", 3: "24–27 Jun"}


_SCHEDULE_CACHE = {}


@app.get("/api/wc2026/schedule")
def api_schedule():
    """Full group-stage schedule (all 72 matches) organized by group & matchday,
    each with the model's prediction. Matchups are fixed by the official draw."""
    if _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE
    out = []
    match_no = 1
    for g, teams in GROUPS.items():
        gmatches = []
        for md in (1, 2, 3):
            for i, j in _MATCHDAY_PATTERN[md]:
                a, b = teams[i], teams[j]
                ha = a in HOST_COUNTRIES
                hb = b in HOST_COUNTRIES
                neutral = not (ha or hb)
                home, away = (a, b) if ha or not hb else (b, a)
                p = predict_match(home, away, neutral=neutral)
                gmatches.append({
                    "match_no": None,  # filled after ordering
                    "matchday": md, "window": _MATCHDAY_WINDOW[md],
                    "home": home, "away": away, "neutral": neutral,
                    "home_win": p["probabilities"]["home_win"],
                    "draw": p["probabilities"]["draw"],
                    "away_win": p["probabilities"]["away_win"],
                    "xg_home": p["expected_goals"]["home"],
                    "xg_away": p["expected_goals"]["away"],
                    "top_score": p["top_scores"][0] if p["top_scores"] else None,
                })
        out.append({"group": g, "matches": gmatches})
    _SCHEDULE_CACHE["group_stage"] = out
    _SCHEDULE_CACHE["note"] = ("Group matchups are fixed by the official draw; "
                               "exact per-match venues/dates follow FIFA's matchday windows.")
    return _SCHEDULE_CACHE


@app.get("/api/wc2026/predictions")
def api_wc_predictions():
    rows = []
    for g, teams in GROUPS.items():
        for a, b in combinations(teams, 2):
            ha = a in HOST_COUNTRIES
            hb = b in HOST_COUNTRIES
            neutral = not (ha or hb)
            home, away = (a, b) if ha or not hb else (b, a)
            p = predict_match(home, away, neutral=neutral)
            rows.append({
                "group": g, "home": home, "away": away, "neutral": neutral,
                "home_win": p["probabilities"]["home_win"],
                "draw":     p["probabilities"]["draw"],
                "away_win": p["probabilities"]["away_win"],
                "xg_home":  p["expected_goals"]["home"],
                "xg_away":  p["expected_goals"]["away"],
                "top_score": p["top_scores"][0] if p["top_scores"] else None,
            })
    return rows


def _slot_label(slot) -> str:
    """Human-readable slot spec, e.g. ('1','E')->'1E', ('3','ABCDF')->'3rd A/B/C/D/F', ('M',74)->'W74'."""
    typ, val = slot
    if typ == "1":
        return f"Winner {val}"
    if typ == "2":
        return f"Runner-up {val}"
    if typ == "3":
        return "3rd " + "/".join(val)
    if typ == "M":
        return f"Winner M{val}"
    if typ == "L":
        return f"Loser M{val}"
    return str(slot)


@app.get("/api/wc2026/bracket")
def api_bracket(n: int = 10000):
    """Official knockout bracket with most-likely occupants (runs the simulation)."""
    sim = api_simulation(n=n)
    return {"n_simulations": sim["n_simulations"], "bracket": sim["bracket"],
            "rounds": ROUND_ORDER}


@app.get("/api/wc2026/simulation")
def api_simulation(n: int = 10000):
    """Full tournament Monte-Carlo: group stage → R32 → R16 → QF → SF → Final → Champion.

    Knockout stage uses a seeded bracket (by post-group strength score). When a
    knockout match ends level, the winner is decided by an Elo-weighted shootout.
    """
    n = max(1000, min(n, 50000))
    rng = np.random.default_rng(42)

    # ----- pre-compute group-stage λ's (these never change across sims) -----
    group_edges = []
    for g, teams in GROUPS.items():
        for a, b in combinations(teams, 2):
            ha = a in HOST_COUNTRIES
            hb = b in HOST_COUNTRIES
            neutral = not (ha or hb)
            home, away = (a, b) if ha or not hb else (b, a)
            try:
                lh, la = DC.lambdas(home, away, neutral=neutral)
            except KeyError:
                X = build_feature_row(home, away, neutral)
                lh = float(REG_H.predict(X)[0])
                la = float(REG_A.predict(X)[0])
            group_edges.append((g, home, away, lh, la))

    # ----- knockout λ cache (all matches are neutral) -----
    k_cache: dict[tuple, tuple] = {}

    def k_lambdas(a: str, b: str) -> tuple[float, float]:
        key = (a, b)
        cached = k_cache.get(key)
        if cached is not None:
            return cached
        try:
            lh, la = DC.lambdas(a, b, neutral=True)
        except KeyError:
            X = build_feature_row(a, b, True)
            lh = float(REG_H.predict(X)[0])
            la = float(REG_A.predict(X)[0])
        k_cache[(a, b)] = (lh, la)
        k_cache[(b, a)] = (la, lh)
        return (lh, la)

    def play_knockout(a: str, b: str):
        """Returns (winner, loser)."""
        lh, la = k_lambdas(a, b)
        hg = rng.poisson(max(lh, 0.05))
        ag = rng.poisson(max(la, 0.05))
        if hg > ag:
            return a, b
        if ag > hg:
            return b, a
        # Penalty shootout — soft Elo bias (smaller than open-play Elo gap)
        ea = ELO.get(a, 1500.0); eb = ELO.get(b, 1500.0)
        p_a = 1.0 / (1.0 + 10 ** ((eb - ea) / 800.0))
        return (a, b) if rng.random() < p_a else (b, a)

    # ----- counters -----
    win_grp = defaultdict(int)
    top2 = defaultdict(int)
    third = defaultdict(int)
    reach = {r: defaultdict(int) for r in ROUND_ORDER}   # round -> team -> count reached
    champion = defaultdict(int)
    # per-match occupancy: match_id -> {"home": Counter, "away": Counter, "winner": Counter}
    slot_occ = {m["id"]: {"home": defaultdict(int), "away": defaultdict(int),
                          "winner": defaultdict(int)} for m in BRACKET}
    ROUND_OF = {m["id"]: m["round"] for m in BRACKET}

    # Group-stage edges indexed by group (for inner loop)
    per_group_edges: dict[str, list] = defaultdict(list)
    for g, home, away, lh, la in group_edges:
        per_group_edges[g].append((home, away, lh, la))

    # ============= main simulation loop =============
    for _ in range(n):
        # ---- 1. group stage ----
        group_table: dict[str, list] = {}   # group -> ranked team list
        group_pts: dict[str, dict] = {}
        group_gd: dict[str, dict] = {}
        group_gf: dict[str, dict] = {}
        for g, teams in GROUPS.items():
            pts = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            ga = {t: 0 for t in teams}
            for home, away, lh, la in per_group_edges[g]:
                hg = rng.poisson(max(lh, 0.05))
                ag = rng.poisson(max(la, 0.05))
                gf[home] += hg; ga[home] += ag
                gf[away] += ag; ga[away] += hg
                if hg > ag: pts[home] += 3
                elif hg < ag: pts[away] += 3
                else: pts[home] += 1; pts[away] += 1
            ranking = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
            win_grp[ranking[0]] += 1
            top2[ranking[0]] += 1
            top2[ranking[1]] += 1
            third[ranking[2]] += 1
            group_table[g] = ranking  # ordered team list
            group_pts[g] = pts; group_gd[g] = {t: gf[t] - ga[t] for t in teams}; group_gf[g] = gf

        # ---- 2. qualifiers: winners, runners-up, 8 best 3rds (FIFA tie-breakers) ----
        winners = {g: group_table[g][0] for g in GROUPS}
        runners = {g: group_table[g][1] for g in GROUPS}
        third_team_by_group = {g: group_table[g][2] for g in GROUPS}
        thirds_ranked = sorted(
            GROUPS.keys(),
            key=lambda g: (group_pts[g][third_team_by_group[g]],
                           group_gd[g][third_team_by_group[g]],
                           group_gf[g][third_team_by_group[g]]),
            reverse=True,
        )[:8]
        for g in GROUPS:
            reach["R32"][winners[g]] += 1
            reach["R32"][runners[g]] += 1
        for g in thirds_ranked:
            reach["R32"][third_team_by_group[g]] += 1

        # ---- 3. assign 8 best 3rds to the official slots, resolve R32 ----
        third_assignment = assign_thirds(thirds_ranked)
        if third_assignment is None:  # fallback (shouldn't happen)
            third_assignment = {mid: thirds_ranked[i] for i, mid in enumerate(THIRD_SLOTS)}
        r32 = resolve_r32_concrete(winners, runners, third_team_by_group, third_assignment)

        # ---- 4. play the real bracket R32 -> Final (+ 3rd-place play-off) ----
        match_winner: dict[int, str] = {}
        match_loser: dict[int, str] = {}
        for m in BRACKET:
            mid = m["id"]
            if m["round"] == "R32":
                home, away = r32[mid]
            else:
                hk = m["home"]; ak = m["away"]
                home = (match_loser if hk[0] == "L" else match_winner)[hk[1]]
                away = (match_loser if ak[0] == "L" else match_winner)[ak[1]]
            slot_occ[mid]["home"][home] += 1
            slot_occ[mid]["away"][away] += 1
            w, l = play_knockout(home, away)
            match_winner[mid] = w
            match_loser[mid] = l
            slot_occ[mid]["winner"][w] += 1
            # reach-next-round credit (knockout rounds only)
            nxt = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "F"}.get(m["round"])
            if nxt:
                reach[nxt][w] += 1
        champion[match_winner[104]] += 1

    # ============= shape response =============
    def stat(t):
        return {
            "win_group": win_grp[t] / n, "top2": top2[t] / n, "third": third[t] / n,
            "r32": reach["R32"][t] / n, "r16": reach["R16"][t] / n,
            "qf": reach["QF"][t] / n, "sf": reach["SF"][t] / n,
            "final": reach["F"][t] / n, "champion": champion[t] / n,
        }

    groups_out = []
    for g, teams in GROUPS.items():
        rows = [{"team": t, **stat(t)} for t in teams]
        rows.sort(key=lambda r: -r["top2"])
        groups_out.append({"group": g, "teams": rows})

    all_teams_flat = []
    for g, teams in GROUPS.items():
        for t in teams:
            all_teams_flat.append({"team": t, "group": g, "elo": ELO.get(t, 1500.0), **stat(t)})
    all_teams_flat.sort(key=lambda r: -r["champion"])

    # Bracket view: most-likely occupant + win prob for every match slot.
    def modal(counter):
        if not counter:
            return None
        team, c = max(counter.items(), key=lambda kv: kv[1])
        return {"team": team, "prob": c / n}

    def _src(slot):
        # feeder match id if this slot is fed by another match's winner/loser
        return slot[1] if slot[0] in ("M", "L") else None

    bracket_out = []
    for m in BRACKET:
        mid = m["id"]
        occ = slot_occ[mid]
        bracket_out.append({
            "id": mid, "round": m["round"], "venue": m["venue"], "date": m["date"],
            "home_slot": _slot_label(m["home"]), "away_slot": _slot_label(m["away"]),
            "home_src": _src(m["home"]), "away_src": _src(m["away"]),
            "home": modal(occ["home"]), "away": modal(occ["away"]),
            "winner": modal(occ["winner"]),
        })

    return {
        "n_simulations": n,
        "groups": groups_out,
        "teams": all_teams_flat,
        "bracket": bracket_out,
        "stages": ["r32", "r16", "qf", "sf", "final", "champion"],
    }


@app.get("/")
def root():
    """API index — the UI is served separately by the Next.js app (web/)."""
    return {
        "service": "Football ML — World Cup 2026 API",
        "model": METRICS.get("model"),
        "docs": "/docs",
        "endpoints": [
            "/api/teams", "/api/metrics", "/api/markets", "/api/calibration",
            "/api/predict (POST)", "/api/squads", "/api/squad/{team}",
            "/api/wc2026/groups", "/api/wc2026/schedule",
            "/api/wc2026/simulation", "/api/wc2026/bracket",
        ],
    }


if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("app.main:app", host=os.getenv("APP_HOST", "127.0.0.1"),
                port=int(os.getenv("APP_PORT", "8000")), reload=False)
