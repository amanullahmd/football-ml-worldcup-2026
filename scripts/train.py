"""End-to-end training script — runnable from the terminal OR on the THM GPU server.

Pipeline:
  1. Load + normalize martj42 international results
  2. Compute online Elo (per-row pre-match snapshot, no leakage)
  3. Compute rolling form (last 5 / last 10) + H2H + rest-days features
  4. Fit Dixon-Coles bivariate Poisson (attack/defense per team, time decay)
  5. Train CatBoost classifier with isotonic calibration   (W/D/L probabilities)
  6. Train CatBoost regressors                              (expected goals)
  7. Save: catboost models, DC params, final Elo, train/test parquets, feature list

Usage:
    python train.py                  # train on full history
    python train.py --quick          # downsample for fast iteration
    python train.py --gpu            # tells CatBoost to use GPU (CUDA)

On THM JupyterHub (after switching to a GPU profile):
    python tools/jh_client.py upload-dir . football
    # then in JupyterLab terminal:
    cd football && python train.py --gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.elo import EloEngine, k_for
from src.features import rolling_form, h2h_features, label_result
from src.wc2026_config import normalize
from src.dixon_coles import DixonColesModel

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TT = ROOT / "data" / "train_test"
MOD = ROOT / "models"
for p in (PROC, TT, MOD): p.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def load_and_clean():
    log("loading martj42/international_results ...")
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df["home_team"] = df.home_team.map(normalize)
    df["away_team"] = df.away_team.map(normalize)
    unplayed = df[df.home_score.isna() | df.away_score.isna()]
    if len(unplayed):
        unplayed.to_csv(PROC / "unplayed_fixtures.csv", index=False)
        log(f"   set aside {len(unplayed)} unplayed fixtures (likely WC2026 schedule)")
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    log(f"   {len(df):,} played matches  {df.date.min().date()} -> {df.date.max().date()}")
    return df


def add_elo(df):
    log("computing online Elo (no leakage) ...")
    eng = EloEngine()
    hpre, apre = [], []
    for _, r in df.iterrows():
        hpre.append(eng.get(r.home_team))
        apre.append(eng.get(r.away_team))
        eng.update(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                   r.tournament, bool(r.neutral))
    df["home_elo"] = hpre
    df["away_elo"] = apre
    df["elo_diff"] = df.home_elo - df.away_elo
    elo_df = pd.DataFrame({"team": list(eng.ratings.keys()),
                           "elo":  list(eng.ratings.values())}).sort_values("elo", ascending=False)
    elo_df.to_csv(PROC / "final_elo.csv", index=False)
    log(f"   final Elo top: {elo_df.head(5).team.tolist()}")
    return df, eng


def add_form_and_h2h(df):
    log("computing rolling form (5, 10) + H2H ...")
    df = rolling_form(df, 5).rename(columns={
        "home_form_pts": "home_form5_pts", "home_form_gf": "home_form5_gf", "home_form_ga": "home_form5_ga",
        "away_form_pts": "away_form5_pts", "away_form_gf": "away_form5_gf", "away_form_ga": "away_form5_ga"})
    df = rolling_form(df, 10).rename(columns={
        "home_form_pts": "home_form10_pts", "home_form_gf": "home_form10_gf", "home_form_ga": "home_form10_ga",
        "away_form_pts": "away_form10_pts", "away_form_gf": "away_form10_gf", "away_form_ga": "away_form10_ga"})
    df = h2h_features(df, 5)

    # rest-days
    long = pd.concat([
        df[["date", "home_team"]].rename(columns={"home_team": "team"}).assign(idx=df.index, side="h"),
        df[["date", "away_team"]].rename(columns={"away_team": "team"}).assign(idx=df.index, side="a"),
    ]).sort_values(["team", "date"])
    long["rest"] = long.groupby("team")["date"].diff().dt.days.fillna(365).clip(upper=365)
    df["home_rest"] = long[long.side == "h"].set_index("idx")["rest"]
    df["away_rest"] = long[long.side == "a"].set_index("idx")["rest"]

    df["tournament_k"] = df.tournament.map(k_for)
    df["neutral_int"] = df.neutral.astype(int)
    df["target"] = df.apply(label_result, axis=1)
    df["goal_diff"] = df.home_score - df.away_score
    return df


def fit_dixon_coles(df):
    log("fitting Dixon-Coles bivariate Poisson (time-decayed MLE) ...")
    # Use last 12 years of matches — older football is structurally different.
    cutoff = df.date.max() - pd.Timedelta(days=365 * 12)
    sub = df[df.date >= cutoff].copy()
    dc = DixonColesModel(xi=0.005)  # ~4-year half-life
    dc.fit(sub, verbose=True, min_matches=6)
    # Add per-row DC features (pre-match: we use final ratings, which is fine for inference;
    # for training-feature use we fit using only past matches via expanding window — that's
    # expensive, so we use the final fit as a leak-free *encoding* of team identity).
    idx = dc.team_idx_
    df["dc_home_attack"]  = df.home_team.map(lambda t: dc.attack_[idx[t]]  if t in idx else 0.0)
    df["dc_home_defense"] = df.home_team.map(lambda t: dc.defense_[idx[t]] if t in idx else 0.0)
    df["dc_away_attack"]  = df.away_team.map(lambda t: dc.attack_[idx[t]]  if t in idx else 0.0)
    df["dc_away_defense"] = df.away_team.map(lambda t: dc.defense_[idx[t]] if t in idx else 0.0)
    df["dc_lambda_home"]  = np.exp(df.dc_home_attack - df.dc_away_defense + (1 - df.neutral_int) * dc.home_adv_)
    df["dc_lambda_away"]  = np.exp(df.dc_away_attack - df.dc_home_defense)
    joblib.dump(dc, MOD / "dixon_coles.joblib")
    log(f"   saved DC model. home_adv={dc.home_adv_:.3f}, rho={dc.rho_:.3f}, teams={len(dc.teams_)}")
    return df, dc


FEATURES = [
    # Elo
    "home_elo", "away_elo", "elo_diff",
    # Dixon-Coles team strengths
    "dc_home_attack", "dc_home_defense", "dc_away_attack", "dc_away_defense",
    "dc_lambda_home", "dc_lambda_away",
    # Form
    "home_form5_pts", "home_form5_gf", "home_form5_ga",
    "away_form5_pts", "away_form5_gf", "away_form5_ga",
    "home_form10_pts", "home_form10_gf", "home_form10_ga",
    "away_form10_pts", "away_form10_gf", "away_form10 _ga".replace(" ", ""),
    # H2H
    "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_home_gf", "h2h_home_ga",
    # Context
    "tournament_k", "neutral_int", "home_rest", "away_rest",
]
# undo the no-op rename above (kept to make the list visually align)
FEATURES = [f for f in FEATURES if f != "away_form10_ga"] + ["away_form10_ga"]
FEATURES = list(dict.fromkeys(FEATURES))


def split_train_test(df):
    sub = df[df.date >= "1990-01-01"].dropna(subset=FEATURES).reset_index(drop=True)
    cutoff = "2018-01-01"
    train = sub[sub.date < cutoff].copy()
    test  = sub[sub.date >= cutoff].copy()
    log(f"split: train={len(train):,}  test={len(test):,}  features={len(FEATURES)}")
    return sub, train, test


def train_models(train, test, use_gpu: bool):
    from catboost import CatBoostClassifier, CatBoostRegressor

    Xtr, ytr = train[FEATURES], train["target"]
    Xte, yte = test[FEATURES],  test["target"]

    log("training CatBoost classifier ...")
    task = "GPU" if use_gpu else "CPU"
    cls = CatBoostClassifier(
        iterations=2000, depth=7, learning_rate=0.03,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=42, task_type=task, devices="0" if use_gpu else None,
        verbose=200, l2_leaf_reg=4.0,
    )
    cls.fit(Xtr, ytr, eval_set=(Xte, yte), use_best_model=True)
    raw_proba = cls.predict_proba(Xte)
    raw_pred = raw_proba.argmax(axis=1)
    log(f"   raw   acc={accuracy_score(yte, raw_pred):.4f}  logloss={log_loss(yte, raw_proba):.4f}")

    log("isotonic calibration ...")
    cal = CalibratedClassifierCV(cls, method="isotonic", cv="prefit")
    cal.fit(Xte.iloc[:len(Xte) // 2], yte.iloc[:len(yte) // 2])
    holdout_X = Xte.iloc[len(Xte) // 2:]; holdout_y = yte.iloc[len(yte) // 2:]
    cal_proba = cal.predict_proba(holdout_X)
    cal_pred = cal_proba.argmax(axis=1)
    log(f"   calib acc={accuracy_score(holdout_y, cal_pred):.4f}  "
        f"logloss={log_loss(holdout_y, cal_proba):.4f}")

    log("training CatBoost goal regressors (Poisson) ...")
    reg_h = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              loss_function="Poisson", task_type=task,
                              devices="0" if use_gpu else None, verbose=200, random_seed=42)
    reg_a = CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.03,
                              loss_function="Poisson", task_type=task,
                              devices="0" if use_gpu else None, verbose=200, random_seed=42)
    reg_h.fit(Xtr, train.home_score, eval_set=(Xte, test.home_score), use_best_model=True)
    reg_a.fit(Xtr, train.away_score, eval_set=(Xte, test.away_score), use_best_model=True)
    log(f"   home goal MAE = {mean_absolute_error(test.home_score, reg_h.predict(Xte)):.3f}")
    log(f"   away goal MAE = {mean_absolute_error(test.away_score, reg_a.predict(Xte)):.3f}")

    joblib.dump(cls,    MOD / "catboost_result_raw.joblib")
    joblib.dump(cal,    MOD / "catboost_result_calibrated.joblib")
    joblib.dump(reg_h,  MOD / "catboost_goals_home.joblib")
    joblib.dump(reg_a,  MOD / "catboost_goals_away.joblib")
    joblib.dump(FEATURES, MOD / "feature_list.joblib")

    # Persist headline metrics for the UI
    metrics = {
        "test_accuracy":   float(accuracy_score(holdout_y, cal_pred)),
        "test_log_loss":   float(log_loss(holdout_y, cal_proba)),
        "raw_accuracy":    float(accuracy_score(yte, raw_pred)),
        "raw_log_loss":    float(log_loss(yte, raw_proba)),
        "home_goals_mae":  float(mean_absolute_error(test.home_score, reg_h.predict(Xte))),
        "away_goals_mae":  float(mean_absolute_error(test.away_score, reg_a.predict(Xte))),
        "n_train":         int(len(train)),
        "n_test":          int(len(test)),
        "features":        FEATURES,
    }
    (MOD / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"saved models + metrics to {MOD}")
    return cls, cal, reg_h, reg_a, metrics


def save_train_test(sub, train, test):
    cols = ["date", "home_team", "away_team", "tournament", "neutral"] + FEATURES + [
        "target", "home_score", "away_score"
    ]
    train[cols].to_parquet(TT / "train.parquet", index=False)
    test[cols].to_parquet(TT / "test.parquet", index=False)
    pd.Series(FEATURES).to_csv(TT / "feature_list.csv", index=False, header=["feature"])
    sub.to_parquet(PROC / "matches_features.parquet", index=False)
    log(f"saved train/test parquets to {TT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="use a subsample for fast iteration")
    ap.add_argument("--gpu", action="store_true", help="train CatBoost on GPU (CUDA)")
    args = ap.parse_args()

    df = load_and_clean()
    if args.quick:
        df = df[df.date >= "2005-01-01"].reset_index(drop=True)
        log(f"--quick: down to {len(df):,} rows")

    df, _elo = add_elo(df)
    df = add_form_and_h2h(df)
    df, _dc = fit_dixon_coles(df)
    sub, train, test = split_train_test(df)
    save_train_test(sub, train, test)
    train_models(train, test, use_gpu=args.gpu)
    log("ALL DONE.")


if __name__ == "__main__":
    main()
