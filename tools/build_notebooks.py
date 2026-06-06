"""Generate the JupyterLab notebooks for the project.

Run once:  python tools/build_notebooks.py
Produces .ipynb files under notebooks/.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(name, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("wrote", path)


# ---------------------------------------------------------------------------
# 01 — download data
# ---------------------------------------------------------------------------
nb01 = [
    md("# 01 — Download datasets\n\n"
       "Downloads the **martj42/international_results** dataset (no API key needed) "
       "into `data/raw/`. ~47k international matches since 1872.\n\n"
       "Run this notebook first."),
    code(
        "import os, sys, pathlib, requests\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "RAW = ROOT / 'data' / 'raw'\n"
        "RAW.mkdir(parents=True, exist_ok=True)\n"
        "print('Raw data dir:', RAW)\n"
    ),
    code(
        "BASE = 'https://raw.githubusercontent.com/martj42/international_results/master/'\n"
        "FILES = ['results.csv', 'goalscorers.csv', 'shootouts.csv', 'former_names.csv']\n"
        "\n"
        "for f in FILES:\n"
        "    url = BASE + f\n"
        "    dest = RAW / f\n"
        "    if dest.exists() and dest.stat().st_size > 0:\n"
        "        print(f'skip (exists): {f}  {dest.stat().st_size:,} bytes')\n"
        "        continue\n"
        "    print(f'downloading {url} ...')\n"
        "    r = requests.get(url, timeout=60)\n"
        "    r.raise_for_status()\n"
        "    dest.write_bytes(r.content)\n"
        "    print(f'  saved {dest} ({len(r.content):,} bytes)')\n"
    ),
    code(
        "import pandas as pd\n"
        "df = pd.read_csv(RAW / 'results.csv')\n"
        "print('rows:', len(df))\n"
        "print('date range:', df.date.min(), '->', df.date.max())\n"
        "df.head()\n"
    ),
    md("### Optional: FIFA rankings\n\n"
       "FIFA does not publish a clean CSV. If you want rankings, scrape "
       "https://www.fifa.com/fifa-world-ranking/ or use a Kaggle FIFA-rankings dataset. "
       "The Elo we build in notebook 03 is a strong substitute and free."),
]
save("01_download_data.ipynb", nb01)

# ---------------------------------------------------------------------------
# 02 — explore
# ---------------------------------------------------------------------------
nb02 = [
    md("# 02 — Explore the data\n\nSanity checks, missing values, distributions, and team-name audit against the World Cup 2026 squad."),
    code(
        "import sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from src.wc2026_config import all_teams, NAME_ALIASES\n"
        "\n"
        "df = pd.read_csv(ROOT / 'data' / 'raw' / 'results.csv', parse_dates=['date'])\n"
        "df = df.sort_values('date').reset_index(drop=True)\n"
        "print(df.shape)\n"
        "df.head()\n"
    ),
    code(
        "print(df.dtypes)\n"
        "print('\\nmissing:\\n', df.isna().sum())\n"
        "print('\\ntournaments (top 15):\\n', df.tournament.value_counts().head(15))\n"
    ),
    code(
        "df['year'] = df.date.dt.year\n"
        "df.groupby('year').size().plot(figsize=(10,3), title='Matches per year')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    code(
        "df['result'] = np.where(df.home_score > df.away_score, 'H',\n"
        "                np.where(df.home_score < df.away_score, 'A', 'D'))\n"
        "print(df.result.value_counts(normalize=True).round(3))\n"
    ),
    code(
        "# Audit: which WC2026 team names don't appear in the dataset under the FIFA spelling?\n"
        "names_in_data = set(df.home_team).union(df.away_team)\n"
        "missing = [t for t in all_teams() if t not in names_in_data]\n"
        "print('WC2026 names missing from dataset:', missing)\n"
        "print('\\nIf this is non-empty, extend NAME_ALIASES in src/wc2026_config.py.')\n"
    ),
]
save("02_explore.ipynb", nb02)

# ---------------------------------------------------------------------------
# 03 — feature engineering
# ---------------------------------------------------------------------------
nb03 = [
    md("# 03 — Feature engineering\n\n"
       "Builds the feature table used for training. All features are **leak-free** — "
       "they use only past matches relative to each row.\n\n"
       "**Features**: pre-match Elo (home/away), rolling form (last 5 & 10), "
       "head-to-head record (last 5), tournament importance, neutral-ground flag, days since each team's last match."),
    code(
        "import sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "from src.elo import EloEngine, k_for\n"
        "from src.features import rolling_form, h2h_features, label_result\n"
        "from src.wc2026_config import normalize\n"
        "\n"
        "df = pd.read_csv(ROOT / 'data' / 'raw' / 'results.csv', parse_dates=['date'])\n"
        "df = df.sort_values('date').reset_index(drop=True)\n"
        "df['home_team'] = df.home_team.map(normalize)\n"
        "df['away_team'] = df.away_team.map(normalize)\n"
        "# Drop unplayed fixtures (e.g. the WC2026 schedule that ships in this CSV).\n"
        "unplayed = df[df.home_score.isna() | df.away_score.isna()]\n"
        "print('unplayed fixtures dropped:', len(unplayed))\n"
        "if len(unplayed):\n"
        "    unplayed.to_csv(ROOT / 'data' / 'processed' / 'unplayed_fixtures.csv', index=False)\n"
        "df = df.dropna(subset=['home_score','away_score']).reset_index(drop=True)\n"
        "print(len(df), 'played matches')\n"
    ),
    code(
        "# --- Pre-match Elo (online update, walk-forward, no leakage) ---\n"
        "engine = EloEngine()\n"
        "h_elo_pre, a_elo_pre = [], []\n"
        "for _, row in df.iterrows():\n"
        "    h_elo_pre.append(engine.get(row.home_team))\n"
        "    a_elo_pre.append(engine.get(row.away_team))\n"
        "    engine.update(row.home_team, row.away_team,\n"
        "                  int(row.home_score), int(row.away_score),\n"
        "                  tournament=row.tournament, neutral=bool(row.neutral))\n"
        "df['home_elo'] = h_elo_pre\n"
        "df['away_elo'] = a_elo_pre\n"
        "df['elo_diff'] = df.home_elo - df.away_elo\n"
        "print('Elo computed. sample:')\n"
        "df[['date','home_team','away_team','home_elo','away_elo']].tail()\n"
    ),
    code(
        "# Persist final Elo table (used by the WC2026 prediction notebook).\n"
        "final = pd.DataFrame({'team': list(engine.ratings.keys()), 'elo': list(engine.ratings.values())})\n"
        "final = final.sort_values('elo', ascending=False)\n"
        "(ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)\n"
        "final.to_csv(ROOT / 'data' / 'processed' / 'final_elo.csv', index=False)\n"
        "final.head(20)\n"
    ),
    code(
        "# --- Rolling form (last 5 and last 10) ---\n"
        "df = rolling_form(df, window=5).rename(columns={\n"
        "    'home_form_pts':'home_form5_pts','home_form_gf':'home_form5_gf','home_form_ga':'home_form5_ga',\n"
        "    'away_form_pts':'away_form5_pts','away_form_gf':'away_form5_gf','away_form_ga':'away_form5_ga',\n"
        "})\n"
        "df = rolling_form(df, window=10).rename(columns={\n"
        "    'home_form_pts':'home_form10_pts','home_form_gf':'home_form10_gf','home_form_ga':'home_form10_ga',\n"
        "    'away_form_pts':'away_form10_pts','away_form_gf':'away_form10_gf','away_form_ga':'away_form10_ga',\n"
        "})\n"
        "print('form done')\n"
    ),
    code(
        "# --- Head-to-head (last 5 meetings) ---\n"
        "df = h2h_features(df, window=5)\n"
        "print('h2h done')\n"
    ),
    code(
        "# --- Other features ---\n"
        "df['tournament_k'] = df.tournament.map(k_for)\n"
        "df['neutral_int'] = df.neutral.astype(int)\n"
        "df['year'] = df.date.dt.year\n"
        "\n"
        "# Days since each team's last match (rest)\n"
        "long = pd.concat([\n"
        "    df[['date','home_team']].rename(columns={'home_team':'team'}).assign(idx=df.index, side='h'),\n"
        "    df[['date','away_team']].rename(columns={'away_team':'team'}).assign(idx=df.index, side='a'),\n"
        "]).sort_values(['team','date'])\n"
        "long['rest'] = long.groupby('team')['date'].diff().dt.days.fillna(365).clip(upper=365)\n"
        "rest_h = long[long.side=='h'].set_index('idx')['rest']\n"
        "rest_a = long[long.side=='a'].set_index('idx')['rest']\n"
        "df['home_rest'] = rest_h\n"
        "df['away_rest'] = rest_a\n"
        "\n"
        "df['target'] = df.apply(label_result, axis=1)\n"
        "df['goal_diff'] = df.home_score - df.away_score\n"
        "print('total features done, shape:', df.shape)\n"
    ),
    code(
        "out = ROOT / 'data' / 'processed' / 'matches_features.parquet'\n"
        "df.to_parquet(out, index=False)\n"
        "print('wrote', out, df.shape)\n"
        "df.head()\n"
    ),
]
save("03_feature_engineering.ipynb", nb03)

# ---------------------------------------------------------------------------
# 04 — train/test split
# ---------------------------------------------------------------------------
nb04 = [
    md("# 04 — Train / Test split\n\n"
       "**Time-based split** (no random shuffle — football is temporal):\n"
       "- **Train**: matches before 2018-01-01 (~80%)\n"
       "- **Test**:  matches from 2018-01-01 to today (~20%)\n\n"
       "We also drop the earliest matches where Elo hasn't converged."),
    code(
        "import sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "df = pd.read_parquet(ROOT / 'data' / 'processed' / 'matches_features.parquet')\n"
        "df = df[df.date >= '1950-01-01'].reset_index(drop=True)  # let Elo settle\n"
        "print('rows after burn-in:', len(df))\n"
    ),
    code(
        "FEATURES = [\n"
        "    'home_elo','away_elo','elo_diff',\n"
        "    'home_form5_pts','home_form5_gf','home_form5_ga',\n"
        "    'away_form5_pts','away_form5_gf','away_form5_ga',\n"
        "    'home_form10_pts','home_form10_gf','home_form10_ga',\n"
        "    'away_form10_pts','away_form10_gf','away_form10_ga',\n"
        "    'h2h_home_wins','h2h_draws','h2h_away_wins','h2h_home_gf','h2h_home_ga',\n"
        "    'tournament_k','neutral_int','home_rest','away_rest',\n"
        "]\n"
        "TARGET_CLS = 'target'        # 0=home,1=draw,2=away\n"
        "TARGET_HGOALS = 'home_score'\n"
        "TARGET_AGOALS = 'away_score'\n"
        "\n"
        "df = df.dropna(subset=FEATURES).reset_index(drop=True)\n"
        "print('rows after dropna:', len(df))\n"
    ),
    code(
        "CUTOFF = '2018-01-01'\n"
        "train = df[df.date <  CUTOFF].copy()\n"
        "test  = df[df.date >= CUTOFF].copy()\n"
        "print(f'train: {len(train):,}  test: {len(test):,}')\n"
        "print('train target dist:\\n', train.target.value_counts(normalize=True).round(3))\n"
        "print('test  target dist:\\n', test.target.value_counts(normalize=True).round(3))\n"
    ),
    code(
        "OUT = ROOT / 'data' / 'train_test'\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "cols = ['date','home_team','away_team','tournament','neutral'] + FEATURES + [TARGET_CLS, TARGET_HGOALS, TARGET_AGOALS]\n"
        "train[cols].to_parquet(OUT / 'train.parquet', index=False)\n"
        "test[cols].to_parquet(OUT / 'test.parquet', index=False)\n"
        "pd.Series(FEATURES).to_csv(OUT / 'feature_list.csv', index=False, header=['feature'])\n"
        "print('wrote', OUT)\n"
    ),
]
save("04_train_test_split.ipynb", nb04)

# ---------------------------------------------------------------------------
# 05 — train models
# ---------------------------------------------------------------------------
nb05 = [
    md("# 05 — Train models\n\n"
       "Two models:\n"
       "1. **XGBoost classifier** → P(home_win, draw, away_win)\n"
       "2. **Poisson regressors** (one per side) → expected goals\n\n"
       "Saved under `models/` for the WC2026 prediction notebook."),
    code(
        "import sys, pathlib, joblib\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "from sklearn.metrics import accuracy_score, log_loss, classification_report, mean_absolute_error\n"
        "from sklearn.linear_model import PoissonRegressor\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n"
        "import xgboost as xgb\n"
        "\n"
        "TT = ROOT / 'data' / 'train_test'\n"
        "train = pd.read_parquet(TT / 'train.parquet')\n"
        "test  = pd.read_parquet(TT / 'test.parquet')\n"
        "FEATURES = pd.read_csv(TT / 'feature_list.csv').feature.tolist()\n"
        "print('train', train.shape, 'test', test.shape, '| features:', len(FEATURES))\n"
    ),
    code(
        "X_train, y_train = train[FEATURES], train['target']\n"
        "X_test,  y_test  = test[FEATURES],  test['target']\n"
        "\n"
        "clf = xgb.XGBClassifier(\n"
        "    n_estimators=600, max_depth=5, learning_rate=0.05,\n"
        "    subsample=0.9, colsample_bytree=0.8,\n"
        "    objective='multi:softprob', num_class=3,\n"
        "    eval_metric='mlogloss', tree_method='hist', n_jobs=-1,\n"
        ")\n"
        "clf.fit(X_train, y_train)\n"
        "proba = clf.predict_proba(X_test)\n"
        "pred = proba.argmax(axis=1)\n"
        "print('accuracy:', round(accuracy_score(y_test, pred), 4))\n"
        "print('log loss:', round(log_loss(y_test, proba), 4))\n"
        "print(classification_report(y_test, pred, target_names=['home','draw','away']))\n"
    ),
    code(
        "# Feature importances\n"
        "imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)\n"
        "imp.head(15)\n"
    ),
    code(
        "# --- Poisson regressors for expected goals (scaled — Elo is on a 1500-2200 scale) ---\n"
        "def poisson_pipe():\n"
        "    return Pipeline([('scale', StandardScaler()), ('poi', PoissonRegressor(alpha=0.01, max_iter=1000))])\n"
        "pr_home = poisson_pipe().fit(X_train, train['home_score'])\n"
        "pr_away = poisson_pipe().fit(X_train, train['away_score'])\n"
        "pred_h = pr_home.predict(X_test)\n"
        "pred_a = pr_away.predict(X_test)\n"
        "print('home goals MAE:', round(mean_absolute_error(test.home_score, pred_h), 3))\n"
        "print('away goals MAE:', round(mean_absolute_error(test.away_score, pred_a), 3))\n"
    ),
    code(
        "# --- Save models ---\n"
        "MOD = ROOT / 'models'; MOD.mkdir(parents=True, exist_ok=True)\n"
        "joblib.dump(clf,     MOD / 'xgb_result.joblib')\n"
        "joblib.dump(pr_home, MOD / 'poisson_home.joblib')\n"
        "joblib.dump(pr_away, MOD / 'poisson_away.joblib')\n"
        "joblib.dump(FEATURES, MOD / 'feature_list.joblib')\n"
        "print('saved models to', MOD)\n"
    ),
]
save("05_train_models.ipynb", nb05)

# ---------------------------------------------------------------------------
# 06 — WC2026 prediction
# ---------------------------------------------------------------------------
nb06 = [
    md("# 06 — Predict FIFA World Cup 2026\n\n"
       "Uses trained models + final Elo + the official 2026 groups to predict every group match, "
       "then runs a **10,000-rollout Monte Carlo** simulation of the group stage to estimate "
       "qualification probabilities for each team."),
    code(
        "import sys, pathlib, joblib\n"
        "ROOT = pathlib.Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd, numpy as np\n"
        "from itertools import combinations\n"
        "from collections import defaultdict\n"
        "from src.wc2026_config import GROUPS, HOST_COUNTRIES, normalize, all_teams\n"
        "from src.elo import k_for, HOME_ADV\n"
        "\n"
        "clf      = joblib.load(ROOT / 'models' / 'xgb_result.joblib')\n"
        "pr_home  = joblib.load(ROOT / 'models' / 'poisson_home.joblib')\n"
        "pr_away  = joblib.load(ROOT / 'models' / 'poisson_away.joblib')\n"
        "FEATURES = joblib.load(ROOT / 'models' / 'feature_list.joblib')\n"
        "elo      = pd.read_csv(ROOT / 'data' / 'processed' / 'final_elo.csv').set_index('team').elo.to_dict()\n"
        "matches  = pd.read_parquet(ROOT / 'data' / 'processed' / 'matches_features.parquet')\n"
        "matches['home_team'] = matches.home_team.map(normalize)\n"
        "matches['away_team'] = matches.away_team.map(normalize)\n"
        "print('models + elo loaded. elo entries:', len(elo))\n"
    ),
    code(
        "# Most recent form snapshot per team (from feature table).\n"
        "def latest_form(team):\n"
        "    h = matches[matches.home_team == team].tail(1)\n"
        "    a = matches[matches.away_team == team].tail(1)\n"
        "    cand = pd.concat([h.assign(side='h'), a.assign(side='a')])\n"
        "    if cand.empty:\n"
        "        return None\n"
        "    row = cand.sort_values('date').tail(1).iloc[0]\n"
        "    if row.side == 'h':\n"
        "        return dict(form5_pts=row.home_form5_pts, form5_gf=row.home_form5_gf, form5_ga=row.home_form5_ga,\n"
        "                    form10_pts=row.home_form10_pts, form10_gf=row.home_form10_gf, form10_ga=row.home_form10_ga)\n"
        "    return dict(form5_pts=row.away_form5_pts, form5_gf=row.away_form5_gf, form5_ga=row.away_form5_ga,\n"
        "                form10_pts=row.away_form10_pts, form10_gf=row.away_form10_gf, form10_ga=row.away_form10_ga)\n"
        "\n"
        "form_cache = {t: latest_form(t) for t in all_teams()}\n"
        "missing = [t for t,v in form_cache.items() if v is None]\n"
        "print('teams with no form data:', missing)\n"
    ),
    code(
        "def build_row(home, away, tournament='FIFA World Cup', neutral=True):\n"
        "    fh = form_cache.get(home) or dict(form5_pts=1, form5_gf=1, form5_ga=1, form10_pts=1, form10_gf=1, form10_ga=1)\n"
        "    fa = form_cache.get(away) or dict(form5_pts=1, form5_gf=1, form5_ga=1, form10_pts=1, form10_gf=1, form10_ga=1)\n"
        "    h_elo = elo.get(home, 1500.0)\n"
        "    a_elo = elo.get(away, 1500.0)\n"
        "    # crude H2H proxy = recent meetings in matches table\n"
        "    h2h = matches[((matches.home_team==home)&(matches.away_team==away)) |\n"
        "                  ((matches.home_team==away)&(matches.away_team==home))].tail(5)\n"
        "    hw=dr=aw=0; gf=ga=0.0\n"
        "    for _, r in h2h.iterrows():\n"
        "        if r.home_team == home:\n"
        "            hg, ag = r.home_score, r.away_score\n"
        "        else:\n"
        "            hg, ag = r.away_score, r.home_score\n"
        "        if hg>ag: hw+=1\n"
        "        elif hg<ag: aw+=1\n"
        "        else: dr+=1\n"
        "        gf += hg; ga += ag\n"
        "    n = max(len(h2h), 1)\n"
        "    return {\n"
        "        'home_elo': h_elo, 'away_elo': a_elo, 'elo_diff': h_elo-a_elo,\n"
        "        'home_form5_pts': fh['form5_pts'], 'home_form5_gf': fh['form5_gf'], 'home_form5_ga': fh['form5_ga'],\n"
        "        'away_form5_pts': fa['form5_pts'], 'away_form5_gf': fa['form5_gf'], 'away_form5_ga': fa['form5_ga'],\n"
        "        'home_form10_pts': fh['form10_pts'], 'home_form10_gf': fh['form10_gf'], 'home_form10_ga': fh['form10_ga'],\n"
        "        'away_form10_pts': fa['form10_pts'], 'away_form10_gf': fa['form10_gf'], 'away_form10_ga': fa['form10_ga'],\n"
        "        'h2h_home_wins': hw, 'h2h_draws': dr, 'h2h_away_wins': aw,\n"
        "        'h2h_home_gf': gf/n, 'h2h_home_ga': ga/n,\n"
        "        'tournament_k': k_for(tournament),\n"
        "        'neutral_int': int(neutral), 'home_rest': 4, 'away_rest': 4,\n"
        "    }\n"
        "\n"
        "def predict(home, away, neutral=True):\n"
        "    row = build_row(home, away, neutral=neutral)\n"
        "    X = pd.DataFrame([row])[FEATURES]\n"
        "    p = clf.predict_proba(X)[0]\n"
        "    eh = float(pr_home.predict(X)[0])\n"
        "    ea = float(pr_away.predict(X)[0])\n"
        "    return {'home_win': float(p[0]), 'draw': float(p[1]), 'away_win': float(p[2]),\n"
        "            'expected_home_goals': eh, 'expected_away_goals': ea}\n"
        "\n"
        "predict('Brazil', 'Morocco')\n"
    ),
    code(
        "# Predict every group match\n"
        "rows = []\n"
        "for g, teams in GROUPS.items():\n"
        "    for a, b in combinations(teams, 2):\n"
        "        # Treat host countries as home when they play in their own group; else neutral.\n"
        "        host_a = a in HOST_COUNTRIES\n"
        "        host_b = b in HOST_COUNTRIES\n"
        "        neutral = not (host_a or host_b)\n"
        "        home, away = (a, b) if host_a or not host_b else (b, a)\n"
        "        p = predict(home, away, neutral=neutral)\n"
        "        rows.append(dict(group=g, home=home, away=away, neutral=neutral, **p))\n"
        "preds = pd.DataFrame(rows)\n"
        "(ROOT / 'outputs').mkdir(exist_ok=True)\n"
        "preds.to_csv(ROOT / 'outputs' / 'group_stage_predictions.csv', index=False)\n"
        "preds.head(12)\n"
    ),
    code(
        "# Monte Carlo group-stage simulator using Poisson expected goals.\n"
        "rng = np.random.default_rng(42)\n"
        "N_SIM = 10_000\n"
        "\n"
        "# Precompute expected goals per (home, away) ordered pair in each group.\n"
        "lookup = {(r.home, r.away): (r.expected_home_goals, r.expected_away_goals) for r in preds.itertuples()}\n"
        "\n"
        "advance_count = defaultdict(int)   # advance as top-2\n"
        "winner_count  = defaultdict(int)\n"
        "third_count   = defaultdict(int)   # finished 3rd (best-third candidates)\n"
        "\n"
        "for _ in range(N_SIM):\n"
        "    for g, teams in GROUPS.items():\n"
        "        pts = {t: 0 for t in teams}\n"
        "        gf  = {t: 0 for t in teams}\n"
        "        ga  = {t: 0 for t in teams}\n"
        "        for a, b in combinations(teams, 2):\n"
        "            if (a, b) in lookup:\n"
        "                eh, ea = lookup[(a, b)]; home, away = a, b\n"
        "            else:\n"
        "                eh, ea = lookup[(b, a)]; home, away = b, a\n"
        "            hg = rng.poisson(max(eh, 0.05))\n"
        "            ag = rng.poisson(max(ea, 0.05))\n"
        "            gf[home] += hg; ga[home] += ag; gf[away] += ag; ga[away] += hg\n"
        "            if hg > ag: pts[home] += 3\n"
        "            elif hg < ag: pts[away] += 3\n"
        "            else: pts[home] += 1; pts[away] += 1\n"
        "        ranking = sorted(teams, key=lambda t: (pts[t], gf[t]-ga[t], gf[t]), reverse=True)\n"
        "        winner_count[ranking[0]] += 1\n"
        "        advance_count[ranking[0]] += 1\n"
        "        advance_count[ranking[1]] += 1\n"
        "        third_count[ranking[2]]   += 1\n"
        "\n"
        "summary = []\n"
        "for g, teams in GROUPS.items():\n"
        "    for t in teams:\n"
        "        summary.append({\n"
        "            'group': g, 'team': t,\n"
        "            'win_group_%':   round(100 * winner_count[t]  / N_SIM, 1),\n"
        "            'top2_%':        round(100 * advance_count[t] / N_SIM, 1),\n"
        "            'third_place_%': round(100 * third_count[t]   / N_SIM, 1),\n"
        "        })\n"
        "summary_df = pd.DataFrame(summary).sort_values(['group','top2_%'], ascending=[True, False])\n"
        "summary_df.to_csv(ROOT / 'outputs' / 'group_stage_simulation.csv', index=False)\n"
        "summary_df\n"
    ),
    md("### Next steps\n\n"
       "- Add **player-level features** (FBref club stats, Transfermarkt market value) per docs1.md §C–E.\n"
       "- Build the **bracket generator** for the new Round of 32 (docs3.md) and extend the simulator end-to-end to champion probabilities.\n"
       "- Ensemble: blend XGBoost with CatBoost + a neural net (docs1.md Layer 6).\n"
       "- Daily refresh: re-run notebook 03 every day until kickoff to keep Elo + form current."),
]
save("06_predict_wc2026.ipynb", nb06)

print('\nAll notebooks generated.')
