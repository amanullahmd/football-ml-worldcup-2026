<div align="center">

# ⚽ Football ML — FIFA World Cup 2026 Prediction Engine

**Calibrated football match & tournament forecasting.**
Dixon‑Coles bivariate Poisson ⊕ a calibrated CatBoost ensemble, trained on
**49k+ real international matches (1872–2026)**, served via a FastAPI ML API and a
modern Next.js 16 frontend.

</div>

---

## ✨ Features

- **Any‑match predictor** — Win/Draw/Loss probabilities, expected goals, full
  Poisson **score‑matrix**, BTTS & over/under markets, and a **"who's‑missing"
  what‑if** (remove injured players → attack drops → prediction shifts).
- **World Cup 2026** — the official **104‑match schedule** (72 group + 32 knockout),
  the real FIFA **bracket with connector arrows** (R32 → Final + 3rd‑place), and a
  **Monte‑Carlo simulation** (up to 25k runs) to champion probability.
- **Squad & XI intelligence** — squad strength for **135 nations** from FIFA‑24
  ratings, a **projected starting XI**, and each player's **real international
  goal form** (plus optional live club xG).
- **Honest metrics** — accuracy, **Ranked Probability Score (RPS)**, Brier, a
  **calibration reliability diagram**, and a **binary‑market backtest** showing
  where 85–97% accuracy genuinely lives.

> **On accuracy:** 3‑way W/D/L has a structural ceiling of ~62–64% (even bookmaker
> closing odds resolve to ~55–58%). This model hits **60.4%** with **RPS 0.167** and
> excellent calibration — the metrics professional forecasters actually optimize.
> All predictions are **model‑derived from real data — no mock/dummy values.**

---

## 🏗️ Architecture

```
                 ┌─────────────────────┐      HTTP/JSON      ┌────────────────────┐
   Browser  ───► │  Next.js 16 (web/)  │ ─────────────────► │  FastAPI (api/)    │
                 │  TS · Tailwind      │ ◄───────────────── │  ML prediction API │
                 └─────────────────────┘                    └─────────┬──────────┘
                                                                       │ loads
                                              ┌────────────────────────┼───────────────┐
                                              ▼                        ▼               ▼
                                        models/ (.joblib)     src/ (ML library)   data/ (parquet)
                                        DC + CatBoost stack    Elo · DC · features  features + Elo
```

The Python side owns **all** modelling; the frontend is a pure consumer of the API.

---

## 📁 Project structure

```
football/
├── api/                  # FastAPI backend (ML prediction API)
│   └── main.py
├── src/                  # Python ML library
│   ├── elo.py                # online Elo rating engine
│   ├── dixon_coles.py        # Dixon-Coles bivariate Poisson model
│   ├── features.py           # rolling form, head-to-head, rest-days
│   ├── squad.py              # FIFA-24 squad strength + projected XI
│   ├── player_perf.py        # real international goal form + club xG
│   ├── wc2026_config.py      # official 2026 groups + name aliases
│   └── wc2026_bracket.py     # official knockout bracket + 3rd-place rule
├── scripts/              # runnable entry points
│   ├── download_data.py      # fetch all datasets (no API key)
│   ├── train.py              # baseline pipeline (CatBoost + DC)
│   └── train_pro.py          # "pro" stacked ensemble + calibration + backtest
├── web/                  # Next.js 16 frontend (TS + Tailwind)
│   ├── app/                  # Home, Predict, World Cup, Squads pages
│   ├── components/           # Nav, ExtensionErrorGuard
│   └── lib/api.ts            # typed API client
├── notebooks/            # 01–06 exploratory pipeline (JupyterLab)
├── tools/                # jh_client.py (THM GPU), build_notebooks.py
├── docs/                 # design notes
├── data/                 # datasets (gitignored — regenerate via scripts)
├── models/               # trained artifacts (gitignored — regenerate)
├── outputs/              # saved predictions/sims (gitignored)
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick start

### 1. Backend (Python ≥ 3.10)

```bash
pip install -r requirements.txt

python scripts/download_data.py     # download datasets (~3 min, no key needed)
python scripts/train_pro.py          # train the pro ensemble (CPU; --gpu for CUDA)

python -m uvicorn api.main:app --port 8000     # ML API → http://127.0.0.1:8000
```

> A fresh clone has no `models/` or `data/` (they're gitignored). Run the
> `download_data.py` → `train_pro.py` steps once to generate them, then start the API.

### 2. Frontend (Node ≥ 20.9)

```bash
cd web
npm install
npm run dev                          # http://localhost:3000  (Next.js 16, Turbopack)
```

The frontend reads `NEXT_PUBLIC_API_BASE` (defaults to `http://127.0.0.1:8000`);
CORS for `localhost:3000` is enabled on the API.

---

## 🧠 Model stack

| Component | What it does |
|---|---|
| **Elo engine** (`src/elo.py`) | Online team ratings; K scaled by competition + goal diff; leak‑free pre‑match snapshot |
| **Dixon‑Coles** (`src/dixon_coles.py`) | Bivariate Poisson with attack/defense per team, home advantage, low‑score correction; time‑decayed MLE → full score matrix |
| **CatBoost stack** (`scripts/train_pro.py`) | Gradient‑boosted W/D/L classifier (+optional LightGBM/XGBoost), **isotonic‑calibrated** |
| **Ensemble blend** | 50% calibrated ML + 50% Dixon‑Coles |
| **Monte‑Carlo** (`api/main.py`) | Plays the **real FIFA bracket** with the official best‑third assignment → stage & champion probabilities |

### Measured performance (held‑out 2018–2026)

| Metric | Value |
|---|---|
| W/D/L accuracy (calibrated) | **60.4%** |
| Ranked Probability Score ↓ | **0.167** |
| Home / Away goals MAE | **1.00 / 0.82** |
| Over 0.5 goals (backtest) | **91.4%** |
| Double‑chance favourite | **84.3%** |

---

## 🔌 API reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/teams` | All teams + Elo |
| `GET` | `/api/metrics` | Model metrics |
| `GET` | `/api/markets` | Backtested binary‑market accuracy |
| `GET` | `/api/calibration` | Reliability‑diagram bins |
| `POST`| `/api/predict` | `{home, away, neutral, home_out[], away_out[]}` → probabilities, xG, score matrix, markets |
| `GET` | `/api/squads` | Squad strength for all nations |
| `GET` | `/api/squad/{team}` | Strength + projected XI + intl goal form (`?exclude=`) |
| `GET` | `/api/wc2026/schedule` | All 72 group matches (predicted) |
| `GET` | `/api/wc2026/simulation?n=` | Full‑tournament Monte‑Carlo |
| `GET` | `/api/wc2026/bracket?n=` | Official bracket with modal occupants |

Interactive docs at `/docs` (Swagger).

---

## 📊 Data sources (all free, no API key)

| Source | Data |
|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | 49k international matches + goalscorers (1872–2026) |
| [football-data.co.uk](https://www.football-data.co.uk) | Club leagues + closing odds |
| FIFA‑24 player dataset | Player ratings by country (squad strength) |
| [StatsBomb open data](https://github.com/statsbomb/open-data) | Event data incl. xG (index) |

---

## ⚡ Optional: GPU training on THM JupyterHub

```bash
python tools/jh_client.py whoami
python tools/jh_client.py upload-dir . football
# then in a GPU JupyterLab terminal:
cd football && python scripts/train_pro.py --gpu
```

Credentials live in `.env` (gitignored — copy from `.env.example`). Rotate the THM
token at `https://jl.mni.thm.de/hub/token` if it was ever shared.

---

## 🗺️ Roadmap

- Live club xG enrichment (Understat/FBref) for the projected XI
- Encode FIFA's exact best‑third combination table (vs. valid bipartite matching)
- Hyperparameter tuning (Optuna) + tuned ensemble weights
- Dockerfile + CI for one‑command deploy

---

## 📄 License

MIT — see [LICENSE](LICENSE).
