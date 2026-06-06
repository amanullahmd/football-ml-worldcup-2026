# Football ML — Next.js frontend

Modern frontend (Next.js 16 App Router + TypeScript + Tailwind) for the Football ML
World Cup 2026 prediction engine. It consumes the **FastAPI ML API** (the Python
backend) — all models/predictions stay in Python.

## Run

```bash
# 1. start the ML API (from repo root)
python -m uvicorn api.main:app --port 8000

# 2. start the frontend (from web/)
npm install      # first time only
npm run dev      # http://localhost:3000
```

The API base URL is set in `.env.local` (`NEXT_PUBLIC_API_BASE`, defaults to
`http://127.0.0.1:8000`). CORS for `localhost:3000` is enabled on the FastAPI side.

## Pages
- `/`          — landing: live metrics, binary-market accuracy, calibration diagram
- `/predict`   — any-match predictor + who's-missing what-if + score matrix/markets
- `/worldcup`  — tournament outlook (champion %), full 104-match schedule, bracket w/ arrows
- `/squads`    — squad strength (135 nations), projected XI, real intl goal form
