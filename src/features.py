"""Feature engineering helpers for international match prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_result(row) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if row["home_score"] > row["away_score"]:
        return 0
    if row["home_score"] == row["away_score"]:
        return 1
    return 2


def rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling form features per team using only PAST matches (no leakage).

    Adds for each row:
      home_form_pts, home_gf, home_ga,
      away_form_pts, away_gf, away_ga
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Long format: one row per team per match.
    home = df[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
    )
    home["pts"] = np.where(home["gf"] > home["ga"], 3, np.where(home["gf"] == home["ga"], 1, 0))
    home["row_id"] = df.index

    away = df[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
    )
    away["pts"] = np.where(away["gf"] > away["ga"], 3, np.where(away["gf"] == away["ga"], 1, 0))
    away["row_id"] = df.index

    long = pd.concat([home.assign(side="home"), away.assign(side="away")], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    g = long.groupby("team", group_keys=False)
    long["form_pts"] = g["pts"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_gf"] = g["gf"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_ga"] = g["ga"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())

    home_feat = long[long.side == "home"].set_index("row_id")[["form_pts", "form_gf", "form_ga"]]
    home_feat.columns = ["home_form_pts", "home_form_gf", "home_form_ga"]
    away_feat = long[long.side == "away"].set_index("row_id")[["form_pts", "form_gf", "form_ga"]]
    away_feat.columns = ["away_form_pts", "away_form_gf", "away_form_ga"]

    out = df.join(home_feat).join(away_feat)
    return out


def h2h_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Past head-to-head record between the two teams (leak-free).

    Adds: h2h_home_wins, h2h_draws, h2h_away_wins, h2h_home_gf, h2h_home_ga (last `window` meetings).
    """
    df = df.sort_values("date").reset_index(drop=True)

    out_cols = {
        "h2h_home_wins": np.zeros(len(df)),
        "h2h_draws": np.zeros(len(df)),
        "h2h_away_wins": np.zeros(len(df)),
        "h2h_home_gf": np.zeros(len(df)),
        "h2h_home_ga": np.zeros(len(df)),
    }

    history: dict[frozenset, list] = {}
    for i, row in df.iterrows():
        key = frozenset((row["home_team"], row["away_team"]))
        past = history.get(key, [])[-window:]
        hw = dw = aw = 0
        gf = ga = 0.0
        for m in past:
            # m = (date, home, away, hg, ag) — orient to current home perspective
            ph, pa, phg, pag = m[1], m[2], m[3], m[4]
            if ph == row["home_team"]:
                cur_hg, cur_ag = phg, pag
            else:
                cur_hg, cur_ag = pag, phg
            if cur_hg > cur_ag:
                hw += 1
            elif cur_hg == cur_ag:
                dw += 1
            else:
                aw += 1
            gf += cur_hg
            ga += cur_ag
        n = max(len(past), 1)
        out_cols["h2h_home_wins"][i] = hw
        out_cols["h2h_draws"][i] = dw
        out_cols["h2h_away_wins"][i] = aw
        out_cols["h2h_home_gf"][i] = gf / n
        out_cols["h2h_home_ga"][i] = ga / n

        history.setdefault(key, []).append(
            (row["date"], row["home_team"], row["away_team"], row["home_score"], row["away_score"])
        )

    for c, vals in out_cols.items():
        df[c] = vals
    return df
