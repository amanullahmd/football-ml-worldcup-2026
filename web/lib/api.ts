// Central API client for the FastAPI ML backend.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${await r.text()}`);
  return r.json();
}

// ---- Types ----
export type TeamElo = { team: string; elo: number };
export type Metrics = {
  model?: string;
  calibrated?: { accuracy: number; log_loss: number; rps: number; brier: number };
  goals?: { home_mae: number; away_mae: number };
  binary_markets?: Record<string, { accuracy: number; base_rate: number; n: number }>;
  calibration_bins?: { bin_lo: number; bin_hi: number; mean_confidence: number; accuracy: number; count: number }[];
  n_train?: number; n_test?: number;
};
export type Prediction = {
  home: string; away: string; neutral: boolean;
  probabilities: { home_win: number; draw: number; away_win: number };
  ml_probabilities: { home_win: number; draw: number; away_win: number };
  dc_probabilities: { home_win: number; draw: number; away_win: number } | null;
  expected_goals: { home: number; away: number };
  elo: { home: number; away: number };
  squad?: { home: any; away: any; home_attack_mult: number; away_attack_mult: number; home_out: string[]; away_out: string[] };
  score_matrix: number[][] | null;
  top_scores: { home: number; away: number; prob: number }[];
  markets: { btts: number | null; over_1_5: number | null; over_2_5: number | null; over_3_5: number | null };
  h2h?: { played: number; home_wins: number; draws: number; away_wins: number; home_goals: number; away_goals: number; last5: { home_goals: number; away_goals: number }[] };
  players?: { home: XiPlayer[]; away: XiPlayer[] };
};
export type SquadRow = {
  team: string; squad_overall: number; top11_overall: number; attack_rating: number;
  midfield_rating: number; defense_rating: number; gk_rating: number; avg_age: number | null;
  squad_value_m: number | null; n_players: number;
};
export type XiPlayer = {
  player: string; position: string; overall: number; club: string; age: number | null;
  intl_goals?: number; club_goals?: number | null; club_assists?: number | null; club_xg?: number | null;
};
export type SquadDetail = {
  strength: SquadRow & { excluded: string[] };
  xi: XiPlayer[];
  full: SquadRow | null;
  intl_form: { total_intl_goals: number; top_scorers: { scorer: string; goals: number }[]; years: number } | null;
};

export const api = {
  teams: () => get<TeamElo[]>("/api/teams"),
  metrics: () => get<Metrics>("/api/metrics"),
  markets: () => get<{ is_pro: boolean; markets: Metrics["binary_markets"] }>("/api/markets"),
  calibration: () => get<{ is_pro: boolean; bins: Metrics["calibration_bins"]; rps: number; brier: number; accuracy: number }>("/api/calibration"),
  predict: (b: { home: string; away: string; neutral: boolean; home_out?: string[]; away_out?: string[] }) =>
    post<Prediction>("/api/predict", b),
  squads: () => get<{ available: boolean; teams: SquadRow[] }>("/api/squads"),
  squad: (team: string, exclude?: string) =>
    get<SquadDetail>(`/api/squad/${encodeURIComponent(team)}${exclude ? `?exclude=${encodeURIComponent(exclude)}` : ""}`),
  groups: () => get<{ groups: Record<string, string[]>; hosts: string[] }>("/api/wc2026/groups"),
  schedule: () => get<{ group_stage: { group: string; matches: any[] }[]; note: string }>("/api/wc2026/schedule"),
  simulation: (n = 10000) => get<any>(`/api/wc2026/simulation?n=${n}`),
  bracket: (n = 10000) => get<any>(`/api/wc2026/bracket?n=${n}`),
};

export const pct = (x: number, d = 1) => (x * 100).toFixed(d) + "%";
export const HOSTS = new Set(["United States", "Canada", "Mexico"]);
