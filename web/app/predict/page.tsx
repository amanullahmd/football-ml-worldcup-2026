"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, pct, HOSTS, TeamElo, Prediction } from "@/lib/api";

const POS_COLOR: Record<string, string> = { GK: "#fbbf24", DEF: "#60a5fa", MID: "#34e39b", ATT: "#f87171" };

function Combo({ label, value, onChange, teams }: {
  label: string; value: string; onChange: (v: string) => void; teams: TeamElo[];
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState(value);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => setQ(value), [value]);
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h); return () => document.removeEventListener("mousedown", h);
  }, []);
  const list = useMemo(() => {
    const s = q.trim().toLowerCase();
    return teams.filter((t) => t.team.toLowerCase().includes(s)).slice(0, 40);
  }, [q, teams]);
  return (
    <div className="card p-5">
      <div className="label">{label}</div>
      <div className="relative mt-2" ref={ref}>
        <input className="input" value={q} placeholder="Type a team…" autoComplete="off"
          onChange={(e) => { setQ(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)} />
        {open && list.length > 0 && (
          <div className="absolute z-20 mt-1 w-full max-h-60 overflow-y-auto rounded-xl bg-[#0c1219] ring-1 ring-white/10">
            {list.map((t) => (
              <div key={t.team} className="px-3 py-2 text-sm flex justify-between cursor-pointer hover:bg-brand/10"
                onClick={() => { onChange(t.team); setQ(t.team); setOpen(false); }}>
                <span>{t.team}</span><span className="text-slate-500">{t.elo.toFixed(0)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Bar({ label, val, color }: { label: string; val: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1"><span>{label}</span><span className="font-mono">{pct(val)}</span></div>
      <div className="h-3 rounded-full bg-white/5 overflow-hidden">
        <div className="h-full" style={{ width: `${val * 100}%`, background: color }} />
      </div>
    </div>
  );
}

export default function PredictPage() {
  const [teams, setTeams] = useState<TeamElo[]>([]);
  const [home, setHome] = useState("Brazil");
  const [away, setAway] = useState("France");
  const [neutral, setNeutral] = useState(true);
  const [homeOut, setHomeOut] = useState("");
  const [awayOut, setAwayOut] = useState("");
  const [res, setRes] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.teams().then(setTeams).catch((e) => setErr(String(e))); }, []);

  const run = async () => {
    if (!home || !away || home === away) { setErr("Pick two different teams"); return; }
    setLoading(true); setErr(null);
    try {
      setRes(await api.predict({
        home, away, neutral,
        home_out: homeOut.split(",").map((s) => s.trim()).filter(Boolean),
        away_out: awayOut.split(",").map((s) => s.trim()).filter(Boolean),
      }));
    } catch (e) { setErr(String(e)); } finally { setLoading(false); }
  };

  const matrixMax = res?.score_matrix ? Math.max(...res.score_matrix.flat()) : 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Match predictor</h1>
        <p className="text-slate-400 text-sm mt-1">Calibrated probabilities, expected goals, full score matrix, betting markets, and a who's-missing what-if.</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <Combo label="Home / Team A" value={home} onChange={setHome} teams={teams} />
        <Combo label="Away / Team B" value={away} onChange={setAway} teams={teams} />
        <div className="card p-5 flex flex-col">
          <div className="label">Venue</div>
          <select className="input mt-2" value={neutral ? "1" : "0"} onChange={(e) => setNeutral(e.target.value === "1")}>
            <option value="1">Neutral ground</option>
            <option value="0">Team A at home</option>
          </select>
          <button onClick={run} disabled={loading} className="btn-brand mt-auto">{loading ? "Predicting…" : "Predict match →"}</button>
        </div>
      </div>

      <div className="card p-5">
        <div className="label mb-1">Who's missing? <span className="text-slate-500 normal-case">(optional · injuries / suspensions)</span></div>
        <div className="text-xs text-slate-500 mb-3">Comma-separated names. Removing key players lowers that team's attack — e.g. type <i>Mbappe</i> for Team A.</div>
        <div className="grid md:grid-cols-2 gap-4">
          <input className="input" placeholder="Team A players out" value={homeOut} onChange={(e) => setHomeOut(e.target.value)} />
          <input className="input" placeholder="Team B players out" value={awayOut} onChange={(e) => setAwayOut(e.target.value)} />
        </div>
      </div>

      {err && <div className="card p-4 text-red-300 text-sm">{err}</div>}

      {res && (
        <div className="space-y-5">
          <div className="card p-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="text-center md:text-left">
                <div className="text-xs text-slate-400">{res.neutral ? "Neutral ground" : `Home advantage to ${res.home}`}</div>
                <div className="text-3xl font-extrabold mt-1">{res.home} <span className="text-slate-500 mx-2">vs</span> {res.away}</div>
                <div className="text-slate-400 text-sm mt-1">Elo {res.elo.home.toFixed(0)} · {res.elo.away.toFixed(0)}</div>
              </div>
              <div className="text-center md:text-right">
                <div className="text-xs text-slate-400">Expected score</div>
                <div className="text-3xl font-extrabold text-brand">{res.expected_goals.home.toFixed(2)} – {res.expected_goals.away.toFixed(2)}</div>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-4 mt-7">
              <Bar label={`${res.home} win`} val={res.probabilities.home_win} color="linear-gradient(90deg,#34e39b,#0f9d68)" />
              <Bar label="Draw" val={res.probabilities.draw} color="linear-gradient(90deg,#64748b,#475569)" />
              <Bar label={`${res.away} win`} val={res.probabilities.away_win} color="linear-gradient(90deg,#3b82f6,#1d4ed8)" />
            </div>
            {res.squad && (res.squad.home_out?.length || res.squad.away_out?.length) ? (
              <div className="text-amber-400 text-xs mt-4">
                ⚠ {res.squad.home_out?.length ? `${res.home} without ${res.squad.home_out.join(", ")} (attack ×${res.squad.home_attack_mult.toFixed(2)})` : ""}
                {res.squad.away_out?.length ? `  ${res.away} without ${res.squad.away_out.join(", ")} (attack ×${res.squad.away_attack_mult.toFixed(2)})` : ""}
              </div>
            ) : null}
          </div>

          <div className="grid lg:grid-cols-3 gap-4">
            <div className="card p-6 lg:col-span-2">
              <div className="label mb-3">Score probability matrix</div>
              <div className="text-xs text-slate-500 mb-3">Rows = {res.home} goals · Cols = {res.away} goals · % chance of exact score</div>
              {res.score_matrix && (
                <table className="mx-auto text-[11px]">
                  <tbody>
                    <tr><td></td>{res.score_matrix[0].map((_, j) => <td key={j} className="px-1.5 py-1 text-slate-500 text-center">{j}</td>)}</tr>
                    {res.score_matrix.slice(0, 7).map((row, i) => (
                      <tr key={i}>
                        <td className="px-1.5 text-slate-500">{i}</td>
                        {row.slice(0, 7).map((v, j) => {
                          const a = Math.min(1, v / matrixMax);
                          return <td key={j} className="px-1.5 py-1 text-center" style={{ background: `rgba(52,227,155,${a * 0.85})`, color: v > 0.04 ? "#000" : "#94a3b8", fontWeight: v > 0.04 ? 600 : 400 }}>{(v * 100).toFixed(1)}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
            <div className="card p-6">
              <div className="label mb-3">Most likely scores</div>
              {res.top_scores.map((s, i) => (
                <div key={i} className="flex justify-between py-2 border-b border-white/5">
                  <span className="text-lg font-bold">{s.home} – {s.away}</span>
                  <span className="text-brand font-mono">{pct(s.prob)}</span>
                </div>
              ))}
              <div className="label mt-6 mb-3">Markets</div>
              <div className="space-y-2 text-sm">
                {([["Both teams score", res.markets.btts], ["Over 1.5 goals", res.markets.over_1_5], ["Over 2.5 goals", res.markets.over_2_5], ["Over 3.5 goals", res.markets.over_3_5]] as const).map(([l, v]) =>
                  v != null ? <div key={l} className="flex justify-between rounded-lg px-3 py-2 bg-white/5"><span>{l}</span><span className="font-mono">{pct(v)}</span></div> : null
                )}
              </div>
            </div>
          </div>

          <div className="card p-6">
            <div className="label mb-3">Ensemble breakdown</div>
            <div className="grid md:grid-cols-2 gap-6 text-sm">
              <div>
                <div className="text-xs text-slate-500 mb-2">CatBoost (calibrated)</div>
                <Bar label="Home" val={res.ml_probabilities.home_win} color="#34e39b" /><div className="h-2" />
                <Bar label="Draw" val={res.ml_probabilities.draw} color="#64748b" /><div className="h-2" />
                <Bar label="Away" val={res.ml_probabilities.away_win} color="#3b82f6" />
              </div>
              <div>
                <div className="text-xs text-slate-500 mb-2">Dixon-Coles (Poisson)</div>
                {res.dc_probabilities ? (<>
                  <Bar label="Home" val={res.dc_probabilities.home_win} color="#34e39b" /><div className="h-2" />
                  <Bar label="Draw" val={res.dc_probabilities.draw} color="#64748b" /><div className="h-2" />
                  <Bar label="Away" val={res.dc_probabilities.away_win} color="#3b82f6" />
                </>) : <div className="text-slate-500">Team not in DC set — ML only</div>}
              </div>
            </div>
          </div>

          {/* Head-to-head */}
          {res.h2h && res.h2h.played > 0 && (
            <div className="card p-6">
              <div className="label mb-3">Head-to-head · {res.h2h.played} meetings</div>
              <div className="grid grid-cols-3 gap-4 text-center mb-4">
                <div><div className="text-2xl font-extrabold text-brand">{res.h2h.home_wins}</div><div className="text-xs text-slate-400">{res.home} wins</div></div>
                <div><div className="text-2xl font-extrabold text-slate-300">{res.h2h.draws}</div><div className="text-xs text-slate-400">Draws</div></div>
                <div><div className="text-2xl font-extrabold text-blue-400">{res.h2h.away_wins}</div><div className="text-xs text-slate-400">{res.away} wins</div></div>
              </div>
              <div className="text-xs text-slate-500 text-center">
                Goals {res.h2h.home_goals}–{res.h2h.away_goals} · last meetings:{" "}
                {res.h2h.last5.map((m, i) => <span key={i} className="chip mx-0.5">{m.home_goals}-{m.away_goals}</span>)}
              </div>
            </div>
          )}

          {/* Current 26-man squads — player performance */}
          {res.players && (res.players.home.length > 0 || res.players.away.length > 0) && (
            <div className="card p-6">
              <div className="label mb-1">Current squads · player performance (26)</div>
              <div className="text-xs text-slate-500 mb-4">Ability from current club tier; ⚽ = real international goals (last 4y). Top 11 in bold.</div>
              <div className="grid md:grid-cols-2 gap-6">
                {([["home", res.home], ["away", res.away]] as const).map(([side, name]) => (
                  <div key={side}>
                    <div className="font-semibold mb-2">{name}</div>
                    <div className="space-y-1">
                      {res.players![side].map((p, i) => (
                        <div key={i} className={`flex items-center justify-between text-sm py-1 border-b border-white/5 ${i < 11 ? "font-semibold" : "text-slate-400"}`}>
                          <div className="truncate">
                            <span className="font-mono text-[10px] w-7 inline-block" style={{ color: POS_COLOR[p.position] }}>{p.position}</span>
                            {p.player}
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {p.club_goals != null ? <span className="chip" title="club goals/assists 25/26">🏟 {p.club_goals}G {p.club_assists}A</span> : null}
                            {p.intl_goals ? <span className="chip" title="international goals (4y)">⚽ {p.intl_goals}</span> : null}
                            <span className="text-slate-500 text-xs truncate max-w-[100px]">{p.club}</span>
                            <span className="font-mono">{p.overall.toFixed(0)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
