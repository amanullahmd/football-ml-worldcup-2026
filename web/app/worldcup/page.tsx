"use client";
import { useEffect, useState } from "react";
import { api, pct, HOSTS } from "@/lib/api";

const ROUND_LABEL: Record<string, string> = { R32: "Round of 32", R16: "Round of 16", QF: "Quarter-finals", SF: "Semi-finals", "3P": "3rd place", F: "Final" };
const ROW_H = 172; // vertical pitch of a Round-of-32 slot; later rounds are centred between feeders

export default function WorldCupPage() {
  const [tab, setTab] = useState<"outlook" | "schedule">("outlook");
  const n = 25000;
  const [sim, setSim] = useState<any>(null);
  const [sched, setSched] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setErr(null);
    try { setSim(await api.simulation(n)); } catch (e) { setErr(String(e)); } finally { setLoading(false); }
  };
  useEffect(() => { run(); /* eslint-disable-next-line */ }, []);
  useEffect(() => { if (tab === "schedule" && !sched) api.schedule().then(setSched).catch(() => {}); }, [tab, sched]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">FIFA World Cup 2026 — Forecast</h1>
          <p className="text-slate-400 text-sm mt-1">48 teams · 104 matches · official bracket. Hosts 🇺🇸🇨🇦🇲🇽 · Monte-Carlo to champion.</p>
        </div>
        <div className="flex gap-3 items-center">
          <span className="chip" title="Number of full tournament simulations run to estimate each team's chances">25,000 simulations</span>
          <button className="btn-brand" onClick={run} disabled={loading}>{loading ? "Running…" : "Run simulation"}</button>
        </div>
      </div>
      {err && <div className="card p-4 text-red-300 text-sm">{err}</div>}

      <div className="flex gap-1 border-b border-white/10 text-sm font-medium">
        {(["outlook", "schedule"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-3 border-b-2 -mb-px ${tab === t ? "text-brand border-brand" : "text-slate-400 border-transparent hover:text-white"}`}>
            {t === "outlook" ? "🏆 Tournament outlook" : "🗺️ Full schedule"}
          </button>
        ))}
      </div>

      {tab === "outlook" && sim && <Outlook sim={sim} />}
      {tab === "schedule" && <Schedule sim={sim} sched={sched} />}
    </div>
  );
}

function Outlook({ sim }: { sim: any }) {
  const top = sim.teams.slice(0, 8);
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        {top.map((t: any, i: number) => (
          <div key={t.team} className={`card p-4 text-center ${i === 0 ? "ring-2 ring-amber-400" : ""}`}>
            {i === 0 && <div className="text-2xl mb-1">🏆</div>}
            <div className="text-xs text-slate-500">#{i + 1} · Grp {t.group}</div>
            <div className="font-bold mt-1">{HOSTS.has(t.team) ? "🏠 " : ""}{t.team}</div>
            <div className={`text-2xl font-extrabold mt-2 ${i === 0 ? "text-amber-400" : "text-brand"}`}>{pct(t.champion, 2)}</div>
            <div className="text-[10px] text-slate-400">champion</div>
          </div>
        ))}
      </div>

      <div className="card p-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-[11px] uppercase tracking-wider text-slate-400">
            <tr>{["#", "Team", "Grp", "Elo", "R32", "R16", "QF", "SF", "Final", "🏆"].map((h) => <th key={h} className="text-left px-3 py-2">{h}</th>)}</tr>
          </thead>
          <tbody>
            {sim.teams.map((t: any, i: number) => (
              <tr key={t.team} className="border-t border-white/5 hover:bg-brand/5">
                <td className="px-3 py-2 text-slate-500 font-mono">{i + 1}</td>
                <td className="px-3 py-2 font-semibold">{HOSTS.has(t.team) ? "🏠 " : ""}{t.team}</td>
                <td className="px-3 py-2"><span className="chip">{t.group}</span></td>
                <td className="px-3 py-2 font-mono text-slate-400">{t.elo.toFixed(0)}</td>
                {["r32", "r16", "qf", "sf", "final"].map((k) => <td key={k} className="px-3 py-2 font-mono">{pct(t[k], 0)}</td>)}
                <td className="px-3 py-2 font-mono text-amber-400 font-bold">{pct(t.champion, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Schedule({ sim, sched }: { sim: any; sched: any }) {
  return (
    <div className="space-y-8">
      {/* Group stage */}
      <section>
        <div className="label mb-3">Group stage · 72 matches (predicted)</div>
        {!sched ? <div className="text-slate-400 text-sm">loading group schedule…</div> : (
          <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {sched.group_stage.map((g: any) => (
              <div key={g.group} className="card p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-8 h-8 rounded-lg grid place-items-center font-extrabold text-black text-sm" style={{ background: "linear-gradient(135deg,#34e39b,#0f9d68)" }}>{g.group}</div>
                  <span className="font-bold">Group {g.group}</span>
                </div>
                {[1, 2, 3].map((md) => (
                  <div key={md}>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 mt-2 mb-1">Matchday {md} · {g.matches.find((x: any) => x.matchday === md)?.window}</div>
                    {g.matches.filter((x: any) => x.matchday === md).map((m: any, i: number) => {
                      const win = m.home_win > Math.max(m.draw, m.away_win) ? m.home : m.away_win > m.draw ? m.away : "Draw";
                      const p = Math.max(m.home_win, m.draw, m.away_win);
                      return (
                        <div key={i} className="flex items-center justify-between text-sm py-1 border-b border-white/5">
                          <div className="truncate">{HOSTS.has(m.home) ? "🏠 " : ""}{m.home} <span className="text-slate-600">v</span> {m.away}</div>
                          <div className="flex items-center gap-2 shrink-0">
                            {m.top_score && <span className="font-mono text-xs text-brand">{m.top_score.home}–{m.top_score.away}</span>}
                            <span className="text-[10px] text-slate-500 w-20 text-right truncate">{win} {pct(p, 0)}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Bracket */}
      <section>
        <div className="label mb-3">Knockout bracket · R32 → Final {sim ? `(${sim.n_simulations.toLocaleString()} sims)` : ""}</div>
        {sim ? <Bracket bracket={sim.bracket} /> : <div className="text-slate-400 text-sm">Run a simulation to see the bracket.</div>}
      </section>
    </div>
  );
}

function Bracket({ bracket }: { bracket: any[] }) {
  const byId: Record<number, any> = {}; bracket.forEach((m) => (byId[m.id] = m));
  const feeders = (id: number) => [byId[id].home_src, byId[id].away_src].filter((x) => x != null) as number[];

  // Split the tree at the Final: the two semi-finals each anchor one half of the bracket.
  const sf = feeders(104);                       // [left SF, right SF]
  const half = (sfId: number) => {               // build a half from its semi-final, R32 → SF
    const SF = [sfId];
    const QF = SF.flatMap(feeders);
    const R16 = QF.flatMap(feeders);
    const R32 = R16.flatMap(feeders);
    return { R32, R16, QF, SF };
  };
  const L = half(sf[0]); const R = half(sf[1]);

  const COL_W = 196, COL_GAP = 56, COL_PITCH = COL_W + COL_GAP;
  const CARD_H = 140; // approx rendered card height, used to vertically centre the absolute card on its midpoint
  const totalH = L.R32.length * ROW_H;           // 8 slots per half
  const NCOLS = 9;                               // R32 R16 QF SF | F | SF QF R16 R32
  const totalW = NCOLS * COL_PITCH - COL_GAP;

  // column index per side: left half grows rightward (0→3), right half grows leftward (8→5), Final centred at 4
  const colL: Record<string, number> = { R32: 0, R16: 1, QF: 2, SF: 3 };
  const colR: Record<string, number> = { R32: 8, R16: 7, QF: 6, SF: 5 };
  const colOf: Record<number, number> = { 104: 4 };
  const yOf: Record<number, number> = {};

  const place = (side: { R32: number[]; R16: number[]; QF: number[]; SF: number[] }, col: Record<string, number>) => {
    side.R32.forEach((id, i) => { yOf[id] = i * ROW_H + ROW_H / 2; colOf[id] = col.R32; });
    (["R16", "QF", "SF"] as const).forEach((r) =>
      side[r].forEach((id) => {
        const fs = feeders(id);
        yOf[id] = fs.reduce((s, f) => s + yOf[f], 0) / fs.length;
        colOf[id] = col[r];
      })
    );
  };
  place(L, colL); place(R, colR);
  yOf[104] = (yOf[sf[0]] + yOf[sf[1]]) / 2;       // Final centred between the two semis
  yOf[103] = CARD_H / 2 + 28;                      // 3rd-place play-off sits below its own label

  // Elbow connectors — direction derived from whether the feeder sits left or right of its parent.
  const paths: { d: string; champ: boolean }[] = [];
  bracket.forEach((m) => {
    if (m.round === "3P") return;
    const pc = colOf[m.id]; if (pc == null) return;
    [m.home_src, m.away_src].forEach((src: number | null) => {
      if (src == null || colOf[src] == null) return;
      const cc = colOf[src];
      const leftFeed = cc < pc;
      const x1 = leftFeed ? cc * COL_PITCH + COL_W : cc * COL_PITCH;        // edge of feeder facing the parent
      const x2 = leftFeed ? pc * COL_PITCH : pc * COL_PITCH + COL_W;        // edge of parent facing the feeder
      const mx = (x1 + x2) / 2;
      const champ = !!(m.winner && byId[src]?.winner && m.winner.team === byId[src].winner.team);
      paths.push({ d: `M${x1},${yOf[src]} H${mx} V${yOf[m.id]} H${x2}`, champ });
    });
  });

  const card = (m: any, mirror = false, ci?: number) => {
    const col = ci ?? colOf[m.id] ?? 0;
    return (
      <div
        id={`bk-${m.id}`}
        key={m.id}
        className={`card p-3 absolute ${m.round === "F" ? "ring-1 ring-amber-400/70" : ""}`}
        style={{ width: COL_W, left: col * COL_PITCH, top: (yOf[m.id] ?? ROW_H / 2) - CARD_H / 2, zIndex: 1 }}
      >
        <div className={`flex items-center mb-2 ${mirror ? "flex-row-reverse" : ""} justify-between`}>
          <span className="text-[10px] font-bold text-brand">{m.round === "F" ? "🏆 FINAL" : `M${m.id}`}</span>
          <span className="text-[10px] text-slate-500 truncate">{m.venue} · {m.date.slice(5)}</span>
        </div>
        {[["home_slot", "home"], ["away_slot", "away"]].map(([sl, sd], idx) => {
          const t = m[sd as string];
          const isWin = !!(m.winner && t && m.winner.team === t.team);
          return (
            <div key={sd as string}>
              {idx === 1 && <div className="border-t border-white/5 my-2" />}
              <div className={`text-[9px] text-slate-600 mb-1 ${mirror ? "text-right" : ""}`}>{m[sl as string]}</div>
              {t ? (
                <div className={`flex items-center gap-2 justify-between ${mirror ? "flex-row-reverse" : ""} ${isWin ? "text-amber-300 font-bold" : ""}`}>
                  <span className="truncate text-sm font-semibold">{isWin ? "▸ " : ""}{HOSTS.has(t.team) ? "🏠 " : ""}{t.team}</span>
                  <span className="font-mono text-[10px] text-slate-500">{pct(t.prob, 0)}</span>
                </div>
              ) : <div className={`text-slate-600 text-xs ${mirror ? "text-right" : ""}`}>—</div>}
            </div>
          );
        })}
      </div>
    );
  };

  const headers: { label: string; col: number }[] = [
    { label: ROUND_LABEL.R32, col: 0 }, { label: ROUND_LABEL.R16, col: 1 }, { label: ROUND_LABEL.QF, col: 2 }, { label: ROUND_LABEL.SF, col: 3 },
    { label: ROUND_LABEL.F, col: 4 },
    { label: ROUND_LABEL.SF, col: 5 }, { label: ROUND_LABEL.QF, col: 6 }, { label: ROUND_LABEL.R16, col: 7 }, { label: ROUND_LABEL.R32, col: 8 },
  ];

  return (
    <div>
      <div className="overflow-x-auto pb-4">
        <div className="relative mb-2" style={{ width: totalW, minWidth: totalW }}>
          {headers.map((h, i) => (
            <div key={i} className="label absolute" style={{ left: h.col * COL_PITCH, width: COL_W, textAlign: "center" }}>{h.label}</div>
          ))}
        </div>
        <div className="relative" style={{ width: totalW, minWidth: totalW, height: totalH }}>
          <svg className="absolute inset-0 pointer-events-none" width={totalW} height={totalH} style={{ zIndex: 0 }}>
            {paths.map((p, i) => (
              <path key={i} d={p.d} fill="none" stroke={p.champ ? "#fbbf24" : "#34e39b"} strokeWidth={p.champ ? 2.2 : 1.4} opacity={p.champ ? 0.9 : 0.4} />
            ))}
          </svg>
          {[L.R32, L.R16, L.QF, L.SF].flat().map((id) => card(byId[id]))}
          {card(byId[104])}
          {[R.R32, R.R16, R.QF, R.SF].flat().map((id) => card(byId[id], true))}
        </div>
      </div>
      <div className="mt-6 max-w-xs relative" style={{ height: CARD_H + 28 }}>
        <div className="label mb-2 text-center">3rd-place play-off</div>
        {card(byId[103], false, 0)}
      </div>
    </div>
  );
}
