"use client";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, pct, HOSTS } from "@/lib/api";

const ROUND_LABEL: Record<string, string> = { R32: "Round of 32", R16: "Round of 16", QF: "Quarter-finals", SF: "Semi-finals", "3P": "3rd place", F: "Final" };
const GAP: Record<string, number> = { R32: 10, R16: 64, QF: 180, SF: 400, F: 430 };

export default function WorldCupPage() {
  const [tab, setTab] = useState<"outlook" | "schedule">("outlook");
  const [n, setN] = useState(10000);
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
          <select className="input !w-auto" value={n} onChange={(e) => setN(+e.target.value)}>
            <option value={2000}>2,000 sims</option><option value={10000}>10,000 sims</option><option value={25000}>25,000 sims</option>
          </select>
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
  const order: Record<string, number[]> = { F: [104], SF: [], QF: [], R16: [], R32: [], "3P": [103] };
  order.SF = feeders(104); order.QF = order.SF.flatMap(feeders);
  order.R16 = order.QF.flatMap(feeders); order.R32 = order.R16.flatMap(feeders);
  const rounds = ["R32", "R16", "QF", "SF", "F"];

  const wrapRef = useRef<HTMLDivElement>(null);
  const [paths, setPaths] = useState<string[]>([]);
  const [dims, setDims] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const draw = () => {
      const wrap = wrapRef.current; if (!wrap) return;
      const base = wrap.getBoundingClientRect();
      const ps: string[] = [];
      bracket.forEach((m) => {
        if (m.round === "3P") return;
        [m.home_src, m.away_src].forEach((src: number | null) => {
          if (src == null) return;
          const a = wrap.querySelector(`#bk-${src}`) as HTMLElement;
          const t = wrap.querySelector(`#bk-${m.id}`) as HTMLElement;
          if (!a || !t) return;
          const ra = a.getBoundingClientRect(), rt = t.getBoundingClientRect();
          const x1 = ra.right - base.left, y1 = ra.top + ra.height / 2 - base.top;
          const x2 = rt.left - base.left, y2 = rt.top + rt.height / 2 - base.top;
          const mx = (x1 + x2) / 2;
          ps.push(`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
        });
      });
      setDims({ w: base.width, h: base.height }); setPaths(ps);
    };
    const id = requestAnimationFrame(() => requestAnimationFrame(draw));
    window.addEventListener("resize", draw);
    return () => { cancelAnimationFrame(id); window.removeEventListener("resize", draw); };
  }, [bracket]);

  const card = (m: any) => (
    <div id={`bk-${m.id}`} key={m.id} className="card p-3 relative" style={{ width: 200, zIndex: 1 }}>
      <div className="flex justify-between items-center mb-2">
        <span className="text-[10px] font-bold text-brand">M{m.id}</span>
        <span className="text-[10px] text-slate-500 truncate">{m.venue} · {m.date.slice(5)}</span>
      </div>
      {[["home_slot", "home"], ["away_slot", "away"]].map(([sl, sd], idx) => (
        <div key={sd as string}>
          {idx === 1 && <div className="border-t border-white/5 my-2" />}
          <div className="text-[9px] text-slate-600 mb-1">{m[sl as string]}</div>
          {m[sd as string] ? (
            <div className="flex justify-between items-center gap-2">
              <span className="font-semibold truncate text-sm">{HOSTS.has(m[sd as string].team) ? "🏠 " : ""}{m[sd as string].team}</span>
              <span className="font-mono text-[10px] text-slate-500">{pct(m[sd as string].prob, 0)}</span>
            </div>
          ) : <div className="text-slate-600 text-xs">—</div>}
        </div>
      ))}
      {m.winner && <div className="mt-2 pt-2 border-t border-white/5 text-[10px] text-amber-400">▶ {m.winner.team} ({pct(m.winner.prob, 0)})</div>}
    </div>
  );

  return (
    <div>
      <div ref={wrapRef} className="relative overflow-x-auto pb-4">
        <svg className="absolute inset-0 pointer-events-none" width={dims.w} height={dims.h} style={{ zIndex: 0 }}>
          <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#34e39b" /></marker></defs>
          {paths.map((d, i) => <path key={i} d={d} fill="none" stroke="#34e39b" strokeWidth={1.6} opacity={0.5} markerEnd="url(#ah)" />)}
        </svg>
        <div className="relative flex gap-12 min-w-max" style={{ zIndex: 1 }}>
          {rounds.map((r) => (
            <div key={r} className="flex flex-col items-center" style={{ gap: GAP[r] || 16 }}>
              <div className="label mb-1">{ROUND_LABEL[r]}</div>
              {order[r].map((id) => card(byId[id]))}
            </div>
          ))}
        </div>
      </div>
      <div className="mt-6 max-w-xs">
        <div className="label mb-2 text-center">3rd-place play-off</div>
        {card(byId[103])}
      </div>
    </div>
  );
}
