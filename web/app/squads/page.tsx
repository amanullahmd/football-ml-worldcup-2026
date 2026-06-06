"use client";
import { useEffect, useMemo, useState } from "react";
import { api, HOSTS, SquadRow, SquadDetail } from "@/lib/api";

const POS_COLOR: Record<string, string> = { GK: "#fbbf24", DEF: "#60a5fa", MID: "#34e39b", ATT: "#f87171" };

export default function SquadsPage() {
  const [rows, setRows] = useState<SquadRow[]>([]);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<SquadDetail | null>(null);
  const [exclude, setExclude] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.squads().then((d) => {
      if (!d.available) { setErr("Squad layer unavailable."); return; }
      setRows(d.teams); if (d.teams[0]) load(d.teams[0].team);
    }).catch((e) => setErr(String(e)));
  }, []);

  const load = async (team: string, ex = "") => {
    setSel(team);
    try { setDetail(await api.squad(team, ex)); } catch (e) { setErr(String(e)); }
  };

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    return s ? rows.filter((r) => r.team.toLowerCase().includes(s)) : rows;
  }, [q, rows]);

  const full = detail?.full;
  const s = detail?.strength;
  const da = s && full ? s.attack_rating - full.attack_rating : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Squad strength & projected XI</h1>
        <p className="text-slate-400 text-sm mt-1">Per-nation strength from FIFA-24 ratings + each player's <b>real international goal form</b>. Click a country; run a who's-missing what-if.</p>
      </div>
      {err && <div className="card p-4 text-red-300 text-sm">{err}</div>}

      <div className="grid lg:grid-cols-2 gap-6 items-start">
        {/* Ranking */}
        <div className="card p-2">
          <div className="flex items-center justify-between gap-3 px-2 py-2">
            <input className="input flex-1" placeholder="Search country…" value={q} onChange={(e) => setQ(e.target.value)} />
            <span className="text-xs text-slate-500 whitespace-nowrap">{filtered.length}/{rows.length}</span>
          </div>
          <div className="max-h-[70vh] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="text-[11px] uppercase tracking-wider text-slate-400">
                <tr>{["#", "Team", "OVR", "ATT", "MID", "DEF", "GK", "Players"].map((h) => <th key={h} className="text-left px-3 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={r.team} onClick={() => load(r.team, "")}
                    className={`border-t border-white/5 cursor-pointer hover:bg-brand/5 ${sel === r.team ? "bg-brand/10" : ""}`}>
                    <td className="px-3 py-2 text-slate-500 font-mono">{i + 1}</td>
                    <td className="px-3 py-2 font-semibold">{HOSTS.has(r.team) ? "🏠 " : ""}{r.team}</td>
                    <td className="px-3 py-2 font-mono">{r.squad_overall.toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono" style={{ color: POS_COLOR.ATT }}>{r.attack_rating.toFixed(0)}</td>
                    <td className="px-3 py-2 font-mono" style={{ color: POS_COLOR.MID }}>{r.midfield_rating.toFixed(0)}</td>
                    <td className="px-3 py-2 font-mono" style={{ color: POS_COLOR.DEF }}>{r.defense_rating.toFixed(0)}</td>
                    <td className="px-3 py-2 font-mono" style={{ color: POS_COLOR.GK }}>{r.gk_rating.toFixed(0)}</td>
                    <td className="px-3 py-2 font-mono text-slate-500">{r.n_players}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Detail */}
        <div className="card p-6">
          {!s ? <div className="text-slate-400 text-sm">Select a country.</div> : (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="label">Team</div>
                  <div className="text-2xl font-bold">{HOSTS.has(sel!) ? "🏠 " : ""}{sel}</div>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-extrabold text-brand">{s.squad_overall.toFixed(1)}</div>
                  <div className="text-xs text-slate-400">overall · {s.n_players} players</div>
                </div>
              </div>

              <div className="space-y-3 mb-5">
                {([["Attack", s.attack_rating, POS_COLOR.ATT], ["Midfield", s.midfield_rating, POS_COLOR.MID], ["Defense", s.defense_rating, POS_COLOR.DEF], ["Goalkeeper", s.gk_rating, POS_COLOR.GK]] as const).map(([l, v, c]) => (
                  <div key={l}>
                    <div className="flex justify-between text-sm mb-1"><span style={{ color: c }}>{l}</span><span className="font-mono">{v.toFixed(1)}</span></div>
                    <div className="h-2 rounded bg-white/5 overflow-hidden"><div className="h-full" style={{ width: `${v}%`, background: c }} /></div>
                  </div>
                ))}
              </div>

              {detail?.intl_form && (
                <div className="text-xs text-slate-400 mb-4">
                  Real intl goals (last {detail.intl_form.years}y): <b className="text-slate-200">{detail.intl_form.total_intl_goals}</b>
                  {" · "}top: {detail.intl_form.top_scorers.slice(0, 3).map((x) => `${x.scorer} (${x.goals})`).join(", ")}
                </div>
              )}

              <div className="mb-3">
                <div className="label">Who's missing?</div>
                <div className="flex gap-2 mt-2">
                  <input className="input flex-1" placeholder="e.g. Mbappe, Griezmann" value={exclude} onChange={(e) => setExclude(e.target.value)} />
                  <button className="btn-brand" onClick={() => load(sel!, exclude)}>Apply</button>
                </div>
                {exclude && full && (
                  <div className="text-xs mt-2 text-slate-400">
                    Excluding <span className="text-slate-200">{exclude}</span>: attack {full.attack_rating.toFixed(1)} → <span style={{ color: da < 0 ? "#f87171" : "#34e39b" }}>{s.attack_rating.toFixed(1)}</span> ({da >= 0 ? "+" : ""}{(da / full.attack_rating * 100).toFixed(1)}%)
                  </div>
                )}
              </div>

              <div className="label mb-2">Projected XI · real intl goals</div>
              <div className="space-y-1">
                {detail?.xi.map((p, i) => (
                  <div key={i} className="flex items-center justify-between text-sm py-1.5 border-b border-white/5">
                    <div className="truncate">
                      <span className="font-mono text-xs w-8 inline-block" style={{ color: POS_COLOR[p.position] }}>{p.position}</span>
                      <span className="font-medium">{p.player}</span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      {p.intl_goals ? <span className="chip">⚽ {p.intl_goals}</span> : null}
                      <span className="text-slate-500 text-xs truncate max-w-[120px]">{p.club}</span>
                      <span className="font-mono font-bold">{p.overall.toFixed(0)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
