"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, pct, Metrics } from "@/lib/api";

export default function Home() {
  const [m, setM] = useState<Metrics | null>(null);
  const [markets, setMarkets] = useState<Metrics["binary_markets"] | null>(null);
  const [bins, setBins] = useState<Metrics["calibration_bins"] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.metrics().then(setM).catch((e) => setErr(String(e)));
    api.markets().then((d) => setMarkets(d.markets || null)).catch(() => {});
    api.calibration().then((d) => setBins(d.bins || null)).catch(() => {});
  }, []);

  const acc = m?.calibrated?.accuracy;
  const marketLabels: Record<string, string> = {
    over_0_5: "Over 0.5 goals", over_1_5: "Over 1.5 goals", over_2_5: "Over 2.5 goals",
    btts: "Both teams score", double_chance_fav: "Favourite ≥ draw",
  };

  return (
    <div className="space-y-10">
      {err && <div className="card p-4 text-red-300 text-sm">API not reachable at the configured base — start the FastAPI server. ({err})</div>}

      {/* Hero */}
      <section className="text-center pt-8 pb-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-brand/10 text-brand ring-1 ring-brand/30 mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" /> Live model · Dixon-Coles ⊕ CatBoost
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-br from-white via-emerald-100 to-brand bg-clip-text text-transparent">
          Predict the beautiful game.
        </h1>
        <p className="text-slate-400 mt-4 max-w-2xl mx-auto text-lg">
          Production-grade football forecasting on 154 years of international matches — calibrated
          probabilities, expected goals, full World Cup 2026 simulation, and squad/XI intelligence.
        </p>
        <div className="mt-8 flex gap-3 justify-center">
          <Link href="/predict" className="btn-brand">Predict a match →</Link>
          <Link href="/worldcup" className="btn-ghost font-medium">Run WC2026 simulation</Link>
        </div>
      </section>

      {/* Metric cards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { v: acc !== undefined ? pct(acc) : "—", l: "W/D/L accuracy (calibrated)" },
          { v: m?.calibrated?.rps?.toFixed(3) ?? "—", l: "Ranked Probability Score ↓" },
          { v: m?.goals?.home_mae?.toFixed(2) ?? "—", l: "Home goals MAE" },
          { v: m ? ((m.n_train || 0) + (m.n_test || 0)).toLocaleString() : "—", l: "Matches modelled" },
        ].map((c) => (
          <div key={c.l} className="card p-6">
            <div className="text-3xl font-extrabold">{c.v}</div>
            <div className="text-slate-400 text-sm mt-1">{c.l}</div>
          </div>
        ))}
      </section>

      {/* Tools */}
      <section className="grid md:grid-cols-3 gap-4">
        {[
          { href: "/predict", t: "Match predictor", d: "Any two teams → W/D/L, expected goals, score-matrix heatmap, BTTS & over/under, and a who's-missing what-if." },
          { href: "/worldcup", t: "World Cup 2026", d: "Full 104-match schedule, official bracket with connector arrows, and 10k Monte-Carlo to champion %." },
          { href: "/squads", t: "Squads & XI", d: "Squad strength for 135 nations, projected starting XI, and each player's real international goal form." },
        ].map((c) => (
          <Link key={c.href} href={c.href} className="card card-hover p-6 block">
            <div className="text-xl font-bold mb-2">{c.t}</div>
            <p className="text-slate-400 text-sm">{c.d}</p>
            <div className="text-brand font-medium mt-4 text-sm">Open →</div>
          </Link>
        ))}
      </section>

      {/* Honest accuracy note */}
      <section className="card p-6">
        <div className="label text-amber-400 mb-2">Why “95% accuracy” is the wrong target</div>
        <p className="text-slate-400 text-sm leading-relaxed">
          3-way Win/Draw/Loss has a structural ceiling of ~62–64% — even bet365's closing odds resolve to ~55–58%.
          Football is low-scoring and high-variance. The professional KPIs are <b>calibration</b> and <b>RPS</b>, not raw
          accuracy. Genuine 85–97% lives in <b>binary markets</b> and <b>group qualification</b>, shown below.
        </p>
      </section>

      {/* Binary markets */}
      {markets && (
        <section className="card p-6">
          <div className="label text-brand mb-1">Where 85–97% accuracy genuinely lives</div>
          <div className="text-slate-500 text-xs mb-4">Backtested on the held-out test set (2018→2026).</div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(markets).map(([k, v]) => (
              <div key={k} className="text-center">
                <div className="text-3xl font-extrabold text-brand">{pct(v.accuracy)}</div>
                <div className="text-slate-300 text-sm mt-1">{marketLabels[k] || k}</div>
                <div className="text-slate-500 text-xs mt-1">base {pct(v.base_rate, 0)} · n={v.n.toLocaleString()}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Calibration */}
      {bins && bins.length > 0 && (
        <section className="card p-6">
          <div className="label text-blue-400 mb-1">Calibration — is the model trustworthy?</div>
          <div className="text-slate-500 text-xs mb-4">When the model says X%, how often is it right? Closer to X = better.</div>
          <div className="space-y-2">
            {bins.map((b, i) => {
              const conf = b.mean_confidence * 100, acc2 = b.accuracy * 100;
              const diff = Math.abs(b.mean_confidence - b.accuracy);
              const col = diff < 0.05 ? "#34e39b" : diff < 0.12 ? "#eab308" : "#ef4444";
              return (
                <div key={i} className="flex items-center gap-3 text-sm">
                  <div className="w-28 text-slate-400">{Math.round(b.bin_lo * 100)}–{Math.round(b.bin_hi * 100)}% conf</div>
                  <div className="flex-1 h-5 rounded bg-white/5 overflow-hidden">
                    <div className="h-full" style={{ width: `${acc2}%`, background: col }} />
                  </div>
                  <div className="w-44 text-right text-slate-400">pred {conf.toFixed(0)}% · actual {acc2.toFixed(0)}% <span className="text-slate-600">(n={b.count})</span></div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
