"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/predict", label: "Match predictor" },
  { href: "/worldcup", label: "World Cup 2026" },
  { href: "/squads", label: "Squads & XI" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-50 backdrop-blur bg-ink/70 border-b border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand to-brand-deep grid place-items-center font-bold text-black">⚽</div>
          <div className="font-bold tracking-tight">Football&nbsp;ML</div>
        </Link>
        <nav className="flex gap-1 text-sm font-medium">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link key={l.href} href={l.href}
                className={`px-3 py-2 rounded-lg transition ${active ? "text-brand bg-brand/10" : "text-slate-300 hover:text-white hover:bg-white/5"}`}>
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
