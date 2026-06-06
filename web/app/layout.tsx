import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import ExtensionErrorGuard from "@/components/ExtensionErrorGuard";

export const metadata: Metadata = {
  title: "Football ML — World Cup 2026 Prediction Engine",
  description: "Calibrated football match & World Cup 2026 forecasting (Dixon-Coles ⊕ CatBoost ensemble).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="font-sans antialiased">
        <ExtensionErrorGuard />
        <Nav />
        <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
        <footer className="max-w-7xl mx-auto px-6 py-10 text-xs text-slate-500">
          Predictions are model-derived (Dixon-Coles + calibrated CatBoost on 41k+ real international matches). No mock data.
        </footer>
      </body>
    </html>
  );
}
