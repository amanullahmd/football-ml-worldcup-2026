"use client";
import { useEffect } from "react";

/**
 * Browser extensions (notably MetaMask's inpage.js) inject themselves into every
 * page and sometimes emit unhandled promise rejections like "Failed to connect to
 * MetaMask". Our app never uses web3 — these are 100% external. Next.js's dev
 * overlay otherwise surfaces them as "Unhandled Runtime Error". This guard quietly
 * swallows errors that originate from a chrome-/moz-extension or mention MetaMask,
 * and leaves all genuine app errors untouched.
 */
export default function ExtensionErrorGuard() {
  useEffect(() => {
    const fromExtension = (s?: string) =>
      !!s && (/chrome-extension:\/\//.test(s) || /moz-extension:\/\//.test(s) || /MetaMask/i.test(s));

    const onRejection = (e: PromiseRejectionEvent) => {
      const r: any = e.reason;
      const msg = typeof r === "string" ? r : r?.message || r?.stack || "";
      if (fromExtension(msg)) { e.preventDefault(); e.stopImmediatePropagation?.(); }
    };
    const onError = (e: ErrorEvent) => {
      if (fromExtension(e.filename) || fromExtension(e.message)) {
        e.preventDefault(); e.stopImmediatePropagation?.();
      }
    };
    window.addEventListener("unhandledrejection", onRejection, true);
    window.addEventListener("error", onError, true);
    return () => {
      window.removeEventListener("unhandledrejection", onRejection, true);
      window.removeEventListener("error", onError, true);
    };
  }, []);
  return null;
}
