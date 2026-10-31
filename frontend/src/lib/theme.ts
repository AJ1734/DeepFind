import { useSyncExternalStore } from "react";

/**
 * Colour themes. The look itself lives in styles/tokens.css (one block per theme, selected by
 * <html data-theme="…">). This file knows the list, remembers the choice and applies it.
 *
 * The choice is saved in the engine's settings (so it survives restarts — the app's address changes
 * on every start, which would lose browser storage) and cached in localStorage so a browser/dev
 * session paints in the right colours on the first frame.
 */

export type ThemeId = "daylight" | "sunset" | "mint" | "midnight" | "aurora" | "ember" | "contrast";
export type ThemeChoice = ThemeId | "auto";

export interface ThemeInfo {
  id: ThemeId;
  label: string;
  blurb: string;
  dark: boolean;
  /** preview colours: background, card, accent, text */
  swatch: [string, string, string, string];
}

export const THEMES: ThemeInfo[] = [
  { id: "daylight", label: "Daylight", blurb: "Clean white and blue", dark: false, swatch: ["#f4f6fa", "#ffffff", "#2646d6", "#1b2333"] },
  { id: "sunset",   label: "Sunset",   blurb: "Warm cream and orange", dark: false, swatch: ["#fbf3ea", "#fffdf9", "#b53a12", "#33241a"] },
  { id: "mint",     label: "Mint",     blurb: "Fresh and calm", dark: false, swatch: ["#eef6f3", "#ffffff", "#0a7060", "#12302a"] },
  { id: "midnight", label: "Midnight", blurb: "Easy on the eyes at night", dark: true, swatch: ["#0e1320", "#161d2e", "#7c9bff", "#e8ecf6"] },
  { id: "aurora",   label: "Aurora",   blurb: "Neon violet and cyan", dark: true, swatch: ["#0f0a1f", "#1a1231", "#b78cff", "#f2edff"] },
  { id: "ember",    label: "Ember",    blurb: "Warm charcoal and amber", dark: true, swatch: ["#14110f", "#1e1a17", "#ff8c42", "#f5eee7"] },
  { id: "contrast", label: "High contrast", blurb: "Black and yellow, easiest to read", dark: true, swatch: ["#000000", "#0b0b0b", "#ffd60a", "#ffffff"] },
];

export const THEME_IDS = THEMES.map((t) => t.id) as ThemeId[];
const KEY = "deepfind.theme.v1";

const isTheme = (v: unknown): v is ThemeId => typeof v === "string" && (THEME_IDS as string[]).includes(v);
export const isChoice = (v: unknown): v is ThemeChoice => v === "auto" || isTheme(v);

const prefersDark = () => {
  try { return window.matchMedia("(prefers-color-scheme: dark)").matches; } catch { return false; }
};

/** "auto" follows the operating system: Daylight in light mode, Midnight in dark mode. */
export const resolveTheme = (c: ThemeChoice): ThemeId => (c === "auto" ? (prefersDark() ? "midnight" : "daylight") : c);

export function applyTheme(c: ThemeChoice) {
  const id = resolveTheme(c);
  const root = document.documentElement;
  if (id === "daylight") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", id);
  root.style.colorScheme = THEMES.find((t) => t.id === id)?.dark ? "dark" : "light";
}

function cached(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    if (isChoice(v)) return v;
  } catch { /* storage unavailable */ }
  return "auto";
}

let current: ThemeChoice = cached();
const subs = new Set<() => void>();

/** Choose a theme (also what the engine's saved setting calls when it loads). */
export function setTheme(c: ThemeChoice) {
  current = c;
  try { localStorage.setItem(KEY, c); } catch { /* ignore */ }
  applyTheme(c);
  subs.forEach((f) => f());
}

export function initTheme() {
  applyTheme(current);
  try {   // "auto" tracks the operating system while the app is open
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (current === "auto") applyTheme("auto");
    });
  } catch { /* old webview */ }
}

export function useTheme(): ThemeChoice {
  return useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, () => current);
}
