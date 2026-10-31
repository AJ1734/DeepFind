import { useEffect, useRef, useState } from "react";
import { useEngine } from "../lib/engine";
import { THEMES, ThemeChoice, ThemeInfo, setTheme, useTheme } from "../lib/theme";
import { IcCheck, IcPalette } from "./Icons";

/** Choose a theme now, and remember it in the engine's settings so it survives restarts. */
function useChoose() {
  const { saveSettings } = useEngine();
  return (c: ThemeChoice) => {
    setTheme(c);
    saveSettings({ theme: c });
  };
}

function Preview({ t }: { t: ThemeInfo }) {
  const [bg, card, accent, ink] = t.swatch;
  return (
    <span className="tp-preview" style={{ background: bg, borderColor: ink + "33" }} aria-hidden="true">
      <span className="tp-card" style={{ background: card }}>
        <i style={{ background: ink, width: "62%" }} />
        <i style={{ background: ink, opacity: 0.35, width: "40%" }} />
        <b style={{ background: accent }} />
      </span>
    </span>
  );
}

function AutoPreview() {
  const l = THEMES[0].swatch, d = THEMES[3].swatch;
  return (
    <span className="tp-preview tp-auto" aria-hidden="true"
      style={{ background: `linear-gradient(115deg, ${l[0]} 50%, ${d[0]} 50%)` }}>
      <span className="tp-card" style={{ background: `linear-gradient(115deg, ${l[1]} 50%, ${d[1]} 50%)` }}>
        <i style={{ background: `linear-gradient(115deg, ${l[3]} 50%, ${d[3]} 50%)`, width: "62%" }} />
        <b style={{ background: `linear-gradient(115deg, ${l[2]} 50%, ${d[2]} 50%)` }} />
      </span>
    </span>
  );
}

/** Settings → Appearance: every theme as a small preview card. */
export function ThemeGrid() {
  const choice = useTheme();
  const choose = useChoose();
  return (
    <div className="tp-grid" role="radiogroup" aria-label="Colour theme">
      <button role="radio" aria-checked={choice === "auto"} className={`tp-item ${choice === "auto" ? "on" : ""}`}
        onClick={() => choose("auto")}>
        <AutoPreview />
        <span className="tp-text"><b>Match Windows</b><em>Light by day, dark when Windows is dark</em></span>
        {choice === "auto" && <IcCheck className="tp-tick" width={18} height={18} />}
      </button>
      {THEMES.map((t) => (
        <button key={t.id} role="radio" aria-checked={choice === t.id} className={`tp-item ${choice === t.id ? "on" : ""}`}
          onClick={() => choose(t.id)}>
          <Preview t={t} />
          <span className="tp-text"><b>{t.label}</b><em>{t.blurb}</em></span>
          {choice === t.id && <IcCheck className="tp-tick" width={18} height={18} />}
        </button>
      ))}
    </div>
  );
}

/** Palette button in the top bar: pick a theme without leaving the page. */
export function ThemeMenu() {
  const choice = useTheme();
  const choose = useChoose();
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const down = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", down);
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("mousedown", down); window.removeEventListener("keydown", key); };
  }, [open]);

  const items: { id: ThemeChoice; label: string; dots: string[] }[] = [
    { id: "auto", label: "Match Windows", dots: [THEMES[0].swatch[0], THEMES[3].swatch[0]] },
    ...THEMES.map((t) => ({ id: t.id as ThemeChoice, label: t.label, dots: [t.swatch[0], t.swatch[2]] })),
  ];
  return (
    <div className="theme-menu" ref={box}>
      <button className={`tb-icon ${open ? "on" : ""}`} onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu" aria-expanded={open} title="Change colour theme">
        <IcPalette width={18} height={18} />
        <span className="sr-only">Colour theme</span>
      </button>
      {open && (
        <div className="menu-pop" role="menu" aria-label="Colour theme">
          {items.map((it) => (
            <button key={it.id} role="menuitemradio" aria-checked={choice === it.id}
              className={`menu-item ${choice === it.id ? "on" : ""}`}
              onClick={() => { choose(it.id); setOpen(false); }}>
              <span className="mi-dots" aria-hidden="true">
                <i style={{ background: it.dots[0] }} /><i style={{ background: it.dots[1] }} />
              </span>
              <span>{it.label}</span>
              {choice === it.id && <IcCheck width={16} height={16} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
