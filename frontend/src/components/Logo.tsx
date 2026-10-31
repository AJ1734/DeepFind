import { useId } from "react";

/** DeepFind mark: a neon magnifier with a spark in the lens. Same artwork as public/logo.svg
 *  (favicon + the .ico used by Windows). Brand colours are fixed so it looks the same in every theme. */
export function Logo({ size = 30 }: { size?: number }) {
  const u = useId().replace(/:/g, "");
  const id = (n: string) => `${n}-${u}`;
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" className="logo-mark">
      <defs>
        <linearGradient id={id("bg")} x1="6" y1="2" x2="58" y2="62" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#2a1a78" />
          <stop offset="0.55" stopColor="#17104a" />
          <stop offset="1" stopColor="#0b0827" />
        </linearGradient>
        <radialGradient id={id("glow")} cx="26" cy="26" r="26" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#7c5cff" stopOpacity="0.75" />
          <stop offset="0.6" stopColor="#4b2fd0" stopOpacity="0.22" />
          <stop offset="1" stopColor="#4b2fd0" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={id("ring")} x1="12" y1="10" x2="42" y2="42" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5ef2ff" />
          <stop offset="0.55" stopColor="#9f7bff" />
          <stop offset="1" stopColor="#ff6fd8" />
        </linearGradient>
        <linearGradient id={id("handle")} x1="38" y1="38" x2="54" y2="54" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#c58bff" />
          <stop offset="1" stopColor="#ff6fd8" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${id("bg")})`} />
      <rect x="0.75" y="0.75" width="62.5" height="62.5" rx="14.3" fill="none" stroke="#ffffff" strokeOpacity="0.14" strokeWidth="1.5" />
      <circle cx="26" cy="26" r="26" fill={`url(#${id("glow")})`} />
      <circle cx="26" cy="26" r="13.5" fill="#0b0827" fillOpacity="0.55" />
      <circle cx="26" cy="26" r="8.2" fill="none" stroke="#5ef2ff" strokeOpacity="0.55" strokeWidth="1.4" strokeDasharray="2.2 3" />
      <path d="M26 17.6c.7 4.6 3.8 7.7 8.4 8.4-4.6.7-7.7 3.8-8.4 8.4-.7-4.6-3.8-7.7-8.4-8.4 4.6-.7 7.7-3.8 8.4-8.4z" fill="#ffffff" />
      <circle cx="26" cy="26" r="13.5" fill="none" stroke={`url(#${id("ring")})`} strokeWidth="5" />
      <path d="M19.2 17.4a9.6 9.6 0 0 1 5-3.2" fill="none" stroke="#ffffff" strokeOpacity="0.75" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M36.6 36.6 50 50" stroke={`url(#${id("handle")})`} strokeWidth="7.4" strokeLinecap="round" />
    </svg>
  );
}
