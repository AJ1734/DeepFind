import { Family, FAMILIES, fmtInt, folderName } from "../lib/files";
import { IcFilter, IcFolder, IcPlus, IcX } from "./Icons";

export interface FolderOption { path: string; count?: number; custom?: boolean }

interface Props {
  open: boolean;
  setOpen: (v: boolean) => void;
  counts: Partial<Record<Family, number>> | null;     // indexed files per family (null = not known yet)
  types: Family[];
  setTypes: (t: Family[]) => void;
  folders: FolderOption[];
  picked: string[];
  setPicked: (p: string[]) => void;
  onPickFolder: () => void;
  onRemoveCustom: (path: string) => void;
}

/** One tick box. Real <input type=checkbox>, so keyboard and screen readers work as usual. */
function Check({ checked, onChange, label, count, title, extra }: {
  checked: boolean; onChange: (v: boolean) => void; label: string; count?: number; title?: string; extra?: React.ReactNode;
}) {
  return (
    <label className={`chk ${checked ? "on" : ""}`} title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="chk-box" aria-hidden="true">
        <svg viewBox="0 0 16 16" width="12" height="12"><path d="m3.5 8.5 3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </span>
      <span className="chk-label">{label}</span>
      {count != null && <b className="chk-n">{fmtInt(count)}</b>}
      {extra}
    </label>
  );
}

const toggle = <T,>(list: T[], v: T, on: boolean) => (on ? (list.includes(v) ? list : [...list, v]) : list.filter((x) => x !== v));

/** "Narrow your search": tick file types and folders BEFORE searching; only those are searched.
 *  Nothing ticked in a group means "everything" for that group. */
export function Filters({ open, setOpen, counts, types, setTypes, folders, picked, setPicked, onPickFolder, onRemoveCustom }: Props) {
  const active = types.length + picked.length;
  const fams = FAMILIES.filter((f) => !counts || (counts[f.id] ?? 0) > 0 || types.includes(f.id));
  const clear = () => { setTypes([]); setPicked([]); };

  return (
    <div className="filterbar">
      <div className="fb-row">
        <button className={`fb-toggle ${open ? "on" : ""} ${active ? "has" : ""}`} onClick={() => setOpen(!open)}
          aria-expanded={open} aria-controls="filter-panel" title="Choose which file types and folders to search">
          <IcFilter width={16} height={16} />
          Filters
          {active > 0 && <b className="fb-count">{active}</b>}
        </button>
        {!open && active > 0 && (
          <div className="fb-chips" aria-label="Active filters">
            {types.map((t) => (
              <button key={t} className="fb-chip" onClick={() => setTypes(types.filter((x) => x !== t))}
                title="Remove this filter">{FAMILIES.find((f) => f.id === t)?.label}<IcX width={12} height={12} /></button>
            ))}
            {picked.map((p) => (
              <button key={p} className="fb-chip" onClick={() => setPicked(picked.filter((x) => x !== p))}
                title={`Remove this filter — ${p}`}><IcFolder width={12} height={12} />{folderName(p)}<IcX width={12} height={12} /></button>
            ))}
          </div>
        )}
        {active > 0 && <button className="linkish fb-clear" onClick={clear}>Clear filters</button>}
      </div>

      {open && (
        <div className="filter-panel" id="filter-panel" role="group" aria-label="Narrow your search">
          <div className="fp-group">
            <div className="fp-title">File type</div>
            <div className="fp-options">
              {fams.map((f) => (
                <Check key={f.id} label={f.label} count={counts?.[f.id]} checked={types.includes(f.id)}
                  onChange={(on) => setTypes(toggle(types, f.id, on))} />
              ))}
            </div>
          </div>
          <div className="fp-group">
            <div className="fp-title">Folder</div>
            <div className="fp-options">
              {folders.map((f) => (
                <Check key={f.path} label={folderName(f.path)} count={f.count} title={f.path} checked={picked.includes(f.path)}
                  onChange={(on) => setPicked(toggle(picked, f.path, on))}
                  extra={f.custom ? (
                    <button type="button" className="chk-x" aria-label={`Forget ${f.path}`} title="Forget this folder"
                      onClick={(e) => { e.preventDefault(); e.stopPropagation(); onRemoveCustom(f.path); }}><IcX width={12} height={12} /></button>
                  ) : undefined} />
              ))}
              <button className="fp-add" onClick={onPickFolder}><IcPlus width={14} height={14} /> Choose another folder…</button>
            </div>
          </div>
          <p className="fp-hint">Tick what you want to search in. Nothing ticked means everything. Only ticked types and folders are searched.</p>
        </div>
      )}
    </div>
  );
}
