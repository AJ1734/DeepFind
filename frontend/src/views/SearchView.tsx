import { KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, SearchFilters, SearchResult } from "../lib/api";
import { copyText, pickFolders } from "../lib/desktop";
import { semanticReady, useEngine } from "../lib/engine";
import { Family, familyCounts, fmtInt, fmtTime, shortFolder } from "../lib/files";
import { pushRecent, setPrefs, usePrefs } from "../lib/prefs";
import { FileVisual } from "../components/FileVisual";
import { Filters, FolderOption } from "../components/Filters";
import { Highlight } from "../components/Highlight";
import { IcCopy, IcFolder, IcOpen, IcPlus, IcSearch, IcX } from "../components/Icons";

const MATCH: Record<SearchResult["match_type"], { label: string; hint: string }> = {
  keyword: { label: "Keyword match", hint: "The words you typed appear in this file" },
  semantic: { label: "Meaning match", hint: "Matched by meaning, e.g. what a picture shows" },
  hybrid: { label: "Keyword + meaning", hint: "Matched by both the words and the meaning" },
  filename: { label: "File name match", hint: "The file's name fits the pattern you typed, like *.jpg" },
};

type Mode = "keyword" | "semantic" | "filename";
const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "keyword", label: "Keyword", hint: "The words you typed appear in the file" },
  { id: "semantic", label: "Meaning", hint: "Matched by meaning, e.g. what a picture shows" },
  { id: "filename", label: "Filename", hint: "A word you typed appears in the file name" },
];
const queryWords = (s: string) =>
  Array.from(new Set(s.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) ?? []));

const MIN_CHARS = 2;   // searching starts at two letters
const TRY = ["invoice total", "photo of a whiteboard", "*.pdf", "report*"];

/** Words for a 0-100 match score. */
const quality = (pct: number) => (pct >= 85 ? "Excellent match" : pct >= 60 ? "Good match" : pct >= 35 ? "Fair match" : "Weak match");
const tone = (pct: number) => (pct >= 85 ? "high" : pct >= 60 ? "good" : pct >= 35 ? "fair" : "low");

/** How strongly a result matches, as a number and a bar. */
function Score({ pct }: { pct: number }) {
  return (
    <span className={`score tone-${tone(pct)}`} title={`${quality(pct)} (${pct} out of 100)`}
      aria-label={`Match score ${pct} out of 100`}>
      <span className="score-bar"><span style={{ width: `${pct}%` }} /></span>
      <b>{pct}%</b>
    </span>
  );
}

export function SearchView({ active, goFolders }: { active: boolean; goFolders: () => void }) {
  const { status, settings, library, models, saveSettings, toast } = useEngine();
  const prefs = usePrefs();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searched, setSearched] = useState("");
  const [took, setTook] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode | "all">("all");
  // what the search is limited to (set before searching; nothing ticked = everything)
  const [types, setTypes] = useState<Family[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [extraFolders, setExtraFolders] = useState<string[]>([]);
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [parsed, setParsed] = useState<{ text: string; patterns: string[] }>({ text: "", patterns: [] });
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const ctrl = useRef<AbortController | null>(null);

  // ---- focus ---------------------------------------------------------------------------
  useEffect(() => { if (active) inputRef.current?.focus(); }, [active]);
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (!active) return;
      const inField = (e.target as HTMLElement)?.tagName === "INPUT";
      if ((e.ctrlKey && e.key.toLowerCase() === "k") || (e.key === "/" && !inField)) {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active]);

  // ---- debounced search -----------------------------------------------------------------
  const run = useCallback(async (query: string, filters: SearchFilters) => {
    ctrl.current?.abort();
    const t = query.trim();
    if (t.length < MIN_CHARS) {        // too short to search: clear, don't call the engine
      setResults([]); setSearched(""); setLoading(false); setErr(null); setTook(null); return;
    }
    const c = new AbortController();
    ctrl.current = c;
    setLoading(true);
    const t0 = performance.now();
    try {
      const r = await api.search(t, t.includes("*") || t.includes("?") || t.startsWith("/") ? 100 : 50, c.signal, filters);
      if (c.signal.aborted) return;
      setResults(r.results);
      setParsed({ text: r.text ?? t, patterns: r.patterns ?? [] });
      setSearched(t);
      setTook(Math.round(performance.now() - t0));
      setSel(0);
      setErr(null);
      if (r.results.length) setPrefs((p) => pushRecent(p, t));
    } catch (e) {
      if (!c.signal.aborted) { setErr(e instanceof Error ? e.message : "Search failed"); setResults([]); }
    } finally {
      if (!c.signal.aborted) setLoading(false);
    }
  }, []);

  const filters = useMemo<SearchFilters>(() => ({ types, folders: picked }), [types, picked]);
  useEffect(() => {
    const id = setTimeout(() => run(q, filters), q.trim().length < 2 ? 0 : 200);
    return () => clearTimeout(id);
  }, [q, filters, run]);

  // folders you can tick: the watched folders plus any you chose here
  const folderOptions = useMemo<FolderOption[]>(() => {
    const watched = settings?.watched_folders ?? [];
    return [
      ...watched.map((p) => ({ path: p, count: library?.folders?.[p] })),
      ...extraFolders.filter((p) => !watched.includes(p)).map((p) => ({ path: p, custom: true })),
    ];
  }, [settings?.watched_folders, library?.folders, extraFolders]);
  useEffect(() => {      // a ticked folder that is no longer in the list is dropped
    const all = new Set(folderOptions.map((f) => f.path));
    setPicked((cur) => (cur.every((p) => all.has(p)) ? cur : cur.filter((p) => all.has(p))));
  }, [folderOptions]);
  const chooseFolder = async () => {
    const got = await pickFolders();
    if (!got.length) return;
    setExtraFolders((cur) => Array.from(new Set([...cur, ...got])));
    setPicked((cur) => Array.from(new Set([...cur, ...got])));
  };
  const famCounts = useMemo(() => familyCounts(library?.types), [library?.types]);

  // ---- filtering -------------------------------------------------------------------------
  // (file types and folders are limited by the engine before it searches; these tabs only slice what came back)
  const words = useMemo(() => queryWords(parsed.text), [parsed.text]);
  const inMode = useCallback(
    (r: SearchResult, m: Mode) =>
      m === "keyword" ? r.match_type === "keyword" || r.match_type === "hybrid"
      : m === "semantic" ? r.match_type === "semantic" || r.match_type === "hybrid"
      : r.match_type === "filename" || words.some((w) => r.file_name.toLowerCase().includes(w)),
    [words],
  );
  const modeCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const m of MODES) c[m.id] = results.filter((r) => inMode(r, m.id)).length;
    return c;
  }, [results, inMode]);
  const namesOnly = results.length > 0 && results.every((r) => r.match_type === "filename");
  const shown = useMemo(
    () => results.filter((r) => mode === "all" || inMode(r, mode)),
    [results, mode, inMode],
  );
  useEffect(() => { setSel(0); }, [mode]);
  useEffect(() => { if (mode !== "all" && !modeCounts[mode]) setMode("all"); }, [modeCounts, mode]);
  const current = shown[Math.min(sel, shown.length - 1)];
  // Relative to the best hit: a perfect keyword-only hit has a raw fused score of ~0.5
  // (it can only win one of the two lists), which would otherwise read as "50%".
  const top = Math.max(1e-6, ...results.map((r) => r.score));
  const strength = (r: SearchResult) => Math.max(0.05, r.score / top);

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-i="${sel}"]`)?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  // ---- actions ---------------------------------------------------------------------------
  const act = useCallback(async (kind: "open" | "reveal" | "copy", r?: SearchResult) => {
    if (!r) return;
    try {
      if (kind === "copy") {
        toast((await copyText(r.file_path)) ? "Path copied to clipboard" : "Could not copy", "ok");
      } else if (kind === "open") {
        await api.open(r.file_path);
        toast(`Opening ${r.file_name}`, "ok");
      } else {
        await api.reveal(r.file_path);
        toast("Shown in File Explorer", "ok");
      }
    } catch (e) {
      toast(e instanceof Error ? e.message : "That didn't work", "error");
    }
  }, [toast]);

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, shown.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    else if (e.key === "Enter" && current) { e.preventDefault(); act(e.shiftKey ? "reveal" : "open", current); }
    else if (e.key === "Escape") { setQ(""); }
    else if (e.key.toLowerCase() === "c" && e.ctrlKey && current
      && inputRef.current?.selectionStart === inputRef.current?.selectionEnd) {
      e.preventDefault(); act("copy", current);
    }
  };

  const addFolder = async () => {
    const picked = await pickFolders();
    if (!picked.length) return;
    const s = await saveSettings({ watched_folders: [...(settings?.watched_folders ?? []), ...picked] });
    if (s) toast("Folder added — indexing started", "ok");
  };

  // One search box, one tree: the box is never unmounted when the layout switches between the
  // start screen and the results, so it keeps focus (and your cursor) while you type.
  const trimmed = q.trim();
  const idle = trimmed.length < MIN_CHARS;
  const noFolders = settings && settings.watched_folders.length === 0;
  // the filter panel is open on the start screen and tucks away (leaving chips) once results are showing
  useEffect(() => { setFiltersOpen(idle); }, [idle]);
  const semanticOff = !semanticReady(status, models);
  const shownPct = (r: SearchResult) => r.match_percent ?? Math.round(strength(r) * 100);
  const rankText = namesOnly ? "sorted by name" : semanticOff ? "ranked by keyword relevance" : "ranked by keyword + meaning";
  const limited = types.length + picked.length > 0;

  return (
    <section className={`view search-view ${idle ? "is-idle" : "is-active"}`} aria-label="Search">
      <div className="search-head">
        {idle && <h1 className="headline">Search your files</h1>}
        {idle && <p className="idle-sub">Type a word from inside a document, or describe a picture.</p>}
        <div className={`search-box ${loading ? "is-loading" : ""}`}>
          <span className="sb-icon"><IcSearch width={22} height={22} /></span>
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search by a word, describe what you're looking for, or use * like *.jpg"
            spellCheck={false}
            autoComplete="off"
            aria-label="Search your files"
            aria-controls="results"
            aria-activedescendant={current && !idle ? `r-${sel}` : undefined}
          />
          {q ? (
            <button className="sb-clear" onClick={() => { setQ(""); inputRef.current?.focus(); }} aria-label="Clear search">
              <IcX width={16} height={16} />
            </button>
          ) : (
            <kbd className="sb-keys">Ctrl K</kbd>
          )}
          <span className="sb-rule" aria-hidden="true" />
        </div>
        <Filters open={filtersOpen} setOpen={setFiltersOpen} counts={famCounts} types={types} setTypes={setTypes}
          folders={folderOptions} picked={picked} setPicked={setPicked} onPickFolder={chooseFolder}
          onRemoveCustom={(p) => { setExtraFolders((c) => c.filter((x) => x !== p)); setPicked((c) => c.filter((x) => x !== p)); }} />
      </div>

      {idle ? (
        <div className="idle-body">
          {trimmed.length === 1 && <p className="idle-hint">Keep typing — enter at least {MIN_CHARS} letters to search.</p>}
          {trimmed.length === 0 && !noFolders && (
            <p className="idle-tip"><b>Tip:</b> <code>*</code> stands for any letters. <code>*.jpg</code> finds every JPG picture,
              <code>report*</code> finds names that start with “report”, and <code>invoice *.pdf</code> looks for “invoice” only inside PDFs.</p>
          )}

          {trimmed.length === 0 && (noFolders ? (
            <div className="first-run">
              <h2>Choose a folder to get started</h2>
              <p>DeepFind reads the documents, PDFs, sheets and pictures inside it, so you can search them.
                Everything stays on this computer.</p>
              <button className="btn btn-primary" onClick={addFolder}><IcPlus width={16} height={16} /> Choose a folder</button>
            </div>
          ) : (
            <div className="lines">
              {prefs.recent.length > 0 && (
                <div className="line-group">
                  <span className="lg-label">Recent</span>
                  {prefs.recent.slice(0, 4).map((r) => (
                    <button key={r} className="line-link" onClick={() => { setQ(r); inputRef.current?.focus(); }}>{r}</button>
                  ))}
                </div>
              )}
              <div className="line-group">
                <span className="lg-label">Try</span>
                {TRY.map((r) => <button key={r} className="line-link" onClick={() => { setQ(r); inputRef.current?.focus(); }}>{r}</button>)}
              </div>
            </div>
          ))}

          {!noFolders && (
            <p className="idle-foot">
              {fmtInt(library?.total ?? status?.db_rows)} files in{" "}
              <button className="linkish" onClick={goFolders}>
                {settings?.watched_folders.length ?? 0} {settings?.watched_folders.length === 1 ? "folder" : "folders"}
              </button>
              {" · "}
              {semanticOff ? "Searching by keyword only. Install the AI models in Settings to search by meaning." : "Searching by keyword and meaning."}
            </p>
          )}
        </div>
      ) : (
        <>
          <div className="results-bar">
            <div className="filters">
              {!namesOnly && (
              <div className="tabs-mini" role="tablist" aria-label="Filter by match">
                <button role="tab" aria-selected={mode === "all"} className={mode === "all" ? "on" : ""}
                  onClick={() => setMode("all")}>Any match <b>{results.length}</b></button>
                {MODES.filter((m) => !(m.id === "semantic" && semanticOff)).map((m) => (
                  <button key={m.id} role="tab" aria-selected={mode === m.id} className={mode === m.id ? "on" : ""}
                    title={m.hint} onClick={() => setMode(m.id)}>
                    {m.label} <b>{modeCounts[m.id]}</b>
                  </button>
                ))}
              </div>
              )}
            </div>
            {took != null && !loading && (
              <span className="results-meta" title="Best match first. The match score runs from 1 to 100.">
                {fmtInt(results.length)} {results.length === 1 ? "file" : "files"} found
                {parsed.patterns.length > 0 && <> · names like {parsed.patterns.map((p) => <code key={p} className="pat">{p}</code>)}</>}
                {" · "}{rankText}
                {semanticOff && <span className="tag" title="Install AI models in Settings to search by meaning">keyword only</span>}
              </span>
            )}
          </div>

          <div className="results-wrap">
            <div className="results" id="results" role="listbox" ref={listRef} aria-label="Results">
              {err && <div className="notice bad"><b>Search failed.</b> {err}</div>}
              {!err && !loading && searched && shown.length === 0 && (
                <div className="empty">
                  <p className="empty-big">No files found for “{searched}”</p>
                  <span>
                    {limited ? "Only the file types and folders you ticked were searched. " : ""}
                    Try fewer or different words{semanticOff ? "" : ", or describe what the picture shows"}.
                  </span>
                  {limited && <button className="btn btn-line" onClick={() => { setTypes([]); setPicked([]); }}>Search everywhere</button>}
                </div>
              )}
              {shown.map((r, i) => (
                <div
                  key={r.file_path}
                  id={`r-${i}`}
                  data-i={i}
                  role="option"
                  aria-selected={i === sel}
                  className={`result ${i === sel ? "sel" : ""}`}
                  onMouseDown={() => setSel(i)}
                  onDoubleClick={() => act("open", r)}
                >
                  <span className="result-no" title="Rank: 1 is the best match">#{r.rank ?? i + 1}</span>
                  <FileVisual path={r.file_path} ext={r.file_type} size={44} thumb={96} t={r.start_sec} />
                  <div className="result-main">
                    <div className="result-name"><Highlight text={r.file_name} query={parsed.text} /></div>
                    <div className="result-path" title={r.file_path}>{shortFolder(r.file_path, 4)}</div>
                    {r.start_sec != null && <div className="result-at">Matched at {fmtTime(r.start_sec)}</div>}
                    {r.snippet && <div className="result-snip"><Highlight text={r.snippet} query={parsed.text} /></div>}
                  </div>
                  <div className="result-side">
                    <span className={`match m-${r.match_type}`} title={MATCH[r.match_type].hint}>{MATCH[r.match_type].label}</span>
                    {r.match_type !== "filename" && <Score pct={shownPct(r)} />}
                  </div>
                </div>
              ))}
            </div>

            <aside className="preview" aria-label="Preview">
              {current ? (
                <>
                  <div className="pv-scroll">
                  <div className="pv-visual">
                    <FileVisual path={current.file_path} ext={current.file_type} size={84} thumb={480} className="big" t={current.start_sec} />
                  </div>
                  <h2 className="pv-name">{current.file_name}</h2>
                  <p className="pv-path selectable">{current.file_path}</p>

                  {current.match_type !== "filename" && (
                  <div className="pv-score">
                    <div className="pv-score-top">
                      <span className="pv-rank">Rank #{current.rank ?? shown.indexOf(current) + 1}<em> of {results.length}</em></span>
                      <span className="pv-quality">{quality(shownPct(current))}</span>
                    </div>
                    <Score pct={shownPct(current)} />
                  </div>
                  )}

                  <dl className="credits tight">
                    <div><dt>Type</dt><dd>{(current.file_type || "—").toUpperCase()}</dd></div>
                    <div><dt>Found by</dt><dd>{MATCH[current.match_type].label}</dd></div>
                    {current.start_sec != null && (
                      <div><dt>Matched at</dt><dd>{fmtTime(current.start_sec)}<span className="dim"> in the video</span></dd></div>
                    )}
                    {current.keyword_rank != null && (
                      <div><dt>Keyword</dt><dd>#{current.keyword_rank}{current.keyword_score != null && <span className="dim"> · relevance {current.keyword_score.toFixed(2)}</span>}</dd></div>
                    )}
                    {current.semantic_rank != null && (
                      <div><dt>Meaning</dt><dd>#{current.semantic_rank}{current.similarity != null && <span className="dim"> · similarity {Math.round(current.similarity * 100)}%</span>}</dd></div>
                    )}
                  </dl>
                  <p className="pv-note">
                    {current.match_type === "filename"
                      ? "This file's name fits the pattern you typed. Name matches are listed A to Z."
                      : semanticOff
                      ? "Files are ranked by how well the words you typed match."
                      : "Keyword and meaning results are combined (rank fusion) so strong matches in either rise to the top."}
                  </p>
                  {current.snippet && (
                    <p className="pv-snip selectable"><Highlight text={current.snippet} query={parsed.text} /></p>
                  )}
                  </div>
                  <div className="pv-actions">
                    <button className="btn btn-primary" onClick={() => act("open", current)}>
                      <IcOpen width={16} height={16} /> Open file <kbd>Enter</kbd>
                    </button>
                    <button className="btn btn-line" onClick={() => act("reveal", current)} title="Shift+Enter">
                      <IcFolder width={16} height={16} /> Show in folder
                    </button>
                    <button className="btn btn-line" onClick={() => act("copy", current)} title="Ctrl+C">
                      <IcCopy width={16} height={16} /> Copy path
                    </button>
                  </div>
                </>
              ) : (
                <div className="pv-empty">Select a file to see a preview.</div>
              )}
            </aside>
          </div>
        </>
      )}
    </section>
  );
}