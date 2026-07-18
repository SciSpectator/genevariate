#!/usr/bin/env python3
"""LLM-GEO-Label-Extractor — headless CLI runner (no GUI, cluster-friendly).

Same pipeline as ``llm_label_extractor.py`` but driven entirely by command-line
arguments and stdout / stderr logging. No tkinter import, no display required.
Designed for SLURM / PBS submission via ``submit.sh``.

Phases run for each GSE in turn:

    Phase 1   verbatim per-label LLM extraction
    Phase 1b  GSE-context inference for "Not Specified" fields
    Phase 1c  deterministic consensus (+ optional LLM semantic curator)
    Phase 2   multi-agent MeSH cascade canonicalisation

Inputs
------
  --samples samples.json   list of {gsm, title, source_name_ch1,
                           characteristics_ch1, treatment_protocol_ch1,
                           description, gse} dicts
  --gpl GPLxxxx            (optional) dump every GSM on this platform from
                           GEOmetadb.sqlite into ``samples.json`` first
  --output out.json        final merged JSON written at the end
  --checkpoint ckpt.jsonl  per-GSE JSONL append; rerun resumes from here

Run options
-----------
  --workers N              N collapsers + 1 verifier (1..3, default 1)
  --backend ollama|vllm    LLM backend (default ollama)
  --model NAME             model id (default gemma4-e2b-text:latest)
  --no-scrape              skip NCBI GSE-meta scrape (use existing sidecar)
  --no-p1 / --no-p1b /     turn off any phase
  --no-p1c / --no-p2

Example
-------
  python run_cli.py --samples samples.json \\
                    --output  out.json    \\
                    --checkpoint ckpt.jsonl \\
                    --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _pmap(fn, items, workers):
    """Parallel map preserving input order. Falls back to serial when workers<=1."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

# ─── Constants (single source of truth for the headless runner) ─────────────
MAX_WORKERS         = 32   # hard cap; bump cautiously, see README "Tunable parameters"
DEFAULT_WORKERS     = 1    # safe default (conservative, matches the legacy serial path)
DEFAULT_MODEL       = "gemma4-e2b-text:latest"
DEFAULT_OLLAMA_URL  = "http://localhost:11434"
DEFAULT_VLLM_URL    = "http://localhost:8000/v1"
DEFAULT_LLM_BACKEND = "ollama"
LABEL_COLS          = ("Tissue", "Condition", "Treatment")
NS                  = "Not Specified"
SNAPSHOT_EVERY_N    = 100

HERE   = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))


# ─── argparse first, so env can be set BEFORE phase imports ────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_cli.py",
        description="LLM-GEO-Label-Extractor headless CLI runner.")
    p.add_argument("--samples", required=False, type=Path,
                   help="input samples.json (skip if --gpl is set with --dump-only)")
    p.add_argument("--output",  required=False, type=Path,
                   help="final merged output JSON (omit with --search-gpl)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="per-GSE JSONL append for crash-resume")
    p.add_argument("--gpl", default=None,
                   help="optional GPL id; dumps all GSMs into --samples first")
    p.add_argument("--search-gpl", default=None, metavar="QUERY",
                   help="search GEOmetadb for matching GPLs and print them. "
                        "QUERY can be a GPL id (e.g. GPL570), an organism "
                        "(e.g. 'homo sapiens'), or a fragment of the platform "
                        "title. Combine with --tech to filter by technology. "
                        "Standalone mode — exits after printing.")
    p.add_argument("--tech", default=None,
                   help="technology filter for --search-gpl (e.g. "
                        "'in situ oligonucleotide', 'high-throughput sequencing')")
    p.add_argument("--organism", default=None,
                   help="organism / species filter for --search-gpl "
                        "(e.g. 'Homo sapiens', 'Mus musculus')")
    p.add_argument("--list-gpl-limit", type=int, default=50,
                   help="max rows for --search-gpl (default 50)")
    p.add_argument("--geometadb", type=Path, default=None,
                   help="path to GEOmetadb.sqlite (for --gpl or --search-gpl)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"1..{MAX_WORKERS} (default {DEFAULT_WORKERS})")
    p.add_argument("--gse-workers", type=int, default=1,
                   help="how many GSEs to process concurrently (default 1; "
                        "raise ONLY if Phase 2 is disabled or the agent fleet "
                        "is verified deadlock-free on your hardware)")
    p.add_argument("--backend", default=DEFAULT_LLM_BACKEND,
                   choices=["ollama", "vllm", "sglang", "openai"])
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--vllm-url",   default=DEFAULT_VLLM_URL)
    p.add_argument("--limit",  type=int, default=0,
                   help="cap input to first N samples (0 = all)")
    p.add_argument("--no-resume",  action="store_true",
                   help="ignore existing checkpoint")
    p.add_argument("--no-scrape",  action="store_true")
    p.add_argument("--no-p1",      action="store_true")
    p.add_argument("--no-p1b",     action="store_true")
    p.add_argument("--no-p1c",     action="store_true")
    p.add_argument("--no-p2",      action="store_true")
    p.add_argument("--no-semantic-1c", action="store_true",
                   help="skip the heavy semantic Phase 1c curator")
    args = p.parse_args()

    # Reasoning is hard-wired ON in GeneVariate — no toggle. Every LLM
    # call site reads THINK_MODE; setting it here propagates pipeline-wide.
    os.environ["THINK_MODE"] = "true"

    args.workers = max(1, min(MAX_WORKERS, args.workers))
    if args.search_gpl is not None:
        # Standalone search mode — no pipeline, no output needed.
        return args
    if not args.output:
        sys.exit("[err] --output is required (omit only with --search-gpl)")
    if args.gpl and not args.samples:
        sys.exit("[err] --gpl requires --samples to write into")
    if not args.gpl and not args.samples:
        sys.exit("[err] need --samples (or --gpl + --samples)")
    return args


def _search_gpl(query: str, tech: str | None, db_path: Path,
                limit: int, organism: str | None = None) -> list[dict]:
    """Mirror of the GUI's _search_platforms_worker. Returns rows sorted
    by sample count (descending). Two SQL passes — first the gpl table,
    then a per-GPL COUNT(*) on gsm to get sample counts."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    where, args_sql = [], []
    qu = (query or "").strip().upper()
    is_gpl = qu and ((qu.startswith("GPL") and qu[3:].isdigit())
                     or qu.isdigit())
    if is_gpl:
        if qu.isdigit():
            qu = f"GPL{qu}"
        where.append("(UPPER(gpl) = ? OR UPPER(gpl) LIKE ?)")
        args_sql += [qu, f"%{qu}%"]
    elif query:
        pat = f"%{query.lower()}%"
        where.append("LOWER(title) LIKE ?")
        args_sql.append(pat)
    if organism:
        where.append("LOWER(organism) LIKE ?")
        args_sql.append(f"%{organism.lower()}%")
    if tech:
        where.append("LOWER(technology) LIKE ?")
        args_sql.append(f"%{tech.lower()}%")
    sql = ("SELECT gpl, title, technology, organism, data_row_count "
           "FROM   gpl "
           f"WHERE {' AND '.join(where) if where else '1=1'} "
           f"ORDER BY data_row_count DESC LIMIT {int(max(1, limit))}")
    rows = cur.execute(sql, args_sql).fetchall()
    out = [{"gpl": g, "title": t, "technology": tc,
            "organism": o, "probes": p, "samples": None}
           for (g, t, tc, o, p) in rows]
    # second pass: sample counts
    ids = [r["gpl"] for r in out]
    counts: dict[str, int] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        ph = ",".join(["?"] * len(chunk))
        try:
            counts.update(dict(cur.execute(
                f"SELECT gpl, COUNT(*) FROM gsm WHERE gpl IN ({ph}) "
                f"GROUP BY gpl", chunk).fetchall()))
        except Exception:
            pass
    for r in out:
        r["samples"] = int(counts.get(r["gpl"], 0))
    con.close()
    return out


def _print_gpl_table(rows: list[dict]) -> None:
    if not rows:
        print("(no GPLs matched)")
        return
    head = f"{'GPL':<10} {'samples':>9} {'probes':>10} " \
           f"{'organism':<28} {'technology':<32} title"
    print(head)
    print("-" * min(len(head) + 80, 200))
    for r in rows:
        print(f"{r['gpl']:<10} {r['samples']:>9,} "
              f"{(r['probes'] or 0):>10,} "
              f"{(r['organism'] or '')[:28]:<28} "
              f"{(r['technology'] or '')[:32]:<32} "
              f"{(r['title'] or '')[:90]}")


# ─── Module-level helpers (reused from the GUI file, GUI-free) ─────────────
def _gse_of(row: dict) -> str:
    for k in ("gse", "series_id", "series", "GSE"):
        v = row.get(k)
        if v:
            return str(v).split(",")[0].strip()
    return ""


def _build_raw(row: dict) -> dict:
    return {
        "gsm_title":          row.get("title") or "",
        "source_name":        row.get("source_name_ch1") or "",
        "characteristics":    row.get("characteristics_ch1") or "",
        "treatment_protocol": row.get("treatment_protocol_ch1") or "",
        "description":        row.get("description") or "",
    }


def _is_ns(v) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("", NS.lower(), "n/a", "na", "none", "null", "unknown")


def _compact_for_phase1c(row: dict, labels: dict) -> dict:
    return {
        "gsm":                row.get("gsm"),
        "title":              row.get("title") or "",
        "source_name":        row.get("source_name_ch1") or "",
        "characteristics":    row.get("characteristics_ch1") or "",
        "treatment_protocol": row.get("treatment_protocol_ch1") or "",
        "description":        row.get("description") or "",
        "phase1":             labels,
    }


def _read_checkpoint(path: Path):
    seen, rows = set(), []
    if not path or not path.exists():
        return seen, rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add(obj.get("gse"))
            rows.extend(obj.get("samples") or [])
    return seen, rows


def _append_checkpoint(path: Path, gse: str, samples: list) -> None:
    if not path:
        return
    line = json.dumps({"gse": gse, "samples": samples},
                      ensure_ascii=False, default=str)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_snapshot(out_path: Path, rows: list, n_done: int) -> None:
    if not out_path:
        return
    snap = out_path.with_suffix(out_path.suffix + ".partial.json") \
        if out_path.suffix else out_path.with_name(out_path.name + ".partial.json")
    tmp = snap.with_suffix(snap.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"samples_done": n_done, "samples": rows},
                  f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, snap)


def _find_geometadb(arg: Path | None) -> Path:
    if arg and arg.exists():
        return arg
    for p in (HERE / "GEOmetadb.sqlite",
              PARENT / "GEOmetadb.sqlite",
              Path.cwd() / "GEOmetadb.sqlite"):
        if p.exists():
            return p
    sys.exit("[err] GEOmetadb.sqlite not found — pass --geometadb")


def _dump_gpl_samples(gpl: str, db_path: Path, out_path: Path) -> int:
    """Read every GSM on a GPL platform from GEOmetadb into a samples.json."""
    print(f"[gpl] {gpl} -> {out_path} (db={db_path})", flush=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.execute(
        "SELECT gsm, title, source_name_ch1, characteristics_ch1, "
        "       treatment_protocol_ch1, description, series_id "
        "FROM   gsm WHERE gpl = ?",
        (gpl,))
    rows = []
    for r in cur:
        d = dict(r)
        d["gpl"] = gpl
        d["gse"] = (d.pop("series_id") or "").split(",")[0].strip()
        rows.append(d)
    con.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[gpl] wrote {len(rows)} samples", flush=True)
    return len(rows)


# ─── Inline GSE-metadata scraper (zero evals/ dependency) ──────────────────
import re
from urllib import request as _urlreq
from urllib.error import URLError, HTTPError

_GEO_SOFT_URL   = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
                   "?targ=self&form=text&view=quick&acc={gse}")
_GEO_USER_AGENT = ("LLM-GEO-Label-Extractor/1.0 "
                   "(github.com/SciSpectator/LLM-GEO-Label-Extractor)")
_GEO_TIMEOUT_S    = 30
_GEO_RETRY_DELAYS = (1, 2, 5)
_GEO_LINE_RE = re.compile(
    r"^!Series_(title|summary|overall_design)\s*=\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def scrape_gse_meta(gse: str) -> dict:
    url = _GEO_SOFT_URL.format(gse=gse)
    text, last_err = "", None
    for delay in (0,) + _GEO_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            req = _urlreq.Request(url, headers={"User-Agent": _GEO_USER_AGENT})
            with _urlreq.urlopen(req, timeout=_GEO_TIMEOUT_S) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                break
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            last_err = e
    if not text:
        return {"gse_title": "", "gse_summary": "", "gse_design": "",
                "_error": str(last_err) if last_err else ""}
    buckets = {"title": [], "summary": [], "overall_design": []}
    for m in _GEO_LINE_RE.finditer(text):
        v = m.group(2).strip()
        if v:
            buckets[m.group(1).lower()].append(v)
    return {
        "gse_title":   "\n".join(buckets["title"]).strip(),
        "gse_summary": "\n".join(buckets["summary"]).strip(),
        "gse_design":  "\n".join(buckets["overall_design"]).strip(),
    }


# ─── Pipeline orchestrator (mirrors GUI App._run_pipeline, GUI-free) ────────
def run_pipeline(args: argparse.Namespace, stop_evt) -> None:
    # Lazy imports: heavy ML deps only after env is set.
    from phase1            import Phase1Agent
    from phase1b           import Phase1bAgent
    from phase1c_consensus import ConsensusCurator
    from gse_context_cache import GSEContextCache
    from gse_summarizer    import get_or_build_compressed
    from cached_extractors import (
        CachedPhase1Agent, CachedPhase1bAgent,
        PHASE1_PROMPT_VERSION, PHASE1B_PROMPT_VERSION,
    )

    log = lambda m: print(m, flush=True)

    # ── INLINE GUARDRAIL — pre-extraction health check ────────────────
    # Fail loud with actionable message before any LLM call. Catches the
    # 2026-04-28 OOD-mesh cache corruption (FTD→WT silent mapping) and the
    # 2026-05-08 GPU-starvation pattern (BioLORD OOM swallowed silently).
    from mesh_lookup import MeshDB as _MeshDB
    try:
        _MeshDB.verify_pipeline_health(
            ollama_url=getattr(args, "ollama_url", "http://127.0.0.1:11434"),
            model_name=args.model,
            gse_cache_db="gse_context_cache.sqlite",
            require_biolord=not args.no_p2,
            strict=True,
        )
        log("[health] preflight OK (ollama, mesh.sqlite, gse_context_cache, BioLORD)")
    except RuntimeError as _e:
        log(f"[health] PREFLIGHT FAILED: {_e}")
        log("[health] aborting before extraction. Run: "
            "python -c 'from mesh_lookup import MeshDB; MeshDB.repair_ood_mesh()'")
        raise

    # Per-phase error counters — abort if any phase shows >25% failure rate
    # (catches mid-run cache corruption, OOM cascades, Ollama wedges).
    _phase_err: dict = {"p1": 0, "p1b": 0, "p1c": 0, "p2": 0, "samples": 0}
    _PHASE_ERR_FRAC = 0.25
    def _phase_health_assert(phase: str) -> None:
        n = max(1, _phase_err["samples"])
        if _phase_err[phase] / n > _PHASE_ERR_FRAC:
            raise RuntimeError(
                f"[health] phase {phase} error rate "
                f"{_phase_err[phase]}/{n} > {_PHASE_ERR_FRAC:.0%}. "
                f"Likely Ollama wedged, GPU OOM, or DB corrupt. Aborting.")

    # GPL dump (optional)
    if args.gpl:
        db = _find_geometadb(args.geometadb)
        _dump_gpl_samples(args.gpl, db, args.samples)

    with open(args.samples) as f:
        all_rows = json.load(f)
    if args.limit > 0:
        all_rows = all_rows[:args.limit]

    by_gse: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_gse[_gse_of(r)].append(r)
    log(f"[run] {len(all_rows)} samples across {len(by_gse)} GSEs")

    resumed_gses, resumed_rows = set(), []
    if args.checkpoint and not args.no_resume:
        resumed_gses, resumed_rows = _read_checkpoint(args.checkpoint)
        for g in list(by_gse):
            if g in resumed_gses:
                by_gse.pop(g)
        if resumed_gses:
            log(f"[run] resume — {len(resumed_gses)} GSEs already in checkpoint")

    # GSE meta sidecar (scrape on demand)
    meta_path = HERE / "gse_meta_scraped.json"
    gse_meta: dict[str, dict] = {}
    if meta_path.exists():
        try:
            gse_meta = json.load(open(meta_path))
        except Exception:
            gse_meta = {}
    if not args.no_scrape:
        todo = [g for g in by_gse if g and g not in gse_meta]
        log(f"[scrape] need {len(todo)} GSEs from NCBI")
        for i, gse in enumerate(todo, 1):
            if stop_evt["stop"]:
                break
            gse_meta[gse] = scrape_gse_meta(gse)
            if i % 5 == 0 or i == len(todo):
                json.dump(gse_meta, open(meta_path, "w"),
                          indent=2, sort_keys=True)
            log(f"  [{i}/{len(todo)}] {gse}")
            time.sleep(0.3)
        json.dump(gse_meta, open(meta_path, "w"),
                  indent=2, sort_keys=True)

    # Build agents
    cache = GSEContextCache()
    p1_agent  = CachedPhase1Agent(Phase1Agent(), cache,
                                  model_version=args.model,
                                  prompt_version=PHASE1_PROMPT_VERSION)
    p1b_agent = CachedPhase1bAgent(Phase1bAgent(), cache,
                                   model_version=args.model,
                                   prompt_version=PHASE1B_PROMPT_VERSION)
    consensus_1c = ConsensusCurator(cache, threshold=0.80) \
        if not args.no_p1c else None
    semantic_1c = None
    if not args.no_p1c and not args.no_semantic_1c:
        try:
            from phase1c_semantic import SemanticPhase1cCurator
            semantic_1c = SemanticPhase1cCurator()
        except Exception as e:
            log(f"[1c] semantic curator unavailable: {e!r}")

    coordinator = None
    if not args.no_p2:
        try:
            from agents      import Coordinator
            from mesh_lookup import MeshDB
            coordinator = Coordinator(
                n_collapsers=args.workers,
                n_verifiers=1,
                use_router=True,
                db=MeshDB(),
                cache=cache,
            )
            coordinator.start()
            log(f"[phase2] coordinator started (collapsers={args.workers})")
        except Exception as e:
            log(f"[phase2] init failed: {e!r} — skipping Phase 2")
            coordinator = None

    out_rows = list(resumed_rows)
    total = len(by_gse)
    import threading
    _ckpt_lock = threading.Lock()
    _out_lock = threading.Lock()

    def _process_one_gse(idx, gse_id, rows):
        if stop_evt["stop"]:
            return None
        gmeta = dict(gse_meta.get(gse_id, {}) or {})
        if gmeta:
            try:
                gmeta["gse_summary"] = get_or_build_compressed(
                    cache, gse_id,
                    gmeta.get("gse_title", ""),
                    gmeta.get("gse_summary", ""),
                    gmeta.get("gse_design", ""),
                    max_chars=512)
            except Exception as e:
                log(f"  [summarize {gse_id}] {e!r}")
        t0 = time.time()

        # Phase 1
        if not args.no_p1:
            def _do_p1(r):
                try:
                    return p1_agent.extract(
                        _build_raw(r), gsm=r.get("gsm"), gse=gse_id)
                except Exception as e:
                    log(f"  [p1 err {r.get('gsm')}] {e!r}")
                    _phase_err["p1"] += 1
                    return {c: NS for c in LABEL_COLS}
            p1_results = _pmap(_do_p1, rows, args.workers)
            for r, res in zip(rows, p1_results):
                r["_phase1"] = res
        else:
            for r in rows:
                r["_phase1"] = {c: NS for c in LABEL_COLS}

        # Phase 1b
        if not args.no_p1b:
            gse_dist = {c: Counter() for c in LABEL_COLS}
            for r in rows:
                for c in LABEL_COLS:
                    gse_dist[c][r["_phase1"].get(c, NS)] += 1

            def _do_p1b(r):
                p1l = dict(r["_phase1"])
                if not any(_is_ns(p1l.get(c, NS)) for c in LABEL_COLS):
                    return p1l
                sib = {c: Counter(gse_dist[c]) for c in LABEL_COLS}
                for c in LABEL_COLS:
                    own = p1l.get(c, NS)
                    if sib[c][own] > 0:
                        sib[c][own] -= 1
                        if sib[c][own] == 0:
                            del sib[c][own]
                try:
                    return p1b_agent.infer_sample(
                        r["gsm"], _build_raw(r), p1l,
                        gse_id, gmeta, sibling_dist=sib)
                except Exception as e:
                    log(f"  [p1b err {r.get('gsm')}] {e!r}")
                    _phase_err["p1b"] += 1
                    return p1l
            p1b_results = _pmap(_do_p1b, rows, args.workers)
            for r, res in zip(rows, p1b_results):
                r["_phase1b"] = res
        else:
            for r in rows:
                r["_phase1b"] = dict(r["_phase1"])

        # Phase 1c
        if consensus_1c is not None:
            cache.flush_aggregates(gse=gse_id)
            def _do_p1c(r):
                try:
                    decisions = consensus_1c.curate_sample(
                        gse_id, r["gsm"], r["_phase1b"])
                    return {c: decisions[c]["final_value"]
                            for c in LABEL_COLS}
                except Exception as e:
                    log(f"  [p1c-cons err {r.get('gsm')}] {e!r}")
                    _phase_err["p1c"] += 1
                    return dict(r["_phase1b"])
            p1c_results = _pmap(_do_p1c, rows, args.workers)
            for r, res in zip(rows, p1c_results):
                r["_phase1c"] = res
            if semantic_1c is not None:
                try:
                    samples_for_1c = [
                        _compact_for_phase1c(r, r["_phase1c"]) for r in rows]
                    res = semantic_1c.curate(gmeta, samples_for_1c)
                    accepted = {(a["gsm"], a["field"]): a["suggest"]
                                for a in res.get("accepted", [])}
                    for r in rows:
                        for c in LABEL_COLS:
                            sug = accepted.get((r["gsm"], c))
                            if sug is not None:
                                r["_phase1c"][c] = sug
                except Exception as e:
                    log(f"  [p1c-sem err {gse_id}] {e!r}")
        else:
            for r in rows:
                r["_phase1c"] = dict(r["_phase1b"])

        # Phase 1c sibling canonicalization — universal token-fingerprint
        # grouper. For each (gse, col), groups morphologically-equivalent
        # phase1c surfaces (former smoker / former smoking / former) into
        # one canonical surface, so siblings within a GSE describe the
        # same biological state with the same label. The original
        # value is stashed under r["_phase1c_raw"] for audit. Runs
        # in-process (deterministic, no LLM calls).
        if not args.no_p1c:
            try:
                from run_phase1c_canonicalize import canonicalize as _p1c_canon
                _samples_for_canon = [
                    {"gse": gse_id, "phase1c": dict(r["_phase1c"]),
                     "_back": r}
                    for r in rows
                ]
                _stats = _p1c_canon(_samples_for_canon)
                for s in _samples_for_canon:
                    s["_back"]["_phase1c"] = s["phase1c"]
                    if "phase1c_raw" in s:
                        s["_back"]["_phase1c_raw"] = s["phase1c_raw"]
                if _stats.get("samples_changed", 0):
                    log(f"  [p1c-canon {gse_id}] {_stats['samples_changed']} "
                        f"samples updated, {_stats['surfaces_collapsed']} "
                        f"surfaces collapsed")
            except Exception as e:
                log(f"  [p1c-canon err {gse_id}] {e!r}")

        for r in rows:
            for c in LABEL_COLS:
                try:
                    cache.upsert_phase_value(
                        gse_id, r["gsm"], c, "p1c",
                        r["_phase1c"].get(c, NS))
                except Exception:
                    pass

        # Phase 2 — output is MeSH descriptor NAMES (not IDs).
        if coordinator is not None:
            ctx = "\n".join([gmeta.get("gse_title", ""),
                             gmeta.get("gse_summary", ""),
                             gmeta.get("gse_design", "")]).strip()

            # Sibling warm-up pass: resolve every UNIQUE (raw, col)
            # within this GSE serially BEFORE the parallel fan-out below.
            # Populates the per-GSE sibling cache (gse_phase2_canon) so
            # the parallel pass hits Tier 1.5 (exact-raw cache) on every
            # sibling instead of racing the cascade. Without this, N
            # parallel workers that see the same raw simultaneously can
            # each fall through to Tier 4 LLM picker and produce N
            # different canonical IDs (the sibling-divergence pattern).
            # Universal — iterates whatever GSEs/raws are in the input.
            try:
                _unique: set = set()
                for r in rows:
                    for c in LABEL_COLS:
                        raw = (r["_phase1c"].get(c) or "").strip()
                        if raw and not _is_ns(raw):
                            _unique.add((raw, c))
                if _unique:
                    log(f"  [p2-warmup {gse_id}] seeding {len(_unique)} "
                        f"unique (raw,col) into per-GSE sibling cache")
                    for _raw, _col in _unique:
                        try:
                            coordinator.collapse(
                                _raw, _col, context=ctx,
                                gse_id=gse_id, timeout=180.0)
                        except Exception as e:
                            log(f"  [p2-warmup err {gse_id}/{_raw!r}/{_col}] "
                                f"{e!r}")
            except Exception as e:
                log(f"  [p2-warmup err {gse_id}] {e!r}")

            def _do_p2(r):
                p2: dict[str, str] = {}
                for c in LABEL_COLS:
                    raw = r["_phase1c"].get(c, NS)
                    if _is_ns(raw):
                        p2[c] = NS
                        continue
                    try:
                        resp = coordinator.collapse(
                            raw, c, context=ctx,
                            gse_id=gse_id, timeout=180.0)
                        p2[c] = resp.get("canonical") \
                            or resp.get("label") or raw
                    except Exception as e:
                        log(f"  [p2 err {r.get('gsm')}/{c}] {e!r}")
                        _phase_err["p2"] += 1
                        p2[c] = raw
                return p2

            p2_results = _pmap(_do_p2, rows, args.workers)
            for r, p2 in zip(rows, p2_results):
                r["_phase2"] = p2
        else:
            for r in rows:
                r["_phase2"] = dict(r["_phase1c"])

        dt = time.time() - t0
        gse_samples = [{
            "gsm":     r.get("gsm"),
            "gse":     gse_id,
            "gpl":     r.get("gpl"),
            "title":   r.get("title"),
            "source":  r.get("source_name_ch1"),
            "characteristics":    r.get("characteristics_ch1"),
            "treatment_protocol": r.get("treatment_protocol_ch1"),
            "phase1":     r.get("_phase1"),
            "phase1b":    r.get("_phase1b"),
            "phase1c":    r.get("_phase1c"),
            "phase2":     r.get("_phase2"),
        } for r in rows]

        with _out_lock:
            prev_done = len(out_rows)
            out_rows.extend(gse_samples)
            do_snap = (len(out_rows) // SNAPSHOT_EVERY_N
                       > prev_done // SNAPSHOT_EVERY_N)
            n_now = len(out_rows)
        if args.checkpoint:
            with _ckpt_lock:
                _append_checkpoint(args.checkpoint, gse_id, gse_samples)
        if do_snap:
            try:
                _write_snapshot(args.output, out_rows, n_now)
                log(f"  [snapshot] {n_now} samples")
            except Exception as e:
                log(f"  [snapshot err] {e!r}")
        # ── per-GSE phase-health gate ─────────────────────────────────
        # After this GSE is done, check the cumulative error fractions
        # across every phase. >25% in any one phase = systemic failure
        # (Ollama wedge, GPU OOM cascade, or DB drift) — abort the run
        # with a loud message instead of polluting hours more output.
        _phase_err["samples"] += len(rows)
        for _ph in ("p1", "p1b", "p1c", "p2"):
            _phase_health_assert(_ph)

        log(f"  [{idx}/{total}] {gse_id} n={len(rows)} {dt:.1f}s "
            f"err[p1/p1b/p1c/p2]="
            f"{_phase_err['p1']}/{_phase_err['p1b']}/{_phase_err['p1c']}/{_phase_err['p2']}")
        return gse_samples

    try:
        gse_workers = max(1, min(args.workers, args.gse_workers))
        log(f"[run] processing {total} GSEs with gse_workers={gse_workers}")
        if gse_workers <= 1:
            for idx, (gse_id, rows) in enumerate(by_gse.items(), 1):
                if stop_evt["stop"]:
                    log("[run] stopped")
                    break
                _process_one_gse(idx, gse_id, rows)
        else:
            with ThreadPoolExecutor(max_workers=gse_workers) as ex:
                futs = []
                for idx, (gse_id, rows) in enumerate(by_gse.items(), 1):
                    futs.append(ex.submit(_process_one_gse, idx, gse_id, rows))
                for fut in futs:
                    try:
                        fut.result()
                    except Exception as e:
                        log(f"[gse-fut-err] {e!r}")
    finally:
        if coordinator is not None:
            try:
                coordinator.stop()
            except Exception:
                pass
        try:
            cache.close()
        except Exception:
            pass

    with open(args.output, "w") as f:
        json.dump({"samples":   out_rows,
                   "n_samples": len(out_rows),
                   "n_gses":    len(by_gse) + len(resumed_gses)},
                  f, ensure_ascii=False, indent=2, default=str)
    log(f"[run] wrote {args.output} ({len(out_rows)} samples)")


def main() -> None:
    args = _parse_args()

    # --- Standalone GPL search: no pipeline, no LLM, no env mutation ---
    if args.search_gpl is not None:
        db = _find_geometadb(args.geometadb)
        rows = _search_gpl(args.search_gpl, args.tech, db,
                           args.list_gpl_limit, organism=args.organism)
        _print_gpl_table(rows)
        sys.exit(0)

    # Env BEFORE any phase-module import (modules read at import-time).
    os.environ["LLM_BACKEND"]         = args.backend
    os.environ["PHASE1_BACKEND"]      = args.backend
    os.environ["PHASE1_MODEL"]        = args.model
    os.environ["OLLAMA_URL"]          = args.ollama_url
    os.environ["OLLAMA_HOST"]         = args.ollama_url
    os.environ["VLLM_URL"]            = args.vllm_url
    os.environ["OLLAMA_NUM_PARALLEL"] = str(args.workers)
    # DB defaults: prefer parent-dir copies if present, else repo-local.
    for env_name, fname in (("MESH_DB",           "mesh.sqlite"),
                            ("GSE_CONTEXT_CACHE", "gse_context_cache.sqlite")):
        if env_name in os.environ:
            continue
        parent_path = PARENT / fname
        local_path  = HERE / fname
        os.environ[env_name] = str(parent_path if parent_path.exists()
                                   else local_path)

    # Cooperative stop on SIGINT / SIGTERM (sbatch sends SIGTERM at walltime).
    stop_evt = {"stop": False}

    def _on_sig(sig, _frame):
        stop_evt["stop"] = True
        print(f"[run] signal {sig} — finishing current GSE then exiting",
              flush=True)
    signal.signal(signal.SIGINT,  _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    print(f"[run] backend={args.backend} workers={args.workers} "
          f"model={args.model}", flush=True)
    run_pipeline(args, stop_evt)


if __name__ == "__main__":
    main()
