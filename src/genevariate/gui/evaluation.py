#!/usr/bin/env python3
"""
GeneVariate — Fully Automatic Label Extraction Evaluation
=========================================================
Platforms evaluated: GPL570, GPL6947, GPL96, GPL10558
                     (samples drawn proportionally from all 4, then shuffled)

Pipeline phases evaluated (independently at each stage):
  Phase 1    Raw LLM extraction (no .title(), no synonyms, no post-processing)
  Phase 1.5  Per-GSE label collapsing (exact match + abbreviation initials only)
  Phase 2    NS recovery (Condition, Tissue, Treatment — GSE context + sibling consensus)
  Phase 3    LLM Curator (cross-experiment, LLM judges label pairs — optional)

Evaluation methodology:
  Phase 1 accuracy:     Compare raw LLM output vs gold standard (human/judge)
  Phase 1→1.5 delta:    Did collapsing help? Compare accuracy before vs after
  Phase 1.5 collapsing: True positive merges vs false positive merges
  Phase 2 NS recovery:  Precision/recall of recovered labels
  Phase 3 LLM Curator:  Precision of proposed cross-experiment merges

Human evaluation pipeline (200 samples):
  STEP 1  Sample 200 GSMs across 4 GPL platforms (shuffled)
  STEP 2  Phase 1 raw extraction (LLM, GPU-accelerated)
  STEP 3  Phase 1.5 — per-GSE collapsing (exact + abbreviation)
  STEP 4  Human GUI — Round 1: evaluate Phase 1 raw + Phase 1.5 labels
  STEP 5  Phase 2 NS recovery (ContextRecallExtractor)
  STEP 6  Human GUI — Round 2: evaluate NS corrections
  STEP 7  Phase 3 LLM Curator scan + human review of proposals

LLM judge pipeline (1000 samples):
  STEP 8   Sample 1000 GSMs across 4 GPL platforms (shuffled)
  STEP 9   Phase 1 + Phase 1.5 + Phase 2 labelling
  STEP 10  Parallel LLM judge evaluation
  STEP 11  Phase 3 LLM Curator scan + accuracy assessment
  STEP 12  Final report + per-phase metrics
"""

import os, sys, time, json, gzip, shutil, sqlite3, random, threading
import requests
import ollama
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# ── RAM + VRAM watchdog ─────────────────────────────────────────
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

RAM_PAUSE_PCT  = 90
RAM_RESUME_PCT = 80
RAM_POLL_SEC   = 3

_ram_pause_event = threading.Event()
_ram_pause_event.set()   # set = NOT paused
_watchdog_status = ""
_vram_warn_shown = False


def _ram_watchdog_loop():
    global _watchdog_status, _vram_warn_shown
    if not _PSUTIL:
        return
    while True:
        pct = psutil.virtual_memory().percent
        vram_used, vram_total, vram_pct = _get_vram_usage()
        has_gpu = vram_total > 0

        # Build status
        state = "running" if _ram_pause_event.is_set() else "PAUSED"
        if has_gpu:
            _watchdog_status = (f"RAM:{pct:.0f}% | VRAM:{vram_used:,}/{vram_total:,}MB "
                                f"({vram_pct:.0f}%) | {state}")
        else:
            _watchdog_status = f"RAM:{pct:.0f}% | {state}"

        # Pause ONLY on RAM (prevents OOM crash)
        if pct >= RAM_PAUSE_PCT and _ram_pause_event.is_set():
            _ram_pause_event.clear()
            print(f"\n  [Watchdog] RAM at {pct:.1f}% — PAUSED", flush=True)

        # VRAM full → just warn once, Ollama falls back to CPU
        elif has_gpu and vram_pct >= 90.0 and not _vram_warn_shown:
            _vram_warn_shown = True
            print(f"\n  [Watchdog] VRAM at {vram_pct:.0f}% — Ollama using CPU fallback (no pause)",
                  flush=True)

        # Resume from RAM pause
        elif not _ram_pause_event.is_set():
            if pct < RAM_RESUME_PCT:
                _ram_pause_event.set()
                print(f"\n  [Watchdog] RAM OK — RESUMED", flush=True)

        time.sleep(RAM_POLL_SEC)


def _start_ram_watchdog():
    if not _PSUTIL:
        print("  [Watchdog] psutil not found — monitoring disabled  (pip install psutil)")
        return
    t = threading.Thread(target=_ram_watchdog_loop, daemon=True)
    t.start()
    gpus = detect_gpus()
    if gpus:
        gpu_names = ", ".join(f"{g['name']} ({g['vram_gb']}GB)" for g in gpus)
        print(f"  [Watchdog] GPU: {gpu_names}")
    gpu_status, gpu_vram = check_ollama_gpu()
    if gpu_status == "gpu":
        print(f"  [Watchdog] Ollama: GPU mode ({gpu_vram}GB VRAM)")
    elif gpu_status == "cpu":
        print(f"  [Watchdog] WARNING: Ollama running on CPU — will be slow!")
        print(f"  [Watchdog] Fix: CUDA_VISIBLE_DEVICES=0 OLLAMA_GPU_LAYERS=999 ollama serve")
    print(f"  [Watchdog] Active — RAM>={RAM_PAUSE_PCT}% → pause | VRAM full → CPU fallback")


# ═══════════════════════════════════════════════════════════════
#  Imports from app.py
# ═══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from .app import (CONFIG, classify_sample, SampleClassificationAgent,
                      ContextRecallExtractor, _NOT_SPECIFIED_VALUES,
                      _find_geometadb, _find_data_dir, LLMCurator,
                      detect_gpus, check_ollama_gpu, _get_vram_usage,
                      prefetch_gse_context, init_memory_agent, init_gse_contexts,
                      _HAS_DETERMINISTIC)
except ImportError:
    from app import (CONFIG, classify_sample, SampleClassificationAgent,
                     ContextRecallExtractor, _NOT_SPECIFIED_VALUES,
                     _find_geometadb, _find_data_dir, LLMCurator,
                     detect_gpus, check_ollama_gpu, _get_vram_usage,
                     prefetch_gse_context, init_memory_agent, init_gse_contexts,
                     _HAS_DETERMINISTIC)

FIELDS  = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']
NS_CURATE_FIELDS = ['Condition', 'Tissue', 'Treatment']  # Phase 2 only curates these
NS      = _NOT_SPECIFIED_VALUES

HUMAN_N       = 200
JUDGE_N       = 1000
JUDGE_WORKERS = 8
GPL_PLATFORMS = ['GPL570', 'GPL6947', 'GPL96', 'GPL10558']


# ═══════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════
def setup():
    print("\n" + "=" * 62)
    print("  GeneVariate — Automatic Evaluation Pipeline")
    print("=" * 62)

    geo_path = CONFIG['paths']['geo_db']
    if not os.path.exists(geo_path):
        geo_path = _find_geometadb()
    if not os.path.exists(geo_path):
        print(f"[FATAL] GEOmetadb not found at {geo_path}")
        sys.exit(1)
    print(f"  GEOmetadb : {geo_path}")

    print("  Loading database ...")
    from genevariate.core.db_loader import open_geometadb
    conn = open_geometadb(geo_path)
    if conn is None:
        print("  ERROR: Could not open GEOmetadb")
        sys.exit(1)

    data_dir   = CONFIG['paths']['data']
    mem_path   = os.path.join(data_dir, "gse_cache", "gse_cache.json")
    output_dir = os.path.join(SCRIPT_DIR, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Data dir  : {data_dir}")
    print(f"  Memory    : {mem_path} {'(exists)' if os.path.exists(mem_path) else '(new)'}")
    print(f"  Output    : {output_dir}")

    model = CONFIG.get('ai', {}).get('model', 'gemma4:e2b')
    print(f"  LLM model : {model}")
    print(f"  Platforms : {', '.join(GPL_PLATFORMS)}")

    # GPU detection
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            print(f"  GPU       : {g['name']} ({g['vram_gb']}GB total, {g['free_vram_gb']}GB free)")
    else:
        print("  GPU       : None detected (will use CPU — slow)")
    gpu_status, gpu_vram = check_ollama_gpu()
    if gpu_status == "gpu":
        print(f"  Ollama    : GPU mode ({gpu_vram}GB VRAM)")
    elif gpu_status == "cpu":
        print(f"  Ollama    : CPU mode (WARNING: slow!)")
        print(f"              Fix: CUDA_VISIBLE_DEVICES=0 OLLAMA_GPU_LAYERS=999 ollama serve")
    else:
        print(f"  Ollama    : unknown (no model loaded yet)")

    print(f"\n  {'Platform':<12} {'Available GSMs':>16}")
    print(f"  {'─'*30}")
    for gpl in GPL_PLATFORMS:
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM gsm WHERE UPPER(gpl)=?",
                               [gpl]).fetchone()[0]
        except Exception:
            cnt = 0
        print(f"  {gpl:<12} {cnt:>16,}")
    print()

    return conn, mem_path, output_dir, model


# ═══════════════════════════════════════════════════════════════
#  Multi-platform proportional sampling
# ═══════════════════════════════════════════════════════════════
def sample_multi_gpl(conn, n):
    """
    Draw n samples proportionally across GPL_PLATFORMS, then shuffle.
    Returns a combined shuffled DataFrame.
    """
    cols = ("gsm, title, source_name_ch1, characteristics_ch1, "
            "description, extract_protocol_ch1, treatment_protocol_ch1, "
            "organism_ch1, series_id, gpl")

    available = {}
    for gpl in GPL_PLATFORMS:
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM gsm WHERE UPPER(gpl)=?",
                               [gpl]).fetchone()[0]
            available[gpl] = cnt
        except Exception:
            available[gpl] = 0

    total_avail = sum(available.values())
    if total_avail == 0:
        print("[FATAL] No GSMs found for any target platform.")
        sys.exit(1)

    # Proportional quotas
    quotas    = {}
    remainder = n
    gpls_desc = sorted(available, key=lambda g: available[g], reverse=True)
    for i, gpl in enumerate(gpls_desc):
        if i == len(gpls_desc) - 1:
            quotas[gpl] = max(remainder, 0)
        else:
            q = min(round(n * available[gpl] / total_avail), available[gpl], remainder)
            quotas[gpl] = max(q, 0)
            remainder   = max(remainder - q, 0)

    print(f"  Sampling {n} GSMs proportionally across platforms:")
    frames = []
    for gpl in GPL_PLATFORMS:
        q = quotas.get(gpl, 0)
        if q == 0:
            print(f"    {gpl}: 0  (skipped)")
            continue
        df = pd.read_sql_query(
            f"SELECT {cols} FROM gsm WHERE UPPER(gpl)=? ORDER BY RANDOM() LIMIT ?",
            conn, params=[gpl, q])
        print(f"    {gpl}: {len(df)} samples")
        frames.append(df)

    if not frames:
        print("[FATAL] No samples collected.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=random.randint(0, 99999)
                               ).reset_index(drop=True)
    print(f"  Total: {len(combined)} samples (shuffled across all platforms)\n")
    return combined


# ═══════════════════════════════════════════════════════════════
#  Shared progress-bar renderer (used by Phase 1, Phase 2, Judge)
# ═══════════════════════════════════════════════════════════════
def _render_pbar(done, total, t0, label="", width=44, extra=""):
    """
    Print a single updating progress line with:
      [████░░░░]  done/total  pct%  N.Ns/smp  ETA hh:mm:ss  elapsed hh:mm:ss  label  extra
    """
    pct    = done / total if total > 0 else 0
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)

    elapsed   = time.time() - t0
    sps       = done / elapsed if elapsed > 0 else 0          # samples/sec
    spt       = elapsed / done if done > 0 else 0             # sec/sample
    remaining = (total - done) / sps if sps > 0 else 0

    eta_str  = str(timedelta(seconds=int(remaining)))
    el_str   = str(timedelta(seconds=int(elapsed)))
    spt_str  = f"{spt:.1f}s/smp" if spt < 60 else f"{spt/60:.1f}min/smp"
    sps_str  = f"{sps:.2f}smp/s" if sps >= 0.01 else f"{sps*60:.2f}smp/min"

    sys.stdout.write(
        f"\r  [{bar}] {done}/{total} ({pct*100:.1f}%)"
        f"  {sps_str}  {spt_str}"
        f"  ETA {eta_str}  elapsed {el_str}"
        f"{'  ' + label if label else ''}"
        f"{'  ' + extra if extra else ''}   "
    )
    sys.stdout.flush()


def _print_phase_summary(label, t0, total):
    """Print a final summary line after a phase completes."""
    elapsed = time.time() - t0
    spt     = elapsed / total if total > 0 else 0
    spt_str = f"{spt:.1f}s/smp" if spt < 60 else f"{spt/60:.1f}min/smp"
    print(f"\n  ✓ {label} complete — {total} samples in "
          f"{timedelta(seconds=int(elapsed))}  ({spt_str})")


# ═══════════════════════════════════════════════════════════════
#  Phase 1 — LLM extraction only  (with live progress bar)
# ═══════════════════════════════════════════════════════════════
def run_phase1(samples):
    import re
    total = len(samples)
    t0    = time.time()

    # State shared between the log callback and the caller
    _state = {'done': 0, 'ok': 0, 'fail': 0, 'last_gsm': ''}

    # ── Progress-bar log interceptor ─────────────────────────────
    # SampleClassificationAgent emits messages like:
    #   "[Agent] 20/200 (0.2 smp/s, ETA 943s) ok=20 fail=0"
    #   "[Agent] OK Test: GSM123 -> Condition"
    # We parse N/M progress lines and ignore the rest (or print them once).
    _PROG_RE = re.compile(
        r'\[Agent\]\s+(\d+)/(\d+)\s+\(.*?\)\s+ok=(\d+)\s+fail=(\d+)', re.I)
    _PASS_RE = re.compile(r'\[Agent\]\s+OK\s+Test:\s+(\S+)', re.I)

    def _log(msg):
        m = _PROG_RE.search(msg)
        if m:
            done, ttl, ok, fail = int(m.group(1)), int(m.group(2)), \
                                   int(m.group(3)), int(m.group(4))
            _state['done'] = done
            _state['ok']   = ok
            _state['fail'] = fail
            _render_pbar(done, ttl, t0,
                         label="Phase 1",
                         extra=f"ok={ok} fail={fail}")
            return
        mp = _PASS_RE.search(msg)
        if mp:
            _state['last_gsm'] = mp.group(1)
            # Print once, then the progress bar will overwrite on next update
            sys.stdout.write(f"\n    [Agent] test sample OK ({mp.group(1)})\n")
            sys.stdout.flush()
            return
        # Pass-through for other messages (e.g. preflight, completion notice)
        sys.stdout.write(f"\n    {msg}\n")
        sys.stdout.flush()

    print(f"  Phase 1 — extracting {total} samples ...")
    agent = SampleClassificationAgent(
        tools_list=[classify_sample],
        gui_log_func=_log,
        max_workers=10)
    p1 = agent.process_samples(samples, fields=FIELDS)

    _print_phase_summary("Phase 1", t0, total)

    if p1 is None or p1.empty:
        return None

    gc = 'gsm' if 'gsm' in p1.columns else 'GSM'
    p1 = p1.rename(columns={gc: 'GSM'})
    p1['GSM'] = p1['GSM'].astype(str).str.strip().str.upper()

    for mc in ['title', 'source_name_ch1', 'characteristics_ch1', 'series_id', 'gpl']:
        if mc in samples.columns and mc not in p1.columns:
            mapping = samples.set_index(
                samples['gsm'].astype(str).str.strip().str.upper())[mc]
            p1[mc] = p1['GSM'].map(mapping)

    print(f"  Phase 1 NS breakdown:")
    for f in FIELDS:
        if f in p1.columns:
            ns_n = p1[f].astype(str).str.strip().isin(NS).sum()
            print(f"    {f:<20} {ns_n:>4}/{len(p1)}  NS ({ns_n/len(p1)*100:.0f}%)")
    return p1


# ═══════════════════════════════════════════════════════════════
#  Phase 1.5 — Per-GSE Label Collapsing (strict: exact + abbreviation only)
# ═══════════════════════════════════════════════════════════════
def run_phase15(p1):
    """
    Apply per-GSE strict label collapsing to Phase 1 results.
    Two rules only (NO fuzzy, NO substring, NO synonyms):
      1. Exact match after case/space/hyphen normalization
      2. Abbreviation: short uppercase label matches initials of longer label in same GSE
    Numeric guard: different numbers always block merge.
    Returns (p15_df, normalization_stats).
    """
    from collections import Counter
    import re as _re

    total = len(p1)
    t0 = time.time()
    print(f"  Phase 1.5 — per-GSE collapsing on {total} samples ...")

    p15 = p1.copy()
    _SKIP = {'Age', 'Treatment_Time', 'age', 'treatment_time'}

    if 'series_id' not in p15.columns:
        print("  [Phase 1.5] No series_id column — skipping.")
        return p15, {}

    label_cols = [c for c in p15.columns
                  if c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform')
                  and c not in _SKIP
                  and p15[c].dtype == 'object']

    stats = {
        'total_variants_merged': 0,
        'per_field': {},
        'per_gse': {},
        'merge_examples': [],
    }

    if not label_cols:
        print("  [Phase 1.5] No categorical label columns found.")
        return p15, stats

    def _get_initials(text):
        words = _re.split(r'[\s\-_/]+', text.strip())
        return ''.join(w[0].upper() for w in words if len(w) > 1 or w[0].isupper())

    gse_groups = p15.groupby('series_id')
    n_gse = len(gse_groups)
    n_normalized = 0

    for gse_id, group in gse_groups:
        if len(group) < 2:
            continue
        for col in label_cols:
            vals = group[col].fillna('Not Specified').astype(str).str.strip()
            real_vals = [v for v in vals if v.lower() not in
                         ('not specified', 'n/a', 'unknown', 'nan', '')]
            if len(real_vals) < 2:
                continue

            counter = Counter(real_vals)
            if len(counter) <= 1:
                continue

            canonical = counter.most_common(1)[0][0]
            canonical_norm = canonical.lower().replace(' ', '').replace('-', '').replace('_', '')

            merge_map = {}
            for val, cnt in counter.items():
                if val == canonical:
                    continue
                val_norm = val.lower().replace(' ', '').replace('-', '').replace('_', '')

                # GUARD: different numbers = different entities
                val_nums = _re.findall(r'\d+', val)
                can_nums = _re.findall(r'\d+', canonical)
                if val_nums != can_nums:
                    continue

                # Rule 1: Exact match after normalization
                if val_norm == canonical_norm:
                    merge_map[val] = canonical
                    continue

                # Rule 2: Abbreviation initials (strict)
                shorter, longer = (val, canonical) if len(val) <= len(canonical) else (canonical, val)
                if (len(shorter) <= 5 and shorter.replace('-','').replace(' ','').isupper()
                        and len(longer) > len(shorter) * 2):
                    initials = _get_initials(longer)
                    short_clean = shorter.upper().replace('-','').replace(' ','').replace('(','').replace(')','')
                    if short_clean == initials:
                        target = longer if len(canonical) >= len(val) else canonical
                        merge_map[val] = target

                # NO OTHER MATCHING — no substring, no fuzzy, no overlap

            if merge_map:
                idx = group.index
                for old_val, new_val in merge_map.items():
                    mask = p15.loc[idx, col] == old_val
                    n_merged = mask.sum()
                    if n_merged > 0:
                        p15.loc[idx[mask], col] = new_val
                        n_normalized += n_merged

                        if col not in stats['per_field']:
                            stats['per_field'][col] = 0
                        stats['per_field'][col] += n_merged

                        if gse_id not in stats['per_gse']:
                            stats['per_gse'][gse_id] = 0
                        stats['per_gse'][gse_id] += n_merged

                        if len(stats['merge_examples']) < 20:
                            stats['merge_examples'].append({
                                'gse': gse_id, 'field': col,
                                'from': old_val, 'to': new_val,
                                'count': int(n_merged)
                            })

    stats['total_variants_merged'] = n_normalized
    elapsed = time.time() - t0

    print(f"  ✓ Phase 1.5 complete in {timedelta(seconds=int(elapsed))}")
    print(f"    Collapsed {n_normalized} label variants across {n_gse} experiments")
    print(f"    Rules: exact match + abbreviation initials only (no fuzzy)")
    if stats['per_field']:
        print(f"    Per-field merges:")
        for col, cnt in stats['per_field'].items():
            print(f"      {col:<20} {cnt:>4} variants merged")
    if stats['merge_examples']:
        print(f"    Example merges (first {min(5, len(stats['merge_examples']))}):")
        for ex in stats['merge_examples'][:5]:
            print(f"      [{ex['gse']}] {ex['field']}: \"{ex['from']}\" → \"{ex['to']}\" ({ex['count']}x)")

    # Show label diversity before/after
    print(f"\n    Label diversity (unique values):")
    print(f"    {'Field':<20} {'Before':>8} {'After':>8} {'Reduction':>10}")
    print(f"    {'─'*50}")
    for col in label_cols:
        if col in p1.columns and col in p15.columns:
            before = p1[col].nunique()
            after = p15[col].nunique()
            red = before - after
            print(f"    {col:<20} {before:>8} {after:>8} {red:>8} ({red*100/max(1,before):.0f}%)")

    return p15, stats


# ═══════════════════════════════════════════════════════════════
#  Phase 2 — NS handling  (background thread + live ticker)
# ═══════════════════════════════════════════════════════════════
def run_phase2(p1, mem_path):
    """
    Run ContextRecallExtractor.  Because run_recall_pass is sequential with no
    per-item callback, we run it in a background thread and show a live
    elapsed-time / estimated-time ticker in the foreground, calculated from
    Phase 1's observed per-sample speed as a prior.
    """
    import re

    # Count how many NS rows Phase 2 will need to process
    # Phase 2 ONLY curates Condition, Tissue, Treatment (not Age, Treatment_Time)
    ns_count = 0
    for f in NS_CURATE_FIELDS:
        if f in p1.columns:
            ns_count += int(p1[f].astype(str).str.strip().isin(NS).sum())
    total_ns = ns_count   # one LLM call per (GSM × field) that was NS

    saved = None
    if os.path.exists(mem_path):
        try:
            with open(mem_path) as f:
                saved = json.load(f)
        except Exception:
            pass

    # State bucket updated by the log callback
    _state = {
        'done':      0,       # incremented by parsing log messages
        'corrected': 0,
        'confirmed': 0,
        'phase':     'build', # 'build' | 'fetch' | 'recall'
        'fetch_done': 0,
        'fetch_total': 0,
        'finished':  False,
        'result':    None,
        'error':     None,
    }

    # Log patterns emitted by ContextRecallExtractor
    _FETCH_RE   = re.compile(r'Fetched\s+(\d+)/(\d+)', re.I)
    _PHASE2_RE  = re.compile(r'Phase 2:\s+Processing\s+(\d+)', re.I)
    _DONE_RE    = re.compile(
        r'Phase 2 complete:\s+(\d+)\s+corrected,\s+(\d+)\s+confirmed', re.I)
    _SAMPLE_RE  = re.compile(r'\[Phase2\].*?(\d+)/(\d+)', re.I)

    def _log(msg):
        mf = _FETCH_RE.search(msg)
        if mf:
            _state['fetch_done']  = int(mf.group(1))
            _state['fetch_total'] = int(mf.group(2))
            _state['phase']       = 'fetch'
            return
        mp2 = _PHASE2_RE.search(msg)
        if mp2:
            _state['phase'] = 'recall'
            return
        md = _DONE_RE.search(msg)
        if md:
            _state['corrected'] = int(md.group(1))
            _state['confirmed'] = int(md.group(2))
            return
        ms = _SAMPLE_RE.search(msg)
        if ms:
            _state['done'] = int(ms.group(1))
            return
        # Verbose pass-through for other messages
        sys.stdout.write(f"\n    {msg}\n")
        sys.stdout.flush()

    # ── Run ContextRecallExtractor in a background thread ─────────────
    def _worker():
        try:
            recall = ContextRecallExtractor(log_func=_log, saved_cache=saved)
            if recall.build_memory(p1):
                _state['result'] = recall.run_recall_pass(p1)
                # Save memory
                try:
                    os.makedirs(os.path.dirname(mem_path), exist_ok=True)
                    with open(mem_path, 'w') as fh:
                        json.dump({
                            "_info": {"created": datetime.now().isoformat()},
                            "gse_descriptions": recall.gse_descriptions,
                            "gse_consensus": {
                                g: {c: dict(v) for c, v in cols.items()}
                                for g, cols in recall.gse_consensus.items()},
                        }, fh, indent=2)
                except Exception:
                    pass
            else:
                _state['result'] = p1.copy()
        except Exception as e:
            _state['error'] = e
        finally:
            _state['finished'] = True

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    t.start()

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spin_i  = 0

    print(f"  Phase 2 — NS handling for ~{total_ns} (GSM × field) slots ...")

    # ── Live progress ticker (main thread) ───────────────────────
    while not _state['finished']:
        elapsed = time.time() - t0
        el_str  = str(timedelta(seconds=int(elapsed)))
        phase   = _state['phase']
        spin    = SPINNER[spin_i % len(SPINNER)]
        spin_i += 1

        if phase == 'fetch':
            fd, ft = _state['fetch_done'], _state['fetch_total']
            if ft > 0:
                pct    = fd / ft
                filled = int(44 * pct)
                bar    = "█" * filled + "░" * (44 - filled)
                sps    = fd / elapsed if elapsed > 0 else 0
                rem    = (ft - fd) / sps if sps > 0 else 0
                sys.stdout.write(
                    f"\r  {spin} [{bar}] {fd}/{ft} GSE fetched ({pct*100:.0f}%)"
                    f"  {sps:.1f}GSE/s  ETA {timedelta(seconds=int(rem))}"
                    f"  elapsed {el_str}   "
                )
            else:
                sys.stdout.write(
                    f"\r  {spin} Fetching GSE descriptions ...  elapsed {el_str}   "
                )

        elif phase == 'recall':
            done = _state['done']
            corr = _state['corrected']
            if total_ns > 0 and done > 0:
                sps  = done / elapsed if elapsed > 0 else 0
                spt  = elapsed / done
                rem  = (total_ns - done) / sps if sps > 0 else 0
                pct  = done / total_ns
                filled = int(44 * pct)
                bar    = "█" * filled + "░" * (44 - filled)
                spt_s  = f"{spt:.1f}s/smp" if spt < 60 else f"{spt/60:.1f}min/smp"
                sys.stdout.write(
                    f"\r  {spin} [{bar}] {done}/{total_ns} ({pct*100:.0f}%)"
                    f"  {sps:.2f}smp/s  {spt_s}"
                    f"  ETA {timedelta(seconds=int(rem))}"
                    f"  elapsed {el_str}"
                    f"  corrected={corr}   "
                )
            else:
                sys.stdout.write(
                    f"\r  {spin} Phase 2 recall running ...  elapsed {el_str}   "
                )

        else:   # build phase
            sys.stdout.write(
                f"\r  {spin} Building memory index ...  elapsed {el_str}   "
            )

        sys.stdout.flush()
        time.sleep(0.3)

    t.join()
    print()   # newline after ticker

    if _state['error']:
        print(f"  [Phase 2] ERROR: {_state['error']}")
        return p1.copy()

    elapsed = time.time() - t0
    corr    = _state['corrected']
    conf    = _state['confirmed']
    print(f"  ✓ Phase 2 complete in {timedelta(seconds=int(elapsed))}"
          f"  — corrected={corr}  confirmed_NS={conf}")

    p2 = _state['result'] if _state['result'] is not None else p1.copy()

    for mc in ['title', 'source_name_ch1', 'characteristics_ch1', 'series_id', 'gpl']:
        if mc in p1.columns and mc not in p2.columns:
            p2[mc] = p1[mc]
    return p2


# ═══════════════════════════════════════════════════════════════
#  Build evaluation item DataFrames
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
#  Phase 3 — LLM Curator (cross-experiment label harmonization via LLM)
# ═══════════════════════════════════════════════════════════════
def run_phase3_curator(p2):
    """Run LLM Curator: scan labels, find candidates, ask LLM, propose merges.
    Returns (p3_df, curator_stats) where curator_stats contains proposals and metrics.
    """
    total = len(p2)
    t0 = time.time()
    print(f"  Phase 3 — LLM Curator on {total} samples ...")

    # Track before stats
    label_cols = [c for c in p2.columns
                  if c in FIELDS and p2[c].dtype == 'object']
    before_stats = {}
    for c in label_cols:
        before_stats[c] = p2[c].nunique()

    # Run LLM Curator scan
    curator = LLMCurator(log_func=lambda m: sys.stdout.write(f"\n    {m}\n"))

    def _progress(done, total_c, msg):
        _render_pbar(done, total_c, t0, label="Curator", extra=msg[:40])

    proposals = curator.scan_and_propose(p2, progress_fn=_progress)

    # Apply proposals (for evaluation — in production user reviews first)
    p3 = LLMCurator.apply_merges(p2.copy(), proposals,
                                  log_func=lambda m: sys.stdout.write(f"\n    {m}\n"))

    # Stats
    curator_stats = {
        'before': before_stats,
        'after': {},
        'merged': {},
        'proposals': proposals,
        'total_proposals': sum(len(v) for v in proposals.values()),
        'total_samples_changed': 0,
    }
    for c in label_cols:
        if c in p3.columns:
            after_n = p3[c].nunique()
            curator_stats['after'][c] = after_n
            curator_stats['merged'][c] = before_stats.get(c, 0) - after_n

    # Count actual samples changed
    for c in label_cols:
        if c in p2.columns and c in p3.columns:
            curator_stats['total_samples_changed'] += (p2[c] != p3[c]).sum()

    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 3 (LLM Curator) complete in {timedelta(seconds=int(elapsed))}")
    print(f"    Proposals: {curator_stats['total_proposals']} merges proposed")
    print(f"    Samples changed: {curator_stats['total_samples_changed']}")
    print(f"    Label diversity changes:")
    print(f"    {'Field':<20} {'Before':>8} {'After':>8} {'Merged':>8}")
    print(f"    {'─'*48}")
    for c in label_cols:
        b = before_stats.get(c, 0)
        a = curator_stats['after'].get(c, 0)
        m = curator_stats['merged'].get(c, 0)
        if m > 0:
            print(f"    {c:<20} {b:>8} {a:>8} {m:>8} ({m*100/max(1,b):.0f}%)")

    return p3, curator_stats


def _p1_items(p1):
    """Non-NS Phase 1 predictions → human/judge evaluates correctness."""
    rows = []
    for _, r in p1.iterrows():
        gsm  = str(r.get('GSM', '')).strip().upper()
        meta = {
            'GSM':             gsm,
            'series_id':       str(r.get('series_id', '')),
            'title':           str(r.get('title', '')),
            'source_name':     str(r.get('source_name_ch1', '')),
            'characteristics': str(r.get('characteristics_ch1', '')),
            'gpl':             str(r.get('gpl', '')),
        }
        for f in FIELDS:
            if f not in p1.columns:
                continue
            v = str(r.get(f, '')).strip()
            if v not in NS:
                rows.append({**meta, 'field': f,
                             'phase1_value': v, 'phase2_value': v,
                             'is_recall': False})
    return pd.DataFrame(rows)


def _ns_correction_items(p1, p2):
    """Rows that were NS in Phase 1 but got a value in Phase 2.
    Only checks NS_CURATE_FIELDS (Condition, Tissue, Treatment)."""
    rows  = []
    p2_lkp = p2.set_index(p2['GSM'].astype(str).str.strip().str.upper())
    for _, r1 in p1.iterrows():
        gsm = str(r1.get('GSM', '')).strip().upper()
        if gsm not in p2_lkp.index:
            continue
        r2 = p2_lkp.loc[gsm]
        if isinstance(r2, pd.DataFrame):
            r2 = r2.iloc[0]
        meta = {
            'GSM':             gsm,
            'series_id':       str(r1.get('series_id', '')),
            'title':           str(r1.get('title', '')),
            'source_name':     str(r1.get('source_name_ch1', '')),
            'characteristics': str(r1.get('characteristics_ch1', '')),
            'gpl':             str(r1.get('gpl', '')),
        }
        for f in NS_CURATE_FIELDS:
            if f not in p1.columns:
                continue
            v1 = str(r1.get(f, '')).strip()
            v2 = str(r2.get(f, '')).strip()
            if v1 in NS and v2 not in NS:
                rows.append({**meta, 'field': f,
                             'phase1_value': v1, 'phase2_value': v2,
                             'is_recall': True})
    return pd.DataFrame(rows)


def _phase15_items(p1_raw, p15):
    """Rows where Phase 1.5 changed a label (normalization).
    These are variant-unification changes, not NS corrections."""
    rows = []
    for i in range(len(p1_raw)):
        r1 = p1_raw.iloc[i]
        r15 = p15.iloc[i]
        gsm = str(r1.get('GSM', '')).strip().upper()
        meta = {
            'GSM':             gsm,
            'series_id':       str(r1.get('series_id', '')),
            'title':           str(r1.get('title', '')),
            'source_name':     str(r1.get('source_name_ch1', '')),
            'characteristics': str(r1.get('characteristics_ch1', '')),
            'gpl':             str(r1.get('gpl', '')),
        }
        for f in FIELDS:
            if f in ('Age', 'Treatment_Time'):
                continue
            if f not in p1_raw.columns:
                continue
            v1 = str(r1.get(f, '')).strip()
            v15 = str(r15.get(f, '')).strip()
            if v1 != v15 and v1 not in NS:
                rows.append({**meta, 'field': f,
                             'phase1_raw': v1,
                             'phase15_normalized': v15,
                             'is_normalization': True})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  Human Evaluation GUI
# ═══════════════════════════════════════════════════════════════
def human_gui(items_df, conn, round_title="Human Evaluation"):
    import tkinter as tk
    from tkinter import ttk, messagebox

    if items_df is None or items_df.empty:
        print(f"  [{round_title}] No items — skipping.")
        return pd.DataFrame()

    items   = list(items_df.iterrows())
    total   = len(items)
    results = []
    idx     = [0]

    root = tk.Tk()
    root.title(f"GeneVariate — {round_title}  ({total} items)")
    root.geometry("1280x840")
    try:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"1280x840+{(sw-1280)//2}+{(sh-840)//2}")
    except Exception:
        pass

    # Top bar
    bar = tk.Frame(root, bg='#1A237E', pady=9)
    bar.pack(fill=tk.X)
    tk.Label(bar, text=round_title, font=('Segoe UI', 13, 'bold'),
             bg='#1A237E', fg='white').pack(side=tk.LEFT, padx=15)
    prog_lbl = tk.Label(bar, text="", font=('Segoe UI', 12, 'bold'),
                        bg='#1A237E', fg='#FFD54F')
    prog_lbl.pack(side=tk.LEFT, padx=10)
    meta_lbl = tk.Label(bar, text="", font=('Segoe UI', 10),
                        bg='#1A237E', fg='#80CBC4')
    meta_lbl.pack(side=tk.RIGHT, padx=15)

    # Progress strip
    pbar_frame  = tk.Frame(root, bg='#E8EAF6', pady=3)
    pbar_frame.pack(fill=tk.X)
    pbar_canvas = tk.Canvas(pbar_frame, height=8, bg='#E8EAF6', highlightthickness=0)
    pbar_canvas.pack(fill=tk.X)
    pbar_fill   = pbar_canvas.create_rectangle(0, 0, 0, 8, fill='#3F51B5', width=0)

    def _update_pbar(i):
        pbar_canvas.update_idletasks()
        w = pbar_canvas.winfo_width()
        pbar_canvas.coords(pbar_fill, 0, 0, int(w * i / total), 8)

    pw = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    pw.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    lf  = ttk.LabelFrame(pw, text="Sample & Experiment", padding=5)
    pw.add(lf, weight=3)
    txt = tk.Text(lf, wrap=tk.WORD, font=('Consolas', 10))
    sb  = ttk.Scrollbar(lf, orient='vertical', command=txt.yview)
    txt.config(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt.pack(fill=tk.BOTH, expand=True)
    txt.tag_configure('h1',  font=('Consolas', 12, 'bold'), foreground='#1A237E')
    txt.tag_configure('h2',  font=('Consolas', 11, 'bold'), foreground='#2E7D32')
    txt.tag_configure('gpl', font=('Consolas', 10),          foreground='#6A1B9A')
    txt.tag_configure('val', font=('Consolas', 13, 'bold'),  foreground='#E65100',
                      background='#FFF3E0')
    txt.tag_configure('old', font=('Consolas', 10),          foreground='#9E9E9E')

    rf   = ttk.LabelFrame(pw, text="Verdict", padding=12)
    pw.add(rf, weight=1)
    flbl = tk.Label(rf, text="", font=('Segoe UI', 16, 'bold'), fg='#1A237E')
    flbl.pack(pady=(5, 3))
    plbl = tk.Label(rf, text="", font=('Segoe UI', 14, 'bold'), fg='#E65100',
                    wraplength=330, bg='#FFF3E0', padx=10, pady=8, relief=tk.GROOVE)
    plbl.pack(fill=tk.X, pady=(0, 12))

    vvar = tk.StringVar()
    btns = {}
    for label, v, c in [("1  TP — Correct extraction",     "TP", "#2E7D32"),
                          ("2  FP — Wrong / hallucinated",  "FP", "#C62828"),
                          ("3  FN — Should have a value",   "FN", "#E65100"),
                          ("4  TN — Correctly Not Spec.",   "TN", "#1565C0")]:
        b = tk.Button(rf, text=label, width=32, font=('Segoe UI', 11, 'bold'),
                      pady=9, cursor='hand2', bg='#ECEFF1', fg='#333', anchor='w',
                      command=lambda x=v: _pick(x))
        b.pack(fill=tk.X, pady=3)
        btns[v] = (b, c)

    ttk.Label(rf, text="Note (optional):").pack(anchor=tk.W, pady=(10, 2))
    note = ttk.Entry(rf, width=36)
    note.pack(fill=tk.X)

    nav = ttk.Frame(rf)
    nav.pack(fill=tk.X, pady=15)
    tk.Button(nav, text="  Submit & Next  ", bg="#43A047", fg="white",
              font=('Segoe UI', 11, 'bold'), padx=15, pady=5,
              command=lambda: _submit(), cursor='hand2').pack(side=tk.LEFT, padx=3)
    tk.Button(nav, text="Skip",   font=('Segoe UI', 10), padx=8,
              command=lambda: _skip()).pack(side=tk.LEFT, padx=3)
    tk.Button(nav, text="Finish", font=('Segoe UI', 10), padx=8,
              command=root.destroy).pack(side=tk.RIGHT, padx=3)
    ttk.Label(rf, text="Keys: 1=TP  2=FP  3=FN  4=TN  Enter=Submit",
              font=('Segoe UI', 8, 'italic'), foreground='#888').pack(pady=(8, 0))

    def _pick(v):
        vvar.set(v)
        for k, (b, c) in btns.items():
            b.config(bg=c if k == v else '#ECEFF1',
                     fg='white' if k == v else '#333',
                     relief=tk.SUNKEN if k == v else tk.RAISED)

    def _submit():
        v = vvar.get()
        if not v:
            messagebox.showwarning("Select", "Pick TP / FP / FN / TN.", parent=root)
            return
        _, row = items[idx[0]]
        results.append({**row.to_dict(), 'verdict': v, 'notes': note.get().strip()})
        idx[0] += 1
        if idx[0] >= total:
            messagebox.showinfo("Done", f"Evaluated {len(results)} items.", parent=root)
            root.destroy()
            return
        _show(idx[0])

    def _skip():
        idx[0] += 1
        if idx[0] >= total:
            root.destroy()
            return
        _show(idx[0])

    def _show(i):
        _, row    = items[i]
        is_recall = bool(row.get('is_recall', False))
        pred      = str(row.get('phase2_value', row.get('phase1_value', '')))

        prog_lbl.config(text=f"{i+1} / {total}  ({(i+1)/total*100:.0f}%)")
        phase_tag = "NS CORRECTION (Round 2)" if is_recall else "PHASE 1 (Round 1)"
        meta_lbl.config(text=f"{phase_tag}  |  {row.get('gpl','')}  |  {row['GSM']}")
        flbl.config(text=row['field'].replace('_', ' '))
        plbl.config(text=pred)
        vvar.set("")
        note.delete(0, tk.END)
        for k, (b, c) in btns.items():
            b.config(bg='#ECEFF1', fg='#333', relief=tk.RAISED)
        _update_pbar(i + 1)

        txt.config(state=tk.NORMAL)
        txt.delete('1.0', tk.END)
        txt.insert(tk.END, f"SAMPLE : {row['GSM']}\n", 'h1')
        txt.insert(tk.END, f"Platform: {row.get('gpl','?')}\n", 'gpl')
        txt.insert(tk.END, f"Title   : {row.get('title','?')}\n")
        txt.insert(tk.END, f"Source  : {row.get('source_name','?')}\n")
        txt.insert(tk.END, f"Chars   : {row.get('characteristics','?')}\n\n")
        txt.insert(tk.END, f"FIELD:  {row['field']}\n", 'h2')
        txt.insert(tk.END, "  Predicted value:  ", 'h2')
        txt.insert(tk.END, f"{pred}\n\n", 'val')
        if is_recall:
            txt.insert(tk.END,
                       f"  Phase 1 result  : {row.get('phase1_value','')}  (was NS)\n"
                       f"  Phase 2 corrected to the value shown above\n\n", 'old')

        gse = str(row.get('series_id', '')).strip()
        if gse and gse != 'nan':
            try:
                gr = pd.read_sql_query(
                    "SELECT title, summary, overall_design FROM gse WHERE gse=?",
                    conn, params=[gse])
                if not gr.empty:
                    g = gr.iloc[0]
                    txt.insert(tk.END, f"EXPERIMENT : {gse}\n", 'h1')
                    txt.insert(tk.END, f"Title: {g.get('title','?')}\n\n")
                    s = str(g.get('summary',''))[:800]
                    if s and s != 'None':
                        txt.insert(tk.END, f"Summary:\n{s}\n\n")
                    d = str(g.get('overall_design',''))[:500]
                    if d and d != 'None':
                        txt.insert(tk.END, f"Design:\n{d}\n")
            except Exception:
                pass
        txt.config(state=tk.DISABLED)

    root.bind('1', lambda e: _pick("TP"))
    root.bind('2', lambda e: _pick("FP"))
    root.bind('3', lambda e: _pick("FN"))
    root.bind('4', lambda e: _pick("TN"))
    root.bind('<Return>', lambda e: _submit())
    _show(0)
    root.mainloop()
    return pd.DataFrame(results)


def _save(df, path, label):
    if df is None or df.empty:
        print(f"  [{label}] Nothing to save.")
        return
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {label} → {path}  ({len(df)} rows)")


# ═══════════════════════════════════════════════════════════════
#  Parallel LLM Judge
# ═══════════════════════════════════════════════════════════════
def _judge_one(args):
    i, row, conn, model = args

    while not _ram_pause_event.is_set():   # RAM watchdog block
        time.sleep(1)

    is_recall = bool(row.get('is_recall', False))
    pred      = str(row.get('phase2_value', row.get('phase1_value', ''))).strip()
    gse_ctx   = ""
    gse       = str(row.get('series_id', '')).strip()
    if gse and gse != 'nan':
        try:
            gr = pd.read_sql_query("SELECT title, summary FROM gse WHERE gse=?",
                                   conn, params=[gse])
            if not gr.empty:
                gse_ctx = (f"Experiment {gse}: {gr.iloc[0].get('title','')}\n"
                           f"Summary: {str(gr.iloc[0].get('summary',''))[:400]}")
        except Exception:
            pass

    prompt = f"""Evaluate this biomedical metadata extraction.

SAMPLE: {row['GSM']}  |  Platform: {row.get('gpl','?')}
Title: {row.get('title','?')}
Source: {row.get('source_name','?')}
Characteristics: {row.get('characteristics','?')}
{gse_ctx}

Field: {row['field']}
Extracted value: "{pred}"
{"(recovered from Not Specified by Phase 2 re-extraction)" if is_recall else ""}

EVALUATION RULES:
- TP = correct and specific extraction
- FP = wrong, hallucinated, or TOO GENERIC (e.g., just "Cancer" instead of "Breast Cancer",
  just "Leukemia" instead of "Acute Myeloid Leukemia" — these count as FP)
- FN = missed — should have extracted a value but got "Not Specified"
- TN = correctly "Not Specified" (info genuinely absent from metadata)
- For Tissue: "Cell Line: X" or "Cell Type: X" ONLY when explicitly named in metadata.
  If no cell line/type mentioned, tissue should be inferred from disease (e.g., "Breast Cancer" → "Breast").
  "Cell Line: Not Specified" or "Cell Type: Not Specified" counts as FP.
- For Condition: SPECIFIC disease names are required (not generic categories)

Reply with ONLY one token: TP, FP, FN, or TN"""

    try:
        # Use ollama.chat() — matching old working code
        try:
            from .app import _ollama_post
        except ImportError:
            from app import _ollama_post
        raw = _ollama_post(prompt, model=model, timeout=60)
        if raw:
            raw_upper = raw.strip().upper()
            verdict = next((v for v in ['TP', 'TN', 'FP', 'FN'] if v in raw_upper), 'TP')
        else:
            verdict = 'ERROR'
    except ImportError:
        # Fallback: use ollama library directly
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            raw = response.get('message', {}).get('content', '').strip().upper()
            verdict = next((v for v in ['TP', 'TN', 'FP', 'FN'] if v in raw), 'TP')
        except Exception:
            verdict = 'ERROR'
    except Exception:
        verdict = 'ERROR'

    return i, verdict


def _pbar(done, total, t0, tp=0, fp=0):
    ram_s  = f"RAM:{psutil.virtual_memory().percent:.0f}%" if _PSUTIL else ""
    vram_u, vram_t, vram_p = _get_vram_usage()
    vram_s = f"VRAM:{vram_p:.0f}%" if vram_t > 0 else ""
    paused = "PAUSED" if not _ram_pause_event.is_set() else ""
    _render_pbar(done, total, t0,
                 label="Judge",
                 extra=f"TP:{tp} FP:{fp}  {ram_s}  {vram_s}  {paused}")


def judge(items_df, conn, model):
    if items_df is None or items_df.empty:
        return pd.DataFrame()

    total = len(items_df)
    print(f"  [Judge] {total} items · {JUDGE_WORKERS} parallel workers · {model}")
    _start_ram_watchdog()

    rows_list = list(items_df.iterrows())
    tasks     = [(i, row, conn, model) for i, (_, row) in enumerate(rows_list)]
    verdicts  = ['ERROR'] * total
    done = tp_cnt = fp_cnt = 0
    t0   = time.time()

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        fmap = {ex.submit(_judge_one, t): t[0] for t in tasks}
        for future in as_completed(fmap):
            i, verdict = future.result()
            verdicts[i] = verdict
            done += 1
            if verdict == 'TP': tp_cnt += 1
            elif verdict == 'FP': fp_cnt += 1
            _pbar(done, total, t0, tp=tp_cnt, fp=fp_cnt)

    print()
    result            = items_df.copy().reset_index(drop=True)
    result['verdict'] = verdicts
    vc = result['verdict'].value_counts()
    print(f"  [Judge] Done — TP:{vc.get('TP',0)}  FP:{vc.get('FP',0)}  "
          f"FN:{vc.get('FN',0)}  TN:{vc.get('TN',0)}  ERR:{vc.get('ERROR',0)}")
    return result


# ═══════════════════════════════════════════════════════════════
#  Metrics + Report + Plots
# ═══════════════════════════════════════════════════════════════
def calc(df, label=""):
    if df is None or df.empty:
        return dict(label=label, TP=0, FP=0, FN=0, TN=0,
                    prec=0, rec=0, f1=0, acc=0, n=0)
    v  = df['verdict'].value_counts()
    tp, fp, fn, tn = v.get('TP',0), v.get('FP',0), v.get('FN',0), v.get('TN',0)
    t  = tp + fp + fn + tn
    p  = tp / (tp + fp) if tp + fp else 0
    r  = tp / (tp + fn) if tp + fn else 0
    f  = 2 * p * r / (p + r) if p + r else 0
    return dict(label=label, TP=tp, FP=fp, FN=fn, TN=tn,
                prec=p, rec=r, f1=f, acc=(tp+tn)/t if t else 0, n=t)


def _metric_block(L, title, p1_df, rc_df, norm_stats=None, norm_items=None, harm_stats=None):
    L.append(f"\n{'─'*70}\n  {title}\n{'─'*70}")
    for subtitle, df in [("Phase 1 Predictions", p1_df),
                         ("NS Recall Corrections (Condition/Tissue/Treatment only)", rc_df)]:
        if df is None or df.empty:
            continue
        m = calc(df)
        L.append(f"\n  {subtitle} (n={m['n']}):")
        L.append(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
        L.append(f"    Precision={m['prec']:.3f}  Recall={m['rec']:.3f}  "
                 f"F1={m['f1']:.3f}  Accuracy={m['acc']:.3f}")
        L.append(f"\n  {'Field':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
                 f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
        L.append(f"  {'─'*62}")
        for f in FIELDS:
            fd = df[df['field'] == f]
            if fd.empty: continue
            fm = calc(fd)
            L.append(f"  {f:<18} {fm['TP']:>4} {fm['FP']:>4} {fm['FN']:>4} "
                     f"{fm['TN']:>4} {fm['prec']:>6.3f} {fm['rec']:>6.3f} "
                     f"{fm['f1']:>6.3f}")
        # Per-platform
        if 'gpl' in df.columns:
            L.append(f"\n  Per-platform:")
            L.append(f"  {'Platform':<12} {'n':>5} {'TP':>4} {'FP':>4} "
                     f"{'FN':>4} {'TN':>4} {'Acc':>7}")
            L.append(f"  {'─'*50}")
            for gpl in sorted(df['gpl'].dropna().unique()):
                gd = df[df['gpl'] == gpl]
                gm = calc(gd)
                L.append(f"  {gpl:<12} {gm['n']:>5} {gm['TP']:>4} {gm['FP']:>4} "
                         f"{gm['FN']:>4} {gm['TN']:>4} {gm['acc']:>7.3f}")

    # Phase 1.5 normalization metrics
    if norm_stats and norm_stats.get('total_variants_merged', 0) > 0:
        L.append(f"\n  Phase 1.5 — Per-GSE Label Normalization:")
        L.append(f"    Total label variants unified: {norm_stats['total_variants_merged']}")
        if norm_stats.get('per_field'):
            L.append(f"    Per-field merges:")
            for col, cnt in norm_stats['per_field'].items():
                L.append(f"      {col:<20} {cnt:>4} variants")
        if norm_stats.get('per_gse'):
            n_gse = len(norm_stats['per_gse'])
            L.append(f"    Experiments affected: {n_gse}")
        if norm_stats.get('merge_examples'):
            L.append(f"    Example merges:")
            for ex in norm_stats['merge_examples'][:10]:
                L.append(f"      [{ex['gse']}] {ex['field']}: "
                         f"\"{ex['from']}\" → \"{ex['to']}\" ({ex['count']}x)")

    if norm_items is not None and not norm_items.empty:
        L.append(f"\n    Normalization changes to evaluate: {len(norm_items)}")
        for f in norm_items['field'].unique():
            n_f = len(norm_items[norm_items['field'] == f])
            L.append(f"      {f:<20} {n_f:>4} labels normalized")

    # Phase 3 — LLM Curator metrics
    if harm_stats and (harm_stats.get('merged') or harm_stats.get('before')):
        L.append(f"\n  Phase 3 — LLM Curator (Cross-Experiment):")
        L.append(f"    {'Field':<20} {'Before':>8} {'After':>8} {'Merged':>8}")
        L.append(f"    {'─'*48}")
        for col in harm_stats.get('before', {}):
            b = harm_stats['before'].get(col, 0)
            a = harm_stats.get('after', {}).get(col, 0)
            m = harm_stats.get('merged', {}).get(col, 0)
            L.append(f"    {col:<20} {b:>8} {a:>8} {m:>8}")
        total_merged = sum(harm_stats.get('merged', {}).values())
        n_proposals = harm_stats.get('total_proposals', 0)
        if total_merged > 0 or n_proposals > 0:
            L.append(f"    Proposals: {n_proposals} | Labels changed: {total_merged}")


def make_report(hp1, hrc, jp1, jrc, out,
                h_norm_stats=None, h_norm_items=None,
                j_norm_stats=None, j_norm_items=None,
                h_harm_stats=None, j_harm_stats=None):
    os.makedirs(out, exist_ok=True)
    L = ["=" * 70,
         "  GeneVariate — Evaluation Report",
         f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
         f"  Platforms : {', '.join(GPL_PLATFORMS)}",
         f"  Phase 2 NS curation: {', '.join(NS_CURATE_FIELDS)} only",
         f"  Phase 3: LLM Curator (cross-experiment label harmonization via gemma4:e2b)",
         "=" * 70]
    _metric_block(L, "HUMAN EVALUATION", hp1, hrc,
                  norm_stats=h_norm_stats, norm_items=h_norm_items,
                  harm_stats=h_harm_stats)
    _metric_block(L, "LLM JUDGE", jp1, jrc,
                  norm_stats=j_norm_stats, norm_items=j_norm_items,
                  harm_stats=j_harm_stats)

    # Agreement
    valid = lambda d: d is not None and not d.empty
    if valid(hp1) and valid(jp1):
        L.append(f"\n{'─'*70}\n  HUMAN vs JUDGE AGREEMENT\n{'─'*70}")
        ha = pd.concat([x for x in [hp1, hrc] if valid(x)])
        ja = pd.concat([x for x in [jp1, jrc] if valid(x)])
        mg = ha.merge(ja, on=['GSM', 'field'], suffixes=('_h', '_j'))
        if not mg.empty:
            ag = (mg['verdict_h'] == mg['verdict_j']).sum()
            L.append(f"  Overlap: {len(mg)}  |  Agreement: {ag}/{len(mg)} "
                     f"({ag/len(mg)*100:.1f}%)")

    L.append("\n" + "=" * 70)
    txt = "\n".join(L)
    rpt = os.path.join(out, 'evaluation_report.txt')
    with open(rpt, 'w') as f:
        f.write(txt)
    print(f"\n{txt}\n  Report → {rpt}")

    for name, df in [('human_phase1', hp1), ('human_recall', hrc),
                     ('judge_phase1', jp1), ('judge_recall',  jrc)]:
        if valid(df):
            df.to_csv(os.path.join(out, f'{name}.csv'), index=False)

    _plots(hp1, hrc, jp1, jrc, out)


def _plots(hp1, hrc, jp1, jrc, out):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        valid = lambda d: d is not None and not d.empty

        sets = [(t, d) for t, d in
                [('Human Phase 1',   hp1), ('Human NS Recall', hrc),
                 ('Judge Phase 1',   jp1), ('Judge NS Recall',  jrc)]
                if valid(d)]
        if not sets: return

        # Overall confusion matrices
        fig, axes = plt.subplots(1, len(sets), figsize=(5*len(sets), 5))
        if len(sets) == 1: axes = [axes]
        for ax, (t, df) in zip(axes, sets):
            v  = df['verdict'].value_counts()
            cm = np.array([[v.get('TP',0), v.get('FP',0)],
                           [v.get('FN',0), v.get('TN',0)]])
            ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Pos','Neg']); ax.set_yticklabels(['Pos','Neg'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(t, fontsize=11, weight='bold')
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                            fontsize=16, fontweight='bold',
                            color='white' if cm[ii,jj] > cm.max()/2 else 'black')
            s = cm.sum()
            if s:
                ax.text(0.5, -0.15, f"Acc:{(cm[0,0]+cm[1,1])/s:.3f}  n={s}",
                        ha='center', transform=ax.transAxes, fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(out, 'confusion_matrices.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print("  Plot: confusion_matrices.png")

        # Per-field matrices
        for t, df in sets:
            flds = df['field'].unique()
            if not len(flds): continue
            fig2, ax2 = plt.subplots(1, len(flds), figsize=(4*len(flds), 4))
            if len(flds) == 1: ax2 = [ax2]
            for ax, fld in zip(ax2, flds):
                v  = df[df['field']==fld]['verdict'].value_counts()
                cm = np.array([[v.get('TP',0), v.get('FP',0)],
                               [v.get('FN',0), v.get('TN',0)]])
                ax.imshow(cm, cmap='Oranges', aspect='auto')
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(['P','N'], fontsize=8)
                ax.set_yticklabels(['P','N'], fontsize=8)
                ax.set_title(fld.replace('_',' '), fontsize=10, weight='bold')
                for ii in range(2):
                    for jj in range(2):
                        ax.text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                                fontsize=13, fontweight='bold')
            fig2.suptitle(f"Per-Field: {t}", fontsize=12, weight='bold')
            plt.tight_layout()
            fig2.savefig(os.path.join(out, f"cm_{t.replace(' ','_').lower()}.png"),
                         dpi=150, bbox_inches='tight')
            plt.close(fig2)

        # Per-platform accuracy bar chart
        for t, df in [('Judge Phase 1', jp1), ('Judge NS Recall', jrc)]:
            if not valid(df) or 'gpl' not in df.columns: continue
            gpls = sorted(df['gpl'].dropna().unique())
            accs = [calc(df[df['gpl']==g])['acc'] for g in gpls]
            fig3, ax3 = plt.subplots(figsize=(max(6, len(gpls)*2), 4))
            bars = ax3.bar(gpls, accs, color='#3F51B5', edgecolor='white')
            ax3.set_ylim(0, 1.1); ax3.set_ylabel('Accuracy')
            ax3.set_title(f"Per-Platform Accuracy — {t}", fontweight='bold')
            for bar, acc in zip(bars, accs):
                ax3.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                         f"{acc:.2f}", ha='center', fontsize=11)
            plt.tight_layout()
            fig3.savefig(os.path.join(out,
                         f"platform_acc_{t.replace(' ','_').lower()}.png"),
                         dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"  Plot: platform_acc_{t.replace(' ','_').lower()}.png")

    except Exception as e:
        print(f"  Plot error: {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    conn, mem_path, output_dir, model = setup()

    # ═══════════════════════════════════════════════════════════
    #  HUMAN EVALUATION  (200 samples)
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'─'*62}")
    print(f"  STEP 1  Sample {HUMAN_N} GSMs across 4 GPL platforms")
    print(f"{'─'*62}")
    human_samples = sample_multi_gpl(conn, HUMAN_N)

    print(f"\n{'─'*62}")
    print(f"  STEP 2  Phase 1 — Deterministic + LLM extraction")
    print(f"{'─'*62}")
    # Initialize memory agent (deterministic extraction)
    if _HAS_DETERMINISTIC:
        try:
            data_dir = CONFIG['paths']['data']
            mem_ok = init_memory_agent(data_dir)
            if mem_ok:
                print("  Memory Agent: READY (deterministic extraction enabled)")
                init_gse_contexts(human_samples, gds_conn=conn)
            else:
                print("  Memory Agent: NOT AVAILABLE (LLM-only mode)")
        except Exception as e:
            print(f"  Memory Agent: {e}")
    # Pre-fetch GSE context for LLM fallback
    if 'series_id' in human_samples.columns:
        prefetch_gse_context(conn, human_samples['series_id'].dropna().unique().tolist())
    p1h_raw = run_phase1(human_samples)
    if p1h_raw is None:
        print("[FATAL] Phase 1 failed."); sys.exit(1)
    p1h_raw.to_csv(os.path.join(output_dir, 'extraction_phase1_raw_human.csv'), index=False)

    print(f"\n{'─'*62}")
    print(f"  STEP 3  Phase 1.5 — Per-GSE normalization (variant unification)")
    print(f"{'─'*62}")
    p1h, h_norm_stats = run_phase15(p1h_raw)
    p1h.to_csv(os.path.join(output_dir, 'extraction_phase15_human.csv'), index=False)
    h_norm_items = _phase15_items(p1h_raw, p1h)
    if not h_norm_items.empty:
        h_norm_items.to_csv(os.path.join(output_dir, 'phase15_changes_human.csv'), index=False)
        print(f"  Phase 1.5 changed {len(h_norm_items)} labels across experiments")

    print(f"\n{'─'*62}")
    print(f"  STEP 4  Human GUI — Round 1: evaluate Phase 1 + Phase 1.5 labels")
    print(f"  Keys: 1=TP  2=FP  3=FN  4=TN  Enter=Submit")
    print(f"{'─'*62}")
    p1h_items = _p1_items(p1h)
    print(f"  Phase 1+1.5 items to evaluate: {len(p1h_items)}")
    hp1 = human_gui(p1h_items, conn, round_title="Round 1 — Phase 1+1.5 Labels")
    _save(hp1, os.path.join(output_dir, 'human_phase1.csv'), "Human Phase 1+1.5")

    print(f"\n{'─'*62}")
    print(f"  STEP 5  Phase 2 — NS handling (Condition/Tissue/Treatment ONLY)")
    print(f"{'─'*62}")
    p2h = run_phase2(p1h, mem_path)
    p2h.to_csv(os.path.join(output_dir, 'extraction_phase2_human.csv'), index=False)

    print(f"\n{'─'*62}")
    print(f"  STEP 6  Human GUI — Round 2: evaluate NS corrections (Phase 2)")
    print(f"  Keys: 1=TP  2=FP  3=FN  4=TN  Enter=Submit")
    print(f"{'─'*62}")
    rc_items = _ns_correction_items(p1h, p2h)
    print(f"  NS corrections to evaluate: {len(rc_items)}")
    print(f"  (Only Condition, Tissue, Treatment — Age/Treatment_Time excluded)")
    hrc = human_gui(rc_items, conn, round_title="Round 2 — Phase 2 NS Corrections")
    _save(hrc, os.path.join(output_dir, 'human_recall.csv'), "Human Phase 2 Recall")

    print(f"\n{'─'*62}")
    print(f"  STEP 7  Phase 3 — LLM Curator (cross-experiment label harmonization)")
    print(f"{'─'*62}")
    p3h, h_curator_stats = run_phase3_curator(p2h)
    p3h.to_csv(os.path.join(output_dir, 'extraction_phase3_curator_human.csv'), index=False)
    # Save curator proposals for human review
    if h_curator_stats.get('proposals'):
        proposals_rows = []
        for field, items in h_curator_stats['proposals'].items():
            for from_l, to_l, reason, fc, tc in items:
                proposals_rows.append({'field': field, 'from': from_l, 'to': to_l,
                                       'reason': reason, 'from_count': fc, 'to_count': tc})
        if proposals_rows:
            pd.DataFrame(proposals_rows).to_csv(
                os.path.join(output_dir, 'curator_proposals_human.csv'), index=False)
            print(f"  Curator proposals saved: {len(proposals_rows)} merges")

    # Merge all human results into one file
    all_human = [x for x in [hp1, hrc] if x is not None and not x.empty]
    if all_human:
        merged = pd.concat(all_human, ignore_index=True)
        mpath  = os.path.join(output_dir, 'human_all_results.csv')
        merged.to_csv(mpath, index=False)
        print(f"  ✓ All human results merged → {mpath}  ({len(merged)} rows)")

    # ═══════════════════════════════════════════════════════════
    #  LLM JUDGE  (1000 samples)
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'─'*62}")
    print(f"  STEP 8  Sample {JUDGE_N} GSMs across 4 GPL platforms (judge set)")
    print(f"{'─'*62}")
    judge_samples = sample_multi_gpl(conn, JUDGE_N)

    print(f"\n{'─'*62}")
    print(f"  STEP 9  Phase 1 + 1.5 + 2 labelling (judge set)")
    print(f"{'─'*62}")
    if _HAS_DETERMINISTIC:
        try:
            init_gse_contexts(judge_samples, gds_conn=conn)
        except Exception:
            pass
    if 'series_id' in judge_samples.columns:
        prefetch_gse_context(conn, judge_samples['series_id'].dropna().unique().tolist())
    p1j_raw = run_phase1(judge_samples)
    if p1j_raw is None:
        print("[FATAL] Judge Phase 1 failed."); sys.exit(1)
    p1j_raw.to_csv(os.path.join(output_dir, 'extraction_phase1_raw_judge.csv'), index=False)

    p1j, j_norm_stats = run_phase15(p1j_raw)
    p1j.to_csv(os.path.join(output_dir, 'extraction_phase15_judge.csv'), index=False)
    j_norm_items = _phase15_items(p1j_raw, p1j)
    if not j_norm_items.empty:
        j_norm_items.to_csv(os.path.join(output_dir, 'phase15_changes_judge.csv'), index=False)

    p2j = run_phase2(p1j, mem_path)
    p2j.to_csv(os.path.join(output_dir, 'extraction_phase2_judge.csv'), index=False)

    print(f"\n{'─'*62}")
    print(f"  STEP 11  Phase 3 — LLM Curator (judge set)")
    print(f"{'─'*62}")
    p3j, j_curator_stats = run_phase3_curator(p2j)
    p3j.to_csv(os.path.join(output_dir, 'extraction_phase3_curator_judge.csv'), index=False)
    if j_curator_stats.get('proposals'):
        proposals_rows = []
        for field, items in j_curator_stats['proposals'].items():
            for from_l, to_l, reason, fc, tc in items:
                proposals_rows.append({'field': field, 'from': from_l, 'to': to_l,
                                       'reason': reason, 'from_count': fc, 'to_count': tc})
        if proposals_rows:
            pd.DataFrame(proposals_rows).to_csv(
                os.path.join(output_dir, 'curator_proposals_judge.csv'), index=False)

    j_p1_items        = _p1_items(p2j)
    j_rc_items        = _ns_correction_items(p1j, p2j)
    j_p1_items        = j_p1_items.copy(); j_p1_items['is_recall'] = False
    j_rc_items        = j_rc_items.copy(); j_rc_items['is_recall'] = True
    judge_items       = pd.concat([j_p1_items, j_rc_items], ignore_index=True)
    print(f"  Judge items: {len(j_p1_items)} Phase 1+1.5  +  {len(j_rc_items)} NS corrections")

    print(f"\n{'─'*62}")
    print(f"  STEP 10  Parallel LLM Judge  ({len(judge_items)} items)")
    print(f"{'─'*62}")
    judge_results = judge(judge_items, conn, model)

    jp1 = judge_results[~judge_results['is_recall']].copy()
    jrc = judge_results[ judge_results['is_recall']].copy()
    _save(jp1, os.path.join(output_dir, 'judge_phase1.csv'),  "Judge Phase 1+1.5")
    _save(jrc, os.path.join(output_dir, 'judge_recall.csv'),  "Judge Phase 2 Recall")

    print(f"\n{'─'*62}")
    print(f"  STEP 12  Final Report + Per-Phase Metrics")
    print(f"{'─'*62}")
    make_report(hp1, hrc, jp1, jrc, output_dir,
                h_norm_stats=h_norm_stats, h_norm_items=h_norm_items,
                j_norm_stats=j_norm_stats, j_norm_items=j_norm_items,
                h_harm_stats=h_curator_stats, j_harm_stats=j_curator_stats)

    conn.close()
    print(f"\n{'═'*62}")
    print(f"  DONE!  All results saved to: {output_dir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
