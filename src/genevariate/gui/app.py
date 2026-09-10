"""
GeneVariate Main Application Window
Complete implementation with all features - NO SIMPLIFICATIONS
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser

# ── CustomTkinter: modern rounded window frame ──
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    _HAS_CTK = True
except ImportError:
    ctk = None
    _HAS_CTK = False

# ═══════════════════════════════════════════════════════════════
#  FRUTIGER AERO PALETTE - glossy white + sky-blue + nature-green
#  Defined in gui/theme.py so the secondary windows share it.
# ═══════════════════════════════════════════════════════════════
from genevariate.gui.theme import (
    AERO, UI_FONT, MONO_FONT, labelframe, ensure_theme, style_toolbar, style_window,
)


def _aero_vertical_gradient(canvas, width, height, top, bottom, tag="aero_bg"):
    """Draw a vertical gradient on a Tk Canvas by stacking horizontal lines.
    Safe to call repeatedly on resize - removes the previous gradient first."""
    try:
        canvas.delete(tag)
    except Exception:
        pass
    if width <= 0 or height <= 0:
        return
    tr, tg, tb = canvas.winfo_rgb(top)
    br, bg_, bb = canvas.winfo_rgb(bottom)
    tr, tg, tb = tr // 256, tg // 256, tb // 256
    br, bg_, bb = br // 256, bg_ // 256, bb // 256
    steps = max(1, height)
    for y in range(steps):
        t = y / max(1, steps - 1)
        r = int(tr * (1 - t) + br * t)
        g = int(tg * (1 - t) + bg_ * t)
        b = int(tb * (1 - t) + bb * t)
        canvas.create_line(0, y, width, y, fill=f"#{r:02x}{g:02x}{b:02x}", tags=tag)
    canvas.tag_lower(tag)
import pandas as pd
import numpy as np
import queue
import os
import time
import threading
import sqlite3
import tempfile
import gzip
import shutil
import subprocess
import requests
import uuid
import re
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import itertools

# Legacy local-inference "deterministic extraction" engine was removed.
# All label extraction now flows through the vendored geo_label_extractor
# package (see genevariate.core.geo_extract_driver). No in-process model.
_HAS_DETERMINISTIC = False

# MUST set TkAgg BEFORE any pyplot import (default qtagg conflicts with Tk)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

# Every table in the program gets a Save button, a right-click menu and Ctrl+S.
# Done here, once, so it also covers the tables in region_analysis /
# compare_analysis and any table added later.
from genevariate.gui.exporting import (
    install_table_export as _install_table_export,
    save_figure as gv_save_figure,
    attach_figure_export as gv_attach_figure_export,
    _safe as _safe_filename,
)
_install_table_export()
from genevariate.core import label_entities
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

# Unified plot theme (Frutiger Aero). Applies rcParams once, globally.
try:
    from genevariate.utils.viz_style import (
        apply_genevariate_style as _apply_viz_style,
        palette_for as viz_palette_for,
        cmap_for as viz_cmap_for,
        apply_plot_polish as viz_apply_polish,
        apply_aero_background as viz_apply_aero_bg,
        style_axis as viz_style_axis,
        smart_figsize as viz_smart_figsize,
        cap_figsize as viz_cap_figsize,
        make_interactive as viz_make_interactive,
        attach_point_labels as viz_point_labels,
        grid_matrix_cells as _cell_grid,
    )
    _apply_viz_style()
except Exception:
    def viz_palette_for(n, use_case="discrete"):
        return [mcolors.to_hex(c) for c in sns.color_palette("tab10", max(1, n))]
    def viz_cmap_for(kind="intensity"):
        return "viridis"
    def viz_apply_polish(ax, **kw):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    def viz_apply_aero_bg(ax): pass
    def viz_style_axis(ax, xlabel=None, ylabel=None, title=None, grid=True):
        if xlabel: ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        if ylabel: ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        if title:  ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    def viz_smart_figsize(kind="default", n_plots=1, n_rows=1):
        return (10, 6)
    def viz_cap_figsize(w, h):
        return (min(w, 16), min(h, 10))
    def viz_make_interactive(*args, **kwargs):
        return None
    def viz_point_labels(*args, **kwargs):
        return None
    def _cell_grid(ax, nrows, ncols, **kwargs):
        return None
    plt.rcParams['figure.max_open_warning'] = 50
from scipy.stats import ranksums, rankdata, wasserstein_distance
from scipy.signal import find_peaks

# Every density curve the program draws comes from the same estimator the
# Distribution Classifier counts modes on, so no plot can show one shape while
# the label printed beside it names another.
from genevariate.core.analysis.bimodality import (
    BIMODAL_TAGS, density_modes, robust_kde)
try:
    from .region_analysis import RegionAnalysisWindow, _wrap_to_parent
except ImportError:
    from genevariate.gui.region_analysis import (
        RegionAnalysisWindow, _wrap_to_parent)
# compare_analysis defines its own CompareDistributionsWindow with a different
# constructor (title_text rather than skip_autoload). app.py neither used it nor
# CompareRegionsWindow: the class defined below shadowed the import, so importing
# them here only made it ambiguous which window app.py opens. region_analysis
# imports compare_analysis directly where it wants that version.

# ═══════════════════════════════════════════════════════════════
#  GPU DETECTION (informational only)
# ═══════════════════════════════════════════════════════════════
def detect_gpus():
    """Returns list of dicts: [{id, name, vram_gb, free_vram_gb, type}]."""
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "id": int(parts[0]), "name": parts[1],
                    "vram_gb": round(int(parts[2]) / 1024, 1),
                    "free_vram_gb": round(int(parts[3]) / 1024, 1),
                    "type": "nvidia"
                })
    except Exception:
        pass
    if not gpus:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                stderr=subprocess.DEVNULL, text=True, timeout=5)
            for i, line in enumerate(out.strip().splitlines()[1:]):
                parts = line.split(",")
                if len(parts) >= 2:
                    gpus.append({
                        "id": i, "name": f"AMD GPU {i}",
                        "vram_gb": round(int(parts[-1].strip()) / 1e6, 1),
                        "free_vram_gb": 0, "type": "amd"
                    })
        except Exception:
            pass
    return gpus


def _find_geometadb():
    """Locate a local GEOmetadb file (.sqlite or .sqlite.gz).

    Delegates to genevariate.core.db_loader.find_geometadb so that the
    GUI and the core loader use the SAME search logic (one source of truth).
    Returns an existing file path, or a sensible placeholder path if nothing
    is found (so startup code that expects a string path stays happy).
    """
    _PROG_DIR = os.path.dirname(os.path.abspath(__file__))   # genevariate/gui/
    _PROJ_ROOT = os.path.dirname(_PROG_DIR)                  # genevariate/
    _default = os.path.join(_PROJ_ROOT, 'data', 'GEOmetadb.sqlite.gz')

    try:
        from genevariate.core.db_loader import find_geometadb as _core_find
        # Silence the core log here - we log our own line below for clarity
        found = _core_find(log_fn=lambda m: None)
    except Exception:
        found = None

    if found and os.path.exists(found):
        try:
            size_mb = os.path.getsize(found) / (1024 * 1024)
            print(f"[GEOmetadb] Auto-discovered: {found} ({size_mb:.0f} MB)")
        except OSError:
            print(f"[GEOmetadb] Auto-discovered: {found}")
        return found

    # Nothing found - fall back to the configured default (non-existent is fine;
    # downstream code handles the "file missing" case).
    return _default

def _find_data_dir():
    """Find or create data directory.
    
    app.py lives in genevariate/gui/app.py
    Data is at genevariate/data/ (sibling of gui/)
    So the key path is: _PROG_DIR/../data/
    
    Priority: project_root/data > directories with GPL files > standard paths > create default
    """
    _PROG_DIR = os.path.dirname(os.path.abspath(__file__))  # genevariate/gui/
    _PROJ_ROOT = os.path.dirname(_PROG_DIR)                 # genevariate/
    _CWD = os.getcwd()
    
    # Standard candidates - project root/data/ FIRST
    candidates = [
        os.path.join(_PROJ_ROOT, 'data'),       # genevariate/data/ - PRIMARY
        os.path.join(_PROG_DIR, 'data'),         # genevariate/gui/data (fallback)
        os.path.join(_PROG_DIR, '..', 'data'),   # same as PROJ_ROOT/data but via ..
        os.path.join(_CWD, 'data'),
        os.path.expanduser('~/.genevariate/data'),
        os.path.expanduser('~/genevariate_data'),
        './data',
    ]
    
    # Also check for common old output directory patterns
    import glob as _glob
    for parent in [_PROJ_ROOT, _PROG_DIR, _CWD]:
        for pattern in ['AI_agent*', 'results*', 'output*', 'gpl_data*']:
            for match in _glob.glob(os.path.join(parent, pattern)):
                if os.path.isdir(match):
                    candidates.insert(0, match)  # prioritize dirs with data
    
    # First pass: prefer directories that actually contain GPL data
    for p in candidates:
        if os.path.isdir(p):
            resolved = os.path.realpath(p)
            # Check if it has GPL subdirs or GPL files
            try:
                has_gpl = any(
                    (entry.startswith('GPL') and os.path.isdir(os.path.join(resolved, entry)))
                    or ('GPL' in entry.upper() and (entry.endswith('.csv.gz') or entry.endswith('.csv')))
                    for entry in os.listdir(resolved)
                )
                if has_gpl:
                    print(f"[Config] data_dir resolved: {resolved} (contains GPL data)")
                    return resolved
            except PermissionError:
                pass
    
    # Second pass: any existing directory
    for p in candidates:
        if os.path.isdir(p):
            resolved = os.path.realpath(p)
            print(f"[Config] data_dir resolved: {resolved} (from {p})")
            return resolved
    
    # Create default in project root (genevariate/data/, not genevariate/gui/data/)
    default = os.path.join(_PROJ_ROOT, 'data')
    os.makedirs(default, exist_ok=True)
    print(f"[Config] data_dir created: {default}")
    return default

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'threading': {'max_workers': 10},
    'paths': {
        'data': _find_data_dir(),
        'results': os.path.join(_PKG_DIR, 'results'),
        'geo_db': _find_geometadb(),
        'embedding_cache': os.path.join(_PKG_DIR, 'cache', 'embeddings'),
    },
    'database': {'sql_chunk_size': 500},
    'ai': {
        'model': 'gemma4:e2b',
        'embedding_model': 'all-MiniLM-L6-v2',
        'device': 'cpu'
    },
    'plotting': {
        'histogram': {
            'edge_color': 'black',
            'alpha': 0.7,
            'default_color': 'skyblue',
            'min_samples_for_kde': 30,
            'min_variance_for_kde': 0.01
        },
        'selection': {
            'face_color': 'red',
            'edge_color': 'black',
            'alpha': 0.3
        }
    }
}

METADATA_EXCLUSIONS = [
    'GSM', 'gsm', 'series_id', 'gpl', 'platform_id', 
    'submission_date', 'last_update_date', 'type',
    'Unnamed: 0', 'index'
]


def index_gene_columns(df):
    """Map GENE SYMBOL -> column name for every expression column of *df*.

    Downloaded matrices often carry expression as strings, so a column that
    is mostly parseable as a number is coerced in place and counted as
    expression; anything in :data:`METADATA_EXCLUSIONS` never is. Returns
    ``(gene_map, n_coerced)``.
    """
    excluded_upper = {str(c).upper() for c in METADATA_EXCLUSIONS}
    gene_map, coerced = {}, 0
    for col in df.columns:
        if str(col).upper() in excluded_upper:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            gene_map[str(col).upper()] = col
            continue
        try:
            parsed = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            continue
        if parsed.notna().sum() / max(1, len(parsed)) > 0.5:
            df[col] = parsed
            gene_map[str(col).upper()] = col
            coerced += 1
    return gene_map, coerced


# ── Label extraction - the vendored geo_label_extractor pipeline ────────
class SampleClassificationAgent:
    """Runs the vendored ``geo_label_extractor`` pipeline over a sample table.

    This enters the extractor at the same place its own command line does
    (``geo_pipeline.main``, through
    :func:`genevariate.core.geo_extract_driver.run_full_pipeline`), so labels
    produced in the GUI are the labels the extractor publishes: phase 1
    verbatim spans, phase 1b GSE-context recovery, phase 1c consensus, and -
    when the Stage-2 reference artifacts are on the machine - phase 2
    normalization against MeSH, Cellosaurus and BioLORD, with the ontology ids
    that go with it.

    It previously called ``Phase1Extractor.extract_field`` once per sample and
    then collapsed the output with a local abbreviation heuristic. That was
    phase 1 alone: verbatim strings, no ids, no cross-study normalization - and
    the heuristic was a second, different normalizer standing where phase 2
    belongs, so GUI labels and published labels were not the same labels.
    """

    def __init__(self, gui_log_func, max_workers=15):
        self.log = gui_log_func if gui_log_func else print
        self.MAX_WORKERS = max_workers

    def process_samples(self, gsm_df, fields=None, custom_fields=None,
                        stop_flag_fn=None, progress_fn=None,
                        out_dir=None, metadata_columns=None,
                        enrich=False, category=""):
        """Extract labels for ``gsm_df`` and return GSM + one column per field.

        ``out_dir`` receives the pipeline's own run directory - its checkpoint,
        snapshots and (when Stage 2 runs) the normalized corpus. A run left
        there resumes instead of re-billing every sample to the model, so a
        caller with a results directory should pass it rather than accept the
        temporary one.

        ``enrich`` adds a curated study-level block to the prompt, drawn from
        whichever third-party project covers ``category`` (the platform's
        technology category): Expression Atlas for array and bulk RNA-seq,
        CELLxGENE for single cell, nothing for methylation and peak assays.
        """
        from genevariate.core import geo_extract_driver as drv

        if custom_fields:
            self.log("[Extract] Custom fields are not supported by the "
                     "extractor and were ignored.")
        fields = [f for f in (fields or drv.ALL_FIELDS) if f in drv.ALL_FIELDS]
        if not fields:
            fields = list(drv.ALL_FIELDS)
        if gsm_df is None or len(gsm_df) == 0:
            self.log("[Extract] No samples to extract.")
            return pd.DataFrame()

        out_dir = out_dir or tempfile.mkdtemp(prefix="genevariate_extract_")
        os.makedirs(out_dir, exist_ok=True)

        if enrich:
            from genevariate.core import external_enrichment
            gsm_df = gsm_df.copy()
            added = external_enrichment.annotate_samples(
                gsm_df, category, log_func=self.log)
            if added:
                metadata_columns = list(
                    metadata_columns or drv.default_metadata_columns())
                metadata_columns += [c for c in added
                                     if c not in metadata_columns]

        table = os.path.join(out_dir, "gui_samples.csv")
        gsm_df.to_csv(table, index=False)

        refs = drv.reference_paths()
        missing = [k for k, v in refs.items() if not v]
        if missing:
            self.log(f"[Extract] Stage 2 (MeSH/Cellosaurus/BioLORD "
                     f"normalization) is unavailable - missing {', '.join(missing)}. "
                     f"Running extraction only; labels will be verbatim spans "
                     f"without ontology ids.")
        else:
            self.log("[Extract] Stage 2 reference artifacts found - labels "
                     "will be normalized against MeSH/Cellosaurus/BioLORD.")

        picked = drv.resolve_backend()
        self.log(f"[Extract] {len(gsm_df):,} samples via geo_label_extractor "
                 f"pipeline at {picked['url']} "
                 f"(phase 1 {picked['model']}, age/phase 2 {picked['age_model']})")
        self.log(f"[Extract] Run directory: {out_dir}")

        # The pipeline runs its stages as subprocesses, so progress has to be
        # read from what they flush rather than counted here.
        total = len(gsm_df)
        t0 = time.time()
        stop_polling = threading.Event()

        def _poll():
            last = -1
            while not stop_polling.wait(3.0):
                done = drv.pipeline_progress(out_dir)
                if done < 0 or done == last:
                    continue
                last = done
                elapsed = time.time() - t0
                speed = done / max(0.01, elapsed)
                eta = (total - done) / max(0.01, speed)
                self.log(f"[Extract] {done}/{total} "
                         f"({speed:.1f} smp/s, ETA {eta:.0f}s)")
                if progress_fn:
                    try:
                        progress_fn(done, total, speed, eta)
                    except Exception:
                        pass

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            result = drv.run_full_pipeline(
                table, out_dir,
                labels=fields,
                fields=metadata_columns,
                extract_workers=self.MAX_WORKERS,
                **refs)
        except BaseException as exc:
            self.log(f"[Extract ERROR] Pipeline failed: {type(exc).__name__}: {exc}")
            return pd.DataFrame()
        finally:
            stop_polling.set()

        if stop_flag_fn and stop_flag_fn():
            # A stage is a subprocess; it cannot be halted between samples. The
            # checkpoint it leaves means re-running the same out_dir resumes
            # rather than re-extracting, so nothing done so far is lost.
            self.log("[Extract] Stop requested - the pipeline stage ran to "
                     f"completion. Re-run against {out_dir} to resume.")

        labels_df = drv.read_pipeline_labels(out_dir, fields)
        if labels_df.empty:
            self.log("[Extract ERROR] Pipeline produced no labels. "
                     f"Return code {result.get('returncode')}; check {out_dir}.")
            return labels_df

        self.log(f"[Extract] Complete: {len(labels_df):,} samples, deepest "
                 f"stage {labels_df.attrs.get('stage')} "
                 f"(requested {result.get('stop_after')}).")
        return labels_df


class ExtractionThread(threading.Thread):
    """Thread that searches GEOmetadb for experiments matching keywords.

    Strategy (efficient hybrid of SQL + pandas):
      1. Opens its OWN thread-local copy of the database (thread-safe)
      2. Discovers all text columns via PRAGMA (not hard-coded)
      3. Uses SQL LIKE to find matching GSE/GSM rows (C-level speed, not Python .apply)
      4. Loads only matching rows into pandas for rich description building
      5. Tracks per-GSM matched tokens for the review window

    Outputs:
      final_df         - DataFrame of ALL samples from matching experiments
      gse_keywords      - {GSE_ID: [matched_tokens]}
      gse_descriptions  - {GSE_ID: "col: val\\ncol: val\\n..."}  (rich, multi-line)
      gsm_descriptions  - {GSM_ID: "col: val\\n..."}  (for sample-level detail)
      search_tokens     - set of search tokens used
    """

    # Columns to SKIP when building searchable text blobs
    GSM_EXCLUDED = {"gsm", "contact", "supplementary_file", "data_row_count",
                    "channel_count", "organism_ch1", "status", "series_id",
                    "submission_date", "last_update_date", "data_processing", "gpl"}
    GSE_EXCLUDED = {"gse", "status", "submission_date", "last_update_date",
                    "pubmed_id", "contributor"}

    def __init__(self, gz_path, plat_filter, search_tokens, log_func,
                 on_finish, gui_ref, search_sources=None, subfilters=None):
        super().__init__(daemon=True)
        self.gz_path = gz_path
        self.plat_filter = plat_filter
        self.search_tokens_raw = search_tokens
        self.log_func = log_func
        self.on_finish_cb = on_finish
        self.gui_ref = gui_ref
        # Which sources to query. GEOmetadb is opt-in now - every source the
        # user ticks is queried; others are skipped. Hits from all sources
        # merge into final_df with a "Source" column.
        if search_sources is None:
            self.search_sources = {"geo"}
        else:
            self.search_sources = {str(s).lower() for s in search_sources}
        # Per-source sub-filters (organism, tissue/disease/assay, species, …)
        self.subfilters = dict(subfilters or {})
        # Outputs
        self.final_df = None
        self.gse_keywords = {}
        self.gse_descriptions = {}
        self.gsm_descriptions = {}
        self.search_tokens = set()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _log(self, msg):
        self.log_func(msg)

    # ────────────────────────────────────────────────────────────────
    def run(self):
        mem_conn = None
        db_conn = None
        try:
            self._log("PROGRESS: 0")

            if "geo" in self.search_sources:
                self._log("[Step 1] Loading GEOmetadb into thread-local memory...")
                if not os.path.exists(self.gz_path):
                    self._log(f"[Step 1] ERROR: GEOmetadb not found at {self.gz_path}")
                    self.final_df = pd.DataFrame()
                else:
                    # ── Open GEOmetadb (resource-aware: disk or RAM) ──
                    from genevariate.core.db_loader import open_geometadb
                    db_conn = open_geometadb(self.gz_path, log_fn=self._log)
                    if db_conn is None:
                        self._log("[Step 1] ERROR: Could not open GEOmetadb")
                        self.final_df = pd.DataFrame()
                    else:
                        self._log("PROGRESS: 10")
                        self._do_search(db_conn)
            else:
                self._log("[Step 1] GEOmetadb skipped (not selected)")
                # Prime token normalisation + empty frame so merge path works
                self._prime_tokens()
                self.final_df = pd.DataFrame()
                self._log("PROGRESS: 20")
                extra = self.search_sources - {"geo"}
                if extra:
                    self._merge_external_sources(extra)

        except Exception as e:
            import traceback
            self._log(f"[Step 1] THREAD EXCEPTION: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            self.final_df = pd.DataFrame()
        finally:
            try:
                if 'db_conn' in dir() and db_conn is not None:
                    db_conn.close()
            except Exception:
                pass
            if self.gui_ref:
                self.gui_ref.after(0, self.on_finish_cb)

    # ────────────────────────────────────────────────────────────────
    def _prime_tokens(self):
        """Populate ``self.search_tokens`` from the raw keyword string.

        Used when GEOmetadb is skipped - external sources still need the
        normalised token set for description highlighting.
        """
        raw_tokens = {t.strip().lower() for t in self.search_tokens_raw.split(",") if t.strip()}
        tokens = set()
        for t in raw_tokens:
            tokens.add(t)
            cleaned = re.sub(r"[^a-z0-9\s]", "", t).strip()
            if cleaned:
                tokens.add(cleaned)
        self.search_tokens = tokens

    # ────────────────────────────────────────────────────────────────
    def _do_search(self, conn):
        # Normalize tokens: lowercase, strip symbols so "alzheimer's" matches "alzheimers" etc.
        raw_tokens = {t.strip().lower() for t in self.search_tokens_raw.split(",") if t.strip()}
        # Keep both raw and cleaned versions for broader matching
        tokens = set()
        for t in raw_tokens:
            tokens.add(t)  # original lowered: "alzheimer's"
            cleaned = re.sub(r"[^a-z0-9\s]", "", t).strip()  # "alzheimers"
            if cleaned:
                tokens.add(cleaned)
            # Strip trailing 's' for root form: "alzheimers" → "alzheimer"
            if cleaned.endswith('s') and len(cleaned) > 3:
                tokens.add(cleaned[:-1])
            # Also without spaces for compound terms: "breastcancer"
            no_space = cleaned.replace(" ", "")
            if no_space and no_space != cleaned and len(no_space) > 3:
                tokens.add(no_space)
            # Individual words from multi-word tokens (if long enough)
            words = cleaned.split()
            for w in words:
                if len(w) >= 4:
                    tokens.add(w)
                    if w.endswith('s') and len(w) > 4:
                        tokens.add(w[:-1])
        self.search_tokens = tokens  # used by review window for highlighting
        self._display_tokens = raw_tokens  # human-readable version
        if not tokens:
            self._log("[Step 1] ERROR: No search tokens provided.")
            self.final_df = pd.DataFrame()
            return

        # ── 0. Discover schemas ──
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        def _cols(tbl):
            if tbl not in tables:
                return []
            return [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]

        gse_all_cols = _cols('gse')
        gsm_all_cols = _cols('gsm')

        # Searchable text columns (all string-ish minus the exclusion list)
        gse_search_cols = [c for c in gse_all_cols
                           if c.lower() not in self.GSE_EXCLUDED]
        gsm_search_cols = [c for c in gsm_all_cols
                           if c.lower() not in self.GSM_EXCLUDED]

        self._log(f"[Step 1] Tokens: {tokens}")
        self._log(f"[Step 1] GSE searchable cols ({len(gse_search_cols)}): "
                  f"{gse_search_cols[:8]}{'...' if len(gse_search_cols) > 8 else ''}")
        self._log(f"[Step 1] GSM searchable cols ({len(gsm_search_cols)}): "
                  f"{gsm_search_cols[:8]}{'...' if len(gsm_search_cols) > 8 else ''}")

        if self._stop.is_set():
            return

        # ── 1. Search GSE table ──
        gse_ids_from_gse = set()
        if 'gse' in tables and gse_search_cols:
            self._log("[Step 1] Searching experiment (GSE) descriptions...")
            like_parts = []
            params = []
            for tok in tokens:
                for col in gse_search_cols:
                    like_parts.append(f"LOWER({col}) LIKE ?")
                    params.append(f"%{tok}%")

            sel = ', '.join(gse_all_cols)
            q = f"SELECT {sel} FROM gse WHERE {' OR '.join(like_parts)}"
            try:
                gse_df = pd.read_sql_query(q, conn, params=params)
                gse_ids_from_gse = set(gse_df['gse'].tolist())
                self._log(f"[Step 1]  → {len(gse_ids_from_gse)} experiment(s) matched via GSE descriptions")

                # Build rich descriptions & keyword map
                for _, row in gse_df.iterrows():
                    gse_id = row['gse']
                    parts = []
                    blob_lower = ""
                    for c in gse_search_cols:
                        val = row.get(c, None)
                        if pd.notna(val) and str(val).strip():
                            parts.append(f"{c}: {str(val).strip()}")
                            blob_lower += str(val).lower() + " "
                    self.gse_descriptions[gse_id] = "\n".join(parts)
                    # Normalize blob for matching (strip symbols)
                    blob_clean = re.sub(r"[^a-z0-9\s]", "", blob_lower)
                    matched = [t for t in tokens if t in blob_lower or t in blob_clean]
                    self.gse_keywords[gse_id] = matched
            except Exception as e:
                self._log(f"[Step 1] GSE query error: {e}")

        if self._stop.is_set():
            return

        # ── 2. Search GSM table ──
        gse_ids_from_gsm = set()
        matching_gsm_ids = set()
        if 'gsm' in tables and gsm_search_cols:
            self._log("[Step 1] Searching sample (GSM) descriptions...")
            like_parts = []
            params = []
            for tok in tokens:
                for col in gsm_search_cols:
                    like_parts.append(f"LOWER({col}) LIKE ?")
                    params.append(f"%{tok}%")

            where = ' OR '.join(like_parts)
            # Get matching GSMs with their series_id
            sel_cols = ['gsm']
            if 'series_id' in gsm_all_cols:
                sel_cols.append('series_id')
            # Also grab description cols for display
            for c in gsm_search_cols:
                if c not in sel_cols:
                    sel_cols.append(c)

            q = f"SELECT {', '.join(sel_cols)} FROM gsm WHERE {where}"
            try:
                gsm_match_df = pd.read_sql_query(q, conn, params=params)
                matching_gsm_ids = set(str(g).upper() for g in gsm_match_df['gsm'].tolist())
                if 'series_id' in gsm_match_df.columns:
                    gse_ids_from_gsm = set(gsm_match_df['series_id'].dropna().tolist())
                self._log(f"[Step 1]  → {len(matching_gsm_ids):,} sample(s) matched, "
                          f"from {len(gse_ids_from_gsm)} experiment(s)")

                # Build per-GSM descriptions for the review window
                for _, row in gsm_match_df.iterrows():
                    gsm_id = str(row['gsm']).upper()
                    parts = []
                    for c in gsm_search_cols:
                        val = row.get(c, None)
                        if pd.notna(val) and str(val).strip():
                            parts.append(f"{c}: {str(val).strip()}")
                    self.gsm_descriptions[gsm_id] = "\n".join(parts)

            except Exception as e:
                self._log(f"[Step 1] GSM query error: {e}")

            # Fallback: if gsm has no series_id, use gse_gsm mapping
            if not gse_ids_from_gsm and matching_gsm_ids and 'gse_gsm' in tables:
                self._log("[Step 1] Using gse_gsm mapping for GSM → GSE lookup...")
                gsm_list = list(matching_gsm_ids)
                for ci in range(0, len(gsm_list), 500):
                    chunk = gsm_list[ci:ci + 500]
                    ph = ','.join(['?'] * len(chunk))
                    try:
                        r = pd.read_sql_query(
                            f"SELECT DISTINCT gse FROM gse_gsm WHERE UPPER(gsm) IN ({ph})",
                            conn, params=chunk)
                        gse_ids_from_gsm.update(r['gse'].tolist())
                    except:
                        pass
                self._log(f"[Step 1]  → mapped to {len(gse_ids_from_gsm)} experiment(s)")

        self._log("PROGRESS: 30")
        if self._stop.is_set():
            return

        # ── 3. Combine ──
        all_gse_ids = gse_ids_from_gse | gse_ids_from_gsm
        only_gsm = gse_ids_from_gsm - gse_ids_from_gse
        self._log(f"[Step 1] Combined: {len(all_gse_ids)} unique experiment(s) "
                  f"({len(gse_ids_from_gse)} from GSE, {len(only_gsm)} only from GSM)")

        # Fetch GSE descriptions for experiments found only via sample search
        if only_gsm and 'gse' in tables:
            for ci in range(0, len(list(only_gsm)), 200):
                chunk = list(only_gsm)[ci:ci + 200]
                ph = ','.join(['?'] * len(chunk))
                sel = ', '.join(gse_all_cols)
                try:
                    desc_df = pd.read_sql_query(
                        f"SELECT {sel} FROM gse WHERE gse IN ({ph})", conn, params=chunk)
                    for _, row in desc_df.iterrows():
                        gid = row['gse']
                        if gid not in self.gse_descriptions:
                            parts = []
                            for c in gse_search_cols:
                                val = row.get(c, None)
                                if pd.notna(val) and str(val).strip():
                                    parts.append(f"{c}: {str(val).strip()}")
                            self.gse_descriptions[gid] = "\n".join(parts)
                            self.gse_keywords[gid] = list(tokens)
                except:
                    pass

        if not all_gse_ids:
            self._log("[Step 1] No experiments found matching keywords in GSE or GSM tables.")
            self.final_df = pd.DataFrame()
            return

        # ── 4. Load all samples for matching experiments ──
        self._log(f"[Step 1] Fetching all samples for {len(all_gse_ids)} experiment(s)...")
        gse_list = list(all_gse_ids)
        chunk_size = 200
        sample_dfs = []

        for i in range(0, len(gse_list), chunk_size):
            if self._stop.is_set():
                return
            chunk = gse_list[i:i + chunk_size]
            ph = ','.join(['?'] * len(chunk))

            if 'gse_gsm' in tables:
                gsm_sel = ', '.join(f'gsm.{c}' for c in gsm_all_cols)
                q = f"""SELECT {gsm_sel}, gse_gsm.gse AS _gse_map
                        FROM gse_gsm
                        JOIN gsm ON gse_gsm.gsm = gsm.gsm
                        WHERE gse_gsm.gse IN ({ph})"""
            elif 'series_id' in gsm_all_cols:
                q = f"SELECT * FROM gsm WHERE series_id IN ({ph})"
            else:
                continue

            try:
                chunk_df = pd.read_sql_query(q, conn, params=chunk)
                if '_gse_map' in chunk_df.columns:
                    chunk_df['series_id'] = chunk_df['_gse_map']
                    chunk_df.drop(columns=['_gse_map'], inplace=True, errors='ignore')
                sample_dfs.append(chunk_df)
                self._log(f"[Step 1]  chunk {i // chunk_size + 1}: "
                          f"{len(chunk_df):,} samples")
            except Exception as e:
                self._log(f"[Step 1] Sample fetch error: {e}")

        self._log("PROGRESS: 50")

        if not sample_dfs:
            self._log("[Step 1] No samples found for matching experiments.")
            self.final_df = pd.DataFrame()
            return

        all_samples = pd.concat(sample_dfs, ignore_index=True)
        self._log(f"[Step 1] Total samples loaded: {len(all_samples):,}")

        # ── 5. Platform filter ──
        if self.plat_filter.strip() and 'gpl' in all_samples.columns:
            wanted = {p.strip().upper() for p in self.plat_filter.split(",") if p.strip()}
            before = len(all_samples)
            all_samples = all_samples[
                all_samples['gpl'].astype(str).str.strip().str.upper().isin(wanted)
            ].copy()
            self._log(f"[Step 1] Platform filter {wanted}: {before:,} → {len(all_samples):,}")
            if all_samples.empty:
                self._log("[Step 1] No samples match platform filter.")
                self.final_df = pd.DataFrame()
                return

        # ── 6. Mark per-GSM token matches (case-insensitive) ──
        gsm_upper = all_samples['gsm'].astype(str).str.upper()
        all_samples['Token_Match'] = gsm_upper.isin(matching_gsm_ids).astype(int)

        # Build Matched_Tokens column for directly matched GSMs
        def _find_tokens_in_row(row):
            gsm_val = str(row.get('gsm', row.get('GSM', ''))).upper()
            if gsm_val not in matching_gsm_ids:
                return None
            blob = ""
            for c in gsm_search_cols:
                val = row.get(c, None)
                if pd.notna(val):
                    blob += str(val).lower() + " "
            blob_clean = re.sub(r"[^a-z0-9\s]", "", blob)
            return [t for t in tokens if t in blob or t in blob_clean] or None

        all_samples['Matched_Tokens'] = all_samples.apply(_find_tokens_in_row, axis=1)

        # ── 7. Normalize ──
        if 'gsm' in all_samples.columns:
            all_samples.rename(columns={'gsm': 'GSM'}, inplace=True)
        if 'GSM' in all_samples.columns:
            all_samples['GSM'] = all_samples['GSM'].astype(str).str.strip().str.upper()

        if 'series_id' in all_samples.columns:
            all_samples.drop_duplicates(subset=['GSM', 'series_id'], inplace=True)
        else:
            all_samples.drop_duplicates(subset=['GSM'], inplace=True)

        n_gse = all_samples['series_id'].nunique() if 'series_id' in all_samples.columns else 0
        n_gpl = all_samples['gpl'].nunique() if 'gpl' in all_samples.columns else 0
        n_matched = all_samples['Token_Match'].sum()
        self._log(f"[Step 1] OK Final (GEOmetadb): {len(all_samples):,} samples, "
                  f"{n_gse} experiments, {n_gpl} platform(s), "
                  f"{n_matched:,} samples with direct keyword match")
        self._log("PROGRESS: 70")

        # Tag GEOmetadb rows with the source column so the review window can
        # distinguish them from external-source hits merged below.
        all_samples['Source'] = 'GEOmetadb'

        self.final_df = all_samples

        # ── 8. Merge hits from additional sources (ARCHS4, CELLxGENE, Atlas) ──
        extra = self.search_sources - {"geo"}
        if extra:
            self._merge_external_sources(extra)
            self._log("PROGRESS: 90")

    # ────────────────────────────────────────────────────────────────
    def _merge_external_sources(self, sources):
        """Query additional data sources and append their hits to final_df.

        External hits are dataset-level (one row per GSE / dataset_id) since
        sources like CELLxGENE and Expression Atlas don't expose GEO-style
        per-sample metadata. The review window shows these alongside
        GEOmetadb results with a "Source" column so users can tell them
        apart. Step 1.5 download remains GEO-only.
        """
        try:
            from genevariate.sources.discovery import (
                search_sources as _search_sources, hits_to_dataframe)
        except Exception as exc:
            self._log(f"[Step 1] External sources module unavailable: {exc}")
            return

        self._log(f"[Step 1] Querying external sources: {sorted(sources)}")
        try:
            per_source = _search_sources(
                self.search_tokens_raw, sources,
                max_per_source=200, log=self._log,
                subfilters=self.subfilters)
        except Exception as exc:
            self._log(f"[Step 1] External search failed: {exc}")
            return

        all_extra = []
        for src_key, hits in per_source.items():
            if not hits:
                continue
            df = hits_to_dataframe(hits)
            all_extra.append(df)
            # Register descriptions so the review-window detail pane shows
            # something meaningful for non-GEO accessions.
            for h in hits:
                desc_lines = [f"source: {h.source}"]
                if h.title:
                    desc_lines.append(f"title: {h.title}")
                if h.organism:
                    desc_lines.append(f"organism: {h.organism}")
                if h.platform:
                    desc_lines.append(f"platform: {h.platform}")
                if h.n_samples:
                    label = "cells" if h.source == "CELLxGENE" else "samples"
                    desc_lines.append(f"{label}: {h.n_samples:,}")
                if h.summary:
                    desc_lines.append(f"summary: {h.summary}")
                if h.url:
                    desc_lines.append(f"url: {h.url}")
                self.gse_descriptions[h.accession] = "\n".join(desc_lines)
                self.gse_keywords[h.accession] = list(h.matched)

        if not all_extra:
            self._log("[Step 1] No external hits.")
            return

        extra_df = pd.concat(all_extra, ignore_index=True)
        if self.final_df is None or self.final_df.empty:
            self.final_df = extra_df
        else:
            # Align columns before concat (external frames have fewer cols)
            combined = pd.concat([self.final_df, extra_df],
                                  ignore_index=True, sort=False)
            self.final_df = combined

        self._log(f"[Step 1] Merged {len(extra_df)} external hit(s) from "
                  f"{extra_df['Source'].nunique()} source(s)")


# ═══════════════════════════════════════════════════════════════════
#  GSE Review Window - keyword-highlighted experiment browser
# ═══════════════════════════════════════════════════════════════════
class GSEReviewWindow(tk.Toplevel):
    """Interactive experiment review with red keyword highlighting.

    Left pane:  Checkable treeview of experiments (GSE ID, samples, platform, keywords)
    Right pane: Rich text showing full GSE description + sample-level details
                with search keywords highlighted in RED.
    """

    def __init__(self, parent, app_ref, results_df, gse_descriptions,
                 gse_keywords, gsm_descriptions, search_tokens):
        super().__init__(parent)
        self.app = app_ref
        self.results_df = results_df
        self.gse_descriptions = gse_descriptions
        self.gse_keywords = gse_keywords
        self.gsm_descriptions = gsm_descriptions
        self.search_tokens = search_tokens
        self._checks = {}  # {gse_id: BooleanVar}

        self.title("Step 1 - Review Experiments")
        ensure_theme(self)
        style_window(self)
        self.geometry("1200x750")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1200x750+{(_sw-1200)//2}+{(_sh-750)//2}")
            self.minsize(600, 500)
        except Exception: pass
        self.transient(parent)

        self._build_ui()
        self._populate()

    # ── UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ttk.Frame(self)
        hdr.pack(fill=tk.X, padx=8, pady=(8, 2))

        n_gse = self.results_df['series_id'].nunique() if 'series_id' in self.results_df.columns else 0
        n_gsm = len(self.results_df)
        ttk.Label(hdr, text=f"Found {n_gse} experiment(s)  •  {n_gsm:,} total samples  •  "
                             f"Keywords: {', '.join(sorted(self.search_tokens))}",
                  font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT)

        # Mode selector - experiment-wise or sample-wise
        mode_frame = ttk.Frame(hdr)
        mode_frame.pack(side=tk.RIGHT)
        ttk.Label(mode_frame, text="Selection:",
                   font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 4))
        self.selection_mode = tk.StringVar(value="experiment")
        ttk.Radiobutton(mode_frame, text="Experiment-wise",
                         variable=self.selection_mode, value="experiment",
                         command=self._update_count).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Sample-wise",
                         variable=self.selection_mode, value="sample",
                         command=self._update_count).pack(side=tk.LEFT, padx=2)

        # ── PanedWindow (left = tree, right = detail) ──
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # LEFT: Treeview
        left = ttk.Frame(pw)
        pw.add(left, weight=2)

        cols = ("gse", "source", "samples", "platforms", "matched_kw")
        self.tree = ttk.Treeview(left, columns=cols, show='tree headings',
                                  selectmode='browse', height=25)
        self.tree.heading('#0', text='✓')
        self.tree.column('#0', width=45, stretch=False)
        self.tree.heading('gse', text='GSE / GSM')
        self.tree.column('gse', width=150)
        self.tree.heading('source', text='Source')
        self.tree.column('source', width=110)
        self.tree.heading('samples', text='Samples / Match')
        self.tree.column('samples', width=90, anchor='center')
        self.tree.heading('platforms', text='Platform(s)')
        self.tree.column('platforms', width=90)
        self.tree.heading('matched_kw', text='Keywords / Title')
        self.tree.column('matched_kw', width=220)

        sb = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.config(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.tree.bind('<Button-1>', self._on_tree_click)

        # Style tags
        self.tree.tag_configure('checked', foreground=AERO['green_dark'])
        self.tree.tag_configure('partial', foreground=AERO['warn'])
        self.tree.tag_configure('unchecked', foreground=AERO['muted'])
        self.tree.tag_configure('gse_row', font=('Segoe UI', 10, 'bold'))
        self.tree.tag_configure('gsm_row', font=('Segoe UI', 9))

        # RIGHT: Structured detail card (scrollable)
        right = ttk.Frame(pw)
        pw.add(right, weight=3)

        self._detail_canvas = tk.Canvas(right, highlightthickness=0,
                                          background="#FFFFFF", bd=0)
        dsb = ttk.Scrollbar(right, command=self._detail_canvas.yview)
        self._detail_canvas.configure(yscrollcommand=dsb.set)
        self._detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._detail_inner = tk.Frame(self._detail_canvas, bg="#FFFFFF")
        self._detail_window_id = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor='nw')

        def _on_inner_config(event):
            try:
                self._detail_canvas.configure(
                    scrollregion=self._detail_canvas.bbox('all'))
            except Exception:
                pass
        self._detail_inner.bind('<Configure>', _on_inner_config)

        def _on_canvas_config(event):
            try:
                self._detail_canvas.itemconfig(self._detail_window_id,
                                                width=event.width)
            except Exception:
                pass
        self._detail_canvas.bind('<Configure>', _on_canvas_config)

        def _on_mwheel(event):
            try:
                delta = -1 * (event.delta // 120) if event.delta else 0
                if delta == 0 and getattr(event, "num", 0) in (4, 5):
                    delta = -1 if event.num == 4 else 1
                self._detail_canvas.yview_scroll(int(delta), 'units')
            except Exception:
                pass
        self._detail_canvas.bind('<MouseWheel>', _on_mwheel)
        self._detail_canvas.bind('<Button-4>', _on_mwheel)
        self._detail_canvas.bind('<Button-5>', _on_mwheel)

        # Placeholder
        tk.Label(self._detail_inner,
                  text="← Select an experiment to see details",
                  bg="#FFFFFF", fg="#999",
                  font=('Segoe UI', 10, 'italic')
                  ).pack(padx=16, pady=40)

        # ── Bottom buttons ──
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(btn_frame, text="Select All", command=self._select_all,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Deselect All", command=self._deselect_all,
                   style="Destructive.TButton").pack(side=tk.LEFT, padx=4)

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._count_label = ttk.Label(btn_frame, text="", font=('Segoe UI', 10))
        self._count_label.pack(side=tk.LEFT, padx=8)

        ttk.Button(btn_frame, text="🤖 LLM Label Extraction →",
                   command=self._go_to_llm,
                   style="Primary.TButton").pack(side=tk.RIGHT, padx=4)

        ttk.Button(btn_frame, text="💾 Download Data…",
                   command=self._download_data,
                   style="Tool.TButton").pack(side=tk.RIGHT, padx=4)

    # ── Populate ───────────────────────────────────────────────────
    def _populate(self):
        df = self.results_df
        if 'series_id' not in df.columns:
            return

        # GSM iid → parent GSE id lookup
        self._gsm_parent = {}

        gsm_col = 'GSM' if 'GSM' in df.columns else ('gsm' if 'gsm' in df.columns else None)

        for gse_id in sorted(df['series_id'].unique()):
            sub = df[df['series_id'] == gse_id]
            n = len(sub)
            plats = ', '.join(sorted(str(p) for p in sub['gpl'].dropna().unique())) if 'gpl' in sub.columns else '?'
            kws = self.gse_keywords.get(gse_id, [])
            kw_str = ', '.join(kws[:5])
            if len(kws) > 5:
                kw_str += f' (+{len(kws)-5})'

            # Source column
            if 'Source' in sub.columns:
                srcs = sorted(str(s) for s in sub['Source'].dropna().unique() if s)
                src_str = ', '.join(srcs) if srcs else 'GEOmetadb'
            else:
                src_str = 'GEOmetadb'

            # GSE (parent) row
            self._checks[gse_id] = True
            self.tree.insert('', tk.END, iid=gse_id,
                              text='[✓]',
                              values=(gse_id, src_str, f"{n:,}",
                                      plats, kw_str),
                              tags=('checked', 'gse_row'),
                              open=False)

            # GSM (child) rows
            if gsm_col is None:
                continue
            for _, row in sub.iterrows():
                gsm_raw = row.get(gsm_col, '')
                gsm_id = str(gsm_raw).strip()
                if not gsm_id:
                    continue
                iid = f"{gse_id}::{gsm_id}"
                if self.tree.exists(iid):
                    continue
                matched = False
                if 'Token_Match' in sub.columns:
                    try:
                        matched = bool(int(row.get('Token_Match', 0)))
                    except Exception:
                        matched = False
                match_str = '✓ match' if matched else '·'
                title_snip = str(row.get('title', '') or '').strip()[:90]
                if not title_snip:
                    title_snip = str(row.get('source_name_ch1', '') or '').strip()[:90]
                gsm_plat = str(row.get('gpl', '') or '').strip()
                row_src = str(row.get('Source', src_str) or src_str)

                self._checks[iid] = True
                self._gsm_parent[iid] = gse_id
                self.tree.insert(gse_id, tk.END, iid=iid,
                                  text='[✓]',
                                  values=(gsm_id, row_src,
                                          match_str, gsm_plat, title_snip),
                                  tags=('checked', 'gsm_row'))

        self._update_count()

    # ── Tree interactions ──────────────────────────────────────────
    def _on_tree_click(self, event):
        """Toggle checkbox only on the [✓] glyph - clicks elsewhere just select."""
        region = self.tree.identify_region(event.x, event.y)
        if region != 'tree':
            return
        # Skip the disclosure indicator (expand/collapse arrow) and padding -
        # only the 'text' element (where "[✓]" is rendered) should toggle.
        try:
            elem = self.tree.identify_element(event.x, event.y) or ''
        except Exception:
            elem = ''
        if 'text' not in elem.lower():
            return
        iid = self.tree.identify_row(event.y)
        if iid and iid in self._checks:
            self._set_check(iid, not self._checks[iid], cascade=True)
            self._update_count()

    def _set_check(self, iid, value, cascade=False):
        """Set check state for iid; optionally cascade to children / parent."""
        self._checks[iid] = bool(value)
        is_gsm = iid in self._gsm_parent
        row_tag = 'gsm_row' if is_gsm else 'gse_row'

        if value:
            self.tree.item(iid, text='[✓]', tags=('checked', row_tag))
        else:
            self.tree.item(iid, text='[ ]', tags=('unchecked', row_tag))

        if not cascade:
            return

        # GSE toggle → cascade to all its GSM children
        if not is_gsm:
            for child in self.tree.get_children(iid):
                self._set_check(child, value, cascade=False)
            return

        # GSM toggle → recompute parent state from siblings
        parent = self._gsm_parent.get(iid)
        if not parent:
            return
        siblings = self.tree.get_children(parent)
        any_checked = any(self._checks.get(c, False) for c in siblings)
        all_checked = all(self._checks.get(c, False) for c in siblings)
        if all_checked:
            self._checks[parent] = True
            self.tree.item(parent, text='[✓]', tags=('checked', 'gse_row'))
        elif any_checked:
            self._checks[parent] = True
            self.tree.item(parent, text='[▣]', tags=('partial', 'gse_row'))
        else:
            self._checks[parent] = False
            self.tree.item(parent, text='[ ]', tags=('unchecked', 'gse_row'))

    def _on_tree_select(self, event):
        """Show detail for selected row (GSM rows route to their parent GSE)."""
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        gse_id = self._gsm_parent.get(iid, iid)
        self._show_detail(gse_id)

    def _show_detail(self, gse_id):
        """Render GSE + sample details as structured cards."""
        # Clear prior content
        for w in self._detail_inner.winfo_children():
            w.destroy()

        df = self.results_df
        sub = df[df['series_id'] == gse_id] if 'series_id' in df.columns else pd.DataFrame()

        # ── Gather data ──
        n_total = len(sub)
        n_matched = int(sub['Token_Match'].sum()) if 'Token_Match' in sub.columns else 0
        plats = sorted(str(s) for s in sub['gpl'].dropna().unique()) if 'gpl' in sub.columns else []
        plat_str = ', '.join(plats) if plats else '-'

        orgs = []
        for col in ('organism_ch1', 'Organism', 'organism'):
            if col in sub.columns:
                orgs = sorted(str(s) for s in sub[col].dropna().unique() if str(s).strip())
                if orgs:
                    break
        org_str = ', '.join(orgs) if orgs else '-'

        srcs = []
        if 'Source' in sub.columns:
            srcs = sorted(str(s) for s in sub['Source'].dropna().unique() if str(s).strip())
        src_str = ', '.join(srcs) if srcs else 'GEOmetadb'

        desc = self.gse_descriptions.get(gse_id, "") or ""
        matched_kws = self.gse_keywords.get(gse_id, []) or []
        url = self._compute_source_url(gse_id, src_str)

        # ── HEADER CARD ──
        head = tk.Frame(self._detail_inner, bg=AERO["panel_bot"],
                         highlightbackground=AERO["border"], highlightthickness=1)
        head.pack(fill=tk.X, padx=12, pady=(12, 8))

        tk.Label(head, text=str(gse_id),
                  font=('Segoe UI', 16, 'bold'),
                  bg=AERO["panel_bot"], fg=AERO["accent_dark"],
                  anchor='w').pack(anchor='w', padx=14, pady=(10, 0))

        title_line = desc.split('\n', 1)[0][:260] if desc else "(no title available)"
        tk.Label(head, text=title_line,
                  font=('Segoe UI', 10),
                  bg=AERO["panel_bot"], fg=AERO["text"],
                  wraplength=540, justify='left', anchor='w'
                  ).pack(anchor='w', padx=14, pady=(2, 6))

        if url:
            link = tk.Label(head, text=f"🔗 {url}",
                             font=('Segoe UI', 9, 'underline'),
                             bg=AERO["panel_bot"], fg=AERO["accent"],
                             cursor='hand2', anchor='w')
            link.pack(anchor='w', padx=14, pady=(0, 10))
            link.bind('<Button-1>', lambda e, u=url: self._open_url(u))

        # ── STAT BADGES ──
        stats = tk.Frame(self._detail_inner, bg="#FFFFFF")
        stats.pack(fill=tk.X, padx=12, pady=4)
        for label, val, color in (
            ("Source",   src_str,                        "#1E90E0"),
            ("Organism", org_str,                        "#4CAF50"),
            ("Samples",  f"{n_matched:,} / {n_total:,}", "#E67E22"),
            ("Platform", plat_str,                       "#7E57C2"),
        ):
            self._make_badge(stats, label, val, color).pack(
                side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        # ── MATCHED KEYWORD PILLS ──
        if matched_kws:
            kwrow = tk.Frame(self._detail_inner, bg="#FFFFFF")
            kwrow.pack(fill=tk.X, padx=12, pady=(10, 2))
            tk.Label(kwrow, text="Matched keywords",
                      font=('Segoe UI', 9, 'bold'),
                      bg="#FFFFFF", fg=AERO["muted"],
                      anchor='w').pack(anchor='w')
            pill_wrap = tk.Frame(kwrow, bg="#FFFFFF")
            pill_wrap.pack(fill=tk.X, pady=(2, 0))
            for kw in matched_kws[:16]:
                tk.Label(pill_wrap, text=str(kw),
                          bg="#FFE8E6", fg="#C0392B",
                          font=('Segoe UI', 9, 'bold'),
                          padx=8, pady=2,
                          borderwidth=1, relief='solid'
                          ).pack(side=tk.LEFT, padx=2, pady=2)

        # ── SUMMARY CARD ──
        if desc:
            sc = tk.LabelFrame(self._detail_inner, text=' Summary ',
                                bg="#FFFFFF", fg=AERO["text"],
                                font=('Segoe UI', 9, 'bold'),
                                bd=1, relief='solid', padx=4, pady=2)
            sc.pack(fill=tk.X, padx=12, pady=(10, 6))
            self._render_highlighted(sc, desc, max_height=10)

        if sub.empty:
            return

        # ── MATCHED SAMPLES ──
        if 'Token_Match' in sub.columns:
            matched_samples = sub[sub['Token_Match'] == 1]
        else:
            matched_samples = sub.head(10)

        if not matched_samples.empty:
            sec_hdr = tk.Frame(self._detail_inner, bg="#FFFFFF")
            sec_hdr.pack(fill=tk.X, padx=12, pady=(10, 0))
            tk.Label(sec_hdr,
                      text=f"🧪  Matched Samples   "
                           f"({min(30, len(matched_samples))} of {len(matched_samples):,})",
                      bg="#FFFFFF", fg=AERO["accent_dark"],
                      font=('Segoe UI', 10, 'bold'), anchor='w'
                      ).pack(anchor='w')

            gsm_col = 'GSM' if 'GSM' in matched_samples.columns else 'gsm'
            for _, row in matched_samples.head(30).iterrows():
                gsm_id = row.get(gsm_col, '?')
                self._make_sample_card(gsm_id, row).pack(
                    fill=tk.X, padx=12, pady=3)

        # ── OTHER SAMPLES (context) ──
        if 'Token_Match' in sub.columns:
            non_matched = sub[sub['Token_Match'] == 0]
            if not non_matched.empty:
                sec_hdr = tk.Frame(self._detail_inner, bg="#FFFFFF")
                sec_hdr.pack(fill=tk.X, padx=12, pady=(10, 0))
                tk.Label(sec_hdr,
                          text=f"Other Samples   "
                               f"({min(5, len(non_matched))} of {len(non_matched):,})",
                          bg="#FFFFFF", fg=AERO["muted"],
                          font=('Segoe UI', 9, 'bold'), anchor='w'
                          ).pack(anchor='w')
                for _, row in non_matched.head(5).iterrows():
                    gsm_id = row.get('GSM', row.get('gsm', '?'))
                    title = row.get('title', 'N/A')
                    src = row.get('source_name_ch1', '')
                    line_text = f"  {gsm_id}: {str(title)[:140]}"
                    if pd.notna(src) and str(src).strip():
                        line_text += f" | {str(src)[:80]}"
                    tk.Label(self._detail_inner, text=line_text,
                              bg="#FFFFFF", fg=AERO["muted"],
                              font=('Segoe UI', 9),
                              anchor='w', justify='left',
                              wraplength=560
                              ).pack(fill=tk.X, padx=16, pady=1, anchor='w')

        # Reset scroll
        try:
            self._detail_canvas.update_idletasks()
            self._detail_canvas.yview_moveto(0)
        except Exception:
            pass

    # ── Card-render helpers ─────────────────────────────────────────
    def _make_badge(self, parent, label, value, color):
        """Colored stat badge: small coloured label + bold value."""
        f = tk.Frame(parent, bg="#FFFFFF",
                      highlightbackground=color, highlightthickness=2, bd=0)
        tk.Label(f, text=str(label).upper(),
                  bg="#FFFFFF", fg=color,
                  font=('Segoe UI', 8, 'bold'),
                  anchor='w').pack(anchor='w', padx=8, pady=(4, 0))
        tk.Label(f, text=(str(value) if value else "-"),
                  bg="#FFFFFF", fg=AERO["text"],
                  font=('Segoe UI', 10, 'bold'),
                  wraplength=160, justify='left',
                  anchor='w').pack(anchor='w', padx=8, pady=(0, 4))
        return f

    def _make_sample_card(self, gsm_id, row):
        """One matched-sample card: GSM id + wrapped description with kw pills."""
        card = tk.Frame(self._detail_inner, bg=AERO["glass_hilite"],
                         highlightbackground=AERO["border_soft"],
                         highlightthickness=1)
        top = tk.Frame(card, bg=AERO["glass_hilite"])
        top.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(top, text=str(gsm_id), bg=AERO["glass_hilite"],
                  fg=AERO["accent_dark"],
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)

        gsm_desc = (self.gsm_descriptions.get(gsm_id)
                    or self.gsm_descriptions.get(str(gsm_id).upper()))
        if gsm_desc:
            body_text = gsm_desc
        else:
            parts = []
            for c in ('title', 'source_name_ch1', 'characteristics_ch1'):
                val = row.get(c, None)
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{c}: {str(val).strip()}")
            body_text = '\n'.join(parts)

        if body_text:
            self._render_highlighted(card, body_text, max_height=8,
                                      bg=AERO["glass_hilite"],
                                      pad=(8, 0, 8, 6))
        return card

    def _render_highlighted(self, parent, text, *, max_height=10,
                             bg="#FFFFFF", pad=(6, 2, 6, 4)):
        """Borderless tk.Text showing `text` with search tokens in red pills."""
        t = tk.Text(parent, wrap='word',
                     bg=bg, fg=AERO["text"],
                     relief='flat', borderwidth=0, highlightthickness=0,
                     font=('Segoe UI', 9), height=1, cursor='arrow')
        t.tag_configure('keyword', foreground='#C0392B',
                         font=('Segoe UI', 9, 'bold'),
                         background='#FFE8E6')
        self._insert_with_highlights(t, str(text))
        # Size to content
        t.update_idletasks()
        try:
            lines = int(t.index('end-1c').split('.')[0])
        except Exception:
            lines = 1
        t.configure(height=min(max_height, max(1, lines)))
        t.config(state='disabled')
        padl, padt, padr, padb = pad
        t.pack(fill=tk.X, padx=(padl, padr), pady=(padt, padb))
        return t

    def _insert_with_highlights(self, widget, text):
        """Insert `text` into a tk.Text with search tokens tagged 'keyword'."""
        if not self.search_tokens:
            widget.insert(tk.END, text)
            return
        text_lower = text.lower()
        highlights = []
        for tok in self.search_tokens:
            start = 0
            while True:
                idx = text_lower.find(tok, start)
                if idx == -1:
                    break
                highlights.append((idx, idx + len(tok)))
                start = idx + 1
            cleaned_tok = re.sub(r"[^a-z0-9\s]", "", tok)
            if cleaned_tok and cleaned_tok != tok:
                orig_positions = []
                for i, ch in enumerate(text_lower):
                    if re.match(r"[a-z0-9\s]", ch):
                        orig_positions.append(i)
                cleaned_text = re.sub(r"[^a-z0-9\s]", "", text_lower)
                start = 0
                while True:
                    idx = cleaned_text.find(cleaned_tok, start)
                    if idx == -1:
                        break
                    if idx + len(cleaned_tok) - 1 < len(orig_positions):
                        orig_start = orig_positions[idx]
                        orig_end = orig_positions[idx + len(cleaned_tok) - 1] + 1
                        highlights.append((orig_start, orig_end))
                    start = idx + 1
        if not highlights:
            widget.insert(tk.END, text)
            return
        highlights.sort()
        merged = [highlights[0]]
        for s, e in highlights[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        pos = 0
        for s, e in merged:
            if pos < s:
                widget.insert(tk.END, text[pos:s])
            widget.insert(tk.END, text[s:e], 'keyword')
            pos = e
        if pos < len(text):
            widget.insert(tk.END, text[pos:])

    def _compute_source_url(self, gse_id, src_str):
        s = (src_str or "").lower()
        gid = str(gse_id).strip()
        if not gid:
            return ""
        if "cellxgene" in s or "cxg" in s:
            return f"https://cellxgene.cziscience.com/collections?search={gid}"
        if "atlas" in s or "gxa" in s or gid.startswith("E-"):
            return f"https://www.ebi.ac.uk/gxa/experiments/{gid}"
        if "archs4" in s or "geo" in s or gid.startswith("GSE"):
            return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gid}"
        return ""

    def _open_url(self, url):
        try:
            import webbrowser
            webbrowser.open(url, new=2)
        except Exception:
            pass

    # ── Selection helpers ──────────────────────────────────────────
    def _select_all(self):
        for iid in list(self._checks.keys()):
            self._checks[iid] = True
            is_gsm = iid in self._gsm_parent
            self.tree.item(iid, text='[✓]',
                           tags=('checked', 'gsm_row' if is_gsm else 'gse_row'))
        self._update_count()

    def _deselect_all(self):
        for iid in list(self._checks.keys()):
            self._checks[iid] = False
            is_gsm = iid in self._gsm_parent
            self.tree.item(iid, text='[ ]',
                           tags=('unchecked', 'gsm_row' if is_gsm else 'gse_row'))
        self._update_count()

    def _collect_selection_df(self):
        """Return filtered results_df based on current mode + checks."""
        df = self.results_df
        if 'series_id' not in df.columns:
            return df.iloc[0:0]
        mode = (self.selection_mode.get()
                if hasattr(self, 'selection_mode') else 'experiment')
        if mode == 'experiment':
            gse_ids = [k for k, v in self._checks.items()
                        if v and k not in self._gsm_parent]
            return df[df['series_id'].isin(gse_ids)]
        # Sample-wise: only explicitly checked GSM rows
        gsm_ids = set()
        for iid, v in self._checks.items():
            if v and iid in self._gsm_parent:
                gsm_ids.add(iid.split('::', 1)[1])
        gcol = 'GSM' if 'GSM' in df.columns else ('gsm' if 'gsm' in df.columns else None)
        if gcol is None:
            return df.iloc[0:0]
        return df[df[gcol].astype(str).isin(gsm_ids)]

    def _update_count(self):
        mode = (self.selection_mode.get()
                if hasattr(self, 'selection_mode') else 'experiment')
        sel_df = self._collect_selection_df()
        n_samples = len(sel_df)
        n_gse = (sel_df['series_id'].nunique()
                 if 'series_id' in sel_df.columns else 0)
        mode_label = "Experiment-wise" if mode == 'experiment' else "Sample-wise"
        self._count_label.config(
            text=f"Mode: {mode_label}  •  {n_gse} experiment(s)  •  "
                 f"{n_samples:,} sample(s)")

    # ── Primary actions ────────────────────────────────────────────
    def _commit_selection_to_app(self, sel_df):
        """Push the filtered selection into app state so downstream steps use it."""
        if 'series_id' in sel_df.columns:
            gse_list = sorted(sel_df['series_id'].dropna().astype(str).unique())
        else:
            gse_list = []
        self.app.step1_results_df = sel_df.copy()
        self.app.gse_to_keep_for_step2 = gse_list
        return gse_list

    def _go_to_llm(self):
        """Save selection → LLM label extraction (existing Step 2 flow)."""
        sel_df = self._collect_selection_df()
        if sel_df.empty:
            messagebox.showwarning("No Selection",
                                    "Please select at least one experiment or sample.",
                                    parent=self)
            return

        gse_list = self._commit_selection_to_app(sel_df)
        total_samples = len(sel_df)
        mode = (self.selection_mode.get()
                if hasattr(self, 'selection_mode') else 'experiment')

        self.app.enqueue_log(
            f"[Step 1.5] OK Saved {len(gse_list)} experiment(s) "
            f"({total_samples:,} samples, mode={mode}) → LLM extraction")

        # Populate Step 1.5 listbox
        try:
            self.app.gse_listbox.delete(0, tk.END)
            for gse_id in gse_list:
                count = len(sel_df[sel_df['series_id'] == gse_id]) if 'series_id' in sel_df.columns else 0
                desc = self.app.step1_gse_descriptions.get(gse_id, "")
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                kws = self.app.step1_gse_keywords.get(gse_id, [])
                kw_str = ', '.join(kws[:3]) or 'N/A'
                self.app.gse_listbox.insert(
                    tk.END,
                    f"{gse_id} ({count:,} samples) - {desc} | Keywords: {kw_str}")
            self.app.gse_listbox.select_set(0, tk.END)
            self.app.gse_frame.pack(fill=tk.X, padx=5, pady=5,
                                     after=self.app.step1_frame)
            self.app.step2_status_label.config(
                text=f"OK Ready: {len(gse_list)} experiment(s) ({total_samples:,} samples)",
                foreground="green")
            self.app._set_step_status(self.app.step1_frame, self.app._step1_title, "done")
            self.app._set_step_status(self.app.gse_frame, self.app._step15_title, "done")
        except Exception as exc:
            # The selection is committed either way, but if the hand-off panel
            # did not get built the user must not be told to "proceed" to a
            # step that is not on screen.
            self.app.enqueue_log(f"[Step 1.5] Hand-off panel failed: {exc}")
            messagebox.showerror(
                "Selection saved, hand-off incomplete",
                f"Your {len(gse_list)} experiment(s) were saved, but the LLM "
                f"extraction panel could not be prepared:\n\n{exc}\n\n"
                f"Reopen the main window's extraction step before continuing.",
                parent=self)
            self.destroy()
            return

        messagebox.showinfo(
            "Ready for LLM Extraction",
            f"Selected {len(gse_list)} experiment(s) • {total_samples:,} samples.\n\n"
            f"Proceed to LLM Label Extraction in the main window.",
            parent=self)
        self.destroy()

    def _download_data(self):
        """Open a dialog to download metadata / expression data for selection."""
        sel_df = self._collect_selection_df()
        if sel_df.empty:
            messagebox.showwarning("No Selection",
                                    "Please select at least one experiment or sample first.",
                                    parent=self)
            return

        dlg = tk.Toplevel(self)
        ensure_theme(dlg)
        style_window(dlg)
        dlg.title("Download Data")
        dlg.transient(self)
        dlg.geometry("460x320")

        gse_list = sorted(sel_df['series_id'].dropna().astype(str).unique()) \
            if 'series_id' in sel_df.columns else []

        ttk.Label(dlg,
                  text=f"Download for {len(gse_list)} experiment(s)  •  "
                       f"{len(sel_df):,} sample(s)",
                  foreground=AERO["accent_dark"],
                  font=('Segoe UI', 11, 'bold')
                  ).pack(anchor='w', padx=14, pady=(14, 6))

        opt_meta = tk.BooleanVar(value=True)
        opt_expr = tk.BooleanVar(value=False)
        opt_gpl  = tk.BooleanVar(value=False)

        body = ttk.Frame(dlg)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)
        ttk.Checkbutton(body,
                         text="Sample metadata (CSV, with matched keywords)",
                         variable=opt_meta).pack(anchor='w', pady=3)
        ttk.Checkbutton(body,
                         text="Gene expression matrix (via GEO series / platform downloader)",
                         variable=opt_expr).pack(anchor='w', pady=3)
        ttk.Checkbutton(body,
                         text="Platform annotation (GPL)",
                         variable=opt_gpl).pack(anchor='w', pady=3)

        ttk.Separator(dlg).pack(fill=tk.X, padx=14, pady=8)

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, padx=14, pady=(0, 12))

        def _do_download():
            picks = []
            if opt_meta.get(): picks.append('meta')
            if opt_expr.get(): picks.append('expr')
            if opt_gpl.get():  picks.append('gpl')
            if not picks:
                messagebox.showwarning("Nothing selected",
                                        "Please tick at least one data type.",
                                        parent=dlg)
                return
            dlg.destroy()
            self._run_downloads(sel_df, gse_list, picks)

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Download",
                   command=_do_download,
                   style="Primary.TButton").pack(side=tk.RIGHT, padx=4)

    def _run_downloads(self, sel_df, gse_list, picks):
        """Execute the download picks: metadata CSV + optional expression / GPL."""
        import os
        from tkinter import filedialog

        if 'meta' in picks:
            out = filedialog.asksaveasfilename(
                parent=self,
                title="Save sample metadata",
                defaultextension=".csv",
                initialfile=f"step1_selection_{len(gse_list)}gse_{len(sel_df)}gsm.csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if out:
                try:
                    sel_df.to_csv(out, index=False)
                    self.app.enqueue_log(
                        f"[Download] OK Metadata written: {out}  "
                        f"({len(sel_df):,} rows)")
                except Exception as exc:
                    messagebox.showerror("Save failed",
                                          f"Could not write CSV:\n{exc}",
                                          parent=self)

        if 'expr' in picks:
            # Stage selection for app, then trigger expression downloader.
            self._commit_selection_to_app(sel_df)
            try:
                self.app.enqueue_log(
                    f"[Download] → Launching expression downloader for "
                    f"{len(gse_list)} experiment(s)…")
                self.app._download_selected_expression()
            except AttributeError:
                messagebox.showinfo(
                    "Expression download",
                    "Selection staged. Use the main window's "
                    "'Download Expression' / 'Download Platform' controls "
                    "to fetch matrices.",
                    parent=self)

        if 'gpl' in picks:
            try:
                self._commit_selection_to_app(sel_df)
                self.app._open_gpl_downloader_window()
                self.app.enqueue_log(
                    "[Download] → Opened platform (GPL) downloader window.")
            except AttributeError:
                messagebox.showinfo(
                    "Platform download",
                    "Use the main window's 'Download Platform' button.",
                    parent=self)

class LabelingThread(threading.Thread):
    """Thread that runs LLM extraction for a batch of samples."""
    def __init__(self, input_dataframe, ai_agent, gui_log_func, on_finish,
                 fields=None, custom_fields=None, on_progress=None, gui_ref=None,
                 out_dir=None, metadata_columns=None):
        super().__init__(daemon=True)
        self.input_df = input_dataframe
        self.agent = ai_agent
        self.log = gui_log_func or print
        self.on_finish = on_finish
        self.on_progress = on_progress
        self.fields = fields  # list of field names to extract
        self.custom_fields = custom_fields  # list of {'name': ..., 'prompt': ...}
        self.gui_ref = gui_ref  # tkinter widget for after() scheduling
        self.out_dir = out_dir  # pipeline run directory (checkpoint lives here)
        self.metadata_columns = metadata_columns  # GEO columns the model reads
        self.result_df = None
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self.result_df = self.agent.process_samples(
                self.input_df,
                fields=self.fields,
                custom_fields=self.custom_fields,
                stop_flag_fn=lambda: self._stop_flag,
                progress_fn=self.on_progress,
                out_dir=self.out_dir,
                metadata_columns=self.metadata_columns,
            )
        except Exception as e:
            self.log(f"[LLM] Extraction thread error: {e}")
            self.result_df = pd.DataFrame()
        finally:
            if self.on_finish:
                # CRITICAL: Schedule on_finish on MAIN thread to avoid GUI freeze
                if self.gui_ref:
                    try:
                        self.gui_ref.after(0, self.on_finish)
                    except Exception:
                        try: self.on_finish()
                        except: pass
                else:
                    try: self.on_finish()
                    except: pass

class BioAI_Engine:
    @staticmethod
    def analyze_gene_distribution(expr):
        """Classify a gene's expression distribution.

        Delegates to ``analysis.bimodality.classify_gene_distribution``, which
        is the implementation the evaluation measures. This used to be a second
        copy of that algorithm, and the copy went stale: it kept the original
        KDE peak heuristic while the analysis layer gained Hartigan's dip test
        and GMM/BIC mode counting. The heuristic cannot see a mode that is
        separated but not sharply peaked -- a floored microarray gene, say, with
        a spike of absent calls under a broad expressed mode -- so those genes
        fell through to the log-likelihood race between five *unimodal*
        families, which has no goodness-of-fit gate and so always names one.
        Real GPL570 GAPDH came back "Uniform" that way, against a calibrated
        bootstrap-KS p of 1e-29.
        """
        from genevariate.core.analysis.bimodality import (
            classify_gene_distribution)
        return classify_gene_distribution(np.asarray(expr, dtype=np.float64))


class MultiLabelQueryDialog:
    """Reusable dialog for building multi-label compound queries.
    User selects: Column1=Value1 AND Column2=Value2 AND ...
    Returns: (query_name, mask_series) or None if cancelled.
    """
    @staticmethod
    def open(parent, df, title="Multi-Label Query Builder"):
        """Open dialog and return (name, boolean_mask) or None."""
        if df is None or df.empty:
            return None

        result = {'value': None}
        dlg = tk.Toplevel(parent)
        style_window(dlg)
        dlg.title(title)
        dlg.transient(parent)
        dlg.grab_set()

        ttk.Label(dlg, text="Build a compound query - only samples matching ALL criteria will be selected",
                  font=('Segoe UI', 10, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="Example: Tissue=Liver AND Condition=Cancer AND Age=50",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        # Available label columns (string/object columns only)
        label_cols = [c for c in df.columns
                      if c.upper() not in ('GSM', 'GENE', '_PLATFORM', 'SERIES_ID', 'GPL')
                      and df[c].dtype == 'object']

        # Query rows container
        rows_frame = ttk.Frame(dlg)
        rows_frame.pack(fill=tk.X, padx=15, pady=5)
        query_rows = []

        def _add_row():
            row_frame = ttk.Frame(rows_frame)
            row_frame.pack(fill=tk.X, pady=3)

            if query_rows:
                ttk.Label(row_frame, text="AND", font=('Segoe UI', 9, 'bold'),
                          foreground='#C62828').pack(side=tk.LEFT, padx=5)

            # Column selector
            col_var = tk.StringVar(value=label_cols[0] if label_cols else "")
            col_combo = ttk.Combobox(row_frame, textvariable=col_var,
                                      values=label_cols, state='readonly', width=15)
            col_combo.pack(side=tk.LEFT, padx=5)

            ttk.Label(row_frame, text="=", font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=3)

            # Value selector (populated when column changes)
            val_var = tk.StringVar()
            val_combo = ttk.Combobox(row_frame, textvariable=val_var, width=25)
            val_combo.pack(side=tk.LEFT, padx=5)

            def _on_col_change(event=None):
                col = col_var.get()
                if col and col in df.columns:
                    vals = sorted(df[col].fillna('N/A').astype(str).unique().tolist())
                    # Every value, not the first 200: the values past the cut
                    # were simply not selectable, and a column with hundreds of
                    # tissues is exactly the case this dialog is for.
                    val_combo['values'] = vals
                    if vals:
                        val_var.set(vals[0])

            col_combo.bind('<<ComboboxSelected>>', _on_col_change)
            _on_col_change()

            # Remove button
            def _remove():
                query_rows.remove((col_var, val_var, row_frame))
                row_frame.destroy()
                _update_preview()

            ttk.Button(row_frame, text="✕", command=_remove, width=3,
                       style="Destructive.TButton").pack(side=tk.LEFT, padx=5)

            query_rows.append((col_var, val_var, row_frame))
            _update_preview()

        # Preview label
        preview_frame = ttk.Frame(dlg)
        preview_frame.pack(fill=tk.X, padx=15, pady=5)
        preview_label = ttk.Label(preview_frame, text="", font=('Segoe UI', 9),
                                   foreground='#1565C0')
        preview_label.pack()

        def _update_preview(*args):
            mask = pd.Series(True, index=df.index)
            parts = []
            for col_var, val_var, _ in query_rows:
                col = col_var.get()
                val = val_var.get()
                if col and val and col in df.columns:
                    mask = mask & (df[col].fillna('N/A').astype(str) == val)
                    parts.append(f"{col}={val}")
            n = mask.sum()
            query_text = " AND ".join(parts) if parts else "(no criteria)"
            preview_label.config(text=f"Query: {query_text}  →  {n:,} samples match")

        # Bind value changes to update preview
        def _bind_updates():
            for col_var, val_var, _ in query_rows:
                col_var.trace_add('write', _update_preview)
                val_var.trace_add('write', _update_preview)

        # Add row button
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=15, pady=5)
        ttk.Button(btn_row, text="+ Add Criterion",
                   command=lambda: [_add_row(), _bind_updates()],
                   style="Add.TButton").pack(side=tk.LEFT)

        # Query name
        name_frame = ttk.Frame(dlg)
        name_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(name_frame, text="Group name:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)
        name_var = tk.StringVar(value="")
        ttk.Entry(name_frame, textvariable=name_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Label(name_frame, text="(leave empty = auto-fill with values)",
                  font=('Segoe UI', 8), foreground='gray').pack(side=tk.LEFT, padx=3)

        # OK / Cancel
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _ok():
            mask = pd.Series(True, index=df.index)
            parts = []
            for col_var, val_var, _ in query_rows:
                col = col_var.get()
                val = val_var.get()
                if col and val and col in df.columns:
                    mask = mask & (df[col].fillna('N/A').astype(str) == val)
                    parts.append(f"{col}={val}")
            if not parts:
                messagebox.showwarning("No Criteria", "Add at least one criterion.", parent=dlg)
                return
            n = mask.sum()
            if n == 0:
                messagebox.showwarning("No Matches", "No samples match this query.", parent=dlg)
                return
            name = name_var.get().strip()
            if not name:
                name = " + ".join(c.split("=")[1].strip() if "=" in c else c for c in parts)
            result['value'] = (name, mask)
            dlg.destroy()

        ttk.Button(btn_frame, text="Apply Query", command=_ok,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        # Add first row by default
        _add_row()
        _bind_updates()

        dlg.update_idletasks()
        w = max(600, dlg.winfo_reqwidth())
        h = dlg.winfo_reqheight()
        try:
            x = parent.winfo_x() + (parent.winfo_width() - w) // 2
            y = parent.winfo_y() + (parent.winfo_height() - h) // 2
            dlg.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")
        except: pass

        parent.wait_window(dlg)
        return result['value']


class Plotter:
    @staticmethod
    def get_optimal_bins(data, method='auto'):
        """The program's bin rule; see ``core.analysis.bimodality.optimal_bins``.

        The rule itself moved to the analysis layer so the assistant's chart
        builder, which must not import Tkinter, can draw its histogram with the
        same bins this window uses instead of a copy that drifts from it.
        """
        from genevariate.core.analysis.bimodality import optimal_bins
        return optimal_bins(data, method=method)
    @staticmethod
    def get_distinct_colors(n):
        # Central, scalable, colorblind-safe palette (tab10 -> tab20 -> husl),
        # consistent with every other plot in the app. The old gist_ncar path
        # produced near-neon rainbow colors plus a near-black/near-white entry
        # that vanished against the sky-tinted plot background.
        return viz_palette_for(max(1, n), "discrete")

_NOT_SPECIFIED_VALUES = {
    'Not Specified', 'not specified', 'Not specified',
    'N/A', 'n/a', 'NA', 'na', 'nan', 'NaN', 'None', 'none',
    'Unknown', 'unknown', 'UNKNOWN',
    '', 'Parse Error', 'parse error',
}


class ScrollableCanvasFrame(ttk.Frame):
    """Scrollable frame using Canvas."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # Plain tk.Canvas defaults to platform grey and the ttk theme cannot
        # reach it; without this the area below short content is a grey slab.
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                bg=AERO["bg_top"])
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
    def _on_canvas_configure(self, event):
        if self.scrollable_frame.winfo_reqwidth() < event.width:
            self.canvas.itemconfig(self.canvas_window, width=event.width)

# RegionAnalysisWindow -> moved to region_analysis.py (imported above)

class FlowFrame(ttk.Frame):
    """A frame that wraps its children onto as many rows as it needs.

    A packed row of widgets does not wrap and does not scroll: everything past
    the right edge of the window is simply unreachable. That is harmless for a
    fixed toolbar and a defect for anything built from the data - one
    checkbutton per loaded platform, one button per platform found on disk -
    because the number of children is whatever the user's data says it is.

    Children are created with this frame as their master and handed to
    :meth:`add`; they are laid out on a grid whose column count is recomputed
    from the frame's own width, so the last item is always on screen no matter
    how many there are or how narrow the window is.
    """

    def __init__(self, master, spacing=6, **kw):
        super().__init__(master, **kw)
        self._items = []
        self._spacing = spacing
        self._ncols = 0
        self._done = set()      # widths already laid out for the current items
        self.bind("<Configure>", lambda e: self._reflow())

    def add(self, widget):
        self._items.append(widget)
        self._ncols = 0                       # force a re-layout
        self._done.clear()
        self.after_idle(self._reflow)
        return widget

    def clear(self):
        for w in self._items:
            if w.winfo_exists():
                w.destroy()
        self._items = []
        self._ncols = 0
        self._done.clear()

    def _reflow(self):
        if not self.winfo_exists():
            return
        items = [w for w in self._items if w.winfo_exists()]
        width = self.winfo_width()
        # Laying the children out changes this frame's own size, which fires
        # <Configure> again; if two widths each imply the other's column count
        # the two chase each other and the idle queue never drains, hanging
        # every later update_idletasks in the program. The layout is decided by
        # the width alone, so a width already handled has nothing left to do.
        if width in self._done:
            return
        self._done.add(width)
        if not items or width <= 1:
            return
        # One uniform cell, as wide as the widest child: the columns then line
        # up, and no child can be clipped whatever its own label length.
        cell = max(w.winfo_reqwidth() for w in items) + self._spacing
        ncols = max(1, width // cell)
        if ncols == self._ncols:
            return                            # nothing moved; stop the loop
        self._ncols = ncols
        for i, w in enumerate(items):
            w.grid(row=i // ncols, column=i % ncols, sticky="w",
                   padx=(0, self._spacing), pady=2)


def _segmented(parent, options, command, active, expand=True):
    """A row of pill buttons acting as one exclusive choice.

    These were beveled ``tk.Button``s latched with ``relief=SUNKEN`` and a
    per-option colour, which is the one control style the rest of the app no
    longer uses. Built from the shared ttk styles they pick up the same pills
    as every other button in the program.

    ``options`` is a sequence of ``(text, value)``; returns ``{value: button}``.
    """
    btns = {}
    for text, val in options:
        b = ttk.Button(parent, text=text, style="Toggle.TButton",
                       command=lambda v=val: command(v))
        b.pack(side=tk.LEFT, padx=2,
               fill=tk.X if expand else None, expand=expand)
        btns[val] = b
    _paint_segment(btns, active)
    return btns


def _paint_segment(btns, active):
    """Latch one button of a segmented control on and the rest off."""
    for val, b in btns.items():
        b.configure(style="Primary.TButton" if val == active
                    else "Toggle.TButton")


class CompareDistributionsWindow(tk.Toplevel):
    def __init__(self, parent, app_ref, skip_autoload=False):
        super().__init__(parent)
        self.parent = parent
        self.app_ref = app_ref
        ensure_theme(self)
        style_window(self)
        self.title("Distribution Comparison")
        self.geometry("1600x1000")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1600x1000+{(_sw-1600)//2}+{(_sh-1000)//2}")
            self.minsize(600, 500)
        except Exception: pass
        
        # Data State
        self.user_defined_groups = {}
        self.full_dataset = pd.DataFrame() # Metadata
        self.grouping_column = None
        
        # Analysis Results
        self.analysis_results = {}     
        self.current_view_key = None   
        self.current_data_map = {}
        self.bg_data_map = {}
        self.results_cache = {}
        self.group_gsm_map = {} # Maps group label -> list of GSMs for current view
        
        # Visual State
        self.figs = {}
        self.canvases = {}
        self.toolbars = {}
        self.plot_refs = {}
        self.active_artists = {}

        # Linked Brushing State
        self._brushed_gsms = set()          # shared selection across all tabs
        self._scatter_data = {}             # key -> {ax, X_2d, gsms, labels, scatter_artists}
        self._brush_highlights = {}         # key -> list of highlight artists
        self._selectors = []                # keep references to prevent GC

        # Check Matplotlib
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            self.FigureCanvasTkAgg = FigureCanvasTkAgg
            self.NavigationToolbar2Tk = NavigationToolbar2Tk
            self.modules_loaded = True
        except ImportError:
            self.modules_loaded = False
            messagebox.showerror("Error", "Matplotlib is required.")

        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        if not skip_autoload:
            self.after(500, lambda: self.auto_load_subset_data())

    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Distribution Comparison", font=("Segoe UI", 16, "bold"), foreground=AERO["accent_dark"]).pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="Ready", foreground=AERO["muted"])
        self.status_label.pack(side=tk.RIGHT)

        # Splitter
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Controls
        control_pane = ttk.Frame(paned, width=350, padding=(0, 0, 5, 0))
        paned.add(control_pane, weight=0) 
        
        lf_data = labelframe(control_pane, text="1. Load Data")
        lf_data.pack(fill=tk.X, pady=5)
        ttk.Button(lf_data, text="Load Metadata (CSV)", command=self._load_labeled_file).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(lf_data, text="Clear Data", command=self._clear_user_data).pack(fill=tk.X, padx=5, pady=2)
        
        lf_group_info = ttk.Frame(control_pane)
        lf_group_info.pack(fill=tk.X, pady=2)
        ttk.Label(lf_group_info, text="Active Grouping:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_grouping = ttk.Label(lf_group_info, text="-", foreground=AERO["accent_dark"])
        self.lbl_grouping.pack(side=tk.LEFT, padx=5)
        
        lf_group = labelframe(control_pane, text="2. Select Groups")
        lf_group.pack(fill=tk.BOTH, expand=True, pady=5)
        self.loaded_files_listbox = tk.Listbox(lf_group, height=8, selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(lf_group, orient="vertical", command=self.loaded_files_listbox.yview)
        self.loaded_files_listbox.config(yscrollcommand=sb.set)
        self.loaded_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        lf_config = labelframe(control_pane, text="3. Analysis Parameters")
        lf_config.pack(fill=tk.X, pady=5)
        
        ttk.Label(lf_config, text="Target Gene(s):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=5)
        self.gene_entry = ttk.Entry(lf_config, font=("Segoe UI", 10)); self.gene_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(lf_config, text="Comparison Scope:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=5, pady=(5,0))
        self.comparison_mode = tk.StringVar(value="groups_only")
        scope_frame = ttk.Frame(lf_config)
        scope_frame.pack(fill=tk.X, padx=5, pady=3)
        self._cmp_scope_btns = _segmented(
            scope_frame,
            [("Groups Only", "groups_only"), ("vs Gene", "vs_gene"),
             ("vs Platform", "vs_platform")],
            self._set_cmp_scope, "groups_only")


        ttk.Label(lf_config, text="Platform (for BG):").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.platform_vars = {}
        plat_frame = ttk.Frame(lf_config)
        plat_frame.pack(fill=tk.X, padx=5)
        if hasattr(self.app_ref, 'gpl_datasets'):
            for p in sorted(self.app_ref.gpl_datasets.keys()):
                v = tk.BooleanVar()
                ttk.Checkbutton(plat_frame, text=p, variable=v).pack(anchor=tk.W)
                self.platform_vars[p] = v

        ttk.Button(control_pane, text="Run Analysis", command=self._run_analysis,
                   style="Action.TButton").pack(fill=tk.X, pady=10)

        ttk.Button(control_pane, text="Multi-Label Query",
                   command=self._add_query_group,
                   style="Secondary.TButton").pack(fill=tk.X, pady=(0, 5))
        
        # Active Analysis View - scrollable
        nav_outer = labelframe(control_pane, text="Active Analysis View")
        nav_outer.pack(fill=tk.BOTH, expand=True, pady=5)
        nav_canvas = tk.Canvas(nav_outer, height=150, highlightthickness=0)
        nav_scrollbar = ttk.Scrollbar(nav_outer, orient="vertical", command=nav_canvas.yview)
        self.sub_nav_frame = ttk.Frame(nav_canvas)
        self.sub_nav_frame.bind("<Configure>",
            lambda e: nav_canvas.configure(scrollregion=nav_canvas.bbox("all")))
        nav_canvas.create_window((0, 0), window=self.sub_nav_frame, anchor="nw")
        nav_canvas.configure(yscrollcommand=nav_scrollbar.set)
        nav_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nav_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Output Tabs
        self.notebook = ttk.Notebook(paned)
        paned.add(self.notebook, weight=3)
        
        # Tab 0: Data - GSE Experiment Browser
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Data & Grouping")

        # Summary bar
        self._data_summary = ttk.Label(self.tab_data, text="No data loaded",
                                        font=('Segoe UI', 10), foreground='#666')
        self._data_summary.pack(fill=tk.X, padx=5, pady=(5, 2))

        # GSE treeview (the main content)
        gse_tv_frame = ttk.Frame(self.tab_data)
        gse_tv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        gse_cols = ("GSE", "Samples", "Platform", "Top Condition", "Top Tissue")
        self.gse_tree = ttk.Treeview(gse_tv_frame, columns=gse_cols, show="headings", height=25)
        gse_vsb = ttk.Scrollbar(gse_tv_frame, orient="vertical", command=self.gse_tree.yview)
        gse_hsb = ttk.Scrollbar(gse_tv_frame, orient="horizontal", command=self.gse_tree.xview)
        self.gse_tree.configure(yscrollcommand=gse_vsb.set, xscrollcommand=gse_hsb.set)

        gse_tv_frame.grid_rowconfigure(0, weight=1)
        gse_tv_frame.grid_columnconfigure(0, weight=1)
        self.gse_tree.grid(row=0, column=0, sticky="nsew")
        gse_vsb.grid(row=0, column=1, sticky="ns")
        gse_hsb.grid(row=1, column=0, sticky="ew")

        self.gse_tree.heading("GSE", text="GSE Experiment")
        self.gse_tree.heading("Samples", text="Samples")
        self.gse_tree.heading("Platform", text="Platform")
        self.gse_tree.heading("Top Condition", text="Top Condition")
        self.gse_tree.heading("Top Tissue", text="Top Tissue")
        self.gse_tree.column("GSE", width=130)
        self.gse_tree.column("Samples", width=75, anchor='center')
        self.gse_tree.column("Platform", width=100)
        self.gse_tree.column("Top Condition", width=220)
        self.gse_tree.column("Top Tissue", width=180)

        self.gse_tree.bind("<Double-1>", self._on_gse_tree_dblclick)

        ttk.Label(self.tab_data,
                  text="Double-click any experiment to view all its samples and labels",
                  font=('Segoe UI', 8, 'italic'), foreground='#888').pack(pady=(0, 5))

        # Store data for click handler
        self._gse_data = {}  # {gse_id: DataFrame subset}

        # Tab 1: Distributions
        self.tab_dist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dist, text="Distributions & Stats")
        dist_pane = ttk.PanedWindow(self.tab_dist, orient=tk.VERTICAL)
        dist_pane.pack(fill=tk.BOTH, expand=True)
        
        self.dist_plot_container = ttk.Frame(dist_pane)
        dist_pane.add(self.dist_plot_container, weight=3)
        
        dist_ctrl_frame = ttk.Frame(self.dist_plot_container)
        dist_ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        ttk.Label(dist_ctrl_frame, text="Plot:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        self._dist_mode = tk.StringVar(value="both")
        self._dist_mode_btns = _segmented(
            dist_ctrl_frame,
            [("Density", "density"), ("Rug", "rug"), ("Both", "both")],
            self._set_dist_mode, "both", expand=False)

        # ── Color palette selector ──
        ttk.Separator(dist_ctrl_frame, orient='vertical').pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        ttk.Label(dist_ctrl_frame, text="Palette:",
                  font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 4))
        self._palette_var = tk.StringVar(value="husl")
        palette_combo = ttk.Combobox(
            dist_ctrl_frame, textvariable=self._palette_var, state='readonly',
            values=["husl", "tab10", "Set2", "Paired",
                    "colorblind", "dark", "pastel", "bright"],
            width=10)
        palette_combo.pack(side=tk.LEFT, padx=2)
        palette_combo.bind('<<ComboboxSelected>>',
                           lambda e: self._replot_distributions())

        self.dist_scroll_frame = ScrollableCanvasFrame(self.dist_plot_container)
        self.dist_scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.stats_frame = ttk.Frame(dist_pane)
        dist_pane.add(self.stats_frame, weight=1)
        self.stats_tree = ttk.Treeview(self.stats_frame, columns=("A", "B", "Z", "p", "Sig"), show="headings")
        for c in self.stats_tree["columns"]: self.stats_tree.heading(c, text=c)
        sb_stats = ttk.Scrollbar(self.stats_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=sb_stats.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_stats.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tab 2: Matrix
        self.tab_matrix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_matrix, text="Distance Matrix")
        matrix_ctrl = ttk.Frame(self.tab_matrix)
        matrix_ctrl.pack(fill=tk.X, padx=5, pady=5)
        self.metric_var = tk.StringVar(value="Wasserstein")
        ttk.Label(matrix_ctrl, text="Metric:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(2, 4))
        self.metric_combo = ttk.Combobox(matrix_ctrl, textvariable=self.metric_var, values=["Wasserstein", "Euclidean", "Jensen-Shannon"], state="readonly", font=('Segoe UI', 10))
        self.metric_combo.pack(side=tk.LEFT)
        
        ttk.Label(matrix_ctrl, text="  Reference:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(10, 4))
        self.dist_ref_var = tk.StringVar(value="pairwise")
        self._ref_btns = _segmented(
            matrix_ctrl,
            [("Pairwise", "pairwise"), ("Gene Mean", "gene_mean"),
             ("Platform Mean", "platform_mean"), ("Peaks (Mode)", "peaks")],
            self._set_ref_mode, "pairwise", expand=False)

        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self._calculate_matrix())
        
        self.matrix_scroll_frame = ScrollableCanvasFrame(self.tab_matrix)
        self.matrix_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Separation
        self.tab_sep = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sep, text="Class Separation")
        self.sep_scroll_frame = ScrollableCanvasFrame(self.tab_sep)
        self.sep_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Dimensionality Reduction (PCA / t-SNE)
        self.tab_dimred = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dimred, text="PCA / t-SNE")

        dimred_ctrl = ttk.Frame(self.tab_dimred)
        dimred_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(dimred_ctrl, text="Method:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=4)
        self.dimred_method = tk.StringVar(value="PCA")
        self._dimred_btns = _segmented(
            dimred_ctrl, [("PCA", "PCA"), ("t-SNE", "t-SNE")],
            self._set_dimred, "PCA", expand=False)
        ttk.Label(dimred_ctrl, text="Perplexity:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(15, 2))
        self.tsne_perplexity = tk.IntVar(value=30)
        ttk.Entry(dimred_ctrl, textvariable=self.tsne_perplexity, width=5).pack(side=tk.LEFT)
        ttk.Button(dimred_ctrl, text="Run", command=self._run_dimred,
                   style="ToolGreen.TButton").pack(side=tk.LEFT, padx=8)

        self.dimred_scroll_frame = ScrollableCanvasFrame(self.tab_dimred)
        self.dimred_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Tab 5: Clustering (DPC / K-means / DBSCAN)
        self.tab_cluster = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cluster, text="Clustering")

        cluster_ctrl = ttk.Frame(self.tab_cluster)
        cluster_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(cluster_ctrl, text="Method:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=4)
        self.cluster_method = tk.StringVar(value="K-Means")
        self._cluster_btns = _segmented(
            cluster_ctrl,
            [("K-Means", "K-Means"), ("DBSCAN", "DBSCAN"), ("DPC", "DPC")],
            self._set_cluster_method, "K-Means", expand=False)

        # Parameters row
        param_frame = ttk.Frame(self.tab_cluster)
        param_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(param_frame, text="K:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(5, 2))
        self.kmeans_k = tk.IntVar(value=3)
        ttk.Entry(param_frame, textvariable=self.kmeans_k, width=4).pack(side=tk.LEFT)
        ttk.Label(param_frame, text="(K-Means)", font=('Segoe UI', 8, 'italic'),
                  foreground='#888').pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(param_frame, text="eps:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(5, 2))
        self.dbscan_eps = tk.DoubleVar(value=0.5)
        ttk.Entry(param_frame, textvariable=self.dbscan_eps, width=5).pack(side=tk.LEFT)
        ttk.Label(param_frame, text="min_samples:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(10, 2))
        self.dbscan_min_samples = tk.IntVar(value=5)
        ttk.Entry(param_frame, textvariable=self.dbscan_min_samples, width=4).pack(side=tk.LEFT)
        ttk.Label(param_frame, text="(DBSCAN)", font=('Segoe UI', 8, 'italic'),
                  foreground='#888').pack(side=tk.LEFT, padx=2)
        ttk.Button(param_frame, text="Run", command=self._run_clustering,
                   style="ToolGreen.TButton").pack(side=tk.RIGHT, padx=8)

        self.cluster_scroll_frame = ScrollableCanvasFrame(self.tab_cluster)
        self.cluster_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.btn_report = ttk.Button(main_frame, text="Generate Report (Folder)", command=self._generate_report, state=tk.DISABLED)
        self.btn_report.pack(side=tk.BOTTOM, anchor=tk.E, pady=5)

    def _set_ref_mode(self, val):
        """Toggle matrix reference mode with button appearance."""
        self.dist_ref_var.set(val)
        _paint_segment(self._ref_btns, val)
        self._calculate_matrix()

    def _set_dimred(self, val):
        """Toggle dimensionality reduction method."""
        self.dimred_method.set(val)
        _paint_segment(self._dimred_btns, val)

    def _set_cluster_method(self, val):
        """Toggle clustering method."""
        self.cluster_method.set(val)
        _paint_segment(self._cluster_btns, val)

    def _add_query_group(self):
        """Open Multi-Label Query Builder and add matching samples as a new group."""
        if self.full_dataset.empty:
            messagebox.showinfo("No Data", "Load data first (labels or metadata).", parent=self)
            return
        result = MultiLabelQueryDialog.open(self, self.full_dataset,
                                             title="Multi-Label Query - Create Custom Group")
        if result is None:
            return
        name, mask = result
        gsms = self.full_dataset.loc[mask, 'GSM'].tolist() if 'GSM' in self.full_dataset.columns else []
        if not gsms:
            return

        # Determine platform for these GSMs
        best_plat = "Unknown"
        if hasattr(self.app_ref, 'gpl_datasets'):
            for p, pdf in self.app_ref.gpl_datasets.items():
                if 'GSM' in pdf.columns:
                    overlap = len(set(gsms) & set(pdf['GSM'].astype(str).str.upper()))
                    if overlap > 0:
                        best_plat = p
                        break

        label = f"Q: {name} (n={len(gsms):,})"
        self.user_defined_groups[label] = {
            'gsms': gsms,
            'platform': best_plat,
            'raw_val': name,
        }
        self.loaded_files_listbox.insert(tk.END, label)
        self.loaded_files_listbox.select_set(tk.END)
        self.status_label.config(text=f"Added query group: {name} ({len(gsms):,} samples)")

    def _set_cmp_scope(self, val):
        """Toggle comparison scope with button appearance."""
        self.comparison_mode.set(val)
        _paint_segment(self._cmp_scope_btns, val)

    def _set_dist_mode(self, val):
        """Toggle distribution plot mode with button appearance."""
        self._dist_mode.set(val)
        _paint_segment(self._dist_mode_btns, val)
        # Refresh if analysis has been run
        if self.current_view_key:
            try:
                self._toggle_visuals()
            except Exception:
                pass

    def _load_labeled_file(self):
        filepaths = filedialog.askopenfilenames(
            filetypes=[("CSV", "*.csv"), ("GZip", "*.csv.gz"),
                       ("Text", "*.txt"), ("All files", "*.*")])
        if not filepaths:
            return
        fp = filepaths[0]
        try:
            # ── Read file (handle CSV, GZ, and plain text with GSM list) ──
            if fp.endswith('.txt'):
                with open(fp) as f:
                    lines = [l.strip() for l in f if l.strip()]
                # Check if lines look like GSM IDs
                if lines and lines[0].upper().startswith('GSM'):
                    df = pd.DataFrame({'GSM': lines})
                else:
                    # Try treating first line as header
                    df = pd.read_csv(fp, sep='\t' if '\t' in open(fp).readline() else ',')
            else:
                df = pd.read_csv(fp, compression='gzip' if fp.endswith('.gz') else None,
                                 low_memory=False)

            # ── Detect GSM column ──
            cols_map = {c.upper(): c for c in df.columns}
            gsm_key = cols_map.get('GSM') or cols_map.get('ID') or cols_map.get('SAMPLE')
            if not gsm_key:
                for c in df.columns:
                    if str(c).upper().startswith("GSM"):
                        gsm_key = c
                        break
            # Single-column file: if first column values look like GSM IDs
            if not gsm_key and len(df.columns) == 1:
                first_col = df.columns[0]
                sample_vals = df[first_col].astype(str).str.upper()
                if sample_vals.str.startswith('GSM').mean() > 0.5:
                    gsm_key = first_col

            if gsm_key:
                df.rename(columns={gsm_key: 'GSM'}, inplace=True)
            else:
                messagebox.showerror("Error",
                    "Could not detect a 'GSM' column.\n\n"
                    "The file should contain a column named 'GSM', 'ID', or 'Sample',\n"
                    "or be a text file with one GSM ID per line.",
                    parent=self)
                return

            df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()
            # Remove rows that aren't valid GSM IDs. A sample identifier is also
            # valid when a loaded platform already carries it: a pseudo-bulk
            # group from the single-cell browser is named for its donor and cell
            # type, not GSMnnnn, and dropping those rows silently discarded the
            # whole single-cell arm of a cross-platform comparison.
            known = set()
            for p_df in getattr(self.app_ref, 'gpl_datasets', {}).values():
                if 'GSM' in p_df.columns:
                    known |= set(p_df['GSM'].astype(str).str.upper())
            valid_mask = (df['GSM'].str.match(r'^GSM\d+$', na=False)
                          | df['GSM'].isin(known))
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                df = df[valid_mask].reset_index(drop=True)
                self.app_ref.enqueue_log(f"[Load] Removed {n_invalid} rows with invalid GSM IDs")

            if df.empty:
                messagebox.showwarning("Empty", "No valid GSM IDs found in the file.", parent=self)
                return

            # ── Check if series_id and gpl are missing → resolve from GEOmetadb ──
            has_series = 'series_id' in df.columns and df['series_id'].notna().sum() > 0
            has_gpl = any(c.lower() in ('gpl', 'platform', '_platform')
                         for c in df.columns) and True
            # More precise check for gpl
            gpl_col = None
            for c in df.columns:
                if c.lower() in ('gpl', 'platform', '_platform'):
                    gpl_col = c
                    break
            has_gpl = gpl_col is not None and df[gpl_col].notna().sum() > 0 if gpl_col else False

            if not has_series or not has_gpl:
                # Need to resolve GSM metadata from GEOmetadb
                df = self._resolve_gsm_metadata(df, resolve_series=not has_series,
                                                 resolve_gpl=not has_gpl)
                if df is None:
                    return  # user cancelled or db not available

            self.full_dataset = df
            self._refresh_data_table()

            # ── Column selection dialog ──
            try:
                col_dialog = SelectColumnsDialog(self, df.columns.tolist(), Path(fp).name)
                if col_dialog.result:
                    self.grouping_column = col_dialog.result['label_cols'][0]
                    self.lbl_grouping.config(text=self.grouping_column)
                    self._update_group_list()
            except NameError:
                pass

            self.status_label.config(text=f"Loaded {len(df)} samples")
            self.app_ref.enqueue_log(f"[Load] Loaded {len(df):,} samples from {Path(fp).name}")

        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)

    def _resolve_gsm_metadata(self, df, resolve_series=True, resolve_gpl=True):
        """Look up GSM metadata (series_id, gpl, title, etc.) from GEOmetadb.

        Handles:
          - User provides only GSM IDs (no other columns)
          - User provides GSMs + some labels but no series_id/gpl
          - GSMs from multiple platforms (shuffled, mixed)

        Returns enriched DataFrame or None if cancelled.
        """
        if not self.gds_conn:
            messagebox.showwarning(
                "GEOmetadb Required",
                "GEOmetadb is not loaded. Cannot resolve GSM metadata.\n\n"
                "Please load GEOmetadb first (it's needed to look up\n"
                "platform and experiment info for your GSM IDs).",
                parent=self)
            return None

        gsm_list = df['GSM'].unique().tolist()
        total = len(gsm_list)
        self.app_ref.enqueue_log(f"[Load] Resolving metadata for {total:,} GSMs from GEOmetadb...")

        # Query GEOmetadb in chunks (SQLite has variable limit)
        CHUNK = 500
        meta_rows = []
        for i in range(0, total, CHUNK):
            chunk = gsm_list[i:i+CHUNK]
            placeholders = ','.join(['?'] * len(chunk))
            try:
                query = (f"SELECT gsm, series_id, gpl, title, source_name_ch1, "
                         f"characteristics_ch1, description, organism_ch1 "
                         f"FROM gsm WHERE UPPER(gsm) IN ({placeholders})")
                rows = self.gds_conn.execute(query, [g.upper() for g in chunk]).fetchall()
                meta_rows.extend(rows)
            except Exception as e:
                self.app_ref.enqueue_log(f"[Load] DB query error: {e}")

        if not meta_rows:
            messagebox.showwarning(
                "No Matches",
                f"None of the {total:,} GSM IDs were found in GEOmetadb.\n\n"
                f"Possible reasons:\n"
                f"  - GSMs are from a newer dataset not yet in your GEOmetadb\n"
                f"  - GSM IDs are malformed\n"
                f"  - GEOmetadb file is outdated\n\n"
                f"The file will be loaded as-is without metadata enrichment.",
                parent=self)
            return df

        meta_df = pd.DataFrame(meta_rows, columns=[
            'GSM', 'series_id', 'gpl', 'title', 'source_name_ch1',
            'characteristics_ch1', 'description', 'organism_ch1'])
        meta_df['GSM'] = meta_df['GSM'].astype(str).str.strip().str.upper()

        # Handle duplicates (a GSM can appear in multiple GSE series)
        # Keep the first occurrence per GSM
        meta_df = meta_df.drop_duplicates(subset='GSM', keep='first')

        # Merge: keep user's existing columns, add missing ones from GEOmetadb
        matched = len(set(df['GSM']) & set(meta_df['GSM']))
        unmatched = total - matched

        # Merge metadata into df
        merge_cols = []
        if resolve_series and 'series_id' not in df.columns:
            merge_cols.append('series_id')
        if resolve_gpl:
            if 'gpl' not in df.columns and '_platform' not in df.columns:
                merge_cols.append('gpl')
        # Always add metadata columns if user only had GSMs
        user_cols = set(df.columns) - {'GSM'}
        for mc in ['title', 'source_name_ch1', 'characteristics_ch1',
                    'description', 'organism_ch1']:
            if mc not in df.columns:
                merge_cols.append(mc)

        if merge_cols:
            meta_subset = meta_df[['GSM'] + merge_cols].copy()
            df = df.merge(meta_subset, on='GSM', how='left')

        # Platform breakdown
        if 'gpl' in df.columns:
            platform_counts = df['gpl'].value_counts()
            n_platforms = len(platform_counts)
        else:
            n_platforms = 0
            platform_counts = pd.Series(dtype=int)

        # Build summary message
        summary_lines = [
            f"Resolved metadata for {matched:,} of {total:,} GSMs.",
        ]
        if unmatched > 0:
            summary_lines.append(f"{unmatched:,} GSMs not found in GEOmetadb.")
        if n_platforms > 0:
            summary_lines.append(f"\nPlatform breakdown ({n_platforms} platforms):")
            for gpl, cnt in platform_counts.head(10).items():
                summary_lines.append(f"  {gpl}: {cnt:,} samples")
            if n_platforms > 10:
                summary_lines.append(f"  ... and {n_platforms - 10} more")
        if 'series_id' in df.columns:
            n_gse = df['series_id'].nunique()
            summary_lines.append(f"\n{n_gse:,} experiments (GSE) detected.")
        if 'organism_ch1' in df.columns:
            orgs = df['organism_ch1'].value_counts().head(3)
            org_str = ", ".join(f"{o} ({c:,})" for o, c in orgs.items())
            summary_lines.append(f"Species: {org_str}")

        messagebox.showinfo(
            "GSM Metadata Resolved",
            "\n".join(summary_lines),
            parent=self)

        self.app_ref.enqueue_log(
            f"[Load] Resolved {matched:,}/{total:,} GSMs: "
            f"{n_platforms} platforms, "
            f"{df['series_id'].nunique() if 'series_id' in df.columns else '?'} experiments")

        # Offer to auto-load expression data for detected platforms
        if n_platforms > 0 and n_platforms <= 5:
            available = self._discover_available_platforms()
            loadable = [gpl for gpl in platform_counts.index if gpl.upper() in available]
            if loadable:
                load_msg = (
                    f"Expression data is available for {len(loadable)} of "
                    f"{n_platforms} detected platforms:\n"
                    + "\n".join(f"  {g}: {platform_counts.get(g,0):,} samples"
                               for g in loadable)
                    + "\n\nLoad expression data now?\n"
                    "(Required for Gene Explorer and Compare Distributions)")
                if messagebox.askyesno("Load Expression Data?", load_msg, parent=self):
                    for gpl in loadable:
                        if gpl.upper() not in self.gpl_datasets:
                            fpath = available[gpl.upper()]
                            self.app_ref.enqueue_log(f"[Load] Auto-loading {gpl} expression data...")
                            self._load_gpl_data(gpl.upper(), fpath)

        return df

    def _refresh_data_table(self):
        """Populate the GSE experiment list from full_dataset."""
        self.gse_tree.delete(*self.gse_tree.get_children())
        self._gse_data = {}

        if self.full_dataset.empty:
            self._data_summary.config(text="No data loaded")
            return

        df = self.full_dataset
        # Remove Gene column if present (not useful for GSE list)
        label_cols = [c for c in df.columns
                      if c.upper() not in ('GENE', '_PLATFORM')
                      and c not in ('Gene', 'gene')]

        # Group by series_id
        if 'series_id' not in df.columns:
            # Fallback: group by platform or show summary
            self._data_summary.config(
                text=f"{len(df):,} samples loaded - no series_id column for experiment grouping")
            return

        gse_groups = df.groupby('series_id')
        n_gse = 0

        for gse_id, group in gse_groups:
            gse_id = str(gse_id).strip()
            if not gse_id or gse_id == 'nan':
                continue
            n_samples = len(group)

            # Platform
            plat = '?'
            if 'platform' in group.columns:
                plat = str(group['platform'].mode().iloc[0]) if not group['platform'].mode().empty else '?'
            elif '_platform' in group.columns:
                plat = str(group['_platform'].mode().iloc[0]) if not group['_platform'].mode().empty else '?'

            # Top condition
            top_cond = ""
            if 'Condition' in group.columns:
                vc = group['Condition'].fillna("N/A").astype(str).value_counts().head(3)
                top_cond = ", ".join(f"{v} ({n})" for v, n in vc.items())

            # Top tissue
            top_tissue = ""
            if 'Tissue' in group.columns:
                vt = group['Tissue'].fillna("N/A").astype(str).value_counts().head(2)
                top_tissue = ", ".join(f"{v} ({n})" for v, n in vt.items())

            self.gse_tree.insert("", tk.END, values=(gse_id, n_samples, plat, top_cond, top_tissue))
            self._gse_data[gse_id] = group[label_cols].copy()
            n_gse += 1

        self._data_summary.config(
            text=f"{len(df):,} samples across {n_gse:,} experiments - "
                 f"double-click to view samples")

    def _on_gse_tree_dblclick(self, event):
        """Handle double-click on GSE experiment - show all its GSMs."""
        item = self.gse_tree.focus()
        if not item:
            return
        vals = self.gse_tree.item(item, 'values')
        if not vals:
            return
        gse_id = vals[0]

        # Get data from stored GSE data or full dataset
        if gse_id in self._gse_data:
            subset = self._gse_data[gse_id]
        elif not self.full_dataset.empty and 'series_id' in self.full_dataset.columns:
            subset = self.full_dataset[self.full_dataset['series_id'].astype(str).str.strip() == gse_id]
        else:
            return

        if subset.empty:
            return

        # Remove Gene column and _platform
        cols = [c for c in subset.columns
                if c.upper() not in ('GENE', '_PLATFORM')
                and c not in ('Gene', 'gene')]
        subset = subset[cols]

        # Popup window
        top = tk.Toplevel(self)
        style_window(top)
        top.title(f"Experiment {gse_id} - {len(subset)} samples")
        top.geometry("1100x600")
        try:
            _sw, _sh = top.winfo_screenwidth(), top.winfo_screenheight()
            top.geometry(f"1100x600+{(_sw-1100)//2}+{(_sh-600)//2}")
        except: pass

        # Summary
        info_parts = [f"{gse_id}: {len(subset)} samples"]
        for col in ['Condition', 'Tissue', 'Treatment']:
            if col in subset.columns:
                vc = subset[col].fillna("N/A").astype(str).value_counts().head(5)
                info_parts.append(f"{col}: {', '.join(f'{v}({n})' for v, n in vc.items())}")
        ttk.Label(top, text="  |  ".join(info_parts), font=('Segoe UI', 9, 'bold'),
                  wraplength=1050).pack(fill=tk.X, padx=10, pady=(10, 5))

        # Treeview with all GSMs
        tv_frame = ttk.Frame(top)
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tree = ttk.Treeview(tv_frame, columns=cols, show="headings", height=20)
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tv_frame.grid_rowconfigure(0, weight=1)
        tv_frame.grid_columnconfigure(0, weight=1)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, minwidth=80)

        _ROW_CAP = 2000
        for _, row in subset.head(_ROW_CAP).iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, '')) for c in cols])
        if len(subset) > _ROW_CAP:
            note = [f"… {len(subset) - _ROW_CAP:,} more samples "
                    f"(table capped at {_ROW_CAP:,}; use Save for all)"] \
                   + [""] * (len(cols) - 1)
            tree.insert("", tk.END, values=note)

        # Buttons
        btn_frame = ttk.Frame(top, padding=5)
        btn_frame.pack(fill=tk.X)

        def _save():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                initialfile=f"{gse_id}_samples.csv", parent=top)
            if path:
                subset.to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved {len(subset)} samples to:\n{path}", parent=top)

        ttk.Button(btn_frame, text="Save to CSV", command=_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=top.destroy).pack(side=tk.RIGHT, padx=5)


    def _update_group_list(self):
            """
            Populates the 'Select Groups' listbox based on self.grouping_column.
            """
            self.user_defined_groups = {}
            self.loaded_files_listbox.delete(0, tk.END)
    
            # Safety Check
            if not self.grouping_column or self.full_dataset.empty:
                return
            
            if self.grouping_column not in self.full_dataset.columns:
                print(f"Error: Grouping column '{self.grouping_column}' not found in dataset.")
                return
    
            try:
                # Clean data for grouping
                self.full_dataset[self.grouping_column] = self.full_dataset[self.grouping_column].fillna("N/A").astype(str)
                unique_vals = sorted(self.full_dataset[self.grouping_column].unique())
    
                # Map loaded platforms for cross-referencing
                # (Matches samples in the imported file to loaded platforms in memory)
                loaded_platforms = {}
                if hasattr(self.app_ref, 'gpl_datasets') and self.app_ref.gpl_datasets:
                    for p_name, p_df in self.app_ref.gpl_datasets.items():
                        if 'GSM' in p_df.columns:
                            loaded_platforms[p_name] = set(p_df['GSM'].astype(str).str.upper())
    
                for val in unique_vals:
                    # Get GSMs for this specific group
                    group_gsms = self.full_dataset[self.full_dataset[self.grouping_column] == val]['GSM'].tolist()
                    
                    if not group_gsms:
                        continue
    
                    # Find which platform covers these GSMs best
                    best_platform = None
                    max_overlap = 0
                    
                    for p_name, p_gsms_set in loaded_platforms.items():
                        overlap = len(set(group_gsms).intersection(p_gsms_set))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_platform = p_name
    
                    # If no platform matches, label as 'Unknown' (prevents crash)
                    final_platform = best_platform if best_platform else "Unknown_Platform"
    
                    label = f"{val} (n={len(group_gsms)})"
                    
                    # Store data needed for analysis
                    self.user_defined_groups[label] = {
                        "gsms": group_gsms,
                        "platform": final_platform,
                        "raw_val": val
                    }
                    
                    self.loaded_files_listbox.insert(tk.END, label)
                    
            except Exception as e:
                print(f"Error updating group list: {e}")
                import traceback
                traceback.print_exc()

    def _run_analysis(self):
        if not self.modules_loaded: return
        self.analysis_results = {}
        self.current_view_key = None
        for w in self.sub_nav_frame.winfo_children(): w.destroy()
        self._clear_all_plots()
        # Close all matplotlib figures to prevent X11 resource exhaustion
        import matplotlib.pyplot as _plt
        _plt.close('all')

        if self.full_dataset.empty:
            # Everything above already cleared the sub-navigation and closed
            # the figures, so returning in silence here leaves the user
            # staring at a window that just went blank for no stated reason.
            self.status_label.config(text="No data loaded")
            messagebox.showinfo(
                "Nothing to analyse",
                "No dataset is loaded.\n\nLoad an expression file first, "
                "then run the analysis.", parent=self)
            return
        sel_idx = self.loaded_files_listbox.curselection()
        if not sel_idx: 
             self.loaded_files_listbox.select_set(0, tk.END) 
             selected_indices = self.loaded_files_listbox.curselection()
        else: selected_indices = sel_idx

        gene_input = self.gene_entry.get().strip().upper()
        genes = [g.strip() for g in gene_input.split(',') if g.strip()]
        mode = self.comparison_mode.get()
        sel_plats = [p for p, v in self.platform_vars.items() if v.get()]

        self.status_label.config(text="Processing...")
        self.update_idletasks()

        platforms_to_process = sel_plats if sel_plats else []
        if not platforms_to_process:
             inferred = set()
             for i in selected_indices:
                 plat = self.user_defined_groups[self.loaded_files_listbox.get(i)]['platform']
                 if plat: inferred.add(plat)
             platforms_to_process = sorted(list(inferred))

        run_configs = []
        for p in platforms_to_process:
            for g in genes: run_configs.append((f"[{p}] {g}", [p], [g]))
            if len(genes) > 1: 
                run_configs.append((f"[{p}] Collective", [p], genes))
                run_configs.append((f"[{p}] Inter-Gene", [p], genes))
            if not genes: run_configs.append((f"[{p}] Groups", [p], []))

        if len(platforms_to_process) > 1:
            for g in genes: run_configs.append((f"[ALL] {g}", platforms_to_process, [g]))
            if len(genes) > 1: 
                run_configs.append((f"[ALL] Collective", platforms_to_process, genes))
                run_configs.append((f"[ALL] Inter-Gene", platforms_to_process, genes))

        last_prefix = ""
        row_frame = None

        for label, p_list, g_list in run_configs:
            prefix = label.split(']')[0] + ']'
            if prefix != last_prefix:
                row_frame = ttk.Frame(self.sub_nav_frame)
                row_frame.pack(fill=tk.X, padx=2, pady=1)
                last_prefix = prefix
            
            d_map, bg_map, grp_map = self._process_data(selected_indices, p_list, g_list, mode, label)
            if d_map:
                self.analysis_results[label] = {"data": d_map, "bg": bg_map, "grp_map": grp_map}
                ttk.Button(row_frame, text=label, command=lambda k=label: self._switch_view(k)).pack(side=tk.LEFT, padx=2)

        if not self.analysis_results:
            messagebox.showinfo("Info", "No matching data found.")
            return

        self._switch_view(list(self.analysis_results.keys())[0])
        self.btn_report.config(state=tk.NORMAL)
        self.status_label.config(text="Done.")

    def _process_data(self, indices, platforms, genes, mode, label_type):
            """
            Loads data and distinct backgrounds.
            """
            data_map = {}
            bg_map = {}
            grp_gsm_map = {}
            is_inter_gene = "Inter-Gene" in label_type
    
            # 1. Load Active Groups
            for idx in indices:
                lbl = self.loaded_files_listbox.get(idx)
                info = self.user_defined_groups.get(lbl)
                if not info or info['platform'] not in platforms: continue
                
                df_plat = self.app_ref.gpl_datasets.get(info['platform'])
                gmap = self.app_ref.gpl_gene_mappings.get(info['platform'], {})
                subset = df_plat[df_plat['GSM'].isin(info['gsms'])]
                if subset.empty: continue
                
                if genes:
                    for g in genes:
                        col = gmap.get(g)
                        if col and col in subset.columns:
                            v = pd.to_numeric(subset[col], errors='coerce').dropna()
                            if not v.empty:
                                k = f"{info['raw_val']} | {g}" if (is_inter_gene or len(genes) > 1) else info['raw_val']
                                if k in data_map: 
                                    data_map[k] = pd.concat([data_map[k], v])
                                    grp_gsm_map[k].extend(subset.loc[v.index]['GSM'].tolist())
                                else: 
                                    data_map[k] = v
                                    grp_gsm_map[k] = subset.loc[v.index]['GSM'].tolist()
    
            # 2. Load Backgrounds (With Distinct Names)
            if mode != "groups_only":
                for p in platforms:
                    df_bg = self.app_ref.gpl_datasets.get(p)
                    if df_bg is None: continue
                    
                    # A. Whole Platform
                    if mode == "vs_platform":
                        cols = [c for c in df_bg.columns if pd.api.types.is_numeric_dtype(df_bg[c]) and c.upper() not in self.app_ref.METADATA_EXCLUSIONS]
                        if cols:
                            v = df_bg[cols].stack().dropna().sample(n=min(50000, len(df_bg)*100), random_state=1)
                            if not v.empty: bg_map[f"BG: Whole {p}"] = v # Distinct Key
                    
                    # B. Specific Genes
                    elif mode == "vs_gene" or is_inter_gene:
                        for g in genes:
                            c = self.app_ref.gpl_gene_mappings.get(p, {}).get(g)
                            if c and c in df_bg.columns:
                                v = pd.to_numeric(df_bg[c], errors='coerce').dropna()
                                if not v.empty: 
                                    # Distinct Key includes Gene AND Platform
                                    bg_map[f"BG: {g} ({p})"] = v 
    
            return data_map, bg_map, grp_gsm_map
    
    def _switch_view(self, key):
        res = self.analysis_results[key]
        self.current_data_map = res["data"]
        self.bg_data_map = res["bg"]
        self.group_gsm_map = res.get("grp_map", {})
        self.current_view_key = key
        self.status_label.config(text=f"Viewing: {key}")
        self._refresh_current_view()

    def _refresh_current_view(self, event=None):
        self._clear_all_plots()
        self._plot_distributions(self.current_view_key)
        self._calculate_matrix()
        self._plot_separation()
        # PCA/t-SNE and Clustering tabs run on-demand via their Run buttons


    def _replot_distributions(self, event=None):
        """Re-render distribution plot with current palette selection."""
        if self.current_view_key:
            # Clear only the distribution plot, not all tabs
            if "dist" in self.canvases:
                self.canvases["dist"].get_tk_widget().destroy()
                self.toolbars["dist"].destroy()
                plt.close(self.figs["dist"])
                del self.canvases["dist"], self.toolbars["dist"], self.figs["dist"]
            self._plot_distributions(self.current_view_key)

    def _value_axis_label(self):
        """Axis label for the plotted platform, generic when they disagree."""
        get = getattr(self.app_ref, 'platform_measurement_label', None)
        df = self.full_dataset
        if get is None or df is None or df.empty or '_platform' not in df.columns:
            return "expression"
        try:
            labels = {get(p) for p in df['_platform'].dropna().unique()}
        except Exception:
            return "expression"
        return labels.pop() if len(labels) == 1 else "expression"

    def _plot_distributions(self, title):
            """
            Plots distributions with distinct backgrounds, scaled visibility, and uncut legends.
            UPDATED: 
            1. Forces X-axis range to match actual gene expression data min/max.
            2. Normalizes N=1 dashed lines to never exceed the background density peak.
            3. Increases figure size to ensure scrollbars are used instead of clipping.
            """
            # 1. Setup Dynamic Figure Size
            n_groups = len(self.current_data_map)
            n_bgs = len(self.bg_data_map)
            total_items = n_groups + n_bgs
            
            # Calculate Height: Increased multiplier to ensure space for large legends
            # The scrollbar in the GUI will handle the larger size.
            calc_height = max(7, 5 + (total_items * 0.4))
            
            # Calculate Width: Base 12 + extra if we have many backgrounds
            calc_width = 12 + (n_bgs * 0.5) 
       
            fig = Figure(figsize=(calc_width, calc_height))
            ax = fig.subplots()
            
            # 2. Color Setup
            import seaborn as sns
            # Palette for active groups - user-selectable
            _pal_name = getattr(self, '_palette_var', None)
            _pal_name = _pal_name.get() if _pal_name else "husl"
            palette = sns.color_palette(_pal_name, n_groups)
            colors = {k: mcolors.to_hex(c) for k, c in zip(self.current_data_map.keys(), palette)}
            
            # Palette for backgrounds (Greys/Blues/Darks) - distinct for multiple BGs
            bg_palette = sns.color_palette("bone", n_bgs + 2) 
            for i, (k, _) in enumerate(self.bg_data_map.items()):
                colors[k] = mcolors.to_hex(bg_palette[i+1])
       
            self.plot_artists_current = {k: {'rugs':[], 'densities':[], 'main':[]} for k in colors.keys()}
            handles = []
       
            # --- PRE-CALCULATION: Range & Max Background Density ---
            max_group_density = 0
            max_bg_density = 0  # To cap the N=1 lines
            all_data_values = [] 
            
            # A. Analyze Backgrounds first to find the "Mode" (Max Density)
            for data in self.bg_data_map.values():
                valid_vals = data.dropna().tolist()
                all_data_values.extend(valid_vals)
                if data.nunique() > 1:
                    try:
                        kde = robust_kde(data)
                        # Scan a wide range to find the true peak
                        xs = np.linspace(data.min(), data.max(), 200)
                        ys = kde(xs)
                        max_bg_density = max(max_bg_density, ys.max())
                    except: pass
            
            # If no background density found (rare), default to a small value
            if max_bg_density == 0: max_bg_density = 0.5
    
            # B. Analyze Groups
            for data in self.current_data_map.values():
                valid_vals = data.dropna().tolist()
                all_data_values.extend(valid_vals)
                if data.nunique() > 1:
                    try:
                        kde = robust_kde(data)
                        xs = np.linspace(data.min(), data.max(), 100)
                        max_group_density = max(max_group_density, kde(xs).max())
                    except: pass
    
            if max_group_density == 0: max_group_density = 1.0 
       
            # 3. Plot Background Data (Scaled & Distinct)
            for lbl, data in self.bg_data_map.items():
                c = colors[lbl]
                style = '-' if "Whole" in lbl else '--' 
                
                if data.nunique() > 1:
                    try: 
                        kde = robust_kde(data)
                        xs = np.linspace(data.min(), data.max(), 200)
                        ys = kde(xs)
                        
                        # Scale background if it's too small compared to groups, 
                        # but keep it true to the N=1 lines reference.
                        # We plot it 'as is' mostly, unless it's tiny.
                        ax.plot(xs, ys, color=c, linestyle=style, linewidth=2.0, label=lbl, alpha=0.9, zorder=1)
                        ax.fill_between(xs, ys, color=c, alpha=0.1, zorder=0)
                    except: pass
                
                handles.append(mlines.Line2D([],[], color=c, linestyle=style, label=lbl, linewidth=2))
       
            # 4. Plot Active Groups
            for lbl, data in self.current_data_map.items():
                c = colors[lbl]
                n = len(data)
                is_const = data.nunique() <= 1
                l_txt = f"{lbl} (n={n})" + (" (const)" if is_const else "")
                
                # Rugs
                sns.rugplot(data, ax=ax, color=c, height=0.04, alpha=0.8, zorder=5)
                if ax.collections: self.plot_artists_current[lbl]['rugs'].append(ax.collections[-1])
                
                # Density / Vertical Line
                if not is_const:
                    try:
                        sns.kdeplot(data, ax=ax, color=c, fill=False, linewidth=2.5, zorder=6)
                        if ax.lines: self.plot_artists_current[lbl]['densities'].append(ax.lines[-1])
                    except: pass
                else:
                    # --- FIX: Plot Vertical Line scaled to Background Peak ---
                    # Use vlines to set height in DATA coordinates (matching density Y-axis)
                    # Height = max_bg_density (so it doesn't exceed background mode)
                    l_segs = ax.vlines(x=data.iloc[0], ymin=0, ymax=max_bg_density, 
                                       colors=c, linestyles=':', linewidth=3.0, zorder=6)
                    self.plot_artists_current[lbl]['densities'].append(l_segs)
                
                self.plot_artists_current[lbl]['main'] = self.plot_artists_current[lbl]['densities']
                handles.append(mpatches.Patch(color=c, label=l_txt))
       
            # 5. Finalize Layout & Legend
            ax.set_title(f"Distributions: {title}", fontsize=14, pad=20)
            ax.set_xlabel(self._value_axis_label())
            ax.set_ylim(bottom=0)
    
            # Explicitly Set X-Axis Range
            if all_data_values:
                x_min, x_max = min(all_data_values), max(all_data_values)
                x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            
            # Legend: Outside, anchored top-left
            leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), 
                            borderaxespad=0., fontsize='medium', frameon=True)
            
            # Adjust layout to explicitly reserve space for the legend
            fig.subplots_adjust(left=0.08, right=0.70, top=0.9, bottom=0.1)
            
            # Make legend interactive
            self._make_legend_interactive(fig, leg, colors, self.plot_artists_current, list(colors.keys()))
            
            # Embed
            self._embed_plot(fig, self.dist_scroll_frame.scrollable_frame, "dist")
            
            self._calc_stats_table()
    
    def _toggle_visuals(self):
        curr = "density" 
        for k, v in self.plot_artists_current.items():
            if v['rugs'] and v['rugs'][0].get_visible():
                curr = "both" if (v['densities'] and v['densities'][0].get_visible()) else "rugs"
                break
        nxt = "rugs" if curr == "both" else "both" if curr == "density" else "density"
        for k, v in self.plot_artists_current.items():
            show_r = nxt in ["rugs", "both"]
            show_d = nxt in ["density", "both"]
            for r in v['rugs']: r.set_visible(show_r)
            for d in v['densities']: d.set_visible(show_d)
        self.figs["dist"].canvas.draw_idle()

    def _calc_stats_table(self):
        self.stats_tree.delete(*self.stats_tree.get_children())
        keys = list(self.current_data_map.keys())
        stats_res = []
        if len(keys) >= 2:
            for k1, k2 in itertools.combinations(keys, 2):
                try:
                    s, p = ranksums(self.current_data_map[k1], self.current_data_map[k2])
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                    self.stats_tree.insert("", tk.END, values=(k1, k2, f"{s:.3f}", f"{p:.3e}", sig))
                    stats_res.append({"A":k1, "B":k2, "Z":s, "p":p})
                except: pass
        for bk, bv in self.bg_data_map.items():
            for k, v in self.current_data_map.items():
                try:
                    s, p = ranksums(v, bv)
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                    self.stats_tree.insert("", tk.END, values=(k, bk, f"{s:.3f}", f"{p:.3e}", sig))
                    stats_res.append({"A":k, "B":bk, "Z":s, "p":p})
                except: pass
        self.results_cache['stats'] = pd.DataFrame(stats_res)

    def _calculate_matrix(self, event=None):
            from scipy.spatial.distance import jensenshannon
            
            if not self.current_data_map: return
            
            # Clear previous widgets
            for w in self.matrix_scroll_frame.scrollable_frame.winfo_children(): w.destroy()
            
            # Get settings
            metric = self.metric_var.get()
            ref_mode = self.dist_ref_var.get()
            keys = list(self.current_data_map.keys())
            n = len(keys)
            
            # 1. Pre-calculate peaks (modes) for peak-based comparison
            peaks = {}
            for k, v in self.current_data_map.items():
                if v.nunique() > 1:
                    try:
                        kde = robust_kde(v)
                        xs = np.linspace(v.min(), v.max(), 500)
                        peaks[k] = xs[np.argmax(kde(xs))]
                    except: 
                        peaks[k] = v.median()
                else: 
                    peaks[k] = v.iloc[0]
    
            # 2. Calculate Matrix based on Reference Mode
            if ref_mode == "pairwise" or ref_mode == "peaks":
                mat = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        
                        # Mode A: Compare Peaks (Modes)
                        if ref_mode == "peaks": 
                            mat[i, j] = abs(peaks[keys[i]] - peaks[keys[j]])
                        
                        # Mode B: Compare Full Distributions (Pairwise)
                        else:
                            d1, d2 = self.current_data_map[keys[i]], self.current_data_map[keys[j]]
                            val = 0
                            
                            if "Wasserstein" in metric: 
                                val = wasserstein_distance(d1, d2)
                            elif "Euclidean" in metric: 
                                val = abs(d1.mean() - d2.mean())
                            elif "Jensen-Shannon" in metric:
                                # JS requires probability vectors of same length. 
                                # We must discretize using a common grid.
                                min_val = min(d1.min(), d2.min())
                                max_val = max(d1.max(), d2.max())
                                
                                # Create 50 bins over the shared range
                                bins = np.linspace(min_val, max_val, 50)
                                
                                # Get density histograms
                                p, _ = np.histogram(d1, bins=bins, density=True)
                                q, _ = np.histogram(d2, bins=bins, density=True)
                                
                                # Normalize to probability mass (sum to 1) for JS calculation
                                # Avoid division by zero with small epsilon
                                p = p / (p.sum() + 1e-10)
                                q = q / (q.sum() + 1e-10)
                                
                                val = jensenshannon(p, q)
                                
                            mat[i, j] = val
                
                df_mat = pd.DataFrame(mat, index=keys, columns=keys)
                
            else:
                # Mode C: Compare to a Single Reference Value (Mean)
                vals = []
                if ref_mode == "gene_mean": 
                    ref = pd.concat(self.current_data_map.values()).mean()
                elif ref_mode == "platform_mean" and self.bg_data_map: 
                    ref = pd.concat(self.bg_data_map.values()).mean()
                else: 
                    ref = 0
                    
                for k in keys:
                    if ref_mode == "peaks": 
                        vals.append(abs(peaks[k] - ref))
                    else: 
                        vals.append(abs(self.current_data_map[k].mean() - ref))
                
                df_mat = pd.DataFrame(vals, index=keys, columns=["Dist to Ref"])
    
            # 3. Store and Plot Results
            self.results_cache['matrix'] = df_mat
            
            # Dynamic figure size
            fig_width = max(8, n * 0.8)
            fig_height = max(7, n * 0.6)
            
            fig = Figure(figsize=(fig_width, fig_height))
            ax = fig.subplots()
            sns.heatmap(df_mat, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                        linewidths=0.6, linecolor="black")
            
            # Set title
            title_metric = "Peak Diff" if ref_mode == "peaks" else metric
            ax.set_title(f"Distance Matrix ({title_metric}) - Ref: {ref_mode}")
            
            fig.tight_layout()
            self._embed_plot(fig, self.matrix_scroll_frame.scrollable_frame, "matrix")
        
    def _plot_separation(self):
        for w in self.sep_scroll_frame.scrollable_frame.winfo_children(): w.destroy()

        sf = self.sep_scroll_frame.scrollable_frame

        n = len(self.current_data_map)
        if not n: return

        fig = Figure(figsize=(max(9, n*0.5), 6))
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0.18)

        keys = list(self.current_data_map.keys())
        colors = sns.color_palette("husl", len(keys))
        cdict = {k: mcolors.to_hex(c) for k, c in zip(keys, colors)}

        # Build scatter data for pick
        all_x = []
        all_y = []
        all_groups = []
        all_gsms = []

        for gi, k in enumerate(keys):
            vals = self.current_data_map[k]
            gsms = self.group_gsm_map.get(k, ['?'] * len(vals))
            jitter = np.random.uniform(-0.2, 0.2, len(vals))
            x_pos = gi + jitter
            y_pos = vals.values if hasattr(vals, 'values') else np.array(vals)

            ax.scatter(x_pos, y_pos, c=cdict[k], s=20, alpha=0.6,
                       edgecolors='none', zorder=2, picker=True, pickradius=5)

            all_x.extend(x_pos)
            all_y.extend(y_pos)
            all_groups.extend([k] * len(vals))
            if len(gsms) == len(vals):
                all_gsms.extend(gsms)
            else:
                all_gsms.extend(['?'] * len(vals))

            # Mean diamond
            mean_val = np.mean(y_pos)
            ax.scatter(gi, mean_val, marker='D', c='black', s=60, zorder=5)

        all_x = np.array(all_x)
        all_y = np.array(all_y, dtype=float)
        all_groups = np.array(all_groups)
        all_gsms = np.array(all_gsms)

        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel("Expression")
        ax.set_title("Class Separation (strip plot)")

        ax.text(0.5, -0.12,
                'Click to inspect  •  Shift+click multi-select  •  Ctrl+click full detail  •  Dbl-click clear',
                transform=ax.transAxes, fontsize=7.5, ha='center',
                color='#777777', style='italic')

        h = [mpatches.Patch(color=cdict[k], label=k) for k in keys]
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)
        leg = ax.legend(handles=h, title="Groups", loc='upper left', bbox_to_anchor=(1.01, 1))
        self._make_legend_interactive(fig, leg, cdict, {}, keys)

        # Rectangle selector for multi-sample selection
        X_2d = np.column_stack([all_x, all_y])
        self._add_rect_selector(fig, ax, X_2d, all_groups, all_gsms.tolist(), sf, plot_key="sep")

        self._embed_plot(fig, sf, "sep")

        # ── Info table for clicked points ──
        info_frame = labelframe(sf, text="Selected Points (click above)")
        info_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        info_cols = ('GSM', 'Group', 'Expression')
        # Add label columns if available
        extra_cols = []
        if not self.full_dataset.empty:
            extra_cols = [c for c in self.full_dataset.columns
                          if c not in ('GSM', '_platform') and self.full_dataset[c].dtype == 'object']
        all_info_cols = info_cols + tuple(extra_cols[:5])

        info_tree = ttk.Treeview(info_frame, columns=all_info_cols, show='headings', height=5)
        for c in all_info_cols:
            info_tree.heading(c, text=c)
            info_tree.column(c, width=100, anchor=tk.CENTER)
        info_sb = ttk.Scrollbar(info_frame, orient='vertical', command=info_tree.yview)
        info_tree.config(yscrollcommand=info_sb.set)
        info_sb.pack(side=tk.RIGHT, fill=tk.Y)
        info_tree.pack(fill=tk.BOTH, expand=True)

        sel_anns = {}

        def _on_pick(event):
            if event.mouseevent.dblclick:
                for ann in sel_anns.values():
                    ann.remove()
                sel_anns.clear()
                for item in info_tree.get_children():
                    info_tree.delete(item)
                fig.canvas.draw_idle()
                return

            mx, my = event.mouseevent.xdata, event.mouseevent.ydata
            if mx is None or my is None:
                return

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xr = (xlim[1] - xlim[0]) or 1
            yr = (ylim[1] - ylim[0]) or 1
            d = ((all_x - mx)/xr)**2 + ((all_y - my)/yr)**2
            ci = int(np.argmin(d))
            gsm = str(all_gsms[ci])
            key = f"{gsm}_{ci}"

            shift = bool(event.mouseevent.key == 'shift')
            ctrl = bool(event.mouseevent.key == 'control')

            # Ctrl+click → open full detail popup
            if ctrl:
                self._open_sample_detail_popup(
                    gsm, group_label=all_groups[ci],
                    expression_val=all_y[ci])
                return

            if key in sel_anns:
                sel_anns[key].remove()
                del sel_anns[key]
                for item in info_tree.get_children():
                    vals = info_tree.item(item)['values']
                    if vals and str(vals[0]) == gsm:
                        info_tree.delete(item)
                        break
            else:
                if not shift:
                    for ann in sel_anns.values():
                        ann.remove()
                    sel_anns.clear()
                    for item in info_tree.get_children():
                        info_tree.delete(item)

                ann = ax.annotate(
                    gsm, (all_x[ci], all_y[ci]),
                    fontsize=7, fontweight='bold',
                    ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEB3B',
                              edgecolor='#333', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=0.8))
                sel_anns[key] = ann

                # Build row data
                row_vals = [gsm, all_groups[ci], f"{all_y[ci]:.4f}"]
                if extra_cols and not self.full_dataset.empty:
                    gsm_row = self.full_dataset[
                        self.full_dataset['GSM'].str.upper() == gsm.upper()]
                    for ec in extra_cols[:5]:
                        if not gsm_row.empty and ec in gsm_row.columns:
                            row_vals.append(str(gsm_row.iloc[0][ec]))
                        else:
                            row_vals.append('')
                info_tree.insert('', tk.END, values=row_vals)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', _on_pick)

    # ═══════════════════════════════════════════════════════════════
    #  PCA / t-SNE (Dimensionality Reduction)
    # ═══════════════════════════════════════════════════════════════
    def _run_dimred(self):
        """Run PCA or t-SNE on current groups and plot 2D scatter."""
        for w in self.dimred_scroll_frame.scrollable_frame.winfo_children():
            w.destroy()
        if not self.current_data_map:
            return

        method = self.dimred_method.get()
        sf = self.dimred_scroll_frame.scrollable_frame

        # Build feature matrix
        all_vals = []
        all_labels = []
        all_gsms = []

        keys = list(self.current_data_map.keys())
        for k in keys:
            vals = self.current_data_map[k]
            gsms = self.group_gsm_map.get(k, ['?'] * len(vals))
            for v, g in zip(vals.values if hasattr(vals, 'values') else vals, gsms):
                all_vals.append([v])
                all_labels.append(k)
                all_gsms.append(g)

        if len(all_vals) < 5:
            ttk.Label(sf, text="Need at least 5 samples for dimensionality reduction.",
                      font=('Segoe UI', 11), foreground='gray').pack(pady=40)
            return

        X = np.array(all_vals, dtype=np.float64)
        L = np.array(all_labels)

        # Subsample for t-SNE (too slow for >10K)
        max_n = 10000 if method == "PCA" else 5000
        if len(X) > max_n:
            # Stratified subsample - keep proportions per group
            idx_keep = []
            for label in np.unique(L):
                grp_idx = np.where(L == label)[0]
                n_take = max(10, int(max_n * len(grp_idx) / len(X)))
                if len(grp_idx) > n_take:
                    idx_keep.extend(np.random.choice(grp_idx, n_take, replace=False))
                else:
                    idx_keep.extend(grp_idx)
            idx_keep = np.array(idx_keep)
            X = X[idx_keep]
            L = L[idx_keep]
            all_gsms = [all_gsms[i] for i in idx_keep]
            ttk.Label(sf, text=f"Subsampled to {len(X):,} points (from {len(all_vals):,}) for {method}",
                      font=('Segoe UI', 9, 'italic'), foreground='#888').pack(pady=2)

        # Add jittered noise dimension for 2D viz (1D expression → 2D)
        if X.shape[1] < 2:
            noise = np.random.normal(0, max(X.std(), 0.01) * 0.15, size=(len(X), 1))
            X = np.hstack([X, noise])

        try:
            self.status_label.config(text=f"Running {method}...")
            self.update_idletasks()

            if method == "PCA":
                from sklearn.decomposition import PCA
                n_comp = min(2, X.shape[1], X.shape[0])
                reducer = PCA(n_components=n_comp)
                X_2d = reducer.fit_transform(X)
                var_explained = reducer.explained_variance_ratio_
                ax1_label = f"PC1 ({var_explained[0]*100:.1f}%)"
                ax2_label = f"PC2 ({var_explained[1]*100:.1f}%)" if n_comp > 1 else "PC1"
                title = f"PCA - {len(X):,} samples, {len(keys)} groups"
            else:  # t-SNE
                from sklearn.manifold import TSNE
                # Perplexity must be < n_samples / 3
                max_perp = max(2, len(X) // 4)
                perp = min(self.tsne_perplexity.get(), max_perp, 50)
                reducer = TSNE(n_components=2, perplexity=perp, random_state=42,
                               max_iter=1000, init='pca', learning_rate='auto')
                X_2d = reducer.fit_transform(X)
                ax1_label = "t-SNE 1"
                ax2_label = "t-SNE 2"
                title = f"t-SNE (perplexity={perp}) - {len(X):,} samples, {len(keys)} groups"

            # Plot
            fig = Figure(figsize=(12, 8))
            ax = fig.subplots()
            uL = np.unique(L)
            pal = sns.color_palette("husl", len(uL))
            cmap = {l: mcolors.to_hex(c) for l, c in zip(uL, pal)}

            for label in uL:
                mask = (L == label)
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=cmap[label],
                           label=f"{label} ({mask.sum()})", s=25, alpha=0.6,
                           edgecolors='none')

            ax.set_xlabel(ax1_label, fontsize=11)
            ax.set_ylabel(ax2_label, fontsize=11)
            ax.set_title(title, fontsize=13, weight='bold')
            leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                            fontsize=8, framealpha=0.9)
            fig.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.08)

            # Rectangle selector for multi-sample selection
            self._add_rect_selector(fig, ax, X_2d, L, all_gsms, sf, plot_key="dimred")

            self._embed_plot(fig, sf, "dimred")
            self.status_label.config(text=f"{method} complete - {len(X):,} samples plotted")

            if method == "PCA" and hasattr(reducer, 'explained_variance_ratio_'):
                stats_text = f"Variance explained: PC1={var_explained[0]*100:.1f}%"
                if len(var_explained) > 1:
                    stats_text += f", PC2={var_explained[1]*100:.1f}%"
                    stats_text += f", Total={sum(var_explained)*100:.1f}%"
                ttk.Label(sf, text=stats_text, font=('Segoe UI', 10, 'bold')).pack(pady=5)

        except ImportError:
            ttk.Label(sf, text="scikit-learn required.\nInstall: pip install scikit-learn",
                      font=('Segoe UI', 11), foreground='red').pack(pady=40)
        except Exception as e:
            ttk.Label(sf, text=f"{method} Error: {e}", font=('Segoe UI', 10),
                      foreground='red').pack(pady=20)
            import traceback; traceback.print_exc()

    def _selection_table(self):
        """Every loaded sample with its expression and its labels.

        Assembled on the first selection, not on every plot: joining each
        loaded platform to its label table is real work, and a plot the user
        never lassoes should not pay for it. Cached until platforms or labels
        change.
        """
        stamp = (tuple(sorted(self.gpl_datasets)),
                 tuple(sorted(self.platform_labels)),
                 sum(len(v) for v in self.platform_labels.values()))
        cached = getattr(self, "_sel_table_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        parts = []
        for plat, expr in self.gpl_datasets.items():
            if expr is None or expr.empty or "GSM" not in expr.columns:
                continue
            part = expr.drop_duplicates(subset=["GSM"])
            lbl = self.platform_labels.get(plat)
            if (lbl is not None and not lbl.empty and "GSM" in lbl.columns):
                lbl = lbl.drop_duplicates(subset=["GSM"])
                new = [c for c in lbl.columns
                       if c == "GSM" or c not in part.columns]
                part = part.merge(lbl[new], on="GSM", how="left")
            if "platform" not in part.columns:
                part = part.assign(platform=plat)
            parts.append(part)

        table = (pd.concat(parts, ignore_index=True, sort=False)
                 .drop_duplicates(subset=["GSM"]) if parts else pd.DataFrame())
        self._sel_table_cache = (stamp, table)
        return table

    def _add_rect_selector(self, fig, ax, X_2d, labels, gsms, scroll_frame,
                            plot_key=None):
        """
        Add Rectangle + Lasso selection tools to scatter plots.

        Selection feeds into the linked brushing system - selecting samples
        in any plot highlights them across ALL other analysis tabs.
        """
        from matplotlib.widgets import RectangleSelector, LassoSelector
        from matplotlib.path import Path as MplPath

        # Register scatter data for linked brushing
        if plot_key is None:
            plot_key = f"_anon_{id(fig)}"
        self._scatter_data[plot_key] = {
            'ax': ax, 'fig': fig, 'X_2d': X_2d,
            'gsms': list(gsms), 'labels': np.asarray(labels),
        }
        # These axes already own button 1, so PlotInteractor's own
        # shift/ctrl-drag selection stands down here rather than opening a
        # second window for the same drag.
        ax._gv_external_brush = True
        from genevariate.utils.viz_style import attach_sample_table
        attach_sample_table(fig, self._selection_table)

        # ── Control bar ──
        ctrl = ttk.Frame(scroll_frame)
        ctrl.pack(fill=tk.X, pady=2)

        mode_var = tk.StringVar(value="rect")
        info_label = ttk.Label(
            ctrl, text="Drag to select  |  Rectangle mode",
            font=('Segoe UI', 9, 'italic'), foreground='#888')
        info_label.pack(side=tk.LEFT, padx=8)

        # ── Common selection handler ──
        def _handle_selection(mask):
            n_sel = mask.sum()
            if n_sel == 0:
                info_label.config(text="No samples in selection")
                return
            sel_labels = labels[mask]
            sel_gsms = [gsms[i] for i in np.where(mask)[0]]

            # Update linked brushing state
            self._brushed_gsms = set(sel_gsms)
            self._sync_brush(source_key=plot_key)

            from collections import Counter
            counts = Counter(sel_labels)
            summary = ", ".join(f"{k}: {v}" for k, v in counts.most_common(5))
            info_label.config(
                text=f"Selected {n_sel} samples - {summary}  "
                     f"(brushed across all tabs)")

            # The whole record for every selected sample, colourable by any
            # field. This used to be a two-column GSM/Group list, which could
            # not answer the question a selection is made to ask.
            from genevariate.gui.windows.sample_selection import show_selection
            show_selection(fig, sel_gsms, ax,
                           source=f"{plot_key} selection")

        # ── Rectangle callback ──
        def _on_rect(eclick, erelease):
            x1 = min(eclick.xdata, erelease.xdata)
            y1 = min(eclick.ydata, erelease.ydata)
            x2 = max(eclick.xdata, erelease.xdata)
            y2 = max(eclick.ydata, erelease.ydata)
            mask = ((X_2d[:, 0] >= x1) & (X_2d[:, 0] <= x2) &
                    (X_2d[:, 1] >= y1) & (X_2d[:, 1] <= y2))
            _handle_selection(mask)

        # ── Lasso callback ──
        def _on_lasso(verts):
            path = MplPath(verts)
            mask = np.array([path.contains_point(pt) for pt in X_2d])
            _handle_selection(mask)

        # ── Create both selectors (only one active at a time) ──
        rs = RectangleSelector(
            ax, _on_rect, useblit=True, button=[1], interactive=True,
            props=dict(facecolor='yellow', alpha=0.2,
                       edgecolor='red', linewidth=2))

        ls = LassoSelector(ax, _on_lasso,
                            props=dict(color='red', linewidth=2))
        ls.set_active(False)  # start with rectangle mode

        # Store for GC prevention
        self._selectors.extend([rs, ls])

        # ── Mode toggle buttons ──
        def _set_mode(m):
            mode_var.set(m)
            if m == "rect":
                rs.set_active(True)
                ls.set_active(False)
                info_label.config(text="Drag to select  |  Rectangle mode")
            else:
                rs.set_active(False)
                ls.set_active(True)
                info_label.config(text="Draw freehand to select  |  Lasso mode")

        rect_btn = ttk.Button(ctrl, text="Rectangle",
                               command=lambda: _set_mode("rect"), width=10)
        rect_btn.pack(side=tk.RIGHT, padx=2)
        lasso_btn = ttk.Button(ctrl, text="Lasso",
                                command=lambda: _set_mode("lasso"), width=10)
        lasso_btn.pack(side=tk.RIGHT, padx=2)

        def _clear():
            self._brushed_gsms.clear()
            self._sync_brush()
            info_label.config(text="Brush cleared")

        ttk.Button(ctrl, text="Clear Brush",
                   command=_clear, width=12).pack(side=tk.RIGHT, padx=2)

    # ------------------------------------------------------------------
    # Linked Brushing - highlight selected samples across all tabs
    # ------------------------------------------------------------------

    def _sync_brush(self, source_key=None):
        """
        Highlight _brushed_gsms in every registered scatter plot.

        For each plot: overlay orange-edged markers on brushed points,
        and add a rug overlay on the distribution plot.
        """
        brushed = self._brushed_gsms

        # ── Remove previous highlights ──
        for key, artists in list(self._brush_highlights.items()):
            for art in artists:
                try:
                    art.remove()
                except Exception:
                    pass
            self._brush_highlights[key] = []

        if not brushed:
            # Redraw all affected canvases
            for key in self._scatter_data:
                fig = self._scatter_data[key].get('fig')
                if fig and key in self.canvases:
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
            # Also clear distribution brush
            if 'dist_brush' in self._brush_highlights:
                del self._brush_highlights['dist_brush']
            if 'dist' in self.figs:
                try:
                    self.figs['dist'].canvas.draw_idle()
                except Exception:
                    pass
            return

        # ── Highlight in each scatter plot ──
        for key, data in self._scatter_data.items():
            ax = data['ax']
            fig = data['fig']
            X_2d = data['X_2d']
            gsms = data['gsms']

            # Find indices of brushed samples
            indices = [i for i, g in enumerate(gsms) if g in brushed]
            if not indices:
                if key in self.canvases:
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass
                continue

            idx = np.array(indices)
            highlight_x = X_2d[idx, 0]
            highlight_y = X_2d[idx, 1]

            # Draw purple-ringed markers over selected points
            sc = ax.scatter(
                highlight_x, highlight_y,
                s=120, facecolors='none', edgecolors='#7B1FA2',
                linewidths=2.5, zorder=10, label='_brushed')
            self._brush_highlights.setdefault(key, []).append(sc)

            if key in self.canvases:
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

        # ── Highlight on distribution plot (rug overlay) ──
        if 'dist' in self.figs and self.current_data_map:
            try:
                dist_fig = self.figs['dist']
                dist_ax = dist_fig.axes[0] if dist_fig.axes else None
                if dist_ax:
                    # Collect expression values for brushed GSMs
                    brush_vals = []
                    for grp, vals in self.current_data_map.items():
                        grp_gsms = self.group_gsm_map.get(grp, [])
                        for g, v in zip(grp_gsms, vals):
                            if g in brushed:
                                brush_vals.append(v)

                    if brush_vals:
                        ylim = dist_ax.get_ylim()
                        rug_h = (ylim[1] - ylim[0]) * 0.06
                        arts = []
                        for v in brush_vals:
                            line = dist_ax.axvline(
                                v, ymin=0, ymax=0.06,
                                color='#7B1FA2', linewidth=2,
                                alpha=0.8, zorder=10)
                            arts.append(line)
                        self._brush_highlights.setdefault(
                            'dist_brush', []).extend(arts)
                        dist_fig.canvas.draw_idle()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════
    #  Clustering (DPC / K-Means)
    # ═══════════════════════════════════════════════════════════════
    def _run_clustering(self):
        """Dispatch to selected clustering method."""
        method = self.cluster_method.get()
        if method == "DPC":
            self._run_dpc_clustering()
        elif method == "DBSCAN":
            self._run_dbscan_clustering()
        else:
            self._run_kmeans_clustering()

    def _run_dbscan_clustering(self):
        """DBSCAN clustering - finds arbitrary-shaped clusters, detects noise."""
        for w in self.cluster_scroll_frame.scrollable_frame.winfo_children():
            w.destroy()
        sf = self.cluster_scroll_frame.scrollable_frame

        if not self.current_data_map:
            return

        all_v, lbls, gsms_list = [], [], []
        for k, v in self.current_data_map.items():
            all_v.extend(v.tolist())
            lbls.extend([k] * len(v))
            gsms_list.extend(self.group_gsm_map.get(k, ['?'] * len(v)))

        if len(all_v) < 5:
            ttk.Label(sf, text="Need at least 5 samples.", foreground='gray').pack(pady=40)
            return

        X = np.array(all_v).reshape(-1, 1)
        L = np.array(lbls)
        eps = max(0.01, self.dbscan_eps.get())
        min_samp = max(2, self.dbscan_min_samples.get())

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import silhouette_score, adjusted_rand_score
            from sklearn.preprocessing import LabelEncoder, StandardScaler

            # Standardize for DBSCAN (eps is scale-dependent)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            db = DBSCAN(eps=eps, min_samples=min_samp)
            cluster_ids = db.fit_predict(X_scaled)

            n_clusters = len(set(cluster_ids) - {-1})
            n_noise = (cluster_ids == -1).sum()

            # Metrics (only if >1 cluster and not all noise)
            sil = 0; ari = 0
            if n_clusters >= 2 and n_noise < len(X):
                mask_valid = cluster_ids != -1
                if mask_valid.sum() > n_clusters:
                    try:
                        sil = silhouette_score(X[mask_valid], cluster_ids[mask_valid])
                    except: pass
                le = LabelEncoder()
                true_enc = le.fit_transform(L)
                ari = adjusted_rand_score(true_enc, cluster_ids)

            # ── Plot ──
            fig = Figure(figsize=(16, 7))
            ax1, ax2 = fig.subplots(1, 2)

            unique_clusters = sorted(set(cluster_ids))
            pal = sns.color_palette("husl", max(n_clusters, 1))
            cluster_colors = {}
            ci = 0
            for c in unique_clusters:
                if c == -1:
                    cluster_colors[c] = '#999999'  # noise = gray
                else:
                    cluster_colors[c] = mcolors.to_hex(pal[ci % len(pal)])
                    ci += 1

            # Left: samples colored by DBSCAN cluster
            all_scatter_x = []
            all_scatter_y = []
            all_scatter_labels = []
            all_scatter_gsms = []
            for c in unique_clusters:
                mask = cluster_ids == c
                jitter = np.random.uniform(-0.3, 0.3, mask.sum())
                lbl = f"Noise ({mask.sum()})" if c == -1 else f"Cluster {c} ({mask.sum()})"
                ax1.scatter(jitter, X[mask, 0], c=cluster_colors[c],
                            label=lbl, s=25, alpha=0.6,
                            marker='x' if c == -1 else 'o')
                all_scatter_x.extend(jitter)
                all_scatter_y.extend(X[mask, 0])
                all_scatter_labels.extend(L[mask])
                gsms_mask = [gsms_list[i] for i in np.where(mask)[0]]
                all_scatter_gsms.extend(gsms_mask)
            ax1.set_xlabel("Jitter")
            ax1.set_ylabel("Expression")
            ax1.set_title(f"DBSCAN (eps={eps}, min_samples={min_samp})\n"
                         f"{n_clusters} clusters, {n_noise} noise points",
                         fontsize=11, weight='bold')
            ax1.legend(fontsize=8, loc='upper right')

            # Right: contingency - true labels vs DBSCAN clusters
            uL = np.unique(L)
            non_noise = [c for c in unique_clusters if c != -1]
            all_c = non_noise + ([-1] if n_noise > 0 else [])
            contingency = np.zeros((len(uL), len(all_c)), dtype=int)
            for i, label in enumerate(uL):
                for j, c in enumerate(all_c):
                    contingency[i, j] = ((L == label) & (cluster_ids == c)).sum()

            im = ax2.imshow(contingency, aspect='auto', cmap='Blues')
            c_labels = [f"C{c}" if c != -1 else "Noise" for c in all_c]
            ax2.set_xticks(range(len(all_c)))
            ax2.set_xticklabels(c_labels)
            ax2.set_yticks(range(len(uL)))
            ax2.set_yticklabels([str(l)[:25] for l in uL], fontsize=8)
            _cell_grid(ax2, len(uL), len(all_c))
            ax2.set_xlabel("DBSCAN Cluster")
            ax2.set_title("True Labels vs DBSCAN Clusters", fontsize=11, weight='bold')
            fig.colorbar(im, ax=ax2, label="Count", shrink=0.8)
            # Scale annotation font to grid size
            _hm_fs = max(5, min(8, int(72 / max(len(uL), len(all_c), 1))))
            for i in range(len(uL)):
                for j in range(len(all_c)):
                    if contingency[i, j] > 0:
                        ax2.text(j, i, str(contingency[i, j]), ha='center', va='center',
                                fontsize=_hm_fs, color='white' if contingency[i, j] > contingency.max()/2 else 'black')

            fig.subplots_adjust(left=0.06, right=0.85, top=0.90, bottom=0.08, wspace=0.35)
            X_2d_db = np.column_stack([all_scatter_x, all_scatter_y])
            self._add_rect_selector(fig, ax1, X_2d_db, np.array(all_scatter_labels), all_scatter_gsms, sf, plot_key="dbscan")
            self._embed_plot(fig, sf, "dbscan")

            # Stats
            stats_frame = labelframe(sf, text="DBSCAN Results", padding=8)
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            stats_text = (f"eps={eps} | min_samples={min_samp} | "
                         f"Clusters={n_clusters} | Noise={n_noise} ({n_noise*100/len(X):.1f}%) | "
                         f"Silhouette={sil:.4f} | ARI={ari:.4f}")
            ttk.Label(stats_frame, text=stats_text, font=('Segoe UI', 10)).pack()
            ttk.Label(stats_frame,
                      text="Tip: Increase eps to merge nearby clusters. Decrease min_samples for smaller clusters.",
                      font=('Segoe UI', 8, 'italic'), foreground='#888').pack(pady=2)

        except ImportError:
            ttk.Label(sf, text="scikit-learn required.\nInstall: pip install scikit-learn",
                      font=('Segoe UI', 11), foreground='red').pack(pady=40)
        except Exception as e:
            ttk.Label(sf, text=f"DBSCAN Error: {e}", font=('Segoe UI', 10),
                      foreground='red').pack(pady=20)

    def _run_kmeans_clustering(self):
        """K-Means clustering with user-specified K."""
        for w in self.cluster_scroll_frame.scrollable_frame.winfo_children():
            w.destroy()
        sf = self.cluster_scroll_frame.scrollable_frame

        if not self.current_data_map:
            return

        all_v, lbls, gsms_list = [], [], []
        for k, v in self.current_data_map.items():
            all_v.extend(v.tolist())
            lbls.extend([k] * len(v))
            gsms_list.extend(self.group_gsm_map.get(k, ['?'] * len(v)))

        if len(all_v) < 3:
            ttk.Label(sf, text="Need at least 3 samples.", foreground='gray').pack(pady=40)
            return

        X = np.array(all_v).reshape(-1, 1)
        L = np.array(lbls)
        k_val = max(2, min(self.kmeans_k.get(), len(X)))

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, adjusted_rand_score

            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(X)

            # Silhouette score
            sil = silhouette_score(X, cluster_ids) if k_val < len(X) else 0
            # ARI against true labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            true_encoded = le.fit_transform(L)
            ari = adjusted_rand_score(true_encoded, cluster_ids)

            # ── Plot 1: Clusters colored by K-Means assignment ──
            fig = Figure(figsize=(16, 7))
            ax1, ax2 = fig.subplots(1, 2)

            pal = sns.color_palette("husl", k_val)
            all_scatter_x = []
            all_scatter_y = []
            all_scatter_labels = []
            all_scatter_gsms = []
            for ci in range(k_val):
                mask = cluster_ids == ci
                jitter = np.random.uniform(-0.3, 0.3, mask.sum())
                ax1.scatter(jitter, X[mask, 0], c=[mcolors.to_hex(pal[ci])],
                            label=f"Cluster {ci} (n={mask.sum()})", s=25, alpha=0.6)
                all_scatter_x.extend(jitter)
                all_scatter_y.extend(X[mask, 0])
                all_scatter_labels.extend(L[mask])
                gsms_mask = [gsms_list[i] for i in np.where(mask)[0]]
                all_scatter_gsms.extend(gsms_mask)
            ax1.axhline(y=np.mean(X), color='black', ls='--', lw=1, alpha=0.5)
            for ci in range(k_val):
                ax1.axhline(y=kmeans.cluster_centers_[ci, 0], color=mcolors.to_hex(pal[ci]),
                            ls=':', lw=2, alpha=0.8)
            ax1.set_xlabel("Jitter")
            ax1.set_ylabel("Expression")
            ax1.set_title(f"K-Means (K={k_val}) - Silhouette={sil:.3f}, ARI={ari:.3f}",
                         fontsize=11, weight='bold')
            ax1.legend(fontsize=8, loc='upper right')

            # ── Plot 2: True labels vs clusters (contingency) ──
            uL = np.unique(L)
            contingency = np.zeros((len(uL), k_val), dtype=int)
            for i, label in enumerate(uL):
                for ci in range(k_val):
                    contingency[i, ci] = ((L == label) & (cluster_ids == ci)).sum()

            im = ax2.imshow(contingency, aspect='auto', cmap='Blues')
            ax2.set_xticks(range(k_val))
            ax2.set_xticklabels([f"C{i}" for i in range(k_val)])
            ax2.set_yticks(range(len(uL)))
            ax2.set_yticklabels([str(l)[:25] for l in uL], fontsize=8)
            _cell_grid(ax2, len(uL), k_val)
            ax2.set_xlabel("Cluster")
            ax2.set_title("True Labels vs K-Means Clusters", fontsize=11, weight='bold')
            fig.colorbar(im, ax=ax2, label="Count", shrink=0.8)

            # Annotate cells - scale font to grid size
            _hm_fs = max(5, min(8, int(72 / max(len(uL), k_val, 1))))
            for i in range(len(uL)):
                for j in range(k_val):
                    if contingency[i, j] > 0:
                        ax2.text(j, i, str(contingency[i, j]), ha='center', va='center',
                                fontsize=_hm_fs, color='white' if contingency[i, j] > contingency.max()/2 else 'black')

            fig.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.08, wspace=0.35)
            X_2d_km = np.column_stack([all_scatter_x, all_scatter_y])
            self._add_rect_selector(fig, ax1, X_2d_km, np.array(all_scatter_labels), all_scatter_gsms, sf, plot_key="kmeans")
            self._embed_plot(fig, sf, "kmeans")

            # Stats summary
            stats_frame = labelframe(sf, text="K-Means Results", padding=8)
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            stats_text = (f"K = {k_val} | Silhouette Score = {sil:.4f} | "
                         f"Adjusted Rand Index = {ari:.4f} | "
                         f"Inertia = {kmeans.inertia_:.2f}")
            ttk.Label(stats_frame, text=stats_text, font=('Segoe UI', 10)).pack()

            # Cluster centers
            centers_text = " | ".join(f"C{i}: {kmeans.cluster_centers_[i,0]:.3f}" for i in range(k_val))
            ttk.Label(stats_frame, text=f"Cluster Centers: {centers_text}",
                      font=('Consolas', 9)).pack(pady=2)

        except ImportError:
            ttk.Label(sf, text="scikit-learn required.\nInstall: pip install scikit-learn",
                      font=('Segoe UI', 11), foreground='red').pack(pady=40)
        except Exception as e:
            ttk.Label(sf, text=f"K-Means Error: {e}", font=('Segoe UI', 10),
                      foreground='red').pack(pady=20)

    def _run_dpc_clustering(self):
        """Density Peak Clustering with decision graph."""
        for w in self.cluster_scroll_frame.scrollable_frame.winfo_children():
            w.destroy()
        sf = self.cluster_scroll_frame.scrollable_frame

        all_v, lbls = [], []
        for k, v in self.current_data_map.items():
            all_v.extend(v.tolist())
            lbls.extend([k] * len(v))

        if not all_v:
            return

        # Subsample if too large (pdist is O(n^2))
        max_n = 5000
        if len(all_v) > max_n:
            idx = np.random.choice(len(all_v), max_n, replace=False)
            all_v = [all_v[i] for i in idx]
            lbls = [lbls[i] for i in idx]

        X = np.array(all_v).reshape(-1, 1)
        L = np.array(lbls)

        try:
            from scipy.spatial.distance import pdist, squareform
            dists = squareform(pdist(X))
            dc = np.percentile(dists, 2) or 1e-5
            rho = np.sum(np.exp(-(dists / dc) ** 2), axis=1) - 1
            delta = np.zeros(len(X))
            ord_rho = np.argsort(-rho)

            for i, idx in enumerate(ord_rho):
                if i == 0:
                    delta[idx] = dists[idx, :].max()
                else:
                    delta[idx] = dists[idx, ord_rho[:i]].min()

            fig = Figure(figsize=(max(10, len(self.current_data_map) * 0.4), 7))
            ax = fig.subplots()
            uL = np.unique(L)
            pal = sns.color_palette("husl", len(uL))
            cmap_dpc = {l: mcolors.to_hex(c) for l, c in zip(uL, pal)}

            artists = {}
            dpc_scatter_x = []
            dpc_scatter_y = []
            dpc_scatter_labels = []
            dpc_scatter_gsms = []
            for l in uL:
                mask = (L == l)
                grp_idx = np.where(mask)[0]
                if len(grp_idx) > 0:
                    pk = grp_idx[np.argmax(rho[grp_idx])]
                    sc = ax.scatter(rho[pk], delta[pk], c=cmap_dpc[l], label=l,
                                    s=150, alpha=0.9, edgecolors='k', picker=True)
                    artists[l] = [sc]
                    dpc_scatter_x.append(rho[pk])
                    dpc_scatter_y.append(delta[pk])
                    dpc_scatter_labels.append(l)
                    gsms_for_grp = self.group_gsm_map.get(l, ['?'])
                    dpc_scatter_gsms.append(gsms_for_grp[0] if gsms_for_grp else '?')

            ax.set_xlabel("Density (rho)", fontsize=11)
            ax.set_ylabel("Delta (min dist to higher density)", fontsize=11)
            ax.set_title("DPC Decision Graph (Cluster Peaks)", fontsize=13, weight='bold')
            leg = ax.legend(title="Groups", loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
            self._make_legend_interactive(fig, leg, cmap_dpc, artists, uL)
            fig.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.08)
            if dpc_scatter_x:
                X_2d_dpc = np.column_stack([dpc_scatter_x, dpc_scatter_y])
                self._add_rect_selector(fig, ax, X_2d_dpc, np.array(dpc_scatter_labels), dpc_scatter_gsms, sf, plot_key="dpc")
            self._embed_plot(fig, sf, "dpc")

        except Exception as e:
            ttk.Label(sf, text=f"DPC Error: {e}", font=('Segoe UI', 10),
                      foreground='red').pack(pady=20)
                
    def _make_legend_interactive(self, fig, legend, col_map, artist_map, keys):
        if not legend: return
        lmap = {}
        for txt, hnd in zip(legend.get_texts(), legend.legend_handles):
            for k in keys:
                if txt.get_text().startswith(k): lmap[hnd] = k; hnd.set_picker(5); break
        def on_pick(event):
            h = event.artist; k = lmap.get(h)
            if not k: return
            new = colorchooser.askcolor(color=col_map.get(k,'#fff'))[1]
            if not new: return
            col_map[k] = new
            tgt = artist_map.get(k, [])
            if isinstance(tgt, dict): tgt = [x for l in tgt.values() for x in l]
            for art in tgt:
                try:
                    if hasattr(art, 'set_color'): art.set_color(new)
                    if hasattr(art, 'set_facecolor'): art.set_facecolor(new)
                    if hasattr(art, 'set_edgecolor'): art.set_edgecolor(new)
                except: pass
            if hasattr(h, 'set_facecolor'): h.set_facecolor(new)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect('pick_event', on_pick)

    # ------------------------------------------------------------------
    # Sample Detail Popup - shows full metadata for a clicked point
    # ------------------------------------------------------------------

    def _open_sample_detail_popup(self, gsm_id, group_label=None,
                                   expression_val=None):
        """
        Open a detail window showing all available metadata for one sample.

        Searches full_dataset, gene data, and GSE context for the sample.
        """
        top = tk.Toplevel(self)
        style_window(top)
        top.title(f"Sample Detail - {gsm_id}")
        top.geometry("700x520")
        top.transient(self)

        # ── Header ──
        hdr = ttk.Frame(top, padding=8)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text=gsm_id, font=('Segoe UI', 16, 'bold')).pack(
            side=tk.LEFT)
        if group_label:
            ttk.Label(hdr, text=f"  Group: {group_label}",
                      font=('Segoe UI', 11), foreground='#555').pack(
                side=tk.LEFT, padx=(12, 0))

        ttk.Separator(top, orient='horizontal').pack(fill=tk.X, padx=8)

        # ── Collect all metadata ──
        detail_rows = []

        # From full_dataset
        if hasattr(self, 'full_dataset') and not self.full_dataset.empty:
            gsm_col = 'GSM' if 'GSM' in self.full_dataset.columns else None
            if gsm_col:
                match = self.full_dataset[
                    self.full_dataset[gsm_col].astype(str) == str(gsm_id)]
                if not match.empty:
                    row = match.iloc[0]
                    for col in match.columns:
                        val = row[col]
                        if col == gsm_col:
                            continue
                        val_str = str(val) if pd.notna(val) else ''
                        if val_str and val_str.lower() not in ('nan', 'none'):
                            detail_rows.append((col, val_str))

        if expression_val is not None:
            detail_rows.insert(0, ("Expression (selected gene)",
                                   f"{expression_val:.6f}"))

        if not detail_rows:
            detail_rows.append(("(no metadata found)", ""))

        # ── Detail table ──
        tv_frame = ttk.Frame(top, padding=8)
        tv_frame.pack(fill=tk.BOTH, expand=True)

        cols = ('Field', 'Value')
        tree = ttk.Treeview(tv_frame, columns=cols, show='headings',
                             height=min(20, len(detail_rows)))
        tree.heading('Field', text='Field')
        tree.heading('Value', text='Value')
        tree.column('Field', width=180, anchor=tk.W)
        tree.column('Value', width=480, anchor=tk.W)

        vsb = ttk.Scrollbar(tv_frame, orient='vertical', command=tree.yview)
        tree.config(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        for field, value in detail_rows:
            tree.insert('', tk.END, values=(field, value[:500]))

        # ── Buttons ──
        btn_frame = ttk.Frame(top, padding=8)
        btn_frame.pack(fill=tk.X)

        def _copy_to_clipboard():
            text = "\n".join(f"{f}: {v}" for f, v in detail_rows)
            top.clipboard_clear()
            top.clipboard_append(text)

        ttk.Button(btn_frame, text="Copy to Clipboard",
                   command=_copy_to_clipboard).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Close",
                   command=top.destroy).pack(side=tk.RIGHT, padx=4)

    def _embed_plot(self, fig, parent, key):
        if key in self.canvases: self.canvases[key].get_tk_widget().destroy(); self.toolbars[key].destroy(); plt.close(self.figs[key])
        c = self.FigureCanvasTkAgg(fig, master=parent); c.draw()
        viz_make_interactive(fig)
        c.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        t = self.NavigationToolbar2Tk(c, parent); t.update(); style_toolbar(t); t.pack(side=tk.BOTTOM, fill=tk.X)
        self.figs[key] = fig; self.canvases[key] = c; self.toolbars[key] = t

    def _generate_report(self):
        d = filedialog.asksaveasfilename(title="Save Report Folder", initialfile="Analysis_Report")
        if d:
            os.makedirs(d, exist_ok=True)
            for k, df in self.results_cache.items(): df.to_csv(f"{d}/{k}.csv")
            for k, f in self.figs.items(): f.savefig(f"{d}/{k}.png")
            messagebox.showinfo("Success", f"Saved to {d}")

    def _clear_all_plots(self):
        scroll_frames = [self.dist_scroll_frame, self.matrix_scroll_frame, self.sep_scroll_frame]
        if hasattr(self, 'dimred_scroll_frame'):
            scroll_frames.append(self.dimred_scroll_frame)
        if hasattr(self, 'cluster_scroll_frame'):
            scroll_frames.append(self.cluster_scroll_frame)
        for sf in scroll_frames:
            for w in sf.scrollable_frame.winfo_children(): w.destroy()
        if hasattr(self, 'stats_tree'): self.stats_tree.delete(*self.stats_tree.get_children())
        if hasattr(self, 'plot_refs'):
            for f in self.plot_refs.values(): plt.close(f)
        self.plot_refs = {}
        # Clear linked brushing state
        self._scatter_data.clear()
        self._brush_highlights.clear()
        self._brushed_gsms.clear()
        self._selectors.clear()

    def _clear_user_data(self):
        self.user_defined_groups = {}; self.loaded_files_listbox.delete(0, tk.END); self.status_label.config(text="Cleared.")
        self._clear_all_plots()

    def _on_closing(self):
        self._clear_all_plots()
        if hasattr(self, 'app_ref') and self.app_ref:
            self.app_ref.compare_window = None
        self.destroy()


    def auto_load_subset_data(self, file_path="subset_analyzed_show_gene_distribution.csv"):
            """
            Loads the auto-saved subset file.
            REPAIR: Checks if data is already loaded (e.g., from Deep Research) to prevent overwrite.
            """
            # If dataset is already populated (e.g. by CustomCompareWindow), ABORT auto-load.
            if not self.full_dataset.empty:
                return
    
            if not os.path.exists(file_path):
                return
    
            try:
                # 1. Load Data
                df = pd.read_csv(file_path)
                
                # 2. Standardize GSM Column
                cols_map = {c.upper(): c for c in df.columns}
                if 'GSM' in cols_map:
                    df.rename(columns={cols_map['GSM']: 'GSM'}, inplace=True)
                elif 'ID' in cols_map:
                    df.rename(columns={cols_map['ID']: 'GSM'}, inplace=True)
                
                if 'GSM' in df.columns:
                    df['GSM'] = df['GSM'].astype(str).str.upper()
                else:
                    return 
    
                self.full_dataset = df
                self.status_label.config(text=f"Loaded {len(df)} samples (Auto-Import)")
    
                # 3. Detect Grouping
                detected_group_col = None
                priority_prefixes = ["Classified_", "Condition", "Group", "Cluster", "series_id"]
                for prefix in priority_prefixes:
                    candidates = [c for c in df.columns if c.startswith(prefix)]
                    if candidates:
                        detected_group_col = candidates[0]; break
                
                if not detected_group_col:
                    text_cols = [c for c in df.columns if c != 'GSM' and df[c].dtype == 'object']
                    if text_cols: detected_group_col = text_cols[-1]
    
                # 4. Refresh UI
                self._refresh_data_table()
                
                if detected_group_col:
                    self.grouping_column = detected_group_col
                    self.lbl_grouping.config(text=self.grouping_column, foreground=AERO["green_dark"])
                    self._update_group_list()
                    self.loaded_files_listbox.select_set(0, tk.END)
                else:
                    self.lbl_grouping.config(text="Click a Header to Group", foreground=AERO["danger"])
    
            except Exception as e:
                print(f"Auto-load failed: {e}")
            

class CustomCompareWindow(CompareDistributionsWindow):
    """
    A specialized version of CompareDistributionsWindow that accepts 
    pre-processed data from the Analysis Table Popup.
    REPAIRED: Explicitly refreshes UI elements (Table/Listbox) on init so the window isn't empty.
    """
    def __init__(self, parent, app_ref, df_full, data_map, bg_map, grp_gsm_map, title_suffix, grouping_col=None):
        # Initialize the base class
        super().__init__(parent, app_ref)
        
        # Override title to match User's expectation
        self.title(f"BioMetric Analytics: {title_suffix}")
        
        # 1. Inject the Metadata (Critical for DPC Click/Table to work)
        self.full_dataset = df_full.copy()
        
        # 2. Set Grouping Column (Critical for Listbox population)
        self.grouping_column = grouping_col
        if self.grouping_column:
             self.lbl_grouping.config(text=self.grouping_column)

        # 3. REPAIR: Force UI Refresh immediately 
        # (This populates the "Data & Grouping" tab and the "Select Groups" listbox)
        self._refresh_data_table()
        self._update_group_list()
        
        # Select all items in the listbox by default to indicate they are active
        if self.loaded_files_listbox.size() > 0:
            self.loaded_files_listbox.select_set(0, tk.END)

        # 4. Inject the Analysis Data directly
        view_key = "Popup_Selection"
        self.analysis_results[view_key] = {
            "data": data_map,
            "bg": bg_map,
            "grp_map": grp_gsm_map
        }
        
        # 5. Create the navigation button for this view
        row_frame = ttk.Frame(self.sub_nav_frame)
        row_frame.pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(row_frame, text="Current Selection Analysis", 
                   command=lambda: self._switch_view(view_key)).pack(side=tk.LEFT, padx=2)
        
        # 6. Automatically trigger the view rendering
        self.after(100, lambda: self._switch_view(view_key))        
        

class SavePlotsDialog:
    pass

class SubsetDisplayOptionsDialog:
    pass

class SelectColumnsDialog(simpledialog.Dialog):
    """A dialog to select the GSM column and MULTIPLE grouping/label columns."""
    def __init__(self, parent, columns, file_name):
        self.columns = columns
        self.file_name = file_name
        self.result = None
        super().__init__(parent, f"Select Columns for {self.file_name}")

    def body(self, master):
        self.resizable(False, True)
        ttk.Label(master, text="Please specify which columns to use for analysis.",
                  wraplength=350).pack(padx=10, pady=(10, 5))

        gsm_frame = labelframe(master, text="1. Select Sample ID Column (GSM)")
        gsm_frame.pack(padx=10, pady=5, fill=tk.X)
        self.gsm_var = tk.StringVar()
        self.gsm_combo = ttk.Combobox(gsm_frame, textvariable=self.gsm_var,
                                       values=self.columns, state="readonly", width=40)
        for col in self.columns:
            if 'gsm' in col.lower():
                self.gsm_var.set(col)
                break
        self.gsm_combo.pack(padx=5, pady=5)

        label_frame = labelframe(master, text="2. Select Grouping/Label Column(s)")
        label_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        list_frame = ttk.Frame(label_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.label_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=8)
        for col in self.columns:
            self.label_listbox.insert(tk.END, col)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                   command=self.label_listbox.yview)
        self.label_listbox.config(yscrollcommand=scrollbar.set)
        self.label_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return self.gsm_combo

    def apply(self):
        gsm_col = self.gsm_var.get()
        selected_indices = self.label_listbox.curselection()
        label_cols = [self.label_listbox.get(i) for i in selected_indices]
        if not gsm_col or not label_cols:
            messagebox.showerror("Input Error",
                                 "You must select a GSM column and at least one Label column.",
                                 parent=self)
            self.result = None
            return
        if gsm_col in label_cols:
            messagebox.showerror("Input Error",
                                 "GSM column cannot also be a Label column.", parent=self)
            self.result = None
            return
        self.result = {"gsm_col": gsm_col, "label_cols": label_cols}

# Torch compatibility disabled - running in CPU mode
pass  # Placeholder

SPECIES_EXAMPLES = [
    ('Rat 230 2.0', 'GPL1355'),
    ('Canine 2.0', 'GPL3738'),
    ('Rhesus', 'GPL3535'),
    ('Arabidopsis ATH1', 'GPL198'),
    ('E. coli', 'GPL3154'),
    ('Zebrafish', 'GPL1319'),
    ('Porcine', 'GPL3533'),
    ('Drosophila 2.0', 'GPL1322'),
    ('C. elegans', 'GPL200'),
    ('Yeast S98', 'GPL90'),
]

# ═══════════════════════════════════════════════════════════════════
#  Cross-Platform Advanced Analysis Window
# ═══════════════════════════════════════════════════════════════════

# ── Ortholog mapping for common cross-species comparisons ──────────
# Keys: (species_A_token, species_B_token) -> description
COMMON_PLATFORM_COMPARISONS = {
    ('human', 'mouse'): 'Human vs Mouse - ~16,000 one-to-one orthologs',
    ('human', 'rat'): 'Human vs Rat - ~15,000 one-to-one orthologs',
    ('mouse', 'rat'): 'Mouse vs Rat - ~17,000 one-to-one orthologs',
    ('human', 'zebrafish'): 'Human vs Zebrafish - ~10,000 orthologs',
    ('human', 'drosophila'): 'Human vs Drosophila - ~4,000 orthologs',
    ('human', 'c. elegans'): 'Human vs C. elegans - ~3,000 orthologs',
}

# GPL -> species token mapping (common platforms)
GPL_SPECIES = {
    'GPL570': 'human', 'GPL96': 'human', 'GPL571': 'human',
    'GPL6947': 'human', 'GPL10558': 'human', 'GPL6244': 'human',
    'GPL6480': 'human', 'GPL13534': 'human', 'GPL16686': 'human',
    'GPL6885': 'mouse', 'GPL1261': 'mouse', 'GPL7202': 'mouse',
    'GPL6246': 'mouse', 'GPL11180': 'mouse', 'GPL21163': 'mouse',
    'GPL1355': 'rat', 'GPL6101': 'rat', 'GPL85': 'rat',
    'GPL1319': 'zebrafish', 'GPL1322': 'drosophila', 'GPL200': 'c. elegans',
    'GPL3535': 'rhesus', 'GPL3533': 'porcine', 'GPL90': 'yeast',
}

# GEO and CELLxGENE both name organisms scientifically; the comparison only
# needs to know whether two platforms are the same organism, so both spellings
# collapse to one token and anything unrecognised keeps its own name.
_SPECIES_COMMON = {
    'homo sapiens': 'human', 'mus musculus': 'mouse',
    'rattus norvegicus': 'rat', 'danio rerio': 'zebrafish',
    'drosophila melanogaster': 'drosophila',
    'caenorhabditis elegans': 'c. elegans',
    'macaca mulatta': 'rhesus', 'sus scrofa': 'porcine',
    'saccharomyces cerevisiae': 'yeast',
}


def _common_species_name(organism):
    # The Census writes "homo_sapiens", GEO writes "Homo sapiens", and a
    # multi-species design lists every organism separated.
    name = re.split(r"[;\t,]", str(organism or ""))[0]
    name = name.replace("_", " ").strip().lower()
    if not name:
        return "unknown"
    return _SPECIES_COMMON.get(name, name)


def _shorten_platform_title(title, limit=44):
    """Trim a GEO platform title to something that fits beneath a pill.

    GEO titles carry a leading design tag and a trailing organism in
    parentheses -- "[HG-U133_Plus_2] Affymetrix Human Genome U133 Plus 2.0
    Array", "Illumina NextSeq 500 (Homo sapiens)". Both are shown elsewhere
    in the pill, so what is left is the instrument name.
    """
    t = re.sub(r"^\[[^\]]+\]\s*", "", str(title or "")).strip()
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


# The normalizers, the ComBat wrapper and the whole per-gene comparison now
# live in ``core.analysis.cross_platform`` - Tk-free, and shared with the
# assistant so both surfaces report one set of numbers.


class CrossPlatformAnalysisWindow(tk.Toplevel):
    """Advanced multi-platform gene expression comparison with batch correction."""

    def __init__(self, parent, app_ref=None):
        super().__init__(parent)
        # The opener passes the main app as `parent`; app_ref is optional and
        # falls back to parent so `CrossPlatformAnalysisWindow(app)` works.
        self.app = app_ref if app_ref is not None else parent
        ensure_theme(self)
        style_window(self)
        self.title("GeneVariate - Cross-Platform Analysis")
        self.geometry("1200x850")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1200x850+{(_sw-1200)//2}+{(_sh-850)//2}")
            self.minsize(600, 500)
        except Exception: pass
        self.minsize(1000, 700)

        self._results = {}
        self._running = False

        self._build_ui()
        self._populate_platforms()

    # ── UI Construction ─────────────────────────────────────────────
    def _build_ui(self):
        # Frutiger Aero header: glossy sky-blue gradient
        hdr_h = 54
        hdr = tk.Canvas(self, height=hdr_h, highlightthickness=0, bd=0,
                        bg=AERO["sky_top"])
        hdr.pack(fill=tk.X)
        def _paint_cp_hdr(event=None):
            # Same banner the main window wears: sky_top fading to a tinted
            # white, a sheen over the top third, and the cyan underline. This
            # used to fade the other way, into accent_dark, which made the one
            # secondary window with a header the only dark surface in the app.
            w = hdr.winfo_width() or 1200
            _aero_vertical_gradient(hdr, w, hdr_h,
                                    AERO["sky_top"], "#F4FBFF",
                                    tag="cp_hdr_bg")
            hdr.delete("cp_hdr_gloss")
            sheen_h = max(1, hdr_h // 3)
            try:
                tr, tg, tb = hdr.winfo_rgb("#FFFFFF")
                br, bg_, bb = hdr.winfo_rgb(AERO["sky_top"])
                tr, tg, tb = tr // 256, tg // 256, tb // 256
                br, bg_, bb = br // 256, bg_ // 256, bb // 256
                for y in range(sheen_h):
                    t = y / max(1, sheen_h - 1)
                    r = int(tr * (1 - t) + br * t)
                    g = int(tg * (1 - t) + bg_ * t)
                    b = int(tb * (1 - t) + bb * t)
                    hdr.create_rectangle(0, y, w, y + 1,
                                         fill=f"#{r:02x}{g:02x}{b:02x}",
                                         outline="", tags="cp_hdr_gloss")
            except Exception:
                pass
            hdr.delete("cp_hdr_underline")
            hdr.create_rectangle(0, hdr_h - 3, w, hdr_h,
                                 fill=AERO["accent"], outline="",
                                 tags="cp_hdr_underline")
            hdr.delete("cp_hdr_fg")
            hdr.create_text(18, hdr_h // 2 - 8,
                            text="Cross-Platform Gene Expression Analysis",
                            anchor="w", font=('Segoe UI', 14, 'bold'),
                            fill=AERO["accent_dark"], tags="cp_hdr_fg")
            hdr.create_text(18, hdr_h // 2 + 12,
                            text="Compare gene distributions, detect batch effects, find DE & conserved genes",
                            anchor="w", font=('Segoe UI', 9, 'italic'),
                            fill=AERO["muted"], tags="cp_hdr_fg")
        hdr.bind("<Configure>", _paint_cp_hdr)

        # ── Top config panel ──
        config_frame = labelframe(self, text="Analysis Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        # Platform selection
        plat_row = ttk.Frame(config_frame)
        plat_row.pack(fill=tk.X, pady=4)
        ttk.Label(plat_row, text="Platforms to compare:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Label(plat_row, text="(select 2 or more)",
                  font=('Segoe UI', 9, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=8)

        self._plat_checks_frame = FlowFrame(config_frame)
        self._plat_checks_frame.pack(fill=tk.X, pady=2)
        self._plat_vars = {}

        # Reference platform
        ref_row = ttk.Frame(config_frame)
        ref_row.pack(fill=tk.X, pady=4)
        ttk.Label(ref_row, text="Reference platform:",
                  font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self._ref_var = tk.StringVar(value="(auto)")
        self._ref_combo = ttk.Combobox(ref_row, textvariable=self._ref_var,
                                        state='readonly', width=25)
        self._ref_combo.pack(side=tk.LEFT, padx=8)
        ttk.Label(ref_row, text="(other platforms compared against this one)",
                  font=('Segoe UI', 8, 'italic'), foreground='gray').pack(side=tk.LEFT)

        # Options row
        opt_row = ttk.Frame(config_frame)
        opt_row.pack(fill=tk.X, pady=4)

        ttk.Label(opt_row, text="Batch correction:", font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self._batch_var = tk.StringVar(value="none")
        batch_combo = ttk.Combobox(opt_row, textvariable=self._batch_var,
                                    state='readonly', width=32,
                                    values=["none",
                                            "quantile_normalization",
                                            "median_centering",
                                            "combat (preserve biology)"])
        batch_combo.pack(side=tk.LEFT, padx=8)
        self._batch_combo = batch_combo
        # Tooltip-style hint that updates when selection changes
        self._batch_hint = ttk.Label(opt_row, text="", font=('Segoe UI', 8, 'italic'),
                                      foreground='gray')
        self._batch_hint.pack(side=tk.LEFT, padx=4)
        _batch_hints = {
            "none": "",
            "quantile_normalization": "aligns distributions (no labels needed)",
            "median_centering": "shifts medians only (no labels needed)",
            "combat (preserve biology)": "uses extracted labels to protect biological signal",
        }
        def _update_batch_hint(*_):
            self._batch_hint.config(text=_batch_hints.get(self._batch_var.get(), ""))
        self._batch_var.trace_add('write', _update_batch_hint)

        ttk.Label(opt_row, text="p-value threshold:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(20, 0))
        self._pval_var = tk.StringVar(value="0.05")
        ttk.Entry(opt_row, textvariable=self._pval_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(opt_row, text="Min |Δmean|:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(20, 0))
        self._delta_var = tk.StringVar(value="0.5")
        ttk.Entry(opt_row, textvariable=self._delta_var, width=8).pack(side=tk.LEFT, padx=5)

        # Species info row
        self._species_label = ttk.Label(config_frame, text="", font=('Segoe UI', 9),
                                         foreground=AERO['accent_dark'])
        self._species_label.pack(fill=tk.X, pady=2)

        # What kind of comparison the current selection is, and therefore which
        # statistics are defined for it.
        self._scenario_label = ttk.Label(config_frame, text="", font=('Segoe UI', 9),
                                          foreground=AERO['accent_dark'])
        self._scenario_label.pack(fill=tk.X, pady=2)

        # Buttons
        btn_row = ttk.Frame(config_frame)
        btn_row.pack(fill=tk.X, pady=6)

        self._run_btn = ttk.Button(btn_row, text="Run Cross-Platform Analysis",
                                    style="Primary.TButton", cursor="hand2",
                                    command=self._start_analysis)
        self._run_btn.pack(side=tk.LEFT, padx=5)

        self._export_btn = ttk.Button(btn_row, text="Export Report",
                                      style="Add.TButton", cursor="hand2",
                                      command=self._export_full_report,
                                      state=tk.DISABLED)
        self._export_btn.pack(side=tk.LEFT, padx=5)

        self._progress_label = ttk.Label(btn_row, text="", font=('Segoe UI', 9))
        self._progress_label.pack(side=tk.LEFT, padx=15)

        self._progress_bar = ttk.Progressbar(btn_row, mode='determinate', length=200)
        self._progress_bar.pack(side=tk.LEFT, padx=5)

        # ── Results notebook (tabs) ──
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

        # Tab: Overview
        self._overview_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._overview_tab, text=" Overview ")
        self._overview_text = tk.Text(self._overview_tab, font=('Consolas', 10),
                                       wrap=tk.WORD, state=tk.DISABLED)
        ov_sb = ttk.Scrollbar(self._overview_tab, command=self._overview_text.yview)
        ov_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._overview_text.configure(yscrollcommand=ov_sb.set)
        self._overview_text.pack(fill=tk.BOTH, expand=True)

        # Tab: Gene Overlap
        self._overlap_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._overlap_tab, text=" Gene Overlap ")

        # Tab: DE Genes
        self._de_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._de_tab, text=" DE Genes (Cross-Platform) ")

        # Tab: Conserved Genes
        self._conserved_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._conserved_tab, text=" Conserved Genes ")

        # Tab: Platform-Specific
        self._unique_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._unique_tab, text=" Platform-Specific ")

        # Tab: Batch Effects
        self._batch_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._batch_tab, text=" Batch Effects ")

        # Tab: Distribution Metrics
        self._dist_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._dist_tab, text=" Distribution Metrics ")

    def _populate_platforms(self):
        """Fill platform checkboxes from loaded data."""
        self._plat_checks_frame.clear()
        self._plat_vars.clear()

        plats = sorted(self.app.gpl_datasets.keys())
        if not plats:
            self._plat_checks_frame.add(
                ttk.Label(self._plat_checks_frame, text="No platforms loaded",
                          foreground=AERO['danger']))
            return

        for plat in plats:
            var = tk.BooleanVar(value=True)
            self._plat_vars[plat] = var
            n_samples = len(self.app.gpl_datasets[plat])
            n_genes = len(self.app.gpl_gene_mappings.get(plat, {}))
            species = self.app.platform_species(plat)
            self._plat_checks_frame.add(ttk.Checkbutton(
                self._plat_checks_frame,
                text=f"{plat} ({species}, {n_samples:,}s, {n_genes:,}g)",
                variable=var, command=self._on_plat_selection_change))

        self._ref_combo['values'] = ['(auto)'] + plats
        self._ref_var.set('(auto)')
        self._on_plat_selection_change()

    def _selection_technologies(self, selected):
        """Technology category per selected platform, and the resolved set.

        ``custom`` and ``unknown`` are not evidence of a second technology,
        only of a platform whose technology could not be resolved, so they are
        excluded from the set the way ``unknown`` species already are.
        """
        tech_map = {p: self.app._platform_facts(p).get('category', 'unknown')
                    for p in selected}
        known = {t for t in tech_map.values() if t not in ('unknown', 'custom')}
        return tech_map, known

    def _apply_scenario_rules(self, selected):
        """Say what the current selection makes the correction good for.

        What separates two platforms decides what a correction can do. Within
        one technology they differ by roughly an affine rescaling, which is what
        a batch correction is built to remove, so corrected values are
        comparable afterwards. Across technologies they differ by an unknown
        monotone transform instead: array intensity saturates where sequencing
        counts do not, so no shift or scaling maps one onto the other. The
        correction is still worth running there, because it puts the platforms
        on one axis to brush a region on, but the corrected values are a
        preparation step and not a result, and what is compared afterwards is
        the labels the region carries.
        """
        tech_map, known = self._selection_technologies(selected)
        cross_tech = len(known) > 1
        if cross_tech:
            self._scenario_label.config(
                text=("Cross-technology comparison: "
                      + ", ".join(f"{p} = {tech_map.get(p, 'unknown')}"
                                  for p in selected)
                      + ".  These platforms measure different quantities, so a "
                        "correction here prepares an axis to select a region on; "
                        "it does not make the values comparable. Compare the "
                        "labels of the selected region, not the values."),
                foreground='#E65100')
        else:
            tech = next(iter(known), None)
            if tech and len(selected) >= 2:
                self._scenario_label.config(
                    text=(f"Same-technology comparison ({tech}): platforms differ "
                          "by scaling and probe content, which is what a batch "
                          "correction removes."),
                    foreground='#1565C0')
            else:
                self._scenario_label.config(text="")
        return cross_tech

    def _on_plat_selection_change(self):
        """Update species comparison hint."""
        selected = [p for p, v in self._plat_vars.items() if v.get()]
        self._apply_scenario_rules(selected)
        species_set = set()
        for p in selected:
            species_set.add(self.app.platform_species(p))

        if len(species_set) > 1:
            sp_list = sorted(species_set)
            pairs = []
            for i, a in enumerate(sp_list):
                for b in sp_list[i + 1:]:
                    key = (a, b) if (a, b) in COMMON_PLATFORM_COMPARISONS else (b, a)
                    if key in COMMON_PLATFORM_COMPARISONS:
                        pairs.append(COMMON_PLATFORM_COMPARISONS[key])
            hint = f"Cross-species comparison detected: {', '.join(sorted(species_set))}"
            if pairs:
                hint += f"  |  {'; '.join(pairs)}"
            hint += "\nGene symbols will be matched case-insensitively; ortholog mapping uses shared gene symbols."
            self._species_label.config(text=hint, foreground='#E65100')
        elif len(species_set) == 1:
            sp = list(species_set)[0]
            self._species_label.config(
                text=f"Same-species comparison ({sp}) - direct gene symbol matching",
                foreground='#1565C0')
        else:
            self._species_label.config(text="")

    # ── Analysis Engine ─────────────────────────────────────────────
    def _start_analysis(self):
        selected = [p for p, v in self._plat_vars.items() if v.get()]
        if len(selected) < 2:
            messagebox.showwarning("Need 2+ Platforms",
                                    "Select at least two platforms to compare.", parent=self)
            return
        if self._running:
            return

        self._running = True
        self._run_btn.config(state=tk.DISABLED)
        self._export_btn.config(state=tk.DISABLED)
        self._progress_bar['value'] = 0
        self._progress_label.config(text="Starting analysis...")

        # Read the settings here, on the thread that owns the widgets. Tk is
        # not thread-safe, and a worker calling ``.get()`` on a Tk variable
        # reaches into the interpreter from the wrong thread.
        settings = {
            'batch_method': self._batch_var.get(),
            'pval_threshold': float(self._pval_var.get()),
            'delta_threshold': float(self._delta_var.get()),
            'reference': self._ref_var.get(),
        }

        thread = threading.Thread(target=self._run_analysis_thread,
                                   args=(selected, settings), daemon=True)
        thread.start()

    def _run_analysis_thread(self, selected_platforms, settings):
        """Run full cross-platform analysis in background thread."""
        try:
            results = self._analyze_platforms(selected_platforms, settings)
            self.after(0, lambda: self._display_results(results))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # Bind the message now: `except ... as e` deletes `e` at the end of
            # the block, and this lambda only runs later on the Tk thread.
            self.after(0, lambda msg=str(e): self._on_analysis_error(msg, tb))
        finally:
            self.after(0, lambda: self._on_analysis_complete())

    def _update_progress(self, value, text):
        self.after(0, lambda v=value, t=text: (
            self._progress_bar.configure(value=v),
            self._progress_label.config(text=t)))

    def _on_analysis_error(self, err, tb):
        messagebox.showerror("Analysis Error", f"{err}\n\n{tb[:500]}", parent=self)

    def _on_analysis_complete(self):
        self._running = False
        self._run_btn.config(state=tk.NORMAL)

    def _analyze_platforms(self, platforms, settings):
        """Core analysis engine - returns the comprehensive results dict.

        The arithmetic lives in ``core.analysis.cross_platform``. This window
        and the assistant call the same function, so the DE count in the tab
        below and the DE count the assistant quotes cannot drift apart.
        """
        from genevariate.core.analysis import analyze_platforms
        return analyze_platforms(
            platforms,
            {p: self.app.gpl_datasets[p] for p in platforms},
            {p: self.app.gpl_gene_mappings.get(p, {}) for p in platforms},
            species={p: self.app.platform_species(p) for p in platforms},
            technology={p: self.app._platform_facts(p).get('category', 'unknown')
                        for p in platforms},
            labels={p: self.app.platform_labels.get(p) for p in platforms},
            reference=settings['reference'],
            batch_method=settings['batch_method'],
            pval_threshold=settings['pval_threshold'],
            delta_threshold=settings['delta_threshold'],
            progress_cb=self._update_progress,
            log_cb=lambda m: self.app.enqueue_log(f"[XPlat] {m}"))

    # ── Display Results ─────────────────────────────────────────────
    def _display_results(self, results):
        self._results = results
        self._export_btn.config(state=tk.NORMAL)
        self._fill_overview_tab(results)
        self._fill_overlap_tab(results)
        self._fill_de_tab(results)
        self._fill_conserved_tab(results)
        self._fill_unique_tab(results)
        self._fill_batch_tab(results)
        self._fill_dist_tab(results)
        self._notebook.select(0)

    def _fill_overview_tab(self, R):
        txt = self._overview_text
        txt.config(state=tk.NORMAL)
        txt.delete('1.0', tk.END)

        ref = R['reference']
        plats = R['platforms']
        L = []
        L.append("=" * 70)
        L.append("CROSS-PLATFORM ANALYSIS REPORT")
        L.append("=" * 70)
        L.append(f"Platforms analyzed:  {len(plats)}")
        L.append(f"Reference platform: {ref}")
        L.append(f"Batch correction:   {R['batch_correction_used']}")
        L.append(f"p-value threshold:  {R['pval_threshold']}")
        L.append(f"|delta-mean| threshold:  {R['delta_threshold']}")
        L.append("")
        L.append("-" * 70)
        hdr = f"{'Platform':<20} {'Species':<12} {'Samples':>8} {'Genes':>8} {'Median':>10} {'Mean':>10} {'Std':>10}"
        L.append(hdr)
        L.append("-" * 70)
        for plat in plats:
            sp = R['species_map'].get(plat, '?')
            n_s = len(self.app.gpl_datasets[plat])
            n_g = len(R['gene_sets'].get(plat, set()))
            med = R['platform_medians'].get(plat, 0)
            mn = R['platform_means'].get(plat, 0)
            sd = R['platform_stds'].get(plat, 0)
            marker = " << REF" if plat == ref else ""
            L.append(f"{plat:<20} {sp:<12} {n_s:>8,} {n_g:>8,} {med:>10.3f} {mn:>10.3f} {sd:>10.3f}{marker}")

        L.append("")
        L.append("-" * 70)
        L.append("GENE OVERLAP SUMMARY")
        L.append("-" * 70)
        L.append(f"Total unique genes across all platforms: {len(R['all_genes']):,}")
        L.append(f"Common to ALL platforms:                 {len(R['common_all']):,}")
        for plat in plats:
            L.append(f"  Unique to {plat}:  {len(R['unique_genes'].get(plat, set())):>8,}")

        L.append("")
        L.append("-" * 70)
        L.append("PAIRWISE PLATFORM CORRELATIONS (Spearman rho on gene means)")
        L.append("-" * 70)
        for (a, b), info in R['plat_correlations'].items():
            corr = info['spearman']
            n_cmp = info['n_genes_compared']
            shared = len(R['pairwise_overlap'].get((a, b), set()))
            sp_a = R['species_map'].get(a, '?')
            sp_b = R['species_map'].get(b, '?')
            if 'unknown' in (sp_a, sp_b):
                same_sp = "species not resolved"
            elif sp_a == sp_b:
                same_sp = "same-species"
            else:
                same_sp = f"CROSS-SPECIES ({sp_a}<->{sp_b})"
            corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
            L.append(f"  {a} vs {b}: rho={corr_str}  ({n_cmp:,} genes, {shared:,} shared)  [{same_sp}]")

        L.append("")
        L.append("-" * 70)
        L.append("EXPRESSION COMPARISON RESULTS")
        L.append("-" * 70)
        L.append(f"Genes tested:                  {R['n_tested']:,}")
        L.append(f"DE genes (cross-platform):     {R['n_de']:,}  (adj p < {R['pval_threshold']}, |delta| > {R['delta_threshold']})")
        L.append(f"Conserved genes:               {R['n_conserved']:,}  (p > 0.5, |delta| < {R['delta_threshold'] * 0.5})")
        if R.get('has_study_ids'):
            dfe = R.get('median_design_effect', 1.0)
            L.append(f"Study clumping (median):       design effect {dfe:.1f}x  "
                     f"-- p-values widened by this, since samples arrive in studies")
        else:
            L.append("Study clumping:                no series_id on these platforms, "
                     "so p-values assume every sample is an independent experiment")

        L.append("")
        L.append("-" * 70)
        L.append("BATCH EFFECT ASSESSMENT")
        L.append("-" * 70)
        score = R['batch_effect_score']
        detected = R['batch_effect_detected']
        if R.get('batch_score_is_unit_difference'):
            status = "scale difference between technologies, not a batch effect"
        elif detected:
            status = "!! DETECTED -- consider batch correction"
        else:
            status = "OK No significant batch effect"
        L.append(f"Batch effect score:  {score:.4f}  ({status})")
        for plat in plats:
            bm = R['batch_metrics'][plat]
            L.append(f"  {plat}: median_shift={bm['median_shift_from_ref']:+.3f}, "
                     f"variance_ratio={bm['variance_ratio_to_ref']:.3f}")
        if R.get('batch_score_is_unit_difference'):
            techs = R.get('tech_map', {})
            L.append("")
            L.append("  The selected platforms span "
                     + ", ".join(f"{p} ({techs.get(p, 'unknown')})" for p in plats)
                     + ".")
            L.append("  A batch effect is unwanted variation within one assay, where every")
            L.append("  batch measures the same quantity and the offset can be removed. These")
            L.append("  platforms measure different quantities, so the shift above is a")
            L.append("  difference of units and no batch claim is made from it. A correction")
            L.append("  is still useful here as an axis to select a region on; what is")
            L.append("  compared afterwards is the labels that region carries.")

        if R['is_cross_species']:
            L.append("")
            L.append("-" * 70)
            L.append("CROSS-SPECIES NOTES")
            L.append("-" * 70)
            L.append("Gene matching is by shared gene symbol (case-insensitive).")
            L.append("Many orthologs share symbols (TP53/Tp53, GAPDH/Gapdh).")
            L.append(f"Cross-species shared genes found: {len(R['common_all']):,}")
            L.append("For comprehensive ortholog mapping, consider external tools (BioMart, ENSEMBL).")

        txt.insert('1.0', "\n".join(L))
        txt.config(state=tk.DISABLED)

    def _fill_overlap_tab(self, R):
        for w in self._overlap_tab.winfo_children():
            w.destroy()

        plats = R['platforms']
        # Two results share this tab and both grow with the data - the matrix
        # with the number of platforms, the gene list with the overlap. Stacked
        # with pack, the second one was squeezed to a single visible row. A
        # divider lets the reader give the space to whichever they are reading.
        panes = ttk.PanedWindow(self._overlap_tab, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(panes)
        bottom = ttk.Frame(panes)
        panes.add(top, weight=1)
        panes.add(bottom, weight=2)

        ttk.Label(top,
                  text="Gene Overlap Matrix (shared gene count between each pair)",
                  font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        cols = ['Platform'] + plats
        tree = ttk.Treeview(top, columns=cols, show='headings',
                            height=len(plats) + 1)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor=tk.CENTER)
        tree.column('Platform', width=150, anchor=tk.W)

        for plat_a in plats:
            row = [plat_a]
            for plat_b in plats:
                if plat_a == plat_b:
                    row.append(f"{len(R['gene_sets'][plat_a]):,}")
                else:
                    key = (plat_a, plat_b) if (plat_a, plat_b) in R['pairwise_overlap'] else (plat_b, plat_a)
                    row.append(f"{len(R['pairwise_overlap'].get(key, set())):,}")
            tree.insert('', tk.END, values=row)

        tree.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(bottom,
                  text=f"Genes common to ALL {len(plats)} platforms: {len(R['common_all']):,}",
                  font=('Segoe UI', 11, 'bold'), foreground='#2E7D32').pack(anchor=tk.W, padx=10)

        # A table, not a listbox: a list of genes is a result, and every table
        # in the program carries a Save button (gui/exporting.py) while a
        # listbox offers the user no way to take its contents away.
        list_frame = ttk.Frame(bottom)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        common = ttk.Treeview(list_frame, columns=('gene',), show='headings',
                              height=6)
        common.heading('gene', text='Gene')
        common.column('gene', anchor='center', minwidth=120, stretch=True)
        sb = ttk.Scrollbar(list_frame, command=common.yview)
        common.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        common.pack(fill=tk.BOTH, expand=True)
        for gene in sorted(R['common_all']):
            common.insert('', tk.END, values=(gene,))

    def _build_gene_treeview(self, parent, genes, platforms, ref):
        """Build a sortable treeview for gene stats. Returns the tree widget."""
        cols = ['Gene', 'AdjP', 'MaxDelta', 'RefMean', 'RefStd', 'NPlatforms']
        for plat in platforms:
            if plat != ref:
                cols.append(f"D_{plat[:12]}")

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=20)
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        widths = {'Gene': 100, 'AdjP': 90, 'MaxDelta': 90, 'RefMean': 85,
                  'RefStd': 75, 'NPlatforms': 70}
        for c in cols:
            tree.heading(c, text=c,
                         command=lambda _c=c: self._sort_treeview(tree, _c, False))
            tree.column(c, width=widths.get(c, 85), anchor=tk.CENTER)
        tree.column('Gene', anchor=tk.W)

        for g in genes[:2000]:
            row = [
                g['gene'],
                f"{g.get('adj_pval', g['min_pval']):.2e}",
                f"{g['max_abs_delta']:.3f}",
                f"{g['ref_mean']:.3f}",
                f"{g['ref_std']:.3f}",
                str(g['n_platforms']),
            ]
            for plat in platforms:
                if plat != ref:
                    pd_info = g['platform_details'].get(plat, {})
                    row.append(f"{pd_info.get('delta_mean', 0):+.3f}")
            tree.insert('', tk.END, values=row)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        hsb.pack(fill=tk.X, padx=10)
        # Say so when the table is not the whole result: an unmarked cut looks
        # like the analysis found 2,000 genes when it found more.
        if len(genes) > 2000:
            ttk.Label(parent,
                      text=f"Showing the first 2,000 of {len(genes):,} genes "
                           f"(ranked as sorted above).",
                      foreground=AERO["warn"],
                      font=('Segoe UI', 9, 'italic')).pack(anchor=tk.W, padx=10)
        return tree

    def _sort_treeview(self, tree, col, reverse):
        """Sort treeview by clicking column header."""
        data = [(tree.set(k, col), k) for k in tree.get_children('')]
        try:
            data.sort(key=lambda t: float(t[0].replace(',', '')), reverse=reverse)
        except (ValueError, TypeError):
            data.sort(key=lambda t: t[0], reverse=reverse)
        for idx, (val, k) in enumerate(data):
            tree.move(k, '', idx)
        tree.heading(col, command=lambda: self._sort_treeview(tree, col, not reverse))

    def _fill_de_tab(self, R):
        for w in self._de_tab.winfo_children():
            w.destroy()

        de = R['de_genes']
        ref = R['reference']
        plats = R['platforms']

        hdr = ttk.Frame(self._de_tab)
        hdr.pack(fill=tk.X, padx=10, pady=5)
        n_de = len(de)
        color = '#C62828' if n_de > 0 else '#2E7D32'
        ttk.Label(hdr, text=f"Differentially Expressed Genes: {n_de:,}",
                  font=('Segoe UI', 12, 'bold'), foreground=color).pack(side=tk.LEFT)
        ttk.Label(hdr,
                  text=f"  (adj p < {R['pval_threshold']}, |Dmean| > {R['delta_threshold']}, ref={ref})",
                  font=('Segoe UI', 9, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=10)

        if n_de > 0:
            ttk.Label(self._de_tab,
                      text="These genes show significant expression differences BETWEEN platforms "
                           "(likely batch effects or biological differences). "
                           "Consider batch correction before cross-platform DE analysis.",
                      foreground='#E65100', wraplength=1100,
                      font=('Segoe UI', 9)).pack(padx=10, pady=2)
            self._build_gene_treeview(self._de_tab, de, plats, ref)
        else:
            ttk.Label(self._de_tab,
                      text="No significant cross-platform DE genes found - platforms are well-aligned.",
                      foreground='#2E7D32', font=('Segoe UI', 11)).pack(pady=30)

    def _fill_conserved_tab(self, R):
        for w in self._conserved_tab.winfo_children():
            w.destroy()

        conserved = R['conserved_genes']
        ref = R['reference']
        plats = R['platforms']

        ttk.Label(self._conserved_tab,
                  text=f"Conserved Genes (similar expression across platforms): {len(conserved):,}",
                  font=('Segoe UI', 12, 'bold'), foreground='#2E7D32').pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(self._conserved_tab,
                  text="No test found a difference between platforms for these genes, and the "
                       "largest difference in mean is under half the threshold. A large p is not "
                       "proof of agreement, so it is the small delta that qualifies them as "
                       "candidate normalization anchors; the p only says nothing argues against it.",
                  foreground='gray', wraplength=1100,
                  font=('Segoe UI', 9, 'italic')).pack(padx=10, pady=2)

        if conserved:
            self._build_gene_treeview(self._conserved_tab, conserved, plats, ref)
        else:
            ttk.Label(self._conserved_tab,
                      text="No strongly conserved genes found with current thresholds.",
                      foreground='gray', font=('Segoe UI', 10)).pack(pady=30)

    def _fill_unique_tab(self, R):
        for w in self._unique_tab.winfo_children():
            w.destroy()

        plats = R['platforms']
        unique = R['unique_genes']

        ttk.Label(self._unique_tab,
                  text="Platform-Specific Genes (present on only one platform)",
                  font=('Segoe UI', 12, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        # One tab per platform put the platform count in charge of the tab
        # strip, which ttk clips rather than scrolls: past about eight
        # platforms the later tabs could not be reached. A single table
        # carrying the platform as a column holds any number of them, and
        # being a table it comes with a Save button.
        counts = ", ".join(f"{p}: {len(unique.get(p, set())):,}"
                           for p in plats)
        ttk.Label(self._unique_tab, text=counts, foreground=AERO["muted"],
                  font=('Segoe UI', 9)).pack(anchor=tk.W, padx=10)

        holder = ttk.Frame(self._unique_tab)
        holder.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tree = ttk.Treeview(holder, columns=('platform', 'gene'),
                            show='headings')
        for col, title in (('platform', 'Platform'), ('gene', 'Gene')):
            tree.heading(col, text=title)
            tree.column(col, anchor='center', minwidth=120, stretch=True)
        sb = ttk.Scrollbar(holder, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        for plat in plats:
            for gene in sorted(unique.get(plat, set())):
                tree.insert('', tk.END, values=(plat, gene))

    def _fill_batch_tab(self, R):
        for w in self._batch_tab.winfo_children():
            w.destroy()

        plats = R['platforms']
        ref = R['reference']
        score = R['batch_effect_score']
        detected = R['batch_effect_detected']

        unit_diff = R.get('batch_score_is_unit_difference')
        if unit_diff:
            hdr_bg, hdr_fg = "#FFF3E0", "#E65100"
            status = f"Scale difference between technologies (score: {score:.4f})"
        elif detected:
            hdr_bg, hdr_fg = "#FFF3E0", "#E65100"
            status = f"!! Batch Effect DETECTED (score: {score:.4f})"
        else:
            hdr_bg, hdr_fg = "#E8F5E9", "#2E7D32"
            status = f"OK No Significant Batch Effect (score: {score:.4f})"

        hdr = tk.Label(self._batch_tab, text=status, bg=hdr_bg, fg=hdr_fg,
                       font=('Segoe UI', 13, 'bold'), pady=8)
        hdr.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self._batch_tab,
                  text=f"Correction applied: {R['batch_correction_used']}",
                  font=('Segoe UI', 10)).pack(anchor=tk.W, padx=10, pady=2)
        if unit_diff:
            techs = R.get('tech_map', {})
            ttk.Label(
                self._batch_tab,
                text=("These platforms do not measure the same quantity: "
                      + ", ".join(f"{p} = {techs.get(p, 'unknown')}" for p in plats)
                      + ". A batch effect is unwanted variation within one assay, "
                      "where each batch measures the same quantity and the offset "
                      "can be removed. The score below compares the pooled median "
                      "of one measurand against another, so it is large by "
                      "construction and no batch claim is made from it. Use the "
                      "rank-based comparisons, which stay defined across "
                      "technologies."),
                foreground='#8A4B08', wraplength=1100,
                font=('Segoe UI', 9)).pack(anchor=tk.W, padx=10, pady=(2, 6))
        if R.get('batch_correction_caveat'):
            ttk.Label(self._batch_tab, text=R['batch_correction_caveat'],
                      foreground='#8A4B08', wraplength=1100,
                      font=('Segoe UI', 9)).pack(anchor=tk.W, padx=10, pady=(0, 6))
        ttk.Label(self._batch_tab,
                  text="Batch effect score = std(platform medians) / mean(platform stds). "
                       "Values > 0.2 suggest systematic shifts between platforms.",
                  foreground='gray', wraplength=1100,
                  font=('Segoe UI', 9, 'italic')).pack(padx=10, pady=5)

        cols = ['Platform', 'Species', 'Median', 'Mean', 'Std',
                'MedShift_vs_Ref', 'MeanShift_vs_Ref', 'VarRatio']
        tree = ttk.Treeview(self._batch_tab, columns=cols, show='headings',
                            height=len(plats) + 1)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=130, anchor=tk.CENTER)
        tree.column('Platform', width=140, anchor=tk.W)

        for plat in plats:
            bm = R['batch_metrics'][plat]
            sp = R['species_map'].get(plat, '?')
            marker = " << REF" if plat == ref else ""
            tree.insert('', tk.END, values=[
                f"{plat}{marker}", sp,
                f"{bm['median']:.3f}", f"{bm['mean']:.3f}", f"{bm['std']:.3f}",
                f"{bm['median_shift_from_ref']:+.3f}",
                f"{bm['mean_shift_from_ref']:+.3f}",
                f"{bm['variance_ratio_to_ref']:.3f}",
            ])
        tree.pack(fill=tk.X, padx=10, pady=5)

        ttk.Separator(self._batch_tab).pack(fill=tk.X, padx=10, pady=8)
        rec_frame = labelframe(self._batch_tab, text="Recommendations", padding=10)
        rec_frame.pack(fill=tk.X, padx=10, pady=5)

        recs = []
        if detected and R['batch_correction_used'] == 'none':
            recs.append("Batch effect detected -- re-run with 'median_centering' or 'combat' correction")
            recs.append("Median centering: fast, simple, corrects global shift -- good first step")
            recs.append("ComBat: gold standard parametric correction -- install pycombat: pip install pycombat")
        if R['is_cross_species']:
            recs.append("Cross-species comparison: batch effects may overlap with true biological differences")
            recs.append("Consider using conserved genes as normalization anchors")
            recs.append("Gene symbol matching captures ~60-70% of one-to-one orthologs")
        if unit_diff:
            # `detected` is False here because the claim was withheld, not
            # because the platforms agree. Saying they are well-aligned would
            # read that withheld claim backwards.
            recs.append("Different technologies: a correction here prepares one axis to "
                        "select a region on; compare the labels the region carries, in "
                        "Region Analysis > Comparison > Pool across platforms")
            recs.append("Per-gene quantile normalization (feature-specific QN) is the "
                        "published cross-technology method, but it equalizes each gene's "
                        "distribution and so removes shape and modality differences")
            recs.append("ComBat fits a batch offset on a shared measurand, which platforms "
                        "of different technologies do not have")
        elif not detected:
            recs.append("Platforms appear well-aligned -- direct cross-platform analysis is reasonable")
        if R['batch_correction_used'] != 'none':
            recs.append(f"Correction applied: {R['batch_correction_used']} -- check DE genes tab for remaining differences")

        for rec in recs:
            ttk.Label(rec_frame, text=f"  {rec}", font=('Segoe UI', 9),
                      wraplength=1050).pack(anchor=tk.W, pady=1)

    def _fill_dist_tab(self, R):
        for w in self._dist_tab.winfo_children():
            w.destroy()

        gene_stats = R['gene_stats']
        if not gene_stats:
            ttk.Label(self._dist_tab, text="No gene statistics available.",
                      foreground='gray').pack(pady=30)
            return

        ttk.Label(self._dist_tab,
                  text=f"Distribution Metrics for {len(gene_stats):,} genes (click column headers to sort)",
                  font=('Segoe UI', 12, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        self._build_gene_treeview(self._dist_tab, gene_stats, R['platforms'], R['reference'])

    # ── Export ──────────────────────────────────────────────────────
    def _export_full_report(self):
        if not self._results:
            return

        folder = filedialog.askdirectory(title="Select Export Folder", parent=self)
        if not folder:
            return

        R = self._results
        prefix = f"xplat_{'_vs_'.join(R['platforms'][:3])}"

        self._overview_text.config(state=tk.NORMAL)
        with open(os.path.join(folder, f"{prefix}_overview.txt"), 'w') as f:
            f.write(self._overview_text.get('1.0', tk.END))
        self._overview_text.config(state=tk.DISABLED)

        if R['de_genes']:
            rows = []
            for g in R['de_genes']:
                row = {'gene': g['gene'], 'adj_pval': g.get('adj_pval', g['min_pval']),
                       'max_abs_delta_mean': g['max_abs_delta'],
                       'ref_platform': g['ref_platform'],
                       'ref_mean': g['ref_mean'], 'ref_std': g['ref_std']}
                for plat, info in g['platform_details'].items():
                    if plat != g['ref_platform']:
                        row[f'{plat}_mean'] = info['mean']
                        row[f'{plat}_delta'] = info['delta_mean']
                        row[f'{plat}_pval'] = info['pval']
                        row[f'{plat}_effect_size'] = info.get('effect_size', '')
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(folder, f"{prefix}_DE_genes.csv"), index=False)

        if R['conserved_genes']:
            rows = [{'gene': g['gene'], 'max_pval': g['max_pval'],
                     'max_abs_delta': g['max_abs_delta'], 'ref_mean': g['ref_mean']}
                    for g in R['conserved_genes']]
            pd.DataFrame(rows).to_csv(os.path.join(folder, f"{prefix}_conserved_genes.csv"), index=False)

        rows = [{'gene': g['gene'], 'adj_pval': g.get('adj_pval', ''),
                 'max_abs_delta': g['max_abs_delta'], 'ref_mean': g['ref_mean'],
                 'ref_std': g['ref_std'], 'n_platforms': g['n_platforms'],
                 'is_de': g['is_de']} for g in R['gene_stats']]
        pd.DataFrame(rows).to_csv(os.path.join(folder, f"{prefix}_all_gene_stats.csv"), index=False)

        rows = [{'platform': p, 'total_genes': len(R['gene_sets'][p]),
                 'unique_genes': len(R['unique_genes'].get(p, set())),
                 'common_all': len(R['common_all'])} for p in R['platforms']]
        pd.DataFrame(rows).to_csv(os.path.join(folder, f"{prefix}_gene_overlap.csv"), index=False)

        for plat in R['platforms']:
            genes = sorted(R['unique_genes'].get(plat, set()))
            if genes:
                with open(os.path.join(folder, f"{prefix}_{plat}_unique_genes.txt"), 'w') as f:
                    f.write("\n".join(genes))

        messagebox.showinfo("Export Complete",
                            f"Reports exported to:\n{folder}\n\n"
                            f"Files: overview, DE genes, conserved genes,\n"
                            f"all gene stats, gene overlap, unique gene lists",
                            parent=self)
        self.app.enqueue_log(f"[XPlat] Reports exported to {folder}")


class _UIAnimator:
    """Lightweight animation helpers for Tk/ttk widgets.

    All animation callbacks go through root.after() on the UI thread, so it's
    safe to start/stop from background threads via self.root.after(0, ...).

    Provides:
      - spinner(label, base_text):       cycling braille glyph + base text
      - pulse_bar(progressbar):          shimmering color pulse while busy
      - smooth_to(progressbar, target):  interpolated value change ("liquid")
      - fade_color(widget, a, b):        color cross-fade for status labels
      - stop(widget) / stop_all()
    """

    # Smooth "dots" animation using braille patterns (looks like a liquid spinner)
    SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    # Alternate: rotating arc for smaller spaces
    ORBIT_FRAMES = ['◐', '◓', '◑', '◒']
    # Pulse palette - shades of green for "working" bar
    PULSE_GREENS = ['#4CAF50', '#66BB6A', '#81C784', '#66BB6A']
    PULSE_BLUES  = ['#1976D2', '#42A5F5', '#64B5F6', '#42A5F5']

    def __init__(self, root):
        self.root = root
        self._anim = {}  # widget-id -> state dict

    # ---------- core plumbing ----------
    def _cancel(self, key):
        st = self._anim.pop(key, None)
        if not st:
            return
        st['running'] = False
        job = st.get('job')
        if job:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass

    def _schedule(self, key, state, fn, interval_ms):
        if not state.get('running'):
            return
        try:
            state['job'] = self.root.after(interval_ms, fn)
        except Exception:
            state['running'] = False

    # ---------- spinner on status label ----------
    def spinner(self, label, base_text, interval_ms=80, frames=None, color=None):
        """Cycle a spinner glyph in front of base_text on a label.
        Call .stop(label) (or update_spinner_text) when the task finishes.
        """
        frames = frames or self.SPINNER_FRAMES
        key = id(label)
        self._cancel(key)
        state = {'running': True, 'i': 0, 'base': base_text,
                 'frames': frames, 'color': color}
        self._anim[key] = state

        def tick():
            if not state['running']:
                return
            try:
                if not label.winfo_exists():
                    state['running'] = False
                    return
                glyph = state['frames'][state['i'] % len(state['frames'])]
                cfg = {'text': f"{glyph}  {state['base']}"}
                if state['color'] is not None:
                    cfg['foreground'] = state['color']
                try:
                    label.config(**cfg)
                except tk.TclError:
                    state['running'] = False
                    return
                state['i'] += 1
                self._schedule(key, state, tick, interval_ms)
            except Exception:
                state['running'] = False

        tick()

    def update_spinner_text(self, label, new_base_text):
        """Update the base text of a currently-animated spinner label."""
        key = id(label)
        st = self._anim.get(key)
        if st:
            st['base'] = new_base_text

    # ---------- progress bar pulse (liquid shimmer) ----------
    def pulse_bar(self, pbar, style_name=None, palette=None, interval_ms=180):
        """Subtle color pulse for ttk.Progressbar while a task is active.

        Does nothing on CTkProgressBar (fall back is harmless).
        """
        palette = palette or self.PULSE_GREENS
        style_name = style_name or f"Liquid{id(pbar)}.Horizontal.TProgressbar"
        key = ('pulse', id(pbar))
        self._cancel(key)

        try:
            st = ttk.Style()
            # Configure a dedicated style so we don't stomp on other bars
            st.configure(style_name, thickness=18, troughcolor='#E8F5E9',
                         background=palette[0], bordercolor=palette[0],
                         lightcolor=palette[0], darkcolor=palette[0])
            try:
                pbar.configure(style=style_name)
            except Exception:
                return
        except Exception:
            return

        state = {'running': True, 'i': 0, 'style': style_name,
                 'palette': palette}
        self._anim[key] = state

        def tick():
            if not state['running']:
                return
            try:
                if not pbar.winfo_exists():
                    state['running'] = False
                    return
                c = state['palette'][state['i'] % len(state['palette'])]
                ttk.Style().configure(state['style'],
                                      background=c, bordercolor=c,
                                      lightcolor=c, darkcolor=c)
                state['i'] += 1
                self._schedule(key, state, tick, interval_ms)
            except Exception:
                state['running'] = False

        tick()

    def stop_pulse(self, pbar):
        self._cancel(('pulse', id(pbar)))

    # ---------- smooth progress-value interpolation ----------
    def smooth_to(self, pbar, target, duration_ms=260, steps=14):
        """Interpolate pbar['value'] from its current value to `target`.
        Makes 10%→30% jumps feel liquid instead of snapping.
        """
        key = ('smooth', id(pbar))
        self._cancel(key)
        try:
            current = float(pbar['value'])
            maxv = float(pbar['maximum']) or 100.0
        except Exception:
            return
        target = max(0.0, min(target, maxv))
        delta = target - current
        if abs(delta) < 0.3:
            try:
                pbar['value'] = target
            except Exception:
                pass
            return

        step_ms = max(15, duration_ms // steps)
        inc = delta / steps
        state = {'running': True, 'i': 0, 'cur': current}
        self._anim[key] = state

        def tick():
            if not state['running']:
                return
            try:
                if not pbar.winfo_exists():
                    state['running'] = False
                    return
                state['i'] += 1
                if state['i'] >= steps:
                    pbar['value'] = target
                    state['running'] = False
                    return
                state['cur'] += inc
                pbar['value'] = state['cur']
                self._schedule(key, state, tick, step_ms)
            except Exception:
                state['running'] = False

        tick()

    # ---------- soft color fade for status labels ----------
    def flash(self, label, color, revert_to='#555', duration_ms=900):
        """Briefly flash a label's foreground color (e.g. for success/error)."""
        key = ('flash', id(label))
        self._cancel(key)
        try:
            label.config(foreground=color)
        except Exception:
            return
        state = {'running': True}
        self._anim[key] = state

        def revert():
            if not state['running']:
                return
            try:
                if label.winfo_exists():
                    label.config(foreground=revert_to)
            except Exception:
                pass
            state['running'] = False

        try:
            state['job'] = self.root.after(duration_ms, revert)
        except Exception:
            pass

    # ---------- cleanup ----------
    def stop(self, widget):
        self._cancel(id(widget))
        self._cancel(('pulse', id(widget)))
        self._cancel(('smooth', id(widget)))
        self._cancel(('flash', id(widget)))

    def stop_all(self):
        for key in list(self._anim.keys()):
            self._cancel(key)


#: Which columns of a label file are labels. It lives in the Tk-free core
#: because the assistant has to answer that question too, and a chat tool
#: cannot import this module.
semantic_label_columns = label_entities.semantic_label_columns


def column_picker_dialog(parent, col_vars, levels=None, n_rows=0,
                         title="Label columns", on_close=None):
    """A scrollable checklist of *every* column, whatever the file looks like.

    Label files are not standardised: one may call the tissue column
    ``final_Tissue``, another ``tissue``, another ``Site of biopsy``, and a
    file may carry twenty of them. Laying the checkbuttons out in a row works
    for three columns and hides the rest off the edge of the window for
    twenty, so the choice lives in this dialog instead: it scrolls, it shows
    how many distinct values each column has, and it never drops a column the
    program failed to recognise.

    *col_vars* maps column name → ``BooleanVar`` and is edited in place;
    *on_close* is called when the user is done.
    """
    levels = levels or {}
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent)
    ensure_theme(dlg)
    style_window(dlg)

    ttk.Label(dlg, text="Tick the columns that hold labels.",
              font=("Segoe UI", 10, "bold")).pack(
                  anchor="w", padx=12, pady=(12, 0))
    ttk.Label(dlg,
              text=("Values per column are shown on the right. A label has "
                    "a few repeated values; a column with one value per "
                    "sample is an identifier, not a label."),
              foreground=AERO["muted"], font=("Segoe UI", 9, "italic"),
              wraplength=420, justify="left").pack(
                  anchor="w", padx=12, pady=(2, 8))

    # Scrollable body - the number of columns is not bounded.
    body = ttk.Frame(dlg)
    body.pack(fill=tk.BOTH, expand=True, padx=12)
    canvas = tk.Canvas(body, highlightthickness=0, width=440, height=340,
                       bg=AERO["panel"])
    sb = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    # Keep the rows as wide as the canvas so the value counts sit against the
    # right edge instead of hugging the longest column name.
    canvas.bind("<Configure>",
                lambda e: canvas.itemconfigure(win_id, width=e.width))
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(
                        -1 if e.delta > 0 else 1, "units"))

    for col, var in col_vars.items():
        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=1)
        ttk.Checkbutton(row, text=col, variable=var).pack(side=tk.LEFT)
        lv = levels.get(col, 0)
        note = f"{lv} value{'' if lv == 1 else 's'}"
        if n_rows and lv >= n_rows:
            note += " \u2014 one per sample"
        ttk.Label(row, text=note, foreground=AERO["muted"],
                  font=("Segoe UI", 9)).pack(side=tk.RIGHT)

    def set_all(state):
        for v in col_vars.values():
            v.set(state)

    def reset_detected():
        detected = set(semantic_label_columns(col_vars))
        for c, v in col_vars.items():
            v.set(c in detected)

    btns = ttk.Frame(dlg)
    btns.pack(fill=tk.X, padx=12, pady=10)
    ttk.Button(btns, text="\u2611 All", command=lambda: set_all(True),
               style="Secondary.TButton").pack(side=tk.LEFT)
    ttk.Button(btns, text="\u2610 None", command=lambda: set_all(False),
               style="Secondary.TButton").pack(side=tk.LEFT, padx=6)
    ttk.Button(btns, text="\u21bb Detected labels", command=reset_detected,
               style="Secondary.TButton").pack(side=tk.LEFT)

    def close():
        canvas.unbind_all("<MouseWheel>")
        if on_close is not None:
            on_close()
        dlg.destroy()

    ttk.Button(btns, text="Done", command=close,
               style="Primary.TButton").pack(side=tk.RIGHT)
    dlg.protocol("WM_DELETE_WINDOW", close)
    return dlg


def _fmt_p(p) -> str:
    """Format a p/q value, without printing an underflow as an exact zero.

    A Fisher test on thousands of samples routinely returns a p below the
    smallest positive double, and "%.2e" then reads 0.00e+00 - a probability
    of exactly zero, which the test never claims. The leading "<" is stripped
    again by :meth:`LabelEnrichmentWindow._sort_table` so the column still
    sorts numerically.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    if p <= 0.0:
        return "<1e-308"
    return f"{p:.2e}"


class LabelEnrichmentWindow(tk.Toplevel):
    """Fisher / hypergeometric enrichment of LLM-extracted labels.

    Tests whether each categorical label value (e.g. tissue="liver", condition="control")
    is over- or under-represented in a chosen *foreground* group compared to a
    *background* universe. Produces:

        * bar chart (top-N fold-change with significance stars)
        * dot plot (fold-change × k, color = -log10(q))
        * volcano (log2 FC vs -log10 q)
        * enrichment table (CSV export)

    Foreground selection strategies:
        * Gene region: samples whose expression of gene G is in [lo, hi]
        * Gene top-K percent: samples in the top K% of gene G's expression
        * Platform comparison: use platform A as fg, platform B as bg
        * Custom mask (via label value): e.g. "tissue == 'liver'"
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.app = parent  # main GUI reference
        self.title("Label Enrichment - LLM Label × Expression / Platform")
        self.transient(parent)
        ensure_theme(self)
        style_window(self)

        # State ---------------------------------------------------------
        self._current_enrichment = None  # last pd.DataFrame of results
        self._figures = {}               # key → matplotlib Figure
        self._canvases = {}              # key → FigureCanvasTkAgg

        # Layout --------------------------------------------------------
        self._build_ui()

        # Populate pickers based on what the main app has loaded
        self._populate_pickers()

        try:
            self.app._fit_window(self, fallback_w=1150, fallback_h=780)
        except Exception:
            self.geometry("1150x780")

    # ─── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top: configuration panel
        cfg = labelframe(self, text=" Enrichment configuration ", padding=10)
        cfg.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Row 1 - platform + foreground strategy
        r1 = ttk.Frame(cfg); r1.pack(fill=tk.X, pady=4)
        ttk.Label(r1, text="Platform:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.plat_var = tk.StringVar()
        self.plat_combo = ttk.Combobox(r1, textvariable=self.plat_var,
                                        width=18, state="readonly")
        self.plat_combo.pack(side=tk.LEFT, padx=(6, 16))
        self.plat_combo.bind("<<ComboboxSelected>>", lambda e: self._on_platform_change())

        ttk.Label(r1, text="Foreground:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="top_percent")
        strategies = [
            ("Gene - top %",       "top_percent"),
            ("Gene - bottom %",    "bottom_percent"),
            ("Gene - expression range", "range"),
            ("Label value",        "label_value"),
        ]
        for text, val in strategies:
            ttk.Radiobutton(r1, text=text, variable=self.strategy_var, value=val,
                            command=self._on_strategy_change
                            ).pack(side=tk.LEFT, padx=4)

        # Row 2 - gene picker
        r2 = ttk.Frame(cfg); r2.pack(fill=tk.X, pady=4)
        ttk.Label(r2, text="Gene symbol:", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.gene_var = tk.StringVar()
        self.gene_combo = ttk.Combobox(r2, textvariable=self.gene_var,
                                        width=22)
        self.gene_combo.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(r2, text="Threshold:", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="10")
        ttk.Entry(r2, textvariable=self.threshold_var, width=8).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(r2, text="(percent for top/bottom %)",
                  font=("Segoe UI", 8, "italic"),
                  foreground=AERO["muted"]).pack(side=tk.LEFT, padx=(2, 16))

        ttk.Label(r2, text="Range:", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.range_lo_var = tk.StringVar()
        self.range_hi_var = tk.StringVar()
        ttk.Entry(r2, textvariable=self.range_lo_var, width=8).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Label(r2, text="…").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.range_hi_var, width=8).pack(side=tk.LEFT, padx=(2, 4))

        # Row 3 - label columns + min count
        r3 = ttk.Frame(cfg); r3.pack(fill=tk.X, pady=4)
        ttk.Label(r3, text="Label columns to test:",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.label_cols_frame = ttk.Frame(r3)
        self.label_cols_frame.pack(side=tk.LEFT, padx=(8, 16))
        self.label_col_vars = {}       # col name → BooleanVar (every candidate)
        self._label_col_levels = {}    # col name → distinct-value count
        self._label_cols_summary = tk.StringVar(value="(pick a platform)")

        ttk.Label(r3, text="Min count:").pack(side=tk.LEFT)
        self.min_count_var = tk.StringVar(value="3")
        ttk.Entry(r3, textvariable=self.min_count_var, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(r3, text="α:").pack(side=tk.LEFT, padx=(12, 0))
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Entry(r3, textvariable=self.alpha_var, width=5).pack(side=tk.LEFT, padx=4)

        # Row 3b - how much of the result to draw. This is a *view* control:
        # every term is always tested, kept in the table and written to the CSV;
        # these only decide what fits on screen, so nothing here can bias a
        # result towards one dataset.
        r3b = ttk.Frame(cfg); r3b.pack(fill=tk.X, pady=4)
        ttk.Label(r3b, text="Plot display:",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(r3b, text="show top").pack(side=tk.LEFT, padx=(8, 4))
        self.plot_top_var = tk.StringVar(value="0")
        top_entry = ttk.Entry(r3b, textvariable=self.plot_top_var, width=6)
        top_entry.pack(side=tk.LEFT)
        top_entry.bind("<Return>", lambda ev: self._redraw_plots())
        ttk.Label(r3b, text="terms (0 = all)",
                  font=("Segoe UI", 8, "italic"),
                  foreground=AERO["muted"]).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(r3b, text="ranked by").pack(side=tk.LEFT)
        self.rank_by_var = tk.StringVar(value="p-value")
        rank_combo = ttk.Combobox(r3b, textvariable=self.rank_by_var, width=18,
                                  state="readonly",
                                  values=list(self._RANK_KEYS))
        rank_combo.pack(side=tk.LEFT, padx=(6, 16))
        rank_combo.bind("<<ComboboxSelected>>", lambda ev: self._redraw_plots())
        ttk.Button(r3b, text="\u21bb Redraw", command=self._redraw_plots,
                   style="Secondary.TButton").pack(side=tk.LEFT)

        # Row 4 - run / export
        r4 = ttk.Frame(cfg); r4.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(r4, text="Run Enrichment",
                   command=self._run_enrichment,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(r4, text="Export Table (CSV)",
                   command=self._export_csv,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(r4, text="Export All",
                   command=self._export_all_plots,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(r4, textvariable=self.status_var,
                  foreground=AERO["muted"],
                  font=("Segoe UI", 9, "italic")).pack(side=tk.LEFT, padx=16)

        # Bottom: notebook with plots + table ---------------------------
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self.tab_bar = ttk.Frame(self.nb); self.nb.add(self.tab_bar, text="Bar chart")
        self.tab_dot = ttk.Frame(self.nb); self.nb.add(self.tab_dot, text="Dot plot")
        self.tab_volc = ttk.Frame(self.nb); self.nb.add(self.tab_volc, text="Volcano")
        self.tab_tbl = ttk.Frame(self.nb); self.nb.add(self.tab_tbl, text="Table")

        # Table widget. n_gse / n_eff are not decoration: a term carried by one
        # study is one observation however many samples that study deposited,
        # and the last two columns are what say so.
        cols = ("term", "k", "K", "n", "N", "fold_change",
                "odds_ratio", "p_value", "q_value", "n_gse", "n_eff")
        self.table = ttk.Treeview(self.tab_tbl, columns=cols, show="headings", height=18)
        for c in cols:
            self.table.heading(c, text=c, command=lambda cc=c: self._sort_table(cc))
            self.table.column(c, width=100 if c == "term" else 80, anchor="center")
        self.table.column("term", width=260, anchor="w")
        self.table.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        sb = ttk.Scrollbar(self.tab_tbl, command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ─── Data discovery ────────────────────────────────────────────
    def _populate_pickers(self):
        plats = sorted(self.app.gpl_datasets.keys()) if hasattr(self.app, "gpl_datasets") else []
        self.plat_combo["values"] = plats
        if plats:
            self.plat_var.set(plats[0])
            self._on_platform_change()

    def _merged_platform_df(self, plat):
        """Expression frame for *plat* with its curated label columns joined on GSM.

        Curated labels live in ``app.platform_labels`` - a separate DataFrame that
        is never merged into ``gpl_datasets`` - so we join them on GSM here.
        Without this the label pickers stay empty (the raw expression frame holds
        no label columns) and every enrichment run returns nothing.

        Returns ``(df, label_cols)`` where *label_cols* is every column that
        *could* carry a label: the label file is offered in full, whatever its
        column names, plus the non-numeric columns of the expression frame (the
        numeric ones are the genes). Deciding which of them really are labels is
        the user's call - see :meth:`_choose_label_columns`; the picker only
        pre-ticks the ones :func:`semantic_label_columns` recognises.
        """
        expr = self.app.gpl_datasets[plat]
        labels = getattr(self.app, "platform_labels", {}).get(plat)
        df = expr
        label_cols = []
        if (labels is not None and "GSM" in expr.columns
                and "GSM" in labels.columns):
            label_cols = [c for c in labels.columns
                          if c != "GSM" and c not in expr.columns]
            if label_cols:
                e = expr.copy()
                e["GSM"] = e["GSM"].astype(str).str.strip().str.upper()
                lab = labels[["GSM"] + label_cols].copy()
                lab["GSM"] = lab["GSM"].astype(str).str.strip().str.upper()
                df = e.merge(lab, on="GSM", how="left")
        # Also offer label columns already embedded in the expression frame
        # (combined files that carry Classified_Tissue / Condition / …). A gene
        # is a numeric column, so anything non-numeric is a label candidate;
        # a recognised field name counts even when it happens to be numeric.
        recognised = set(semantic_label_columns(expr.columns))
        for c in expr.columns:
            if c == "GSM" or c in label_cols:
                continue
            if c in recognised or not pd.api.types.is_numeric_dtype(expr[c]):
                label_cols.append(c)
        return df, label_cols

    def _on_platform_change(self):
        plat = self.plat_var.get()
        if not plat or plat not in self.app.gpl_datasets:
            return
        df, label_cols = self._merged_platform_df(plat)
        # Gene symbols = numeric columns that are not label columns
        gene_cols = [c for c in df.columns
                     if c not in label_cols
                     and pd.api.types.is_numeric_dtype(df[c])]
        self.gene_combo["values"] = gene_cols[:5000]
        if gene_cols:
            self.gene_var.set(gene_cols[0])

        # Rebuild the label-column picker
        for child in self.label_cols_frame.winfo_children():
            child.destroy()
        self.label_col_vars = {}
        self._label_col_levels = {}
        if not label_cols:
            ttk.Label(self.label_cols_frame,
                      text="(no label columns detected)",
                      foreground=AERO["muted"],
                      font=("Segoe UI", 9, "italic")).pack(side=tk.LEFT)
            return

        # Distinct non-empty values per column - shown in the picker so the user
        # can tell a label (a handful of levels) from an identifier (one level
        # per sample) without leaving the window.
        for col in label_cols:
            try:
                self._label_col_levels[col] = int(
                    df[col].dropna().astype(str).str.strip()
                    .replace("", pd.NA).dropna().nunique())
            except Exception:
                self._label_col_levels[col] = 0

        detected = set(semantic_label_columns(label_cols))
        if not detected:
            # A file that uses none of the extractor's field names still has to
            # start somewhere: tick whatever behaves like a categorical label -
            # at least two values, but not one per sample (an identifier).
            detected = {c for c, lv in self._label_col_levels.items()
                        if 2 <= lv <= max(2, min(50, len(df) - 1))}
        for col in label_cols:
            self.label_col_vars[col] = tk.BooleanVar(value=col in detected)

        ttk.Label(self.label_cols_frame,
                  textvariable=self._label_cols_summary,
                  foreground=AERO["text"],
                  font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Button(self.label_cols_frame, text="Choose\u2026",
                   command=self._choose_label_columns,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))
        self._refresh_label_cols_summary()

    def _refresh_label_cols_summary(self):
        """One-line description of the current label-column selection."""
        chosen = [c for c, v in self.label_col_vars.items() if v.get()]
        total = len(self.label_col_vars)
        if not chosen:
            self._label_cols_summary.set(f"none of {total} columns selected")
            return
        shown = ", ".join(chosen[:3])
        if len(chosen) > 3:
            shown += f", +{len(chosen) - 3} more"
        self._label_cols_summary.set(f"{len(chosen)} of {total}: {shown}")

    def _choose_label_columns(self):
        """Checklist of *every* column found, so any file layout can be used."""
        if not self.label_col_vars:
            return
        self._label_cols_dialog = column_picker_dialog(
            self, self.label_col_vars, self._label_col_levels,
            n_rows=len(self.app.gpl_datasets.get(self.plat_var.get(), [])),
            title="Label columns to test",
            on_close=self._refresh_label_cols_summary)

    def _on_strategy_change(self):
        # Just used for visual cues - heavy work happens in _run_enrichment
        s = self.strategy_var.get()
        if s == "range":
            self.status_var.set("Range mode: enter low and high expression values.")
        elif s == "label_value":
            self.status_var.set(
                "Label-value mode: enter <column>=<value> in the gene box "
                "(e.g. 'Classified_Tissue=liver').")
        elif s == "top_percent":
            self.status_var.set(
                "Top-% mode: threshold = percentile (e.g. 10 = top 10%).")
        else:
            self.status_var.set(
                "Bottom-% mode: threshold = percentile (e.g. 10 = bottom 10%).")

    # ─── Run analysis ──────────────────────────────────────────────
    def _run_enrichment(self):
        try:
            from genevariate.core.analysis import region_label_enrichment
        except Exception as exc:
            messagebox.showerror("Import error",
                                 f"Could not import enrichment helpers:\n{exc}",
                                 parent=self)
            return

        plat = self.plat_var.get()
        if not plat or plat not in self.app.gpl_datasets:
            messagebox.showwarning("No platform",
                                    "Pick a loaded platform first.",
                                    parent=self)
            return
        df, _ = self._merged_platform_df(plat)

        # Build foreground mask
        try:
            fg_mask = self._build_fg_mask(df)
        except Exception as exc:
            messagebox.showerror("Foreground error", str(exc), parent=self)
            return
        if fg_mask.sum() == 0:
            messagebox.showwarning("Empty foreground",
                                    "No samples matched the foreground criteria.",
                                    parent=self)
            return

        # Build label column list
        label_cols = [c for c, v in self.label_col_vars.items() if v.get()]
        if not label_cols:
            messagebox.showwarning("No label columns",
                                    "Tick at least one label column to test.",
                                    parent=self)
            return

        try:
            min_count = max(1, int(self.min_count_var.get()))
            alpha = float(self.alpha_var.get())
        except Exception:
            min_count, alpha = 3, 0.05

        # The same grid the Enrichment tab of region analysis builds, and the
        # same function behind it. One foreground, one row per label column, so
        # BH corrects once over every value of every column rather than once
        # per column - the correction has to know how many tests were run.
        fg_name = self._foreground_name()
        # The split is the program's, not this window's. Building it here meant
        # the labels kept their missing values while the study ids were listed
        # for every row, so the two lengths disagreed, the clumping diagnostics
        # raised, and the raise was swallowed - the window reported enrichment
        # with no study count, no effective sample size and no interval.
        from genevariate.core.analysis import build_enrichment_cells
        ids = df.index.astype(str).to_numpy()
        by_col = {c: pd.Series(df[c].to_numpy(), index=ids) for c in label_cols}
        study_of = (dict(zip(ids, df["series_id"].astype(str)))
                    if "series_id" in df.columns else None)
        cells, groups = build_enrichment_cells(
            by_col, ids[fg_mask.to_numpy() if hasattr(fg_mask, "to_numpy")
                        else fg_mask],
            region_name=fg_name, study_of=study_of)

        result = region_label_enrichment(cells, groups or None, alpha=alpha)
        if result.empty:
            messagebox.showwarning("No enrichments",
                                    "No label value was testable in this "
                                    "foreground.", parent=self)
            return
        # A value seen fewer than min_count times across the platform is noise
        # the user asked not to see; drop it after the test, never before, so
        # the correction still counts it.
        if min_count > 1:
            keep = (result["a"] + result["c"]) >= min_count
            if keep.any():
                result = result[keep]

        result = self._to_plot_frame(result)
        self._current_enrichment = result

        self._redraw_plots()
        n_sig = int((result["q_value"] < alpha).sum())
        self.status_var.set(
            f"Tested {len(result)} label values across {len(label_cols)} "
            f"column(s) | fg n={int(fg_mask.sum())} | {n_sig} significant "
            f"(q<{alpha:g}) | one-sided Fisher, BH over the whole grid, "
            f"p widened for study clumping.")

    def _foreground_name(self):
        """A short name for the current foreground, used as the region label."""
        strat = self.strategy_var.get()
        raw = str(self.gene_var.get()).strip()
        if strat == "label_value":
            return raw or "foreground"
        if strat == "top_percent":
            return f"{raw} top {self.threshold_var.get()}%"
        if strat == "bottom_percent":
            return f"{raw} bottom {self.threshold_var.get()}%"
        return (f"{raw} {self.range_lo_var.get()}-{self.range_hi_var.get()}")

    @staticmethod
    def _to_plot_frame(res):
        """Core's result under the column names the three plots expect.

        Renaming, not recomputing: ``k``/``K``/``n``/``N`` and the fold change
        are read straight off the 2x2 the p-value came from, so the bars and
        the table cannot disagree with the test that drew them.
        """
        out = res.copy()
        out["term"] = (out["label_column"].astype(str) + ": "
                       + out["value"].astype(str))
        out["k"] = out["a"]
        out["K"] = out["a"] + out["c"]
        out["n"] = out["n_region"]
        out["N"] = out["n_region"] + out["n_background"]
        out["fold_change"] = out["enrichment"]
        out["significant"] = out["significance"] != "ns"
        return out.reset_index(drop=True)

    # ─── Foreground construction ───────────────────────────────────
    def _build_fg_mask(self, df):
        strat = self.strategy_var.get()
        if strat == "label_value":
            raw = str(self.gene_var.get()).strip()
            if "=" not in raw:
                raise ValueError(
                    "Label-value mode needs 'column=value' (e.g. Classified_Tissue=liver).")
            col, val = [s.strip() for s in raw.split("=", 1)]
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in platform.")
            return df[col].astype(str) == val

        gene = self.gene_var.get().strip()
        if not gene or gene not in df.columns:
            raise ValueError(f"Gene '{gene}' not found in platform.")
        col = df[gene]
        if not pd.api.types.is_numeric_dtype(col):
            raise ValueError(f"Gene '{gene}' is not numeric.")
        values = col.astype(float)

        if strat in ("top_percent", "bottom_percent"):
            try:
                pct = float(self.threshold_var.get())
            except Exception:
                pct = 10.0
            pct = float(np.clip(pct, 0.1, 99.9))
            q = pct / 100.0
            if strat == "top_percent":
                cut = values.quantile(1 - q)
                return values >= cut
            else:
                cut = values.quantile(q)
                return values <= cut

        # range
        try:
            lo = float(self.range_lo_var.get())
            hi = float(self.range_hi_var.get())
        except Exception:
            raise ValueError("Range mode needs numeric low and high values.")
        if hi < lo:
            lo, hi = hi, lo
        return (values >= lo) & (values <= hi)

    # ─── Plot rendering ────────────────────────────────────────────
    def _embed_figure(self, tab, fig, key, scroll=False):
        # Tear down any previous canvas in this tab
        for child in tab.winfo_children():
            child.destroy()
        old = self._figures.pop(key, None)
        if old is not None:
            try:
                plt.close(old)
            except Exception:
                pass
        holder = None
        if scroll:
            # A term-per-row plot grows with the result, so it will not fit the
            # tab once there are many labels. Scroll it at its natural height
            # rather than squeezing every term into one screen.
            body = ttk.Frame(tab)
            holder = tk.Canvas(body, highlightthickness=0, bg=AERO["panel"])
        canvas = FigureCanvasTkAgg(fig, master=holder if scroll else tab)
        canvas.draw()
        viz_make_interactive(fig)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()
        style_toolbar(toolbar)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        widget = canvas.get_tk_widget()
        if not scroll:
            widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            px_h = max(1, int(fig.get_size_inches()[1] * fig.dpi))
            body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            vsb = ttk.Scrollbar(body, orient="vertical", command=holder.yview)
            holder.configure(yscrollcommand=vsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            item = holder.create_window((0, 0), window=widget, anchor="nw")

            def _fit(ev, item=item, holder=holder, px_h=px_h):
                holder.itemconfigure(item, width=ev.width, height=px_h)
                holder.configure(scrollregion=(0, 0, ev.width, px_h))
            holder.bind("<Configure>", _fit)
            holder.bind("<MouseWheel>",
                        lambda e, h=holder: h.yview_scroll(
                            -1 if e.delta > 0 else 1, "units"))
            holder.bind("<Button-4>", lambda e, h=holder: h.yview_scroll(-1, "units"))
            holder.bind("<Button-5>", lambda e, h=holder: h.yview_scroll(1, "units"))
        self._figures[key] = fig
        self._canvases[key] = canvas

    #: Plot ordering the user can choose from → (column, ascending).
    _RANK_KEYS = {
        "p-value":            ("p_value", True),
        "q-value":            ("q_value", True),
        "fold change":        ("fold_change", False),
        "|log2 fold change|": ("_abs_log2_fc", False),
        "count (k)":          ("k", False),
    }

    def _display_frame(self):
        """The current result, ordered and trimmed the way the user asked.

        A view only: every term stays in ``_current_enrichment``, in the table
        and in the CSV export, and no statistic is recomputed here. Returns
        ``(ordered_df, top_n)`` with *top_n* ``None`` for "draw all of them".
        """
        res = self._current_enrichment
        if res is None or res.empty:
            return res, None
        key, ascending = self._RANK_KEYS.get(
            self.rank_by_var.get(), ("p_value", True))
        df = res
        if key == "_abs_log2_fc":
            fc = pd.to_numeric(res["fold_change"], errors="coerce")
            # fold change 0 means absent from the foreground: |log2| is
            # unbounded, so it ranks above every finite effect, not below.
            l2 = np.abs(np.log2(fc.where(fc > 0)))
            df = res.assign(_abs_log2_fc=l2.fillna(np.inf))
        # Stable sort: terms that tie on the chosen key (all the unbounded ones
        # tie on |log2 FC|, for instance) keep the p-value order they came in.
        df = df.sort_values(key, ascending=ascending, na_position="last",
                            kind="mergesort")
        try:
            top = int(float(self.plot_top_var.get()))
        except Exception:
            top = 0
        return df, (top if top > 0 else None)

    def _redraw_plots(self):
        """Re-draw the three plots and the table from the stored result."""
        if self._current_enrichment is None or self._current_enrichment.empty:
            return
        df, top = self._display_frame()
        self._render_bar(df, top)
        self._render_dot(df, top)
        self._render_volcano(df)
        self._populate_table(df)

    def _term_figsize(self, n):
        """Figure size that leaves every drawn term a readable row of its own."""
        w, h = viz_smart_figsize("default")
        return (w, max(h, 0.26 * max(int(n), 1) + 1.8))

    def _render_bar(self, result, top_n=None):
        from genevariate.utils.plot_types import plot_enrichment_bar
        n = len(result) if top_n is None else min(top_n, len(result))
        fig = Figure(figsize=self._term_figsize(n))
        ax = fig.subplots()
        plot_enrichment_bar(ax, result, top_n=top_n)
        fig.tight_layout()
        self._embed_figure(self.tab_bar, fig, "bar", scroll=True)

    def _render_dot(self, result, top_n=None):
        from genevariate.utils.plot_types import plot_enrichment_dot
        n = len(result) if top_n is None else min(top_n, len(result))
        fig = Figure(figsize=self._term_figsize(n))
        ax = fig.subplots()
        plot_enrichment_dot(ax, result, top_n=top_n)
        fig.tight_layout()
        self._embed_figure(self.tab_dot, fig, "dot", scroll=True)

    def _render_volcano(self, result):
        from genevariate.utils.plot_types import plot_enrichment_volcano
        fig = Figure(figsize=viz_smart_figsize("default"))
        ax = fig.subplots()
        plot_enrichment_volcano(ax, result)
        # The test behind these points is one-sided: it asks whether a value is
        # over-represented in the foreground and nothing else. Points left of
        # zero are depleted terms drawn to scale, but their q was never a test
        # of depletion, so none of them can reach the significance line. Saying
        # so on the axes is the difference between a reader seeing "nothing is
        # depleted here" and "depletion was not tested".
        ax.text(0.5, -0.16,
                "one-sided test (over-representation): terms left of 0 are "
                "shown to scale but were not tested for depletion",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=AERO["muted"])
        fig.tight_layout()
        self._embed_figure(self.tab_volc, fig, "volcano")

    def _populate_table(self, result):
        for iid in self.table.get_children():
            self.table.delete(iid)
        # A whole-platform label enrichment can yield thousands of terms; a
        # Treeview stalls if every row is inserted. Show the top rows (already
        # ranked) and note the remainder - Export CSV still writes everything.
        _CAP = 2000
        for _, row in result.head(_CAP).iterrows():
            fc = row["fold_change"]
            fc_str = f"{fc:.2f}" if np.isfinite(fc) else "inf"
            n_gse = row.get("n_gse")
            n_eff = row.get("n_eff")
            self.table.insert("", "end", values=(
                row["term"], row["k"], row["K"], row["n"], row["N"],
                fc_str,
                f"{row['odds_ratio']:.2f}" if np.isfinite(row["odds_ratio"]) else "inf",
                _fmt_p(row["p_value"]),
                _fmt_p(row["q_value"]),
                # Blank, not "-": the diagnostics run only for the terms that
                # survived correction, so a gap here is a question never asked,
                # not a term found to rest on zero studies.
                "" if n_gse is None or not np.isfinite(float(n_gse or np.nan))
                else f"{int(n_gse)}",
                "" if n_eff is None or not np.isfinite(float(n_eff or np.nan))
                else f"{float(n_eff):.1f}",
            ))
        if len(result) > _CAP:
            self.table.insert("", "end", values=(
                f"… {len(result) - _CAP:,} more terms "
                f"(capped at {_CAP:,}; use Export CSV for all)",
                "", "", "", "", "", "", "", "", "", ""))

    def _sort_table(self, col):
        items = [(self.table.set(k, col), k) for k in self.table.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0].lstrip("<")))
        except ValueError:
            items.sort(key=lambda t: t[0])
        for i, (_, k) in enumerate(items):
            self.table.move(k, "", i)

    # ─── Export ────────────────────────────────────────────────────
    def _export_csv(self):
        if self._current_enrichment is None or self._current_enrichment.empty:
            messagebox.showinfo("Nothing to export",
                                 "Run an enrichment first.", parent=self)
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"label_enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            parent=self)
        if not p:
            return
        try:
            self._current_enrichment.to_csv(p, index=False)
            messagebox.showinfo("Exported", f"Saved to:\n{p}", parent=self)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)

    def _export_all_plots(self):
        if not self._figures:
            messagebox.showinfo("Nothing to export",
                                 "Run an enrichment first.", parent=self)
            return
        d = filedialog.askdirectory(title="Export folder", parent=self)
        if not d:
            return
        try:
            from genevariate.utils.export_manager import PlotExportManager
            mgr = PlotExportManager(base_dir=d)
            plat = self.plat_var.get() or "enrichment"
            paths = mgr.export_batch(self._figures, analysis_id=f"enrichment_{plat}")
            if not paths:
                messagebox.showerror(
                    "Export failed",
                    "None of the figures could be written. Check that the "
                    "folder is writable and has free space.", parent=self)
                return
            mgr.write_html_index(paths, title=f"Enrichment - {plat}",
                                  subdir=list(paths.values())[0].parent.name)
            table = ""
            if self._current_enrichment is not None and not self._current_enrichment.empty:
                csv = Path(d) / f"enrichment_{plat}.csv"
                self._current_enrichment.to_csv(csv, index=False)
                table = f"\n{csv.name}"
            n_missed = len(self._figures) - len(paths)
            note = f"\n\n{n_missed} figure(s) could not be written." if n_missed else ""
            messagebox.showinfo("Exported",
                                 f"Saved {len(paths)} figure(s) to:\n{d}{table}{note}",
                                 parent=self)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)

    def destroy(self):
        for fig in list(self._figures.values()):
            try:
                plt.close(fig)
            except Exception:
                pass
        self._figures.clear()
        self._canvases.clear()
        super().destroy()


class GeoWorkflowGUI(ctk.CTk if _HAS_CTK else tk.Tk):
    """Complete GeneVariate main application window - Modern dark UI."""

    MAX_WORKERS = CONFIG['threading']['max_workers']
    METADATA_EXCLUSIONS = METADATA_EXCLUSIONS

    def __init__(self):
        super().__init__()
        self.title("GeneVariate 0.9.0 - Gene Expression Analysis Platform")
        self.geometry("1200x1050")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1200x1050+{(_sw-1200)//2}+{(_sh-1050)//2}")
            self.minsize(600, 500)
        except Exception: pass
        self.after_id = None

        # UI animation helper (liquid spinners, smooth progress, pulse)
        self._animator = _UIAnimator(self)

        try:
            icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
            if icon_path.exists():
                self.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
        except:
            pass
        self.data_dir = CONFIG['paths']['data']
        self.results_dir = CONFIG['paths']['results']
        print(f"[Startup] data_dir = {self.data_dir}")
        print(f"[Startup] CWD      = {os.getcwd()}")
        print(f"[Startup] app.py   = {os.path.abspath(__file__)}")
        # Auto-add program dir and CWD as extra scan locations if they differ
        _prog_dir = os.path.dirname(os.path.abspath(__file__))
        _cwd = os.getcwd()
        self.log_queue = queue.Queue()
        self.gds_conn = None
        self._gsm_lookup = None
        self.step1_results_df = None
        self.step2_data_df = None
        self.step1_gse_keywords = {}
        self.step1_gse_descriptions = {}
        self.gse_to_keep_for_step2 = []
        self.current_extraction_thread = None
        self.current_labeling_thread = None
        self._progress_owners = 0          # tracks concurrent progress bar users
        self._progress_lock = threading.Lock()  # guard for _progress_owners
        self.gpl_datasets = {}
        self.gpl_gene_mappings = {}
        self.gpl_gene_cache = {}       # {gpl_id: DataFrame} gene-only partial loads
        self.gpl_available_files = {}  # {gpl_id: file_path} discovered but not loaded
        self.gpl_source_paths = {}     # {gpl_id: file_path} of what was loaded
        self._user_data_dirs = []      # user-added directories to scan for GPL files
        self.default_labels_df = None  # loaded labels DataFrame (legacy compat)
        self._pending_gsm_filter = None  # for subset loading after download
        self._llm_update_count_fn = None  # callback for LLM window count updates
        self._dialog_active = False       # prevents concurrent modal dialog crashes (grab conflict)
        
        # ── App-level extraction settings (shared across all extraction paths) ──
        # Only the five labels the vendored geo_label_extractor actually emits
        # (geo_extract_driver.ALL_FIELDS). Treatment_Time and free-text custom
        # fields were old-local-model residue and are not supported.
        self._extraction_fields = ['Sex', 'Condition', 'Tissue', 'Age', 'Treatment']
        self._llm_workers = 0  # parallel in-flight HTTP requests (0 = auto)
        self.platform_labels = {}     # {platform_name: DataFrame} per-platform labels
        self.labels_col_vars = {}      # col_name -> BooleanVar
        self._labels_col_levels = {}   # col_name -> distinct-value count
        self._labels_cols_summary = tk.StringVar(value="")
        self.is_closing = False
        self.token_cache = {}
        self.ai_pipeline = None
        self.gene_dist_popup_root = None
        self.subset_dist_popup_root = None
        self.compare_window = None
        self.tracked_figures = {}
        self.current_device = 'cpu'  # GPU disabled
        self.ai_agent = SampleClassificationAgent(
            gui_log_func=self.enqueue_log,
            max_workers=self.MAX_WORKERS
        )
        self._setup_styles()
        self._setup_menubar()
        self._load_geometadb_connection()
        self._setup_ui()
        self.after(100, self.process_log_queue)
        self.after(100, self._load_ai_pipeline)
        self.after(200, self._startup_discovery)  # discover platforms + log diagnostics
        self.after(400, self._auto_load_labels)  # auto-scan labels directory
        self.after(500, self._show_welcome_tip)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._setup_keyboard_shortcuts()

    def _setup_styles(self):
        """Install the shared Frutiger Aero ttk theme.

        The theme itself lives in ``gui.theme`` so that a window opened without
        the main window - the region window, any dialog - installs the same
        look instead of falling back to bare clam.
        """
        ensure_theme(self)

    def _ensure_series_id(self, df):
        """Ensure series_id column exists and is populated by querying GEOmetadb.
        Uses gsm.series_id first, then gse_gsm junction table as fallback.
        Returns df with series_id filled in.
        """
        if df is None or df.empty:
            return df

        gc = 'GSM' if 'GSM' in df.columns else 'gsm'
        if gc not in df.columns:
            return df

        # Check how many are missing
        if 'series_id' in df.columns:
            missing = df['series_id'].isna() | df['series_id'].astype(str).str.strip().isin(
                ['', 'nan', 'None', 'NaN'])
            n_missing = missing.sum()
        else:
            df['series_id'] = pd.NA
            missing = pd.Series([True] * len(df), index=df.index)
            n_missing = len(df)

        if n_missing == 0:
            return df

        self.enqueue_log(f"[GSM→GSE] {n_missing:,}/{len(df):,} samples missing series_id, querying GEOmetadb...")

        if not self.gds_conn:
            self.enqueue_log("[GSM→GSE] WARNING: GEOmetadb not loaded - cannot look up GSE")
            return df

        # Collect GSMs that need lookup
        gsms_need = df.loc[missing, gc].astype(str).str.strip().str.upper().tolist()
        if not gsms_need:
            return df

        gsm_to_gse = {}

        # Method 1: gsm.series_id column
        try:
            for i in range(0, len(gsms_need), 500):
                chunk = gsms_need[i:i+500]
                ph = ",".join(["?"] * len(chunk))
                rows = self.gds_conn.execute(
                    f"SELECT UPPER(gsm), series_id FROM gsm WHERE UPPER(gsm) IN ({ph})",
                    chunk).fetchall()
                for gsm_val, sid in rows:
                    if sid and str(sid).strip() and str(sid).strip().lower() not in ('nan', 'none', ''):
                        gsm_to_gse[gsm_val] = str(sid).strip()
        except Exception as e:
            self.enqueue_log(f"[GSM→GSE] gsm.series_id lookup error: {e}")

        # Method 2: gse_gsm junction table for remaining
        still_missing = [g for g in gsms_need if g not in gsm_to_gse]
        if still_missing:
            try:
                tables = [r[0] for r in self.gds_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if 'gse_gsm' in tables:
                    self.enqueue_log(f"[GSM→GSE] {len(still_missing):,} still missing, trying gse_gsm table...")
                    for i in range(0, len(still_missing), 500):
                        chunk = still_missing[i:i+500]
                        ph = ",".join(["?"] * len(chunk))
                        rows = self.gds_conn.execute(
                            f"SELECT UPPER(gsm), gse FROM gse_gsm WHERE UPPER(gsm) IN ({ph})",
                            chunk).fetchall()
                        for gsm_val, gse_val in rows:
                            if gse_val and str(gse_val).strip():
                                gsm_to_gse[gsm_val] = str(gse_val).strip()
            except Exception as e:
                self.enqueue_log(f"[GSM→GSE] gse_gsm lookup error: {e}")

        # Apply mapping
        if gsm_to_gse:
            df_gsm_upper = df[gc].astype(str).str.strip().str.upper()
            for idx in df.index:
                if missing.loc[idx]:
                    gsm_val = df_gsm_upper.loc[idx]
                    if gsm_val in gsm_to_gse:
                        df.at[idx, 'series_id'] = gsm_to_gse[gsm_val]

        n_after = df['series_id'].isna().sum() + (df['series_id'].astype(str).str.strip().isin(
            ['', 'nan', 'None', 'NaN'])).sum()
        n_resolved = n_missing - n_after
        n_gse = df['series_id'].dropna().nunique()
        self.enqueue_log(
            f"[GSM→GSE] Resolved {n_resolved:,}/{n_missing:,} missing series_id "
            f"({n_gse} unique experiments)")

        return df

    def _setup_menubar(self):
        """Creates application menu bar with shortcuts."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load External CSV...", 
                             accelerator="Ctrl+O",
                             command=self.load_external_file_for_step2)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", 
                             accelerator="Ctrl+Q",
                             command=self.on_closing)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Gene Distribution Explorer",
                              accelerator="Ctrl+G",
                              command=self.show_gene_distribution_popup)
        tools_menu.add_command(label="Compare Distributions",
                              accelerator="Ctrl+D",
                              command=self.open_compare_window)
        tools_menu.add_command(label="Distribution Classification",
                              accelerator="Ctrl+K",
                              command=self._open_dist_classification)
        tools_menu.add_separator()
        tools_menu.add_command(label="Cross-Platform Analysis",
                              command=self._open_cross_platform_analysis)
        # Not "Curate Labels": nothing here curates. Cross-experiment
        # harmonization is a geo_label_extractor CLI stage, and this entry
        # only reports where it lives and whether its reference artifacts
        # are present on this machine.
        tools_menu.add_command(label="Where label curation lives...",
                              command=self._open_llm_curator)
        tools_menu.add_separator()
        tools_menu.add_command(label="Download Platform...",
                              command=self._open_gpl_downloader_window)
        tools_menu.add_command(label="Add Custom Platform...",
                              command=self._load_custom_gpl_data)
        tools_menu.add_separator()
        tools_menu.add_command(label="Load RNA-seq from ARCHS4...",
                              command=self._open_archs4_window)
        tools_menu.add_command(label="Run Enrichment Analysis...",
                              command=self._open_enrichment_window)
        tools_menu.add_separator()
        tools_menu.add_command(label="Assistant (chat)...",
                              accelerator="Ctrl+/",
                              command=self._toggle_chat_sidebar)
        tools_menu.add_separator()
        tools_menu.add_command(label="Display limits...",
                              command=self._open_display_limits)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Start Guide", command=self._show_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About GeneVariate", command=self._show_about)
    
    def _setup_keyboard_shortcuts(self):
        """Sets up keyboard shortcuts."""
        self.bind('<Control-o>', lambda e: self.load_external_file_for_step2())
        self.bind('<Control-O>', lambda e: self.load_external_file_for_step2())
        self.bind('<Control-q>', lambda e: self.on_closing())
        self.bind('<Control-Q>', lambda e: self.on_closing())
        self.bind('<Control-g>', lambda e: self.show_gene_distribution_popup())
        self.bind('<Control-G>', lambda e: self.show_gene_distribution_popup())
        self.bind('<Control-d>', lambda e: self.open_compare_window())
        self.bind('<Control-D>', lambda e: self.open_compare_window())
        self.bind('<Control-k>', lambda e: self._open_dist_classification())
        self.bind('<Control-K>', lambda e: self._open_dist_classification())
        self.bind('<Control-slash>', lambda e: self._toggle_chat_sidebar())
        self.bind('<F1>', lambda e: self._show_help())
    
    def _show_welcome_tip(self):
        """Shows a welcome tip on first run."""
        tip_text = (
            " Quick Start:\n\n"
            "1. Load GPL platforms (all samples loaded for full analysis)\n"
            "2. Use 'Gene Distribution Explorer' to select expression ranges\n"
            "3. Click and drag on histograms to select regions\n"
            "4. Click 'Analyze Selected Range(s)' for extraction\n\n"
            "Tip: Click legend items to change colors!\n"
            "Press F1 anytime for help."
        )
        
        info_frame = tk.Frame(self, bg=AERO["panel_top"],
                              highlightthickness=1,
                              highlightbackground=AERO["accent"],
                              highlightcolor=AERO["accent"],
                              bd=0)
        info_frame.place(relx=0.5, rely=0.05, anchor=tk.N)

        tk.Label(info_frame, text=" Welcome to GeneVariate!",
                font=('Segoe UI', 11, 'bold'),
                bg=AERO["panel_top"], fg=AERO["accent_dark"]).pack(padx=14, pady=6)
        tk.Label(info_frame, text=tip_text,
                justify=tk.LEFT,
                bg=AERO["panel_top"], fg=AERO["text"]).pack(padx=14, pady=5)

        def close_tip():
            info_frame.destroy()

        ttk.Button(info_frame, text="Got it!", command=close_tip,
                   style="Primary.TButton").pack(pady=8)
        
        self.after(15000, lambda: info_frame.destroy() if info_frame.winfo_exists() else None)
    
    def _show_help(self):
        """Shows comprehensive help dialog."""
        help_win = tk.Toplevel(self)
        style_window(help_win)
        help_win.title("GeneVariate Quick Start Guide")
        help_win.geometry("800x700")
        try:
            _sw, _sh = help_win.winfo_screenwidth(), help_win.winfo_screenheight()
            help_win.geometry(f"800x700+{(_sw-800)//2}+{(_sh-700)//2}")
            help_win.minsize(500, 400)
        except Exception: pass
        help_win.transient(self)
        
        text = tk.Text(help_win, wrap=tk.WORD, font=('Segoe UI', 10), padx=20, pady=20)
        scrollbar = ttk.Scrollbar(help_win, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        help_content = """
GeneVariate 0.9.0 - Quick Start Guide

═══════════════════════════════════════════════════════════

OVERVIEW:
GeneVariate enables comprehensive analysis of gene expression data from 
GEO (Gene Expression Omnibus). Load entire platform datasets, select 
expression ranges visually, and use AI to classify and compare samples.

═══════════════════════════════════════════════════════════

WORKFLOW:

1. LOAD PLATFORMS
   - Click platform buttons to load datasets, or use 'Download Platform'
     (microarray, bulk RNA-seq or single-cell; methylation and peak assays
     are not expression and are refused)
   - ALL samples are loaded - this is essential for distribution analysis
   - Custom platforms: '+ Add Custom Platform' loads your own data

2. GENE DISTRIBUTION EXPLORER (Ctrl+G)
   - Enter gene symbols (comma-separated)
   - Select platforms to compare
   - Click "Plot Distributions"
   - DRAG rectangles on histograms to select expression ranges
   - Multiple regions can be selected per plot
   - Click "Analyze Selected Range(s)" for LLM extraction

3. AI CLASSIFICATION
   - Requires a geo_label_extractor backend (remote HTTP)
   - Classifies samples by condition, tissue, treatment
   - Uses semantic clustering to unify similar labels
   - Results open in the Region Analysis window

4. REGION ANALYSIS
   - Opens on 'Analyze Selected Range(s)'
   - Enrichment, frequency and per-label statistics for the samples
     inside the selected expression range
   - Export tabs write the full table; on-screen tables are capped

5. DISTRIBUTION COMPARISON (Ctrl+D)
   - Load classified CSV files
   - Select multiple groups to compare
   - Statistical tests: Wilcoxon, Wasserstein distance
   - PCA / t-SNE tab and a Clustering tab (K-Means, DBSCAN, DPC)
   - Toggle between rugs and density visualizations

STEP 1 (OPTIONAL): GEO DATABASE SEARCH
   - Search GEO for experiments by keywords
   - Filter by platform (e.g., GPL570)
   - Review and select experiments
   - Proceed to Step 2 for extraction

═══════════════════════════════════════════════════════════

TIPS & TRICKS:

- Click legend items to change colors (Distribution Comparison and
  Region Analysis plots)
- Use the toolbar under each plot to pan, zoom and save the figure
- Select multiple regions across different genes/platforms
- Saved figures are written at 300 DPI

═══════════════════════════════════════════════════════════

KEYBOARD SHORTCUTS:

Ctrl+O     Load external CSV
Ctrl+G     Gene Distribution Explorer
Ctrl+D     Distribution Comparison
Ctrl+K     Distribution Classification
Ctrl+/     Toggle the AI Assistant sidebar
F1         This help screen
Ctrl+Q     Quit application

═══════════════════════════════════════════════════════════

TROUBLESHOOTING:

- "SQLite threading error": Restart analysis, issue is being handled
- "LLM service unavailable": configure the geo_label_extractor backend URL
- Slow loading: Large datasets take time - progress shown in log
- Memory issues: Close unused windows, restart application

═══════════════════════════════════════════════════════════
        """
        
        text.insert('1.0', help_content)
        text.configure(state='disabled')
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _show_shortcuts(self):
        """Shows keyboard shortcuts."""
        shortcuts = """
Keyboard Shortcuts:

Ctrl+O     Load external CSV file
Ctrl+G     Open Gene Distribution Explorer
Ctrl+D     Open Distribution Comparison
Ctrl+K     Open Distribution Classification
Ctrl+/     Toggle the AI Assistant sidebar
F1         Show help
Ctrl+Q     Quit application
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts, parent=self)
    
    def _show_about(self):
        """Shows about dialog."""
        about_text = """
GeneVariate 0.9.0
Gene Expression Variability Analysis Tool

A comprehensive platform for analyzing gene expression 
patterns across large-scale GEO datasets.

Features:
- Multi-platform support (GPL570, GPL96, GPL6947, etc.)
- LLM label extraction from GEO metadata (needs a remote backend)
- Interactive distribution analysis
- PCA / t-SNE and clustering (K-Means, DBSCAN, DPC)
- Figures exported at 300 DPI

Developed with Python, Tkinter, Matplotlib, and scikit-learn.
        """
        messagebox.showinfo("About GeneVariate", about_text, parent=self)

    def enqueue_log(self, msg):
        """Thread-safe logging with enhanced formatting."""
        self.log_queue.put(msg)
    
    def process_log_queue(self):
        """Processes messages from the logging queue with color coding."""
        if self.is_closing:
            return
        
        if hasattr(self, 'after_id') and self.after_id:
            try:
                self.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None
        
        try:
            import queue as q
            while True:
                msg = self.log_queue.get_nowait()
                
                if msg.startswith("PROGRESS:"):
                    try:
                        progress_val = float(msg.split(":", 1)[1])
                        # Route through update_progress so the shimmer animation
                        # is triggered for every process, not just downloads.
                        self.update_progress(value=progress_val)
                        
                        if hasattr(self, 'status_label'):
                            if progress_val >= 100:
                                self.status_label.config(text="OK Complete", foreground="green")
                            else:
                                self.status_label.config(
                                    text=f"Processing... {progress_val:.0f}%", 
                                    foreground=AERO["accent"]
                                )
                    except (ValueError, IndexError):
                        pass
                else:
                    self._render_log_line(msg)
                    self._log_msg_count += 1
                    if hasattr(self, 'log_status_label'):
                        self.log_status_label.config(
                            text=f"Log: {self._log_msg_count} messages",
                            foreground=AERO["accent"])
                    
        except q.Empty:
            pass

        if self.winfo_exists():
            self.after_id = self.after(100, self.process_log_queue)

    # ------------------------------------------------------------------
    # Log formatting - renders one message as timestamp + category chip
    # + level badge + body, with zebra-striped row backgrounds so the
    # activity log looks like a proper product console, not a .txt dump.
    # ------------------------------------------------------------------
    _LOG_CAT_RE = None  # compiled lazily

    def _render_log_line(self, msg):
        import re
        from datetime import datetime

        if self._LOG_CAT_RE is None:
            # Capture a leading [CATEGORY] tag if present. Category allows
            # letters, spaces, digits, arrows and a few punctuation chars.
            type(self)._LOG_CAT_RE = re.compile(r'^\s*\[([^\]]{1,24})\]\s?(.*)$',
                                                 re.DOTALL)

        line_idx = self._log_line_idx
        self._log_line_idx += 1
        zebra = "zebra_odd" if line_idx % 2 else "zebra_even"

        # Level detection (affects level badge + body color)
        msg_u = msg.upper()
        if "[ERROR]" in msg_u or "ERROR:" in msg_u or "FAIL" in msg_u:
            badge, badge_tag, body_tag = " ERROR ", "level_err",  "body_err"
        elif "[WARNING]" in msg_u or "WARN" in msg_u:
            badge, badge_tag, body_tag = " WARN  ", "level_warn", "body_warn"
        elif "[INFO]" in msg_u or " OK" in msg_u or "✓" in msg:
            badge, badge_tag, body_tag = " INFO  ", "level_info", "body"
        else:
            badge, badge_tag, body_tag = "  •    ", "level_info", "body"

        # Category extraction: "[Foo] rest" -> cat="Foo", rest="rest"
        cat, rest = None, msg
        m = self._LOG_CAT_RE.match(msg)
        if m:
            cat = m.group(1).strip()
            rest = m.group(2)

        # Record the starting position, then append the formatted line
        start = self.log_text.index("end-1c")

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{ts}  ", ("ts", zebra))

        # Level badge
        self.log_text.insert(tk.END, f"{badge} ", (badge_tag, zebra))

        # Category chip (bold, colored per-category)
        if cat is not None:
            cat_tag = f"cat_{cat}" if cat in self._log_cat_colors else "cat_default"
            self.log_text.insert(tk.END, f"[{cat}] ", (cat_tag, zebra))

        # Body text
        self.log_text.insert(tk.END, rest.rstrip() + "\n", (body_tag, zebra))

        # Make sure the zebra tag covers the entire line (Text widget treats
        # newline-delimited rows as separate display lines - this keeps the
        # stripe contiguous across the row).
        end = self.log_text.index("end-1c")
        self.log_text.tag_add(zebra, start, end)

        self.log_text.see(tk.END)

    def _load_geometadb_connection(self):
        """Loads GEOmetadb into memory with progress indication."""
        gz_path = CONFIG['paths']['geo_db']
        
        if not os.path.exists(gz_path):
            self.enqueue_log(f"[WARNING] GEOmetadb.sqlite.gz not found at: {gz_path}")
            # Offer to browse for it
            self.enqueue_log("[DB] Prompting user to locate GEOmetadb...")
            result = messagebox.askyesno(
                "GEOmetadb Not Found",
                f"GEOmetadb.sqlite.gz not found at:\n{gz_path}\n\n"
                f"Would you like to browse for it?\n\n"
                f"(Download from: https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz)",
                parent=self
            )
            if result:
                chosen = filedialog.askopenfilename(
                    title="Select GEOmetadb.sqlite.gz",
                    filetypes=[("GEOmetadb", "*.sqlite.gz *.sqlite"), ("All files", "*.*")],
                    parent=self
                )
                if chosen and os.path.exists(chosen):
                    gz_path = chosen
                    CONFIG['paths']['geo_db'] = gz_path
                    self.enqueue_log(f"[DB] User selected: {gz_path}")
                else:
                    self.enqueue_log("[WARNING] No file selected - database not loaded")
                    self.enqueue_log("[WARNING] Step 1 (GSE Extraction) and GPL downloads will not work")
                    return
            else:
                self.enqueue_log("[WARNING] Step 1 (GSE Extraction) and GPL downloads will not work")
                self.enqueue_log("[WARNING] Download from: https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz")
                return

        try:
            file_size_mb = os.path.getsize(gz_path) / (1024*1024)
            self.enqueue_log(f"[DB] Loading GEOmetadb from: {gz_path} ({file_size_mb:.0f} MB)")
            self.enqueue_log("[DB] This may take a minute on first run...")

            from genevariate.core.db_loader import open_geometadb
            self.gds_conn = open_geometadb(
                gz_path, log_fn=self.enqueue_log)
            if self.gds_conn is None:
                self.enqueue_log("[DB ERROR] Could not open GEOmetadb")
                return
            
            # Verify the database has the expected tables
            tables = [r[0] for r in self.gds_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.enqueue_log(f"[DB] Tables found: {', '.join(sorted(tables))}")
            
            # Dump schema for key tables
            for tbl in ['gpl', 'gsm', 'gse', 'gse_gpl', 'gse_gsm']:
                if tbl in tables:
                    cols = [r[1] for r in self.gds_conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
                    n_rows = self.gds_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    self.enqueue_log(f"[DB]   {tbl}: {n_rows:,} rows, columns: {cols}")
                    # Show first 3 sample values from first column
                    sample = self.gds_conn.execute(f"SELECT * FROM {tbl} LIMIT 3").fetchall()
                    if sample:
                        self.enqueue_log(f"[DB]   {tbl} sample: {sample[0][:4]}...")

            if 'gpl' not in tables:
                self.enqueue_log("[DB WARNING] 'gpl' table not found - this may not be a valid GEOmetadb!")
                # Check if table names are different (case?)
                for t in tables:
                    if 'gpl' in t.lower():
                        self.enqueue_log(f"[DB]   But found similar table: '{t}'")
            else:
                # Quick sanity check - count platforms
                n_gpl = self.gds_conn.execute("SELECT COUNT(*) FROM gpl").fetchone()[0]
                self.enqueue_log(f"[DB] Platforms in database: {n_gpl:,}")
                
                # Show GPL column names
                gpl_cols = [r[1] for r in self.gds_conn.execute("PRAGMA table_info(gpl)").fetchall()]
                self.enqueue_log(f"[DB] GPL table columns: {gpl_cols}")
                
                # Test: try to find GPL570 in various ways
                for q_label, q_sql in [
                    ("exact 'GPL570'", "SELECT gpl FROM gpl WHERE gpl = 'GPL570' LIMIT 1"),
                    ("COLLATE NOCASE", "SELECT gpl FROM gpl WHERE gpl = 'GPL570' COLLATE NOCASE LIMIT 1"),
                    ("UPPER match", "SELECT gpl FROM gpl WHERE UPPER(gpl) = 'GPL570' LIMIT 1"),
                    ("LIKE '%GPL570%'", "SELECT gpl FROM gpl WHERE gpl LIKE '%GPL570%' LIMIT 1"),
                    ("LIKE '%570%'", "SELECT gpl FROM gpl WHERE gpl LIKE '%570%' LIMIT 3"),
                ]:
                    try:
                        result = self.gds_conn.execute(q_sql).fetchall()
                        self.enqueue_log(f"[DB]   Search {q_label}: {result}")
                    except Exception as eq:
                        self.enqueue_log(f"[DB]   Search {q_label}: ERROR {eq}")
                
                # Show first 5 GPL IDs to see format
                sample_gpls = self.gds_conn.execute("SELECT gpl FROM gpl ORDER BY gpl LIMIT 5").fetchall()
                self.enqueue_log(f"[DB] First 5 GPL IDs: {[r[0] for r in sample_gpls]}")
            
            print("[DB] OK GEOmetadb loaded successfully")
            self.enqueue_log("[DB] OK GEOmetadb loaded successfully")
            
            self._create_gsm_lookup_table()
            
        except Exception as e:
            print(f"[DB ERROR] Could not load GEOmetadb: {e}")
            self.enqueue_log(f"[DB ERROR] Could not load GEOmetadb: {e}")
            import traceback
            self.enqueue_log(f"[DB ERROR] {traceback.format_exc()}")
            if self.gds_conn:
                self.gds_conn.close()
            self.gds_conn = None
            # No temp file to clean up here: decompression moved into
            # db_loader.open_geometadb, which owns and removes its own.
    
    def _create_gsm_lookup_table(self):
        """Pre-loads GSM->series_id mappings for fast vectorized lookups."""
        try:
            self.enqueue_log("[DB] Creating GSM lookup table...")
            self._gsm_lookup = pd.read_sql_query(
                "SELECT gsm, series_id FROM gsm", 
                self.gds_conn
            )
            self._gsm_lookup['gsm'] = self._gsm_lookup['gsm'].str.upper()
            self.enqueue_log(f"[DB] OK Loaded {len(self._gsm_lookup):,} GSM->GSE mappings")
        except Exception as e:
            self.enqueue_log(f"[DB WARNING] Could not create lookup table: {e}")
            self._gsm_lookup = None
    
    def _fast_gsm_lookup(self, gsm_list):
        """Fast vectorized GSM->series_id lookup using pre-loaded table."""
        if self._gsm_lookup is None:
            return self._sql_gsm_lookup(gsm_list)
        
        df = pd.DataFrame({'GSM': [str(g).upper() for g in gsm_list]})
        result = df.merge(self._gsm_lookup, left_on='GSM', right_on='gsm', how='left')
        return result[['GSM', 'series_id']]
    
    def _sql_gsm_lookup(self, gsm_list):
        """Fallback SQL-based GSM lookup (case-insensitive)."""
        chunk_size = CONFIG['database']['sql_chunk_size']
        results = []
        
        for i in range(0, len(gsm_list), chunk_size):
            chunk = [str(g).upper() for g in gsm_list[i:i + chunk_size]]
            placeholders = ','.join(['?'] * len(chunk))
            query = f"SELECT UPPER(gsm) AS GSM, series_id FROM gsm WHERE UPPER(gsm) IN ({placeholders})"
            results.append(pd.read_sql_query(query, self.gds_conn, params=chunk))
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _load_ai_pipeline(self):
        """Report the label-extraction backend (vendored geo_label_extractor)."""
        self.enqueue_log("[LLM] Label extraction: geo_label_extractor backend")
        try:
            from genevariate.core import geo_extract_driver as _drv
            _drv.configure_backend()
            self.ai_pipeline = True
            self.enqueue_log(
                "[LLM] OK Extraction backend configured "
                "(set the backend URL via geo_extract_driver.configure_backend)")
        except Exception as e:
            self.ai_pipeline = None
            self.enqueue_log(f"[LLM] [!] Extraction backend unavailable: {e}")
    
    def on_closing(self):
        """Enhanced window closing with cleanup and confirmation."""
        active_threads = []
        if self.current_extraction_thread and self.current_extraction_thread.is_alive():
            active_threads.append("GSE extraction")
        if self.current_labeling_thread and self.current_labeling_thread.is_alive():
            active_threads.append("sample labeling")
        
        if active_threads:
            msg = f"The following processes are running:\n" + chr(10).join(f"- {t}" for t in active_threads)
            msg += "\n\nStop these processes and exit?"
            
            if not messagebox.askyesno("Confirm Exit", msg, parent=self):
                return
            
            if self.current_extraction_thread:
                self.current_extraction_thread.stop()
            if self.current_labeling_thread:
                self.current_labeling_thread.stop()
        
        self.is_closing = True
        
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except:
                pass
        
        self._cleanup_all_figures()
        
        if self.gds_conn:
            try:
                self.gds_conn.close()
            except:
                pass
        
        self.destroy()
    
    def _cleanup_all_figures(self):
        """Cleans up all tracked matplotlib figures."""
        import matplotlib.pyplot as plt
        
        for key, (fig, widget, toolbar) in list(self.tracked_figures.items()):
            try:
                if widget and widget.winfo_exists():
                    widget.destroy()
                if toolbar and toolbar.winfo_exists():
                    toolbar.destroy()
                plt.close(fig)
            except:
                pass
        
        self.tracked_figures.clear()
        plt.close('all')

    def _setup_ui(self):
        """Sets up the complete user interface with ALL features - NO SIMPLIFICATIONS."""
        # ── Frutiger Aero header: glossy sky-blue gradient with logo ──
        header_height = 96
        header_canvas = tk.Canvas(self, height=header_height,
                                   highlightthickness=0, bd=0,
                                   bg=AERO["sky_top"])
        header_canvas.pack(fill=tk.X, padx=0, pady=0)
        self._header_canvas = header_canvas

        def _paint_header(event=None):
            w = header_canvas.winfo_width()
            if w < 2:
                w = self.winfo_width() or 1200
            # Smooth background gradient: sky_top -> white (top to bottom).
            # Use a tinted end color instead of pure white so the sheen band
            # added below blends without a visible seam.
            _aero_vertical_gradient(header_canvas, w, header_height,
                                    AERO["sky_top"], "#F4FBFF",
                                    tag="aero_header_bg")
            # Soft glossy sheen across the top third: a second gradient
            # (white -> current-bg) layered over the base. Previously this
            # used stipple="gray25" which renders as a literal dot-dither
            # pattern on modern displays - visible as vertical stripes.
            header_canvas.delete("aero_gloss")
            sheen_h = max(1, header_height // 3)
            try:
                tr, tg, tb = header_canvas.winfo_rgb("#FFFFFF")
                br, bg_, bb = header_canvas.winfo_rgb(AERO["sky_top"])
                tr, tg, tb = tr // 256, tg // 256, tb // 256
                br, bg_, bb = br // 256, bg_ // 256, bb // 256
                for y in range(sheen_h):
                    t = y / max(1, sheen_h - 1)
                    r = int(tr * (1 - t) + br * t)
                    g = int(tg * (1 - t) + bg_ * t)
                    b = int(tb * (1 - t) + bb * t)
                    header_canvas.create_rectangle(
                        0, y, w, y + 1,
                        fill=f"#{r:02x}{g:02x}{b:02x}",
                        outline="", tags="aero_gloss")
            except Exception:
                pass
            # thin cyan underline
            header_canvas.delete("aero_underline")
            header_canvas.create_rectangle(
                0, header_height - 3, w, header_height,
                fill=AERO["accent"], outline="", tags="aero_underline")
            # Keep the AI-assistant launcher (icon + label) pinned to the
            # top-right corner.
            self._place_agent_launcher(w)
            # Re-raise the foreground (logo + titles) above the freshly
            # painted background so the repaint on resize doesn't bury them.
            try:
                header_canvas.tag_raise("aero_header_fg")
            except Exception:
                pass
        header_canvas.bind("<Configure>", _paint_header)

        # Logo
        try:
            from PIL import Image, ImageTk
            _icon_path = Path(__file__).parent.parent / "assets" / "icon.png"
            if _icon_path.exists():
                _pil_img = Image.open(str(_icon_path))
                _pil_img = _pil_img.resize((72, 72), Image.LANCZOS)
                self._header_logo_img = ImageTk.PhotoImage(_pil_img)
                header_canvas.create_image(
                    20, header_height // 2,
                    image=self._header_logo_img, anchor="w",
                    tags="aero_header_fg")
        except Exception:
            pass

        header_canvas.create_text(
            106, header_height // 2 - 10,
            text="GeneVariate", anchor="w",
            font=("Segoe UI", 22, "bold"),
            fill=AERO["accent_dark"], tags="aero_header_fg")
        header_canvas.create_text(
            106, header_height // 2 + 18,
            text="Gene Expression Variability Analysis Platform",
            anchor="w", font=("Segoe UI", 10),
            fill=AERO["muted"], tags="aero_header_fg")

        # ── AI assistant launcher pinned to the header's top-right corner ──
        # Opens the conversational agent sidebar (same action as Tools ▸
        # "Assistant (chat)…" and Ctrl+/). Embedded as a canvas window item so
        # it floats over the glossy header; _paint_header keeps it anchored to
        # the right edge on every resize.
        # Frameless launcher: the branded chatbot icon (speech bubble + DNA
        # helix + AI sparkle) and the "AI Assistant" label are drawn straight
        # onto the header gradient as canvas items - no button widget, so there
        # is no box, just the moving icon + text. _paint_header re-anchors them
        # to the right edge on resize; the icon bobs/wobbles via an after timer.
        self._agent_header_canvas = header_canvas
        self._build_agent_icon_frames(size=30)
        cy = header_height // 2
        self._agent_text_item = header_canvas.create_text(
            0, cy, text="AI Assistant", anchor="e",
            font=("Segoe UI", 12, "bold"), fill=AERO["accent_dark"],
            tags=("aero_header_fg", "agent_launcher"))
        self._agent_icon_item = header_canvas.create_image(
            0, cy, anchor="e", tags=("aero_header_fg", "agent_launcher"))
        if self._agent_icon_frames:
            header_canvas.itemconfig(self._agent_icon_item,
                                     image=self._agent_icon_frames[0])
        self._place_agent_launcher(self.winfo_width() or 1200)
        header_canvas.tag_bind("agent_launcher", "<Button-1>",
                               lambda e: self._toggle_chat_sidebar())
        header_canvas.tag_bind(
            "agent_launcher", "<Enter>",
            lambda e: (header_canvas.itemconfig(self._agent_text_item,
                                                fill=AERO["accent"]),
                       header_canvas.config(cursor="hand2")))
        header_canvas.tag_bind(
            "agent_launcher", "<Leave>",
            lambda e: (header_canvas.itemconfig(self._agent_text_item,
                                                fill=AERO["accent_dark"]),
                       header_canvas.config(cursor="")))
        self._start_agent_icon_anim()

        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        self.status_label = ttk.Label(status_frame, text="Ready",
                                      foreground=AERO["muted"],
                                      background=AERO["bg_top"],
                                      font=('Segoe UI', 9, 'italic'))
        self.status_label.pack(side=tk.LEFT)

        # ── Scrollable main content area ──
        # Progress bar + log stay at bottom outside scroll
        self._bottom_frame = ttk.Frame(self)
        self._bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Horizontal wrapper so a collapsible chat sidebar can sit to the
        # right of the scroll area (created hidden; toggled with Ctrl+/).
        self._content_row = ttk.Frame(self)
        self._content_row.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._chat_frame = ttk.Frame(self._content_row, width=340)
        self._chat_sidebar = None
        self._chat_visible = False

        main_canvas = tk.Canvas(self._content_row, highlightthickness=0, bg=AERO["bg_top"])
        # Subtle Frutiger Aero sky→white→green gradient backdrop
        def _paint_main_bg(event=None):
            try:
                w = main_canvas.winfo_width()
                h = max(main_canvas.winfo_height(),
                        main_canvas.bbox("all")[3] if main_canvas.bbox("all") else 0)
                if w > 2 and h > 2:
                    _aero_vertical_gradient(main_canvas, w, h,
                                             AERO["bg_top"], AERO["bg_bot"],
                                             tag="aero_main_bg")
            except Exception:
                pass
        self._paint_main_bg = _paint_main_bg
        main_vsb = ttk.Scrollbar(self._content_row, orient="vertical", command=main_canvas.yview)
        self._main_canvas = main_canvas
        self._main_vsb = main_vsb
        self._main_sf = ttk.Frame(main_canvas)
        self._main_sf.bind("<Configure>",
                           lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        self._main_cw = main_canvas.create_window((0, 0), window=self._main_sf, anchor="nw")
        main_canvas.configure(yscrollcommand=main_vsb.set)
        main_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        def _on_main_canvas_configure(e):
            main_canvas.itemconfig(self._main_cw, width=e.width)
            _paint_main_bg(e)
        main_canvas.bind("<Configure>", _on_main_canvas_configure)
        # Mouse wheel scrolling for main content area
        def _on_mousewheel_linux(event):
            try:
                # Don't intercept scroll events for child windows (Toplevels)
                w = event.widget
                # Walk up widget hierarchy to check if it's in the main window
                wstr = str(w)
                if '.!toplevel' in wstr:
                    return  # Let child windows handle their own scrolling
                if event.num == 4:
                    main_canvas.yview_scroll(-3, "units")
                elif event.num == 5:
                    main_canvas.yview_scroll(3, "units")
            except: pass
        main_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        main_canvas.bind_all("<Button-5>", _on_mousewheel_linux)
        self._main_canvas = main_canvas

        plat_frame = labelframe(self._main_sf, text=" Load Gene Expression Platforms", padding=10)
        plat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_label = ttk.Label(plat_frame, text="Load platforms to analyze gene distributions. Use 'Download Platform' to fetch a microarray, bulk RNA-seq or single-cell platform from NCBI GEO, or 'Add Custom Platform' to load your own data files.", foreground="gray", font=('Segoe UI', 9, 'italic'), wraplength=1100)
        info_label.pack(fill=tk.X, pady=(0, 10))
        
        # Dynamic platform buttons - discovered from data directory
        self._plat_btn_frame = ttk.Frame(plat_frame)
        self._plat_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._refresh_platform_buttons()
        
        custom_frame = ttk.Frame(plat_frame)
        custom_frame.pack(fill=tk.X, padx=5, pady=8)
        
        ttk.Button(custom_frame, text="+ Add Custom Platform", command=self._load_custom_gpl_data, style="Add.TButton").pack(side=tk.LEFT, padx=5)

        self.cellxgene_btn = ttk.Button(
            custom_frame, text="Single-cell (CELLxGENE)",
            command=self._open_cellxgene_browser,
            style="Action.TButton")
        self.cellxgene_btn.pack(side=tk.LEFT, padx=8)
        self._set_tooltip(self.cellxgene_btn,
                          "Browse and fetch single-cell RNA-seq data from the CELLxGENE Discover Census "
                          "(real public scRNA-seq submissions). Pseudo-bulk the result to use it in every other window, "
                          "or open cell-level plots (composition / UMAP / dot plot / QC).")

        ttk.Button(custom_frame, text="Download Platform", command=self._open_gpl_downloader_window, style="Action.TButton").pack(side=tk.LEFT, padx=8)

        # Normalization is deliberately its own action: a download leaves the
        # raw values on disk, and this rescales them.
        ttk.Button(custom_frame, text="Normalize Platform",
                   command=self._open_normalize_platform_window,
                   style="Action.TButton").pack(side=tk.LEFT, padx=8)

        ttk.Button(custom_frame, text="Refresh", command=self._refresh_platform_buttons,
                   style="Ghost.TButton").pack(side=tk.LEFT, padx=5)

        self.loaded_plat_frame = ttk.Frame(plat_frame)
        self.loaded_plat_frame.pack(fill=tk.X, padx=5, pady=5)
        self.loaded_plat_label = ttk.Label(
            self.loaded_plat_frame, text="No platforms loaded yet",
            foreground=AERO["warn"], font=('Segoe UI', 9, 'bold'),
            justify=tk.LEFT)
        self.loaded_plat_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # This line names every loaded platform, so its length is set by the
        # data: it has to wrap onto further lines rather than be cut off.
        _wrap_to_parent(self.loaded_plat_label, pad=200)
        ttk.Button(self.loaded_plat_frame, text="+ Add Data Directory…",
                   command=self._add_data_directory,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        tools_frame = labelframe(self._main_sf, text=" Analysis Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(tools_frame, text="Always available - each tool can quick-load data on demand:", font=('Segoe UI', 9, 'italic'), foreground='gray').pack(anchor=tk.W, pady=(0, 5))
        
        tools_btn_frame = ttk.Frame(tools_frame)
        tools_btn_frame.pack(fill=tk.X, pady=4)
        tools_btn_frame.columnconfigure((0, 1, 2), weight=1, uniform="toolbtn")

        self.gene_explorer_btn = ttk.Button(
            tools_btn_frame, text="Gene Distribution Explorer",
            command=self.show_gene_distribution_popup,
            style="Tool.TButton")
        self.gene_explorer_btn.grid(row=0, column=0, padx=8, pady=6, sticky="ew")
        self._set_tooltip(self.gene_explorer_btn,
                          "Interactive histograms & KDEs of gene expression across loaded platforms.")

        self.dist_analysis_btn = ttk.Button(
            tools_btn_frame, text="Distribution Analysis",
            command=self._open_distribution_analysis,
            style="ToolGreen.TButton")
        self.dist_analysis_btn.grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        self._set_tooltip(self.dist_analysis_btn,
                          "Compare distributions (KDE, PCA/UMAP, clustering) or classify them "
                          "(unimodal / bimodal / skew + variability metrics).")

        self.label_enrich_btn = ttk.Button(
            tools_btn_frame, text="Label Enrichment",
            command=self._open_label_enrichment,
            style="Tool.TButton")
        self.label_enrich_btn.grid(row=0, column=2, padx=8, pady=6, sticky="ew")
        self._set_tooltip(self.label_enrich_btn,
                          "Fisher / hypergeometric enrichment of LLM-extracted labels across genes / distributions / platforms.")

        # ── Label Source (inside tools_frame - always visible) ──────
        ttk.Separator(tools_frame, orient='horizontal').pack(fill=tk.X, pady=(8, 4))

        lbl_hdr = ttk.Frame(tools_frame)
        lbl_hdr.pack(fill=tk.X)
        ttk.Label(lbl_hdr, text="Sample Labels:",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.label_source_var = tk.StringVar(value="ai")
        ttk.Radiobutton(lbl_hdr, text="LLM Extraction",
                        variable=self.label_source_var, value="ai",
                        command=self._toggle_main_label_source).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Radiobutton(lbl_hdr, text="Label Files (per-platform)",
                        variable=self.label_source_var, value="file",
                        command=self._toggle_main_label_source).pack(side=tk.LEFT, padx=4)

        # File controls row (hidden until "file" selected)
        self.labels_file_row = ttk.Frame(tools_frame)
        ttk.Button(self.labels_file_row, text="+ Add Label File…",
                   command=self._add_label_file,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(self.labels_file_row, text="+ Add Folder…",
                   command=self._browse_labels_folder,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(self.labels_file_row, text="Set Labels Directory",
                   command=self._set_labels_directory,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(self.labels_file_row, text="Clear All",
                   command=self._clear_all_labels,
                   style="Destructive.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(self.labels_file_row, text="Label curation",
                   command=self._open_llm_curator,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=3)
        self.label_entities_btn = ttk.Button(
            self.labels_file_row, text="Entity Links\u2026",
            command=self._open_label_entities,
            style="Secondary.TButton")
        self.label_entities_btn.pack(side=tk.LEFT, padx=3)
        self._set_tooltip(
            self.label_entities_btn,
            "What each label was resolved to: MeSH concept, Cellosaurus cell "
            "line, or a locally minted identifier. Written by the phase 2 "
            "normalization pass; phase 1 and 1b carry no accession.")
        ttk.Label(self.labels_file_row,
                  text="  (GPL ID auto-detected from filename)",
                  font=("Segoe UI", 8, "italic"), foreground="gray").pack(side=tk.LEFT, padx=4)

        # Per-platform loaded labels list
        self.labels_plat_frame = FlowFrame(tools_frame, spacing=14)

        # Column checkboxes row (populated after load)
        self.labels_col_frame = ttk.Frame(tools_frame)
        self.labels_col_vars = {}

        # Status line
        self.labels_status_lbl = ttk.Label(tools_frame,
                                            text="LLM mode: samples labeled via geo_label_extractor during analysis.",
                                            font=("Segoe UI", 8, "italic"), foreground="gray")
        self.labels_status_lbl.pack(fill=tk.X, padx=4, pady=(2, 0))
        # Lightly-padded labelframe so the collapsed state doesn't leave
        # an ugly blank white rectangle - padding grows only when expanded.
        self.step1_frame = labelframe(
            self._main_sf,
            text=" Step 1: Discover Experiments (Optional)",
            padding=(8, 2, 8, 2),
        )
        self.step1_frame.pack(fill=tk.X, padx=5, pady=5)
        self._step1_title = "Step 1: Discover Experiments (Optional)"
        self._set_step_status(self.step1_frame, self._step1_title, "pending")

        collapse_frame = ttk.Frame(self.step1_frame)
        collapse_frame.pack(fill=tk.X, pady=0)

        self.step1_collapsed = tk.BooleanVar(value=True)
        # Compact chevron-style toggle button (ttk - consistent with app theme)
        self.step1_toggle_btn = ttk.Button(
            collapse_frame, text="▸  Show",
            command=self._toggle_step1,
            style="Toggle.TButton",
        )
        self.step1_toggle_btn.pack(side=tk.LEFT)

        ttk.Label(collapse_frame,
                  text="Search GEOmetadb (+ optional ARCHS4 / CELLxGENE / Expression Atlas) by keywords",
                  font=('Segoe UI', 9, 'italic'),
                  foreground=AERO["muted"]
                  ).pack(side=tk.LEFT, padx=10)

        self.step1_content = ttk.Frame(self.step1_frame)

        # Load each source's official logo from assets/icons/ (shipped with
        # the app). Fall back to a coded badge if PIL fails to decode one.
        # Logos remain property of their respective organisations (NCBI,
        # Ma'ayan Lab, CZI, EMBL-EBI) - used for source attribution only.
        _icons_dir = Path(__file__).parent.parent / "assets" / "icons"
        _fallback = {
            "geo":        ("#2E5E9F", "GEO"),
            "archs4":     ("#E87722", "A4"),
            "cellxgene":  ("#7A4FBF", "Cx"),
            "atlas":      ("#008080", "EA"),
        }
        self._source_icons = {}
        TARGET_H = 22
        MAX_W = 80
        for key, (color, initials) in _fallback.items():
            icon = None
            try:
                from PIL import Image, ImageTk
                candidates = list(_icons_dir.glob(f"{key}.*"))
                if candidates:
                    img = Image.open(str(candidates[0])).convert("RGBA")
                    w, h = img.size
                    new_w = max(1, int(round(w * TARGET_H / h)))
                    if new_w > MAX_W:
                        new_w = MAX_W
                        new_h = max(1, int(round(h * MAX_W / w)))
                    else:
                        new_h = TARGET_H
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    icon = ImageTk.PhotoImage(img)
            except Exception:
                icon = None
            if icon is None:
                icon = self._make_source_badge(color, initials)
            self._source_icons[key] = icon

        # State vars - all sources now toggleable, no defaults enabled so the
        # user must pick at least one.
        self.src_geo_var = tk.BooleanVar(value=True)
        self.src_archs4_var = tk.BooleanVar(value=False)
        self.src_cellxgene_var = tk.BooleanVar(value=False)
        self.src_atlas_var = tk.BooleanVar(value=False)

        # ═══ STAGE 1 ═══ Pick data sources ─────────────────────────────
        ttk.Label(self.step1_content,
                  text="1.  Select data sources",
                  font=('Segoe UI', 10, 'bold'),
                  foreground=AERO.get("accent_dark", "#1565C0")
                  ).grid(row=0, column=0, columnspan=2, sticky=tk.W,
                         pady=(4, 2))

        src_frame = ttk.Frame(self.step1_content)
        src_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        def _add_source(parent, key, text, tooltip, var, on_toggle):
            wrapper = ttk.Frame(parent)
            wrapper.pack(side=tk.LEFT, padx=6)
            icon = self._source_icons.get(key)
            if icon is not None:
                lbl = ttk.Label(wrapper, image=icon)
                lbl.pack(side=tk.LEFT, padx=(0, 2))
                self._set_tooltip(lbl, tooltip)
            cb = ttk.Checkbutton(wrapper, text=text, variable=var,
                                  command=on_toggle)
            cb.pack(side=tk.LEFT)
            self._set_tooltip(cb, tooltip)
            return cb

        _add_source(src_frame, "geo", "GEOmetadb",
            "GEOmetadb (NCBI GEO mirror)\n"
            "• SQLite snapshot of NCBI GEO - microarray & bulk RNA-seq\n"
            "• ~4M samples across ~150k experiments (GSE/GSM)\n"
            "• Source of sample-level metadata used in Step 2\n"
            "• Sub-filter: GPL platform IDs (e.g. GPL570).",
            self.src_geo_var, lambda: self._refresh_step1_subfilters())

        _add_source(src_frame, "archs4", "ARCHS4",
            "ARCHS4 (Ma'ayan Lab)\n"
            "• Uniformly reprocessed bulk RNA-seq from GEO/SRA\n"
            "• Human (~720 k samples) + mouse (~400 k samples)\n"
            "• Remote HDF5 range-reads - no 30 GB download\n"
            "• Sub-filter: organism (human / mouse).",
            self.src_archs4_var, lambda: self._refresh_step1_subfilters())

        _add_source(src_frame, "cellxgene", "CELLxGENE",
            "CELLxGENE Discover Census (CZI)\n"
            "• Curated scRNA-seq datasets (~50M+ cells)\n"
            "• Harmonised schema (tissue / disease / assay / cell_type)\n"
            "• Sub-filters: tissue, disease, assay (match the Discover UI).",
            self.src_cellxgene_var, lambda: self._refresh_step1_subfilters())

        _add_source(src_frame, "atlas", "Expression Atlas",
            "Expression Atlas (EMBL-EBI)\n"
            "• Curated differential-expression experiments\n"
            "• Bulk RNA-seq, microarray, proteomics, single-cell\n"
            "• Sub-filter: species (e.g. 'Homo sapiens', 'Mus musculus').",
            self.src_atlas_var, lambda: self._refresh_step1_subfilters())

        # ═══ STAGE 2 ═══ Per-source sub-filters (conditional) ──────────
        self._step1_stage2_header = ttk.Label(
            self.step1_content,
            text="2.  Platform / sub-filters (optional - narrow the search)",
            font=('Segoe UI', 10, 'bold'),
            foreground=AERO.get("accent_dark", "#1565C0"))
        self._step1_stage2_header.grid(row=2, column=0, columnspan=2,
                                        sticky=tk.W, pady=(10, 2))

        self._step1_subfilter_frame = ttk.Frame(self.step1_content)
        self._step1_subfilter_frame.grid(row=3, column=0, columnspan=2,
                                          sticky=tk.EW, pady=2)
        self._step1_subfilter_frame.grid_columnconfigure(1, weight=1)

        # Each source's sub-filter group is a named Frame that we pack / forget
        # depending on whether that source's checkbox is on.

        # GEO - GPL platform filter (reuse the existing name platform_entry)
        self._sf_geo = ttk.Frame(self._step1_subfilter_frame)
        ttk.Label(self._sf_geo, text="GPL IDs (comma):",
                  width=18).pack(side=tk.LEFT)
        self.platform_entry = ttk.Entry(self._sf_geo, width=40)
        self.platform_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._set_tooltip(self.platform_entry,
            "Limit GEOmetadb results to these platform IDs "
            "(e.g. GPL570, GPL96, GPL11154). Leave blank for all platforms.")

        # ARCHS4 - organism radio
        self._sf_archs4 = ttk.Frame(self._step1_subfilter_frame)
        ttk.Label(self._sf_archs4, text="Organism:",
                  width=18).pack(side=tk.LEFT)
        self.archs4_org_var = tk.StringVar(value="human")
        ttk.Radiobutton(self._sf_archs4, text="Human",
                        variable=self.archs4_org_var, value="human"
                        ).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(self._sf_archs4, text="Mouse",
                        variable=self.archs4_org_var, value="mouse"
                        ).pack(side=tk.LEFT, padx=4)

        # CELLxGENE has no sub-filter - keywords alone drive its faceted search
        self._sf_cellxgene = None

        # Expression Atlas - species
        self._sf_atlas = ttk.Frame(self._step1_subfilter_frame)
        ttk.Label(self._sf_atlas, text="Species:",
                  width=18).pack(side=tk.LEFT)
        self.atlas_species_entry = ttk.Entry(self._sf_atlas, width=40)
        self.atlas_species_entry.pack(side=tk.LEFT, padx=4,
                                       fill=tk.X, expand=True)
        self._set_tooltip(self.atlas_species_entry,
            "Expression Atlas species filter "
            "(e.g. 'Homo sapiens', 'Mus musculus', 'Arabidopsis thaliana'). "
            "Leave blank for all species.")

        # Info label that shows when no sub-filter is applicable
        self._sf_empty = ttk.Label(self._step1_subfilter_frame,
            text="(select at least one source above)",
            font=('Segoe UI', 9, 'italic'),
            foreground=AERO.get("muted", "gray"))

        # ═══ STAGE 3 ═══ Keywords + search button ──────────────────────
        ttk.Label(self.step1_content,
                  text="3.  Keywords (comma-separated)",
                  font=('Segoe UI', 10, 'bold'),
                  foreground=AERO.get("accent_dark", "#1565C0")
                  ).grid(row=4, column=0, columnspan=2, sticky=tk.W,
                         pady=(10, 2))

        kw_row = ttk.Frame(self.step1_content)
        kw_row.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=2)
        kw_row.grid_columnconfigure(1, weight=1)
        ttk.Label(kw_row, text="Keywords:",
                  width=18).grid(row=0, column=0, sticky=tk.W)
        self.filter_entry = ttk.Entry(kw_row)
        self.filter_entry.grid(row=0, column=1, sticky=tk.EW, padx=4)
        self._set_tooltip(self.filter_entry,
            "Comma-separated search terms. Applied to every selected "
            "source. Matches titles, descriptions and curated metadata.")

        s1_btn_frame = ttk.Frame(self.step1_content)
        s1_btn_frame.grid(row=7, column=0, columnspan=2, pady=8)
        self._step1_search_btn = ttk.Button(
            s1_btn_frame, text="Search Selected Sources",
            command=self.start_extraction,
            style="Add.TButton")
        self._step1_search_btn.pack(side=tk.LEFT, padx=10)

        self.step1_content.grid_columnconfigure(0, weight=1)
        self.step1_content.grid_columnconfigure(1, weight=1)

        # Initial population of sub-filter frames + active-icons strip
        self._refresh_step1_subfilters()
        
        self.gse_frame = labelframe(self._main_sf, text=" Step 1.5: Selected Experiments for Analysis", padding=10)
        self._step15_title = "Step 1.5: Selected Experiments for Analysis"
        self._set_step_status(self.gse_frame, self._step15_title, "pending")

        gse_info = ttk.Label(self.gse_frame, text="These experiments will be used for Step 2 extraction", font=('Segoe UI', 9, 'italic'), foreground='gray')
        gse_info.pack(fill=tk.X, pady=(0, 5))
        
        gse_list_frame = ttk.Frame(self.gse_frame)
        gse_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.gse_listbox = tk.Listbox(gse_list_frame, selectmode=tk.MULTIPLE, width=80, height=4, font=('Segoe UI', 9))
        self.gse_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        sb_gse = ttk.Scrollbar(gse_list_frame, command=self.gse_listbox.yview)
        sb_gse.pack(side=tk.RIGHT, fill=tk.Y)
        self.gse_listbox.config(yscrollcommand=sb_gse.set)
        
        gse_btn_frame = ttk.Frame(self.gse_frame)
        gse_btn_frame.pack(fill=tk.X, pady=5)
        gse_btn_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="gsebtn")

        ttk.Button(gse_btn_frame, text="Save Selected for Step 2",
                   command=self._save_selected_gses,
                   style="Primary.TButton").grid(row=0, column=0, padx=6, pady=4, sticky="ew")

        ttk.Button(gse_btn_frame, text="Select All",
                   command=lambda: self.gse_listbox.select_set(0, tk.END),
                   style="Secondary.TButton").grid(row=0, column=1, padx=6, pady=4, sticky="ew")

        ttk.Button(gse_btn_frame, text="Clear Selection",
                   command=lambda: self.gse_listbox.selection_clear(0, tk.END),
                   style="Destructive.TButton").grid(row=0, column=2, padx=6, pady=4, sticky="ew")

        ttk.Button(gse_btn_frame, text="Review Details",
                   command=self._review_gse_details,
                   style="Secondary.TButton").grid(row=0, column=3, padx=6, pady=4, sticky="ew")

        # Row 2: Download Expression Data button (spans full width)
        self._download_expr_btn = ttk.Button(
            gse_btn_frame,
            text="Download Expression Data for Selected Experiments",
            command=self._download_selected_expression,
            style="Warn.TButton")
        self._download_expr_btn.grid(row=1, column=0, columnspan=4,
                                      padx=6, pady=4, sticky="ew")

        # Download progress
        self._dl_progress_frame = ttk.Frame(self.gse_frame)
        self._dl_progress_bar = ttk.Progressbar(self._dl_progress_frame, mode='determinate', length=400)
        self._dl_progress_bar.pack(fill=tk.X, padx=5)
        self._dl_progress_label = ttk.Label(self._dl_progress_frame, text="", font=('Consolas', 8), foreground='gray')
        self._dl_progress_label.pack(anchor='w', padx=5)
        lab_frame = labelframe(self._main_sf, text=" Step 2: Sample Classification & Analysis", padding=10)
        lab_frame.pack(fill=tk.X, padx=5, pady=5)
        self._step2_frame = lab_frame
        self._step2_title = "Step 2: Sample Classification & Analysis"
        self._set_step_status(lab_frame, self._step2_title, "pending")
        
        self.step2_status_label = ttk.Label(lab_frame, text="Ready to extract labels from Step 1 or external file", foreground=AERO["accent"], font=('Segoe UI', 9))
        self.step2_status_label.pack(pady=5)
        
        btn_row = ttk.Frame(lab_frame)
        btn_row.pack(pady=8, fill=tk.X)
        # Use grid for perfect symmetry
        btn_row.columnconfigure((0, 1, 2), weight=1, uniform="step2btn")

        self.load_csv_btn = ttk.Button(
            btn_row, text="Load External CSV",
            command=self.load_external_file_for_step2,
            style="Tool.TButton")
        self.load_csv_btn.grid(row=0, column=0, padx=8, pady=6, sticky="ew")

        self.ai_label_btn = ttk.Button(
            btn_row, text="LLM Extraction",
            command=self._open_llm_extraction_window,
            style="Primary.TButton")
        self.ai_label_btn.grid(row=0, column=1, padx=8, pady=6, sticky="ew")

        self.manual_label_btn = ttk.Button(
            btn_row, text="Manual Labeling",
            command=self.run_manual_labeling,
            style="ToolGreen.TButton")
        self.manual_label_btn.grid(row=0, column=2, padx=8, pady=6, sticky="ew")

        

        progress_frame = ttk.Frame(self._bottom_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progressbar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progressbar.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="0%", width=5)
        self.progress_label.pack(side=tk.RIGHT)
        
        self.progress_status = ttk.Label(progress_frame, text="", font=('Segoe UI', 8),
                                          foreground='#666')
        self.progress_status.pack(side=tk.RIGHT, padx=(0, 10))
        
        self.progressbar.bind("<<ProgressUpdate>>", self._update_progress_label)
        
        # Log button (opens separate window)
        log_btn_frame = ttk.Frame(self._bottom_frame)
        log_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(log_btn_frame, text="Show Activity Log", command=self._show_log_window).pack(side=tk.LEFT, padx=5)
        self._log_msg_count = 0
        self.log_status_label = ttk.Label(log_btn_frame, text="Log: 0 messages", foreground="gray", font=('Segoe UI', 9))
        self.log_status_label.pack(side=tk.LEFT, padx=10)
        
        # Create log window (hidden)
        self.log_window = tk.Toplevel(self)
        style_window(self.log_window)
        self.log_window.title("GeneVariate - Activity Log")
        self.log_window.geometry("1100x700")
        self.log_window.withdraw()
        self.log_window.protocol("WM_DELETE_WINDOW", self.log_window.withdraw)
        
        log_ctrl = ttk.Frame(self.log_window)
        log_ctrl.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(log_ctrl, text="Clear Log", command=lambda: (self.log_text.delete('1.0', tk.END), setattr(self, '_log_msg_count', 0), self.log_status_label.config(text="Log: 0 messages"))).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_ctrl, text="Save Log", command=self._save_log).pack(side=tk.LEFT, padx=2)
        
        log_container = ttk.Frame(self.log_window)
        log_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Professional light/corporate-style log widget: soft off-white
        # background, monospace font, timestamps, color-coded category
        # chips, zebra-striped row backgrounds for clean separation.
        self.log_text = tk.Text(
            log_container,
            wrap=tk.WORD,
            font=('Cascadia Mono', 10),   # falls back to Consolas if absent
            bg='#FAFBFC',                  # page-off-white
            fg='#24292F',                  # charcoal body text
            insertbackground='#24292F',
            selectbackground='#D0E4FF',
            selectforeground='#1C2128',
            spacing1=2, spacing3=2,        # extra leading/trailing per line
            padx=10, pady=8,
            borderwidth=1, highlightthickness=0,
            relief='solid',
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_log = ttk.Scrollbar(log_container, command=self.log_text.yview)
        sb_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb_log.set)

        # ── Tag palette ────────────────────────────────────────────────
        # Base content styles (light theme, good contrast on off-white)
        self.log_text.tag_configure("ts",       foreground='#6E7781',
                                     font=('Cascadia Mono', 9))
        self.log_text.tag_configure("level_info",    foreground='#1A7F37',
                                     font=('Cascadia Mono', 10, 'bold'))
        self.log_text.tag_configure("level_warn",    foreground='#9A6700',
                                     font=('Cascadia Mono', 10, 'bold'))
        self.log_text.tag_configure("level_err",     foreground='#CF222E',
                                     font=('Cascadia Mono', 10, 'bold'))
        self.log_text.tag_configure("body",          foreground='#24292F')
        self.log_text.tag_configure("body_warn",     foreground='#7D4E00')
        self.log_text.tag_configure("body_err",      foreground='#A40E26')

        # Alternating row backgrounds (zebra) for clear separation
        self.log_text.tag_configure("zebra_even", background='#FAFBFC')
        self.log_text.tag_configure("zebra_odd",  background='#EEF2F6')

        # Category color palette - each [CATEGORY] gets its own color
        # (picked so they stay readable on the light zebra backgrounds).
        self._log_cat_colors = {
            'DB':             '#0969DA',  # blue
            'Load':           '#116329',  # dark teal-green
            'GEOmetadb':      '#0550AE',  # deeper blue
            'GPL Browser':    '#8250DF',  # purple
            'GPL':            '#8250DF',
            'GeneList':       '#6639BA',  # indigo
            'XPlat':          '#BC4C00',  # dark orange
            'Cache':          '#57606A',  # slate
            'Startup':        '#1F6FEB',  # bright blue
            'GSM→GSE':        '#0550AE',
            'Species Picker': '#8250DF',
            'Config':         '#57606A',
            'SETUP':          '#9A6700',
            'INFO':           '#1A7F37',
            'WARNING':        '#9A6700',
            'ERROR':          '#CF222E',
        }
        # Register the category tags once so inserts are fast
        for cat, col in self._log_cat_colors.items():
            tname = f"cat_{cat}"
            self.log_text.tag_configure(
                tname, foreground=col,
                font=('Cascadia Mono', 10, 'bold'),
            )
        self.log_text.tag_configure("cat_default",
                                     foreground='#1F6FEB',
                                     font=('Cascadia Mono', 10, 'bold'))

        # Legacy tag names - kept so other code that references them still works
        self.log_text.tag_configure("error",
                                     foreground='#CF222E',
                                     font=('Cascadia Mono', 10, 'bold'))
        self.log_text.tag_configure("warning",
                                     foreground='#9A6700',
                                     font=('Cascadia Mono', 10, 'bold'))
        self.log_text.tag_configure("info",   foreground='#1A7F37')
        self.log_text.tag_configure("header", foreground='#0969DA',
                                     font=('Cascadia Mono', 10, 'bold'))

        # Line-counter so the zebra stripes advance correctly
        self._log_line_idx = 0
    
    def _refresh_step1_subfilters(self):
        """Show only the sub-filter frames whose source checkbox is ticked.

        Also refreshes the "active sources" icon strip under stage 3 and
        enables/disables the Search button based on whether at least one
        source is selected.
        """
        # Map: (boolean_var, subfilter_frame_or_None, source_key)
        # CELLxGENE has no sub-filter (entry is None) - it still counts toward
        # any_on so the Search button enables, but no frame is packed for it.
        mapping = [
            (self.src_geo_var,        self._sf_geo,       "geo"),
            (self.src_archs4_var,     self._sf_archs4,    "archs4"),
            (self.src_cellxgene_var,  self._sf_cellxgene, "cellxgene"),
            (self.src_atlas_var,      self._sf_atlas,     "atlas"),
        ]
        any_on = False
        # Detach every subfilter frame first, then pack only the active ones
        for _var, frame, _key in mapping:
            if frame is None:
                continue
            try:
                frame.pack_forget()
            except Exception:
                pass
        try:
            self._sf_empty.pack_forget()
        except Exception:
            pass
        for var, frame, _key in mapping:
            if var.get():
                any_on = True
                if frame is not None:
                    frame.pack(fill=tk.X, padx=4, pady=2)
        if not any_on:
            self._sf_empty.pack(fill=tk.X, padx=4, pady=2)

        # Toggle the Search button
        try:
            state = "normal" if any_on else "disabled"
            self._step1_search_btn.configure(state=state)
        except Exception:
            pass

    def _make_source_badge(self, color_hex: str, text: str, size: int = 20):
        """Draw a small rounded-square badge icon with initials, return PhotoImage.

        We render with PIL so we get anti-aliased rounded corners, then hand
        the result to Tk via ``ImageTk.PhotoImage``. Falls back to ``None``
        (label skipped) if PIL is unavailable.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
        except Exception:
            return None
        # Render at 4x then downscale for a smooth edge
        scale = 4
        w = h = size * scale
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = int(w * 0.22)  # corner radius
        try:
            d.rounded_rectangle([(0, 0), (w - 1, h - 1)],
                                 radius=r, fill=color_hex)
        except AttributeError:
            # Pillow <8.2 - fall back to a plain rectangle
            d.rectangle([(0, 0), (w - 1, h - 1)], fill=color_hex)
        # Pick a font size that fits the badge
        label = (text or "?")[:3]
        target_px = int(h * 0.55)
        font = None
        for fname in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf",
                      "arialbd.ttf", "Helvetica-Bold"):
            try:
                font = ImageFont.truetype(fname, target_px)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        try:
            bbox = d.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = (w - tw) / 2 - bbox[0]
            ty = (h - th) / 2 - bbox[1]
        except AttributeError:
            tw, th = d.textsize(label, font=font)
            tx = (w - tw) / 2
            ty = (h - th) / 2
        d.text((tx, ty), label, fill="white", font=font)
        img = img.resize((size, size), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _set_tooltip(self, widget, text):
        """Attach a lightweight hover tooltip to any widget.

        Implemented as a tiny borderless Toplevel that appears on <Enter>
        and is destroyed on <Leave>. Safe to call repeatedly (re-binds)."""
        if not text:
            return
        state = {"tip": None, "after": None}

        def _show(_e=None):
            if state["tip"] is not None:
                return
            try:
                x = widget.winfo_rootx() + 14
                y = widget.winfo_rooty() + widget.winfo_height() + 6
            except Exception:
                return
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.attributes("-topmost", True)
            try:
                tip.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass
            frame = tk.Frame(tip, bg=AERO["accent_dark"], padx=1, pady=1)
            frame.pack()
            label = tk.Label(frame, text=text,
                             bg="#FFFFFF", fg=AERO["text"],
                             font=('Segoe UI', 9),
                             padx=8, pady=4, justify="left",
                             wraplength=340)
            label.pack()
            state["tip"] = tip

        def _schedule(e=None):
            _cancel()
            state["after"] = widget.after(350, _show)

        def _cancel(_e=None):
            if state["after"] is not None:
                try:
                    widget.after_cancel(state["after"])
                except Exception:
                    pass
                state["after"] = None
            if state["tip"] is not None:
                try:
                    state["tip"].destroy()
                except Exception:
                    pass
                state["tip"] = None

        widget.bind("<Enter>", _schedule, add="+")
        widget.bind("<Leave>", _cancel, add="+")
        widget.bind("<ButtonPress>", _cancel, add="+")

    # ─── Step status badge ─────────────────────────────────────────
    def _set_step_status(self, frame, step_label, state="pending"):
        """Update the status badge on a LabelFrame title.

        state in {'pending', 'running', 'done', 'error'}.
        """
        glyph = {"pending": "○", "running": "◔",
                 "done": "✓", "error": "✗"}.get(state, "○")
        try:
            frame.configure(text=f" {glyph}  {step_label}")
        except Exception:
            pass

    @staticmethod
    def _fit_window(win, fallback_w=900, fallback_h=700):
        """Ensure a Toplevel window is properly sized, centered, and fully visible.
        Call AFTER all widgets are added. Uses reqwidth/reqheight if available,
        otherwise falls back to specified size. Always centers on screen.
        """
        try:
            win.update_idletasks()
            req_w = win.winfo_reqwidth()
            req_h = win.winfo_reqheight()
            w = req_w if req_w > 400 else fallback_w
            h = req_h if req_h > 300 else fallback_h
            scr_w = win.winfo_screenwidth()
            scr_h = win.winfo_screenheight()
            w = min(w, int(scr_w * 0.92))
            h = min(h, int(scr_h * 0.92))
            x = max(0, (scr_w - w) // 2)
            y = max(0, (scr_h - h) // 2)
            win.geometry(f"{w}x{h}+{x}+{y}")
            win.minsize(min(w, 500), min(h, 400))
        except Exception:
            pass

    def _setup_interactive_legend(self, fig, ax, canvas, outside=True,
                                   artist_groups=None, **legend_kwargs):
        """Create a clickable legend where clicking any entry opens a color picker.

        Changes color of both the legend marker and the associated plot artist(s).
        If outside=True, places legend outside the plot area.
        artist_groups: optional list of lists - each inner list is a group of
                       artists to recolor together (e.g., all patches from one hist call).
        Returns the legend object.
        """
        from matplotlib.collections import PathCollection  # scatter
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle

        # Default legend kwargs
        kw = dict(fontsize=8, framealpha=0.9)
        if outside:
            kw.update(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            fig.subplots_adjust(right=0.78)
        kw.update(legend_kwargs)

        leg = ax.legend(**kw)
        if leg is None:
            return None

        # Build map: legend handle index -> artist(s) to recolor
        legend_handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles

        # If artist_groups provided, use that mapping directly
        if artist_groups and len(artist_groups) == len(legend_handles):
            handle_map = {i: artist_groups[i] for i in range(len(legend_handles))}
        else:
            # Auto-detect: match legend handles to plot artists in order
            plot_artists = [c for c in ax.get_children()
                            if isinstance(c, (PathCollection, Line2D))
                            and c not in legend_handles]
            # Deduplicate preserving order
            seen = set()
            unique_artists = []
            for a in plot_artists:
                if id(a) not in seen:
                    seen.add(id(a))
                    unique_artists.append(a)
            handle_map = {i: [unique_artists[i]] for i in range(min(len(legend_handles), len(unique_artists)))}

        for lh in legend_handles:
            lh.set_picker(True)
            # Only line and collection handles carry a pick radius; a legend
            # over histogram bars hands back Rectangles, which are picked by
            # containment and raise AttributeError here.
            if hasattr(lh, "set_pickradius"):
                lh.set_pickradius(10)

        def _on_pick(event):
            artist = event.artist
            # Find which legend handle was clicked
            idx = None
            for i, lh in enumerate(legend_handles):
                if lh is artist:
                    idx = i
                    break
            if idx is None:
                return

            color = colorchooser.askcolor(title="Choose color for this series")
            if color[1] is None:
                return
            new_color = color[1]

            # Update plot artist(s)
            for pa in handle_map.get(idx, []):
                if isinstance(pa, PathCollection):
                    pa.set_facecolors(new_color)
                    pa.set_edgecolors(new_color)
                elif isinstance(pa, Line2D):
                    pa.set_color(new_color)
                elif isinstance(pa, (Rectangle, Patch)):
                    pa.set_facecolor(new_color)
                else:
                    try:
                        pa.set_color(new_color)
                    except:
                        pass

            # Update legend handle
            lh = legend_handles[idx]
            if isinstance(lh, Line2D):
                lh.set_color(new_color)
                lh.set_markerfacecolor(new_color)
            elif isinstance(lh, (Rectangle, Patch)):
                lh.set_facecolor(new_color)
            elif isinstance(lh, PathCollection):
                lh.set_facecolors(new_color)
            else:
                try:
                    lh.set_color(new_color)
                except:
                    pass

            canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', _on_pick)
        return leg
    
    def _toggle_step1(self):
        """Toggles Step 1 visibility + grows/shrinks the LabelFrame padding
        so the collapsed state doesn't leave a blank white rectangle."""
        if self.step1_collapsed.get():
            # Expand: show content and restore breathing room
            try:
                self.step1_frame.configure(padding=(10, 8, 10, 10))
            except Exception:
                pass
            self.step1_content.pack(fill=tk.X, pady=5)
            self.step1_toggle_btn.config(text="▾  Hide")
            self.step1_collapsed.set(False)
        else:
            # Collapse: hide content and tighten padding to kill the empty strip
            self.step1_content.pack_forget()
            try:
                self.step1_frame.configure(padding=(8, 2, 8, 2))
            except Exception:
                pass
            self.step1_toggle_btn.config(text="▸  Show")
            self.step1_collapsed.set(True)
    
    def _update_progress_label(self, event=None):
        """Updates progress percentage label."""
        val = self.progressbar["value"]
        self.progress_label.config(text=f"{val:.0f}%")

    def _acquire_progress(self):
        """Register an operation as using the main progress bar.
        Call this when starting a long-running task that updates progress."""
        with self._progress_lock:
            self._progress_owners += 1

    def _release_progress(self):
        """Unregister an operation from the main progress bar.
        Only resets the bar if no other operations are still using it."""
        with self._progress_lock:
            self._progress_owners = max(0, self._progress_owners - 1)
            should_reset = self._progress_owners == 0
        if should_reset:
            self.update_progress(value=0, _force=True)

    def update_progress(self, value=None, text=None, maximum=None, _force=False):
        """Universal progress update - callable from any process/thread.
        value: 0-100 (or None to keep current)
        text: status message (or None to keep current)
        maximum: set new maximum (default 100)

        If value=0 and another extraction is still running, the reset is
        suppressed so the bar keeps showing the active operation's progress.
        """
        # Guard: don't reset the bar if another operation still owns it
        if value == 0 and not _force and not text:
            with self._progress_lock:
                if self._progress_owners > 0:
                    return  # another operation is still updating the bar

        def _do():
            try:
                if maximum is not None:
                    self.progressbar["maximum"] = maximum
                anim = getattr(self, '_animator', None)
                maxv = self.progressbar["maximum"] or 100
                if value is not None:
                    # Liquid fill: interpolate when animator is available and we're
                    # not snapping to 0 or to maximum (those should feel instant).
                    if (anim is not None and value not in (0, maxv)
                            and hasattr(self.progressbar, 'winfo_exists')):
                        try:
                            anim.smooth_to(self.progressbar, value)
                        except Exception:
                            self.progressbar["value"] = value
                    else:
                        self.progressbar["value"] = value
                    pct = value / max(1, self.progressbar["maximum"]) * 100
                    self.progress_label.config(text=f"{pct:.0f}%")

                    # ── Shimmer animation: on while a task is active ──
                    # Start the color pulse whenever the bar is neither idle (0)
                    # nor complete (>=max). Stop it otherwise. Applies to every
                    # process that funnels through update_progress - search,
                    # downloads, analyses, labeling, etc.
                    if anim is not None:
                        try:
                            active = (value > 0) and (value < maxv)
                            if active and not getattr(self, "_main_bar_pulsing", False):
                                anim.pulse_bar(
                                    self.progressbar,
                                    palette=_UIAnimator.PULSE_BLUES)
                                self._main_bar_pulsing = True
                            elif (not active) and getattr(self, "_main_bar_pulsing", False):
                                anim.stop_pulse(self.progressbar)
                                self._main_bar_pulsing = False
                        except Exception:
                            pass
                if text is not None:
                    self.progress_status.config(text=text)
                if value == 0 or (value is not None and value >= self.progressbar["maximum"]):
                    if value == 0:
                        self.progress_status.config(text="")
                        self.progress_label.config(text="0%")
                        # Stop any lingering pulse animation on the main bar
                        try:
                            if anim is not None:
                                anim.stop_pulse(self.progressbar)
                                self._main_bar_pulsing = False
                        except Exception:
                            pass
            except Exception:
                pass
        try:
            self.after(0, _do)
        except Exception:
            pass
    
    def _save_log(self):
        """Saves log to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"genevariate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get('1.0', tk.END))
                messagebox.showinfo("Log Saved", f"Log saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log:\n{e}")
    
    def _startup_discovery(self):
        """Run platform discovery on startup and log diagnostics."""
        try:
            available = self._discover_available_platforms()
            _prog_dir = os.path.dirname(os.path.abspath(__file__))
            _proj_root = os.path.dirname(_prog_dir)
            _cwd = os.getcwd()
            
            print(f"[Startup] Project root: {_proj_root}")
            print(f"[Startup] app.py dir:   {_prog_dir}")
            print(f"[Startup] Working dir:  {_cwd}")
            print(f"[Startup] data_dir:     {self.data_dir}")
            if self._user_data_dirs:
                print(f"[Startup] User dirs:    {', '.join(self._user_data_dirs)}")
            
            if available:
                print(f"[Startup] Found {len(available)} platform(s):")
                for gpl_id, fpath in sorted(available.items())[:15]:
                    print(f"[Startup]   {gpl_id}: {fpath}")
                if len(available) > 15:
                    print(f"[Startup]   ... +{len(available)-15} more")
                self.enqueue_log(
                    f"[Startup] Found {len(available)} GPL platform(s) on disk: "
                    f"{', '.join(sorted(available.keys())[:8])}"
                    f"{f' +{len(available)-8} more' if len(available) > 8 else ''}")
            else:
                print(f"[Startup] No GPL platform files found!")
                print(f"[Startup] Checked: {self.data_dir}")
                self.enqueue_log(
                    "[Startup] No GPL files found. Use 'Add Data Directory' or 'Download GPL'.")
        except Exception as e:
            print(f"[Startup] Discovery error: {e}")
        
        self._update_platform_status()

    def _update_platform_status(self):
        """Updates the loaded platforms status display."""
        available = self._discover_available_platforms()
        has_loaded = bool(self.gpl_datasets)
        has_available = bool(available)

        if not has_loaded and not has_available:
            self.loaded_plat_label.config(text="No platforms loaded yet",
                                           foreground=AERO["warn"])
            # Tools stay enabled - each one can quick-load on demand.
            self.gene_explorer_btn.config(state=tk.NORMAL)
            self.dist_analysis_btn.config(state=tk.NORMAL)
            if hasattr(self, 'label_enrich_btn'):
                self.label_enrich_btn.config(state=tk.NORMAL)
        elif not has_loaded and has_available:
            # Files on disk but nothing fully loaded - Gene Explorer can quick-load genes
            avail_list = ', '.join(sorted(available.keys())[:6])
            extra = f" +{len(available)-6} more" if len(available) > 6 else ""
            self.loaded_plat_label.config(
                text=f"No platforms fully loaded | {len(available)} available on disk: {avail_list}{extra}\n"
                     f"Use Gene Distribution Explorer for quick gene-only loading",
                foreground=AERO["warn"])
            self.gene_explorer_btn.config(state=tk.NORMAL)
            self.dist_analysis_btn.config(state=tk.NORMAL)
            if hasattr(self, 'label_enrich_btn'):
                self.label_enrich_btn.config(state=tk.NORMAL)
        else:
            plat_names = []
            total_samples = 0

            for plat_name, plat_df in self.gpl_datasets.items():
                samples = len(plat_df)
                total_samples += samples
                plat_names.append(f"{plat_name} ({samples:,} samples)")

            n_extra = len(available) - len(self.gpl_datasets)
            extra_text = f" | {n_extra} more available on disk" if n_extra > 0 else ""
            status_text = (f"✓ Loaded {len(self.gpl_datasets)} platform(s): "
                          + ", ".join(plat_names)
                          + f"\nTotal: {total_samples:,} samples{extra_text}")

            self.loaded_plat_label.config(text=status_text,
                                           foreground=AERO["green_dark"])

            self.gene_explorer_btn.config(state=tk.NORMAL)
            self.dist_analysis_btn.config(state=tk.NORMAL)
            if hasattr(self, 'label_enrich_btn'):
                self.label_enrich_btn.config(state=tk.NORMAL)

            # Platforms fully loaded → Step 2 is ready / complete
            try:
                if hasattr(self, '_step2_frame'):
                    self._set_step_status(self._step2_frame, self._step2_title, "done")
            except Exception:
                pass

        # Refresh platform load buttons after any status change
        try:
            if hasattr(self, '_plat_btn_frame'):
                self._refresh_platform_buttons()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    def _smart_load_gpl(self, gpl_id):
        """Smart GPL loader: tries local preset path -> downloaded file -> offers download."""
        try:
            self._smart_load_gpl_inner(gpl_id)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[CRASH] _smart_load_gpl({gpl_id}): {e}\n{tb}", flush=True)
            self.enqueue_log(f"[CRASH] Platform load failed: {e}\n{tb}")
            try:
                messagebox.showerror("Platform Load Error",
                    f"Failed to load {gpl_id}:\n\n{e}\n\nSee terminal for details.",
                    parent=self)
            except Exception:
                pass
            try:
                self.update_progress(value=0)
                self.status_label.config(text="Ready", foreground="gray")
            except Exception:
                pass

    def _smart_load_gpl_inner(self, gpl_id):
        """Smart GPL loader: checks downloaded files -> discovered files -> offers download."""
        self.status_label.config(text=f"Loading {gpl_id}...", foreground=AERO["accent"])
        self.update_idletasks()

        # Resolve the id once, honouring the precedence _discover_available_platforms
        # sets up (directories the user added explicitly come first).
        available = self._discover_available_platforms()
        discovered_path = available.get(gpl_id) or available.get(gpl_id.upper())
        from_user_dir = bool(discovered_path) and any(
            str(Path(discovered_path)).startswith(str(Path(ud)))
            for ud in getattr(self, '_user_data_dirs', []))

        # 1. Check already-downloaded files in data_dir/GPL_ID/ subdirectory.
        #    Skipped when the user pointed at their own copy of this platform:
        #    otherwise a stale matrix left under data_dir silently shadows the
        #    directory that was just added, and the sample count is wrong with
        #    no visible error.
        if not from_user_dir and getattr(self, 'data_dir', None):
            dl_dir = Path(self.data_dir) / gpl_id
            if dl_dir.exists():
                candidates = sorted(dl_dir.glob("*.csv.gz")) + sorted(dl_dir.glob("*.csv"))
                if candidates:
                    # Quick validation: check file has actual expression data
                    try:
                        test_df = pd.read_csv(candidates[0], compression="gzip" if str(candidates[0]).endswith('.gz') else None,
                                              nrows=5, low_memory=False)
                        # Check for numeric columns beyond GSM/series_id
                        num_cols = [c for c in test_df.columns
                                    if c not in ('GSM', 'gsm', 'series_id')
                                    and pd.api.types.is_numeric_dtype(test_df[c])]
                        has_values = False
                        for c in num_cols[:3]:
                            if test_df[c].notna().any():
                                has_values = True
                                break
                        if has_values:
                            self.enqueue_log(f"[{gpl_id}] Found valid downloaded file: {candidates[0].name}")
                            self._load_gpl_data(gpl_id, str(candidates[0]))
                            self.status_label.config(text="Ready", foreground="gray")
                            return
                        else:
                            self.enqueue_log(f"[{gpl_id}] Downloaded file has no expression data - will re-download")
                    except Exception as ve:
                        self.enqueue_log(f"[{gpl_id}] Downloaded file unreadable ({ve}) - will re-download")

        # 2. Fall back to the discovered file (user-added dirs, data_dir flat
        #    files, subdirectories) resolved at the top of this method.
        if discovered_path and Path(discovered_path).exists():
            self.enqueue_log(f"[{gpl_id}] Found via discovery: {discovered_path}")
            self._load_gpl_data(gpl_id, str(discovered_path))
            self.status_label.config(text="Ready", foreground="gray")
            return

        # 3. Also check data_dir itself for flat files (GPL570_expression.csv.gz)
        if hasattr(self, 'data_dir') and self.data_dir:
            import re as _re_flat
            data_path = Path(self.data_dir)
            if data_path.exists():
                for f in data_path.iterdir():
                    if f.is_file() and gpl_id.upper() in f.name.upper():
                        if f.suffix in ('.gz', '.csv') or f.name.endswith('.csv.gz'):
                            self.enqueue_log(f"[{gpl_id}] Found flat file: {f.name}")
                            self._load_gpl_data(gpl_id, str(f))
                            self.status_label.config(text="Ready", foreground="gray")
                            return

        # 4. Offer to auto-download or browse for file
        self.status_label.config(text="Ready", foreground="gray")
        choice = messagebox.askyesnocancel(
            f"{gpl_id} Not Found Locally",
            f"Pre-processed {gpl_id} data file not found on this machine.\n\n"
            f"Would you like to:\n"
            f"  YES = Auto-download {gpl_id} from GEO (requires GEOmetadb)\n"
            f"  NO = Browse for a local CSV/CSV.GZ file\n"
            f"  CANCEL = Cancel",
            parent=self
        )

        if choice is True:
            # Auto-download
            if not self.gds_conn:
                messagebox.showerror("Database Required",
                                     "GEOmetadb database is required for auto-download.\n"
                                     "Load GEOmetadb first, then try again.", parent=self)
                return
            self._trigger_auto_download(gpl_id)
        elif choice is False:
            # Browse for file
            filepath = filedialog.askopenfilename(
                title=f"Select {gpl_id} expression data file",
                filetypes=[("CSV/GZ files", "*.csv.gz *.csv"), ("All files", "*.*")],
                parent=self
            )
            if filepath:
                self._load_gpl_data(gpl_id, filepath)
                self.status_label.config(text="Ready", foreground="gray")

    def _trigger_auto_download(self, gpl_id):
        """Start auto-download for a GPL platform."""
        # Query platform info locally (same as species browser)
        try:
            info = self._query_gpl_info_local(gpl_id)
        except Exception as e:
            messagebox.showerror("Platform Not Found", str(e), parent=self)
            return

        try:
            from genevariate.core.gpl_downloader import GPLDownloader
            downloader = GPLDownloader(gds_conn=self.gds_conn, output_base_dir=self.data_dir)
            downloader.check_dependencies()
        except ImportError as e:
            messagebox.showerror("Missing Module",
                f"gpl_downloader.py not found:\n{e}\n\n"
                f"Place gpl_downloader.py in:\n"
                f"  genevariate/core/gpl_downloader.py",
                parent=self)
            return
        except Exception as e:
            messagebox.showerror("Download Error", str(e), parent=self)
            return

        if not messagebox.askyesno(f"Download {gpl_id}?",
            f"Platform: {info['title']}\nOrganism: {info['organism']}\n"
            f"Series: {info['total_series']}\n\nProceed with download?", parent=self):
            return

        self.enqueue_log(f"[GPL-DL] Starting {gpl_id}...")
        self.status_label.config(text=f"Downloading {gpl_id}...", foreground=AERO["accent"])

        def worker():
            try:
                result = downloader.run_with_info(info=info,
                    callback=lambda p, s, m: self.after(0, self._gpl_dl_progress, p, s, m))
                self.after(0, lambda: self._gpl_dl_done(result))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda _e=str(e), _tb=tb: self._gpl_dl_error(gpl_id, _e, _tb))

        threading.Thread(target=worker, daemon=True).start()

    # How the picker groups platforms, in display order. Grouping by species
    # alone told the user nothing about what a platform measures, yet the
    # modality is what decides how the values may be analysed: sequencing
    # counts, array intensities and methylation betas are not interchangeable.
    _TECH_GROUPS = [
        ("microarray",       "Microarray - expression"),
        ("bulk-rna-seq",     "Bulk RNA-seq - sequencing counts"),
        ("single-cell",      "Single-cell RNA-seq"),
        ("methylation",      "Methylation array - beta values (not analysable)"),
        ("sequencing-other", "Sequencing - peak-based (not analysable)"),
        ("custom",           "Custom / imported datasets"),
        ("unknown",          "Unclassified"),
    ]

    def register_platform_frame(self, name, df):
        """Store a platform frame and index its gene columns together.

        Every window resolves a gene symbol through ``gpl_gene_mappings``, so
        a frame added to ``gpl_datasets`` without an index is a platform with
        zero genes: it drops out of the shared gene space and every statistic
        derived from that space reads zero. Registration goes through here so
        the frame and its index cannot separate.
        """
        if getattr(self, "gpl_datasets", None) is None:
            self.gpl_datasets = {}
        if getattr(self, "gpl_gene_mappings", None) is None:
            self.gpl_gene_mappings = {}
        gene_map, _ = index_gene_columns(df)
        self.gpl_datasets[name] = df
        self.gpl_gene_mappings[name] = gene_map
        return gene_map

    def platform_species(self, plat):
        """Common-name species for *plat*, or ``'unknown'``.

        GEOmetadb knows the organism of every GPL, which is far more than the
        short built-in table covers, so ask the facts first and keep the table
        only as the offline fallback. Single-cell platforms carry their
        organism on the AnnData instead of a GPL id.
        """
        sc = (getattr(self, "scrna_datasets", None) or {}).get(str(plat))
        if sc is not None:
            for obj in (sc.get("cells"), sc.get("pseudobulk")):
                try:
                    org = str(obj.uns.get("source", {}).get("organism", "") or "")
                except Exception:
                    org = ""
                if org:
                    return _common_species_name(org)
            return "unknown"
        base = str(plat).split('_')[0]
        org = ""
        if re.match(r"^GPL\d+$", base, re.I):
            org = self._platform_facts(base).get("organism") or ""
        return _common_species_name(org) if org else GPL_SPECIES.get(base, "unknown")

    def _platform_facts(self, plat):
        """Return {'title', 'organism', 'category'} for a platform id.

        Answered from GEOmetadb, which is the same source the downloader
        classifies on, so the picker and the download routing can never
        disagree about what a platform is. Cached because the picker is
        rebuilt on every load.
        """
        cache = getattr(self, "_plat_facts_cache", None)
        if cache is None:
            cache = self._plat_facts_cache = {}
        if plat in cache:
            return cache[plat]

        from genevariate.core.gpl_downloader import classify_technology
        facts = {"title": "", "organism": "", "category": "unknown"}

        if str(plat) in (getattr(self, "scrna_datasets", None) or {}):
            # A pseudobulk platform knows what it is without asking GEO, and
            # it has no GPL id to ask with.
            facts["category"] = "single-cell"
        elif not re.match(r"^GPL\d+$", str(plat), re.I):
            # CELLxGENE exports, ARCHS4 pulls and user imports are not GEO
            # platforms and will never resolve against the gpl table.
            facts["category"] = "custom"
        elif getattr(self, "gds_conn", None) is not None:
            try:
                row = self.gds_conn.execute(
                    "SELECT title, organism, technology FROM gpl "
                    "WHERE UPPER(gpl) = ?", (str(plat).upper(),)).fetchone()
            except Exception:
                row = None
            if row:
                facts["title"] = str(row[0] or "").strip()
                facts["organism"] = str(row[1] or "").strip()
                facts["category"] = classify_technology(row[2] or "", row[0] or "")

        if not facts["organism"]:
            facts["organism"] = GPL_SPECIES.get(plat, "").title()
        # Multi-species designs list every organism tab-separated.
        facts["organism"] = re.split(r"[;\t]", facts["organism"])[0].strip()
        cache[plat] = facts
        return facts

    def platform_measurement_label(self, plat):
        """Axis label for the quantity *plat* measures.

        A pseudobulk platform is named after the aggregation it was built with,
        because a sum of cells is a library count while a mean of cells is an
        average of per-cell rates, and the two do not answer the same question
        about a region.

        The aggregation is not the whole story though: a summed library that
        has since been through TMM -> CPM -> log2 is on the log2 scale, and an
        axis that still calls it "sum of cell expression" invites a reader to
        compare it with raw counts. So the normalization, when one ran, is
        what names the axis, and the aggregation is kept in the qualifier.
        """
        from genevariate.core.gpl_downloader import measurement_label
        base = measurement_label(self._platform_facts(plat)["category"])
        sc = (getattr(self, "scrna_datasets", None) or {}).get(str(plat))
        if sc is not None:
            try:
                info = sc["pseudobulk"].uns["pseudobulk"]
            except Exception:
                info = {}
            agg = info.get("agg")
            if agg:
                if info.get("normalization_skipped") or not info.get("normalization"):
                    return f"{agg} of cell expression (pseudobulk, not normalized)"
                if "log2" in str(info.get("normalization", "")).lower():
                    return f"log2 CPM (pseudobulk, {agg})"
                return f"{info['normalization']} (pseudobulk, {agg})"
        return base

    def _refresh_platform_buttons(self):
        """Build the platform pills, grouped by what each platform measures."""
        for w in self._plat_btn_frame.winfo_children():
            w.destroy()

        available = self._discover_available_platforms()
        loaded = set(self.gpl_datasets.keys())

        if not available and not loaded:
            ttk.Label(self._plat_btn_frame,
                      text="No platforms found. Use 'Download Platform' or 'Add Custom Platform' to get started.",
                      font=('Segoe UI', 10), foreground='#888').pack(pady=15)
            return

        groups = {}
        for plat in sorted(set(list(available.keys()) + list(loaded))):
            facts = self._platform_facts(plat)
            groups.setdefault(facts["category"], []).append(
                (plat, plat in loaded, facts))

        names = dict(self._TECH_GROUPS)
        order = [k for k, _ in self._TECH_GROUPS]
        for cat in sorted(groups, key=lambda c: (order.index(c) if c in order
                                                 else len(order), c)):
            items = groups[cat]
            gframe = labelframe(
                self._plat_btn_frame,
                text=f"{names.get(cat, cat.title())}  ({len(items)})")
            gframe.pack(fill=tk.X, padx=5, pady=3)
            # A data directory can hold any number of platforms, so the pills
            # have to wrap onto further rows instead of running off the edge.
            flow = FlowFrame(gframe, spacing=10)
            flow.pack(fill=tk.X, padx=5, pady=4)
            for plat, is_loaded, facts in items:
                self._platform_pill(flow, plat, is_loaded, facts)

    def _platform_pill(self, flow, plat, is_loaded, facts):
        """One platform capsule: accession, instrument, organism, status."""
        bf = flow.add(ttk.Frame(flow))

        # The pill styles draw a fixed-height capsule, so a two-line label
        # has to squeeze both lines against the border. Keep the platform
        # name on the pill and hang the detail underneath as captions.
        if is_loaded:
            n = len(self.gpl_datasets[plat])
            btn = ttk.Button(bf, text=f"✓ {plat}", width=16,
                             style="Secondary.TButton", state=tk.DISABLED)
            caption, fg = f"{n:,} samples", AERO["muted"]
        else:
            btn = ttk.Button(bf, text=plat, width=16,
                             command=lambda p=plat: self._smart_load_gpl(p),
                             style="Primary.TButton")
            caption, fg = "click to load", AERO["accent_dark"]
        btn.pack()

        # An accession on its own identifies nothing to a reader, so name the
        # instrument and the organism; the untruncated title is on hover.
        desc = _shorten_platform_title(facts.get("title"))
        if desc:
            tk.Label(bf, text=desc, bg=AERO["bg_top"], fg=AERO["text"],
                     font=('Segoe UI', 8), wraplength=150,
                     justify=tk.CENTER).pack(pady=(2, 0))
        org = facts.get("organism") or ""
        if org and org.lower() != desc.lower():
            tk.Label(bf, text=org, bg=AERO["bg_top"], fg=AERO["muted"],
                     font=('Segoe UI', 7, 'italic')).pack()
        tk.Label(bf, text=caption, bg=AERO["bg_top"], fg=fg,
                 font=('Segoe UI', 8)).pack(pady=(1, 0))

        tip = plat
        if facts.get("title"):
            tip += f"\n{facts['title']}"
        if org:
            tip += f"\nOrganism: {org}"
        tip += f"\nMeasures: {dict(self._TECH_GROUPS).get(facts.get('category'), 'unknown')}"
        self._set_tooltip(btn, tip)

        # Badge for CELLxGENE-derived platforms: labels already exist
        # (Cell Ontology) so no LLM extraction is needed downstream.
        if plat.startswith("CellxGene_"):
            tk.Label(bf,
                     text="✓ Pre-classified\n(CELLxGENE Cell Ontology)",
                     bg=AERO["accent_light"], fg=AERO["accent_dark"],
                     font=('Segoe UI', 7, 'bold'),
                     relief=tk.SOLID, borderwidth=1,
                     justify=tk.CENTER,
                     padx=2, pady=1).pack(fill=tk.X, pady=(2, 0))

    def _load_custom_gpl_data(self):
        """Load a custom GPL dataset with validation."""
        instructions = (
            "Custom Platform Requirements:\n\n"
            "Your file should be a CSV or CSV.GZ with:\n"
            "- One column named 'GSM' or 'gsm' (sample IDs)\n"
            "- Numeric columns representing gene expression values\n"
            "- Column names as gene symbols or probe IDs\n\n"
            "All samples in the file will be loaded for analysis."
        )
        
        messagebox.showinfo("Custom Platform Format", instructions, parent=self)
        
        filepath = filedialog.askopenfilename(
            title="Select your preprocessed gene expression file",
            filetypes=[("Compressed CSV", "*.csv.gz"), ("CSV files", "*.csv"), ("All files", "*.*")],
            parent=self
        )
        
        if not filepath:
            return
        
        dataset_name = simpledialog.askstring(
            "Dataset Name", 
            "Enter a unique name for this platform\n(e.g., 'MyStudy_GPL12345'):", 
            parent=self
        )
        
        if not dataset_name or not dataset_name.strip():
            messagebox.showwarning("Name Required", "You must provide a name for the dataset.", parent=self)
            return
        
        dataset_name = dataset_name.strip()
        
        if dataset_name in self.gpl_datasets:
            messagebox.showerror("Name Exists", f"Platform '{dataset_name}' is already loaded.\nPlease choose a different name.", parent=self)
            return
        
        self._load_gpl_data(dataset_name, filepath)
    
    @staticmethod
    def _pick_platform_file(candidates):
        """Choose which file in a GPL directory represents the platform.

        A downloaded platform now leaves two matrices side by side: the raw
        one the download wrote and, once the separate normalization step has
        run, the normalized one. Prefer normalized; fall back to raw, then to
        the largest file so directories from older versions still resolve.
        """
        from genevariate.core.gpl_downloader import (
            NORMALIZED_SUFFIX, RAW_SUFFIX)

        for suffix in (NORMALIZED_SUFFIX, RAW_SUFFIX):
            hit = [f for f in candidates if f.name.lower().endswith(suffix)]
            if hit:
                return max(hit, key=lambda f: f.stat().st_size)
        return max(candidates, key=lambda f: f.stat().st_size)

    def _find_raw_platforms(self):
        """Return {gpl_id: (raw_path, normalized_path_or_None)} on disk."""
        from genevariate.core.gpl_downloader import (
            normalized_csv_path, raw_csv_path)

        found = {}
        roots = {Path(self.data_dir)} if getattr(self, 'data_dir', None) else set()
        roots.update(Path(d) for d in getattr(self, '_user_data_dirs', []))
        roots.add(Path(os.path.dirname(os.path.abspath(__file__))).parent)

        for root in roots:
            if not root.exists():
                continue
            try:
                subdirs = [d for d in root.iterdir()
                           if d.is_dir() and d.name.upper().startswith('GPL')]
            except PermissionError:
                continue
            for subdir in subdirs:
                gpl_id = subdir.name.upper()
                raw = raw_csv_path(gpl_id, str(subdir))
                if gpl_id in found or not os.path.exists(raw):
                    continue
                norm = normalized_csv_path(gpl_id, str(subdir))
                found[gpl_id] = (raw, norm if os.path.exists(norm) else None)
        return found

    def _open_normalize_platform_window(self):
        """Run the standalone normalization step on a downloaded raw matrix."""
        from genevariate.core.gpl_downloader import normalize_platform

        platforms = self._find_raw_platforms()
        if not platforms:
            messagebox.showinfo(
                "No Raw Platforms",
                "No raw platform matrices were found.\n\n"
                "Use 'Download Platform' first - it saves the raw values, "
                "and this step rescales them.",
                parent=self)
            return

        win = tk.Toplevel(self)
        style_window(win)
        win.title("Normalize Platform")
        win.geometry("640x460")
        win.transient(self)

        ttk.Label(win, text="Normalize a downloaded platform",
                  font=('Segoe UI', 12, 'bold')).pack(anchor=tk.W,
                                                      padx=16, pady=(14, 2))
        ttk.Label(
            win, justify=tk.LEFT, foreground=AERO["muted"],
            font=('Segoe UI', 9),
            text="RNA-seq counts: TMM effective library size, CPM, "
                 "log2(CPM+1).\nArray intensities: a single matrix-wide log2 "
                 "decision, then NaN-aware quantile normalization.\nThe raw "
                 "file is never modified, so this can be re-run at any "
                 "time.").pack(anchor=tk.W, padx=16, pady=(0, 10))

        list_frame = labelframe(win, text="Available raw matrices")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)
        tree = ttk.Treeview(list_frame, columns=("gpl", "status"),
                            show="headings", height=8)
        tree.heading("gpl", text="Platform")
        tree.heading("status", text="Normalized")
        tree.column("gpl", width=160, anchor='center')
        tree.column("status", width=380, anchor='center')
        tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for gpl_id, (_, norm) in sorted(platforms.items()):
            tree.insert("", tk.END, iid=gpl_id, values=(
                gpl_id,
                "yes - re-running overwrites it" if norm else "not yet"))

        status = ttk.Label(win, text="Select a platform, then Normalize.",
                           foreground=AERO["muted"], font=('Segoe UI', 9))
        status.pack(anchor=tk.W, padx=16, pady=(6, 0))

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=16, pady=12)

        def _run():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Select a Platform",
                                    "Pick a platform to normalize.", parent=win)
                return
            gpl_id = sel[0]
            out_dir = os.path.dirname(platforms[gpl_id][0])
            go.config(state=tk.DISABLED)

            def _cb(pct, stage, msg):
                self.after(0, lambda: status.config(
                    text=msg, foreground=AERO["accent_dark"]))
                self.enqueue_log(f"[Normalize {gpl_id}] {msg}")

            def _work():
                try:
                    res = normalize_platform(gpl_id, out_dir, callback=_cb)
                except Exception as exc:
                    # `exc` is deleted when the except block ends, so the
                    # message has to be captured before the callback is queued.
                    self.after(0, lambda msg=str(exc): status.config(
                        text=f"Failed: {msg}", foreground=AERO["danger"]))
                    self.after(0, lambda: go.config(state=tk.NORMAL))
                    self.enqueue_log(f"[Normalize {gpl_id}] FAILED: {exc}")
                    return
                n_samples, n_cols = res["shape"]
                # Which scale correction ran is the one thing a user has to
                # know to interpret the numbers, so it goes in the summary
                # rather than only in the log.
                if res.get("counts"):
                    how = (f"TMM + CPM + log2, "
                           f"{res.get('genes_dropped', 0):,} low-count genes dropped")
                else:
                    how = f"log2 {'applied' if res['applied_log2'] else 'not needed'}"
                self.after(0, lambda: status.config(
                    text=f"Done - {n_samples:,} samples x {n_cols - 2:,} genes ({how})",
                    foreground=AERO["green_dark"]))
                self.after(0, lambda: tree.item(
                    gpl_id, values=(gpl_id, "yes - re-running overwrites it")))
                self.after(0, lambda: go.config(state=tk.NORMAL))
                self.after(0, self._refresh_platform_buttons)

            threading.Thread(target=_work, daemon=True).start()

        go = ttk.Button(btns, text="Normalize", command=_run,
                        style="Action.TButton")
        go.pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy,
                   style="Ghost.TButton").pack(side=tk.RIGHT)

    def _discover_available_platforms(self):
        """Find available GPL platform files.

        The same GPL id can exist in more than one of the scanned locations
        (a stale ``results/GPL570`` next to a freshly downloaded one, say).
        The first directory that supplies an id wins, so the scan order is
        part of the contract: directories the user added explicitly are
        searched first, then the program's own locations, and the loose
        ``results*``/``output*`` matches last. Ordering used to come from a
        ``set``, which made the winner depend on hash order and let a stale
        copy silently shadow the data the user had just pointed at.

        Scans, in precedence order:
          1. User-added directories
          2. data_dir (genevariate/data/ where GPL Downloader saves)
          3. Project root/data/ and current working directory/data/
          4. Project root (genevariate/) and program directory (genevariate/gui/)
          5. Current working directory
          6. Common output directory names (AI_agent*, results*, output*, gpl_data*)
        Returns dict of {gpl_id: file_path}.
        """
        available = dict(self.gpl_available_files)  # start with cached

        _prog_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _proj_root = _prog_dir.parent  # genevariate/
        _cwd = Path(os.getcwd())

        ordered = []
        # Explicit user intent outranks anything the program found on its own.
        ordered.extend(Path(ud) for ud in getattr(self, '_user_data_dirs', []))
        if getattr(self, 'data_dir', None):
            ordered.append(Path(self.data_dir))
        ordered.extend([_proj_root / 'data', _cwd / 'data',
                        _proj_root, _prog_dir, _cwd])
        for parent in (_proj_root, _prog_dir, _cwd):
            for pattern in ('AI_agent*', 'results*', 'output*', 'gpl_data*'):
                import glob as _glob_mod
                ordered.extend(Path(m) for m in sorted(_glob_mod.glob(str(parent / pattern)))
                               if os.path.isdir(m))

        dirs_to_scan = []
        seen = set()
        for p in ordered:
            key = str(p)
            if key not in seen:
                seen.add(key)
                dirs_to_scan.append(p)

        for base_path in dirs_to_scan:
            if not base_path.exists():
                continue
            try:
                for subdir in base_path.iterdir():
                    if subdir.is_dir() and subdir.name.upper().startswith('GPL'):
                        gpl_id = subdir.name.upper()
                        if gpl_id not in available:
                            candidates = list(subdir.glob('*.csv.gz')) + list(subdir.glob('*.csv'))
                            if candidates:
                                best = self._pick_platform_file(candidates)
                                available[gpl_id] = str(best)
                    elif subdir.is_file():
                        fname = subdir.name.upper()
                        import re as _re
                        m = _re.search(r'(GPL\d+)', fname)
                        if m and (fname.endswith('.CSV.GZ') or fname.endswith('.CSV')):
                            gpl_id = m.group(1)
                            if gpl_id not in available:
                                available[gpl_id] = str(subdir)
            except PermissionError:
                pass

        # Log discovery summary on first scan
        if not self.gpl_available_files and available:
            try:
                self.enqueue_log(
                    f"[Discovery] Scanned {', '.join(str(p) for p in dirs_to_scan)} - "
                    f"found {len(available)} platform(s): "
                    f"{', '.join(sorted(available.keys())[:10])}"
                    f"{f' +{len(available)-10} more' if len(available) > 10 else ''}")
            except: pass

        self.gpl_available_files = available
        return available

    def _add_data_directory(self):
        """Let user pick a directory containing GPL platform files."""
        if getattr(self, '_dialog_active', False):
            return
        self._dialog_active = True
        try:
            d = filedialog.askdirectory(
                title="Select Directory Containing GPL Platform Files (.csv.gz)",
                parent=self)
        except tk.TclError:
            return
        finally:
            self._dialog_active = False
        if not d:
            return
        if d not in self._user_data_dirs:
            self._user_data_dirs.append(d)
        self.gpl_available_files.clear()  # force rescan
        available = self._discover_available_platforms()
        self._update_platform_status()
        self.enqueue_log(f"[DataDir] Added: {d} - found {len(available)} platform(s) total")
        return available

    def _add_data_dir_and_refresh(self, popup):
        """Add data directory and refresh the Gene Explorer platform list."""
        available = self._add_data_directory()
        if available is None:
            return
        # Refresh the platform checkboxes in the popup
        if popup and popup.winfo_exists():
            # Find and rebuild plat_check_frame
            try:
                gpls_loaded = sorted(self.gpl_datasets.keys())
                gpls_available = sorted(k for k in available.keys() if k not in self.gpl_datasets)

                # Clear existing checkboxes - find the frame
                for widget in popup.winfo_children():
                    self._rebuild_plat_checks(popup, widget, gpls_loaded, gpls_available)
            except Exception as e:
                self.enqueue_log(f"[DataDir] Refresh warning: {e}")

    def _rebuild_plat_checks(self, popup, widget, gpls_loaded, gpls_available):
        """Recursively find and rebuild platform checkboxes in the popup."""
        for child in widget.winfo_children():
            if isinstance(child, tk.LabelFrame) and 'Select Platforms' in str(child.cget('text')):
                # Found the platform frame - rebuild checkboxes
                for sub in child.winfo_children():
                    if isinstance(sub, ttk.Frame):
                        # Check if this is the plat_check_frame (has Checkbutton children)
                        has_checks = any(isinstance(w, ttk.Checkbutton) for w in sub.winfo_children())
                        if has_checks:
                            for w in sub.winfo_children():
                                w.destroy()
                            # Rebuild
                            col_idx = 0; row_idx = 0
                            old_sel = {p: v.get() for p, v in popup.gpl_selection_vars.items()}
                            popup.gpl_selection_vars = {}

                            for plat in gpls_loaded:
                                var = tk.BooleanVar(master=popup, value=old_sel.get(plat, False))
                                n = len(self.gpl_datasets[plat])
                                cb = ttk.Checkbutton(sub, text=f"{plat} ({n:,} samples)", variable=var)
                                cb.grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=2)
                                popup.gpl_selection_vars[plat] = var
                                col_idx += 1
                                if col_idx >= 3: col_idx = 0; row_idx += 1

                            if gpls_available:
                                row_idx += 1
                                ttk.Label(sub, text="── Quick Gene Load (not fully loaded) ──",
                                          font=('Segoe UI', 8, 'italic'), foreground='#888'
                                          ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(4, 2))
                                row_idx += 1; col_idx = 0
                                for plat in gpls_available:
                                    var = tk.BooleanVar(master=popup, value=old_sel.get(plat, False))
                                    cb = ttk.Checkbutton(sub, text=f"{plat} (gene-only load)", variable=var)
                                    cb.grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=2)
                                    popup.gpl_selection_vars[plat] = var
                                    col_idx += 1
                                    if col_idx >= 3: col_idx = 0; row_idx += 1

                            self.enqueue_log(f"[DataDir] Refreshed platform list: "
                                             f"{len(gpls_loaded)} loaded + {len(gpls_available)} available")
                            return
            self._rebuild_plat_checks(popup, child, gpls_loaded, gpls_available)

    def _quick_load_genes(self, gpl_id, gene_symbols, file_path=None):
        """Load ONLY specific gene columns from a platform file.
        Much faster than loading the entire platform - reads header first,
        then loads only GSM + matching gene columns.

        Returns True if genes were found and cached, False otherwise.
        """
        if file_path is None:
            available = self._discover_available_platforms()
            file_path = available.get(gpl_id)
        if not file_path or not Path(file_path).exists():
            self.enqueue_log(f"[QuickLoad] {gpl_id}: no data file found")
            return False

        self.enqueue_log(f"[QuickLoad] {gpl_id}: scanning for {len(gene_symbols)} gene(s)...")
        is_gz = str(file_path).endswith('.gz')

        try:
            # Step 1: Read ONLY the header line to get column names
            if is_gz:
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                    header_line = f.readline().strip()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    header_line = f.readline().strip()

            all_cols = [c.strip().strip('"') for c in header_line.split(',')]

            # Step 2: Find GSM column
            gsm_col = None
            for c in all_cols:
                if c.upper() in ('GSM', 'SAMPLE', 'SAMPLE_ID', 'GEO_ACCESSION'):
                    gsm_col = c
                    break
            if gsm_col is None:
                # Check if first column looks like GSMs
                gsm_col = all_cols[0]

            # Step 3: Find gene columns (case-insensitive match)
            gene_upper = {g.upper(): g for g in gene_symbols}
            cols_upper = {c.upper(): c for c in all_cols}

            matched_cols = [gsm_col]  # always include GSM
            gene_map = {}
            for g_upper, g_orig in gene_upper.items():
                if g_upper in cols_upper:
                    real_col = cols_upper[g_upper]
                    matched_cols.append(real_col)
                    gene_map[g_orig] = real_col

            if len(matched_cols) <= 1:
                self.enqueue_log(f"[QuickLoad] {gpl_id}: none of {list(gene_symbols)} found in {len(all_cols):,} columns")
                return False

            self.enqueue_log(f"[QuickLoad] {gpl_id}: found {len(matched_cols)-1} gene(s) in "
                             f"{len(all_cols):,} columns - loading subset...")

            # Step 4: Read only the matched columns (MUCH faster)
            df = pd.read_csv(file_path, compression='gzip' if is_gz else 'infer',
                             usecols=matched_cols, low_memory=False)

            # Normalize GSM column
            if gsm_col != 'GSM':
                df = df.rename(columns={gsm_col: 'GSM'})
            df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()

            # Convert gene columns to numeric
            for col in matched_cols:
                if col == gsm_col:
                    continue
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Step 5: Merge into existing cache (don't overwrite previous genes)
            existing = self.gpl_gene_cache.get(gpl_id)
            if existing is not None and not existing.empty:
                # Add new columns to existing cache
                new_cols = [c for c in df.columns if c != 'GSM' and c not in existing.columns]
                if new_cols:
                    df_new = df[['GSM'] + new_cols]
                    existing = existing.merge(df_new, on='GSM', how='outer')
                    self.gpl_gene_cache[gpl_id] = existing
                    gene_map_existing = self.gpl_gene_mappings.get(f"_cache_{gpl_id}", {})
                    gene_map_existing.update(gene_map)
                    self.gpl_gene_mappings[f"_cache_{gpl_id}"] = gene_map_existing
            else:
                self.gpl_gene_cache[gpl_id] = df
                self.gpl_gene_mappings[f"_cache_{gpl_id}"] = gene_map

            n_samples = len(df)
            n_genes = len(matched_cols) - 1
            self.enqueue_log(f"[QuickLoad] OK {gpl_id}: {n_samples:,} samples × {n_genes} gene(s) loaded")
            return True

        except Exception as e:
            self.enqueue_log(f"[QuickLoad] ERROR {gpl_id}: {e}")
            import traceback
            self.enqueue_log(traceback.format_exc())
            return False

    def _read_gene_columns_from_csv(self, file_path):
        """Return the list of gene-like column names from a GPL CSV header.

        Reads ONLY the header row (no data) and removes known metadata columns
        defined in genevariate.config.METADATA_EXCLUSIONS. Supports .csv and .csv.gz.
        """
        if not file_path or not Path(file_path).exists():
            return []
        try:
            from genevariate.config import METADATA_EXCLUSIONS
        except Exception:
            METADATA_EXCLUSIONS = {'GSM', 'series_id', 'Series', 'series'}

        is_gz = str(file_path).lower().endswith('.gz')
        try:
            import gzip as _gz
            opener = _gz.open if is_gz else open
            with opener(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                header_line = f.readline().strip()
        except Exception as exc:
            self.enqueue_log(f"[GeneList] could not read header of {file_path}: {exc}")
            return []

        all_cols = [c.strip().strip('"') for c in header_line.split(',')]
        exclude_upper = {s.upper() for s in METADATA_EXCLUSIONS}
        genes = [c for c in all_cols if c.upper() not in exclude_upper]
        return genes

    def _show_gene_list_for_selected_platforms(self, popup):
        """Open a window listing genes for every currently-selected platform.

        Also shows union / intersection tabs when >1 platform is selected.
        Includes a search/filter box and 'Save CSV' / 'Copy' actions.
        """
        try:
            self.enqueue_log("[GeneList] 'Show Gene List' clicked")
        except Exception:
            pass

        # Visible feedback on the popup itself so the user sees something
        # happen immediately, even before the window is built.
        feedback_lbl = getattr(popup, '_gene_list_feedback', None)
        if feedback_lbl is None or not feedback_lbl.winfo_exists():
            try:
                feedback_lbl = tk.Label(
                    popup,
                    text="",
                    font=('Segoe UI', 10, 'bold'),
                    fg='#00796B',
                    bg=popup.cget('bg') if hasattr(popup, 'cget') else '#F0F0F0',
                )
                feedback_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 4))
                popup._gene_list_feedback = feedback_lbl
            except Exception:
                feedback_lbl = None

        sel_vars = getattr(popup, 'gpl_selection_vars', {})
        selected = [plat for plat, var in sel_vars.items() if var.get()]
        if not selected:
            try:
                self.enqueue_log("[GeneList] No platforms selected.")
            except Exception:
                pass
            messagebox.showinfo(
                "No Platforms Selected",
                "Please tick at least one platform checkbox above, "
                "then click 'Show Gene List' again.",
                parent=popup
            )
            return

        if feedback_lbl is not None:
            try:
                self._animator.spinner(
                    feedback_lbl,
                    f"Collecting genes from {len(selected)} platform(s)...",
                    color='#00796B',
                )
            except Exception:
                try:
                    feedback_lbl.config(
                        text=f"Collecting genes from {len(selected)} platform(s)...")
                except Exception:
                    pass

        available = self._discover_available_platforms()

        # Collect gene lists per platform
        gene_lists = {}
        try:
            for plat in selected:
                # Prefer an already-loaded dataset (canonical column order)
                if plat in self.gpl_datasets:
                    try:
                        from genevariate.config import METADATA_EXCLUSIONS
                    except Exception:
                        METADATA_EXCLUSIONS = {'GSM'}
                    exclude_upper = {s.upper() for s in METADATA_EXCLUSIONS}
                    cols = list(self.gpl_datasets[plat].columns)
                    gene_lists[plat] = [c for c in cols if c.upper() not in exclude_upper]
                else:
                    fpath = available.get(plat)
                    gene_lists[plat] = self._read_gene_columns_from_csv(fpath)
        except Exception as exc:
            try:
                self._animator.stop(feedback_lbl) if feedback_lbl is not None else None
            except Exception:
                pass
            try:
                self.enqueue_log(f"[GeneList] Failed to collect gene columns: {exc}")
            except Exception:
                pass
            messagebox.showerror(
                "Gene List Error",
                f"Could not collect gene lists:\n{exc}",
                parent=popup,
            )
            return

        try:
            summary = ", ".join(f"{p}={len(g):,}" for p, g in gene_lists.items())
            self.enqueue_log(f"[GeneList] Collected: {summary}")
        except Exception:
            pass

        # If every platform returned zero genes, tell the user plainly rather
        # than opening an empty window they might not even see.
        if not any(gene_lists.values()):
            if feedback_lbl is not None:
                try:
                    self._animator.stop(feedback_lbl)
                    feedback_lbl.config(
                        text="No gene columns could be extracted from the selected platform(s).",
                        fg='#C62828',
                    )
                except Exception:
                    pass
            try:
                self.enqueue_log(
                    "[GeneList] No gene columns found. "
                    "Is the CSV loaded / available on disk?"
                )
            except Exception:
                pass
            messagebox.showwarning(
                "No Genes Found",
                "No gene columns were found for the selected platform(s).\n\n"
                "The platform may not be loaded yet, or the CSV file could not "
                "be read. Check the Activity Log for details.",
                parent=popup,
            )
            return

        # Build the window
        win = tk.Toplevel(popup)
        style_window(win)
        win.title(f"Gene List - {len(selected)} platform(s)")
        win.geometry("780x640")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"780x640+{(_sw-780)//2}+{(_sh-640)//2}")
        except Exception:
            pass
        try:
            win.transient(popup)
        except Exception:
            pass
        # Force the window to the front so the user actually sees it -
        # Toplevel windows sometimes open behind the parent popup.
        try:
            win.deiconify()
            win.lift()
            win.focus_force()
            win.attributes('-topmost', True)
            # Release topmost shortly after so it doesn't cover other dialogs
            win.after(350, lambda w=win: (w.attributes('-topmost', False)
                                          if w.winfo_exists() else None))
        except Exception:
            pass

        # Stop the popup spinner now that the window is open
        if feedback_lbl is not None:
            try:
                self._animator.stop(feedback_lbl)
                feedback_lbl.config(
                    text=f"Gene list window opened for {len(selected)} platform(s).",
                    fg='#1B5E20',
                )
                self._animator.flash(feedback_lbl, '#2E7D32', revert_to='#1B5E20')
            except Exception:
                pass

        # Compute per-gene coverage across the selected platforms (used in
        # every tab so the user sees gene context, not just names).
        # Coverage = how many of the selected platforms contain each gene.
        upper_sets = {p: {g.upper() for g in gs}
                      for p, gs in gene_lists.items()}
        first_seen = {}  # UPPER -> original-case display symbol
        for plat in selected:
            for g in gene_lists.get(plat, []):
                first_seen.setdefault(g.upper(), g)

        all_upper = set().union(*upper_sets.values()) if upper_sets else set()
        gene_platforms = {}   # UPPER -> sorted list of platform names containing it
        for u in all_upper:
            gene_platforms[u] = sorted(
                [p for p, s in upper_sets.items() if u in s]
            )

        n_total = len(selected)

        def _row(u, plat_list_override=None):
            """Build one display row for gene UPPER key u."""
            symbol = first_seen[u]
            plats = plat_list_override if plat_list_override is not None else gene_platforms[u]
            return {
                'symbol':   symbol,
                'coverage': f"{len(plats)}/{n_total}",
                'n_cov':    len(plats),
                'platforms': ", ".join(plats),
            }

        # Search/filter bar (applies to whichever tab is active)
        search_frame = ttk.Frame(win, padding=8)
        search_frame.pack(fill=tk.X)
        ttk.Label(search_frame, text="Filter:", font=('Segoe UI', 10, 'bold')
                  ).pack(side=tk.LEFT, padx=(0, 6))
        search_var = tk.StringVar(master=win)
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=4)
        status_lbl = ttk.Label(search_frame, text="", foreground='#555')
        status_lbl.pack(side=tk.LEFT, padx=10)

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # --- per-tab state so filter can update the right treeview ---
        # tabs[label] = {'tree': Treeview, 'rows': [row_dict, ...]}
        tabs = {}

        # Color tags for coverage (applied to Union tab rows)
        def _coverage_color(n_cov):
            if n_total <= 1:
                return '#263238'
            frac = n_cov / n_total
            if frac >= 1.0:
                return '#1B5E20'     # all platforms - dark green
            if frac >= 0.67:
                return '#2E7D32'     # most - green
            if frac >= 0.34:
                return '#EF6C00'     # some - orange
            return '#B71C1C'         # few - red

        def _make_tab(label, rows, sort_key=None):
            """Create a Treeview tab.
            rows: list of dicts (from _row).
            sort_key: optional fn(row)->key for initial ordering; default = name.
            """
            frame = ttk.Frame(nb, padding=6)
            nb.add(frame, text=label)

            cols = ("idx", "gene", "coverage", "platforms")
            tree = ttk.Treeview(frame, columns=cols, show="headings",
                                selectmode="extended", height=20)
            tree.heading("idx", text="#")
            tree.heading("gene", text="Gene symbol")
            tree.heading("coverage", text="Coverage")
            tree.heading("platforms", text="Platform(s)")
            tree.column("idx", width=60, anchor=tk.CENTER, stretch=False)
            tree.column("gene", width=160, anchor=tk.W, stretch=False)
            tree.column("coverage", width=90, anchor=tk.CENTER, stretch=False)
            tree.column("platforms", width=360, anchor=tk.W, stretch=True)

            vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)

            # Styling: bigger row height + visible row borders, like the
            # GPL list. Apply the same style to every tab so they match.
            try:
                _gst = ttk.Style()
                _gst.configure("GeneList.Treeview",
                               rowheight=24, borderwidth=1, relief="solid")
                _gst.configure("GeneList.Treeview.Heading",
                               font=('Segoe UI', 10, 'bold'))
                tree.configure(style="GeneList.Treeview")
            except Exception:
                pass

            # Zebra-stripe backgrounds so each row is clearly separated
            tree.tag_configure("row_even", background='#FFFFFF')
            tree.tag_configure("row_odd",  background='#F0F4F8')

            # Configure coverage color tags per tree
            for n_cov in range(0, n_total + 1):
                tag = f"cov_{n_cov}"
                tree.tag_configure(tag, foreground=_coverage_color(n_cov))

            ordered = sorted(rows, key=sort_key) if sort_key else rows
            tabs[label] = {'tree': tree, 'rows': ordered}

        # ── Per-platform tabs (keep CSV column order; coverage shows how
        #    widespread each gene is across the other selected platforms) ──
        for plat in selected:
            genes_in_order = gene_lists.get(plat, [])
            plat_rows = []
            for i, g in enumerate(genes_in_order, start=1):
                u = g.upper()
                if u not in first_seen:
                    # Paranoia: platform-local gene that somehow missed first_seen
                    first_seen[u] = g
                    gene_platforms[u] = [plat]
                plat_rows.append(_row(u))
            _make_tab(f"{plat} ({len(plat_rows):,})", plat_rows)  # no re-sort

        # ── Union / intersection tabs (only if >1 platform) ──
        if len(selected) > 1:
            # Union: every gene in any platform, sorted by coverage desc then name
            union_rows = [_row(u) for u in all_upper]
            _make_tab(f"Union ({len(union_rows):,})", union_rows,
                      sort_key=lambda r: (-r['n_cov'], r['symbol'].upper()))

            # Intersection: genes in ALL platforms (coverage = n_total/n_total)
            inter_upper = set.intersection(*upper_sets.values()) if upper_sets else set()
            inter_rows = [_row(u) for u in inter_upper]
            _make_tab(f"Intersection ({len(inter_rows):,})", inter_rows,
                      sort_key=lambda r: r['symbol'].upper())

        # --- Filter handler ---
        def _current_tab_key():
            try:
                return nb.tab(nb.select(), 'text')
            except tk.TclError:
                return None

        def _repopulate(tree, rows, q_upper):
            tree.delete(*tree.get_children())
            shown = 0
            for row in rows:
                if q_upper and q_upper not in row['symbol'].upper() \
                        and q_upper not in row['platforms'].upper():
                    continue
                shown += 1
                tag = f"cov_{row['n_cov']}"
                stripe = "row_odd" if shown % 2 else "row_even"
                tree.insert("", tk.END,
                            values=(shown,
                                    row['symbol'],
                                    row['coverage'],
                                    row['platforms']),
                            tags=(tag, stripe))
            return shown

        def _apply_filter(*_):
            key = _current_tab_key()
            if key not in tabs:
                return
            q = search_var.get().strip().upper()
            tree = tabs[key]['tree']
            rows = tabs[key]['rows']
            shown = _repopulate(tree, rows, q)
            status_lbl.config(
                text=f"{shown:,} of {len(rows):,} genes shown"
            )

        search_var.trace_add('write', _apply_filter)
        nb.bind("<<NotebookTabChanged>>", lambda e: _apply_filter())
        _apply_filter()

        # --- Actions: copy + save ---
        def _selected_or_all_rows():
            key = _current_tab_key()
            if key not in tabs:
                return [], key
            tree = tabs[key]['tree']
            rows = tabs[key]['rows']
            sel_ids = tree.selection()
            if sel_ids:
                # Map tree items back to rows by their displayed symbol
                selected_symbols = {tree.item(i, 'values')[1] for i in sel_ids}
                picked = [r for r in rows if r['symbol'] in selected_symbols]
            else:
                # Honor current filter when nothing is selected
                q = search_var.get().strip().upper()
                picked = [
                    r for r in rows
                    if not q or (q in r['symbol'].upper()
                                 or q in r['platforms'].upper())
                ]
            return picked, key

        def _copy_current():
            picked, _ = _selected_or_all_rows()
            if not picked:
                return
            # Copy just the gene symbols (one per line) - most useful for
            # pasting into enrichment tools.
            payload = "\n".join(r['symbol'] for r in picked)
            try:
                win.clipboard_clear()
                win.clipboard_append(payload)
                status_lbl.config(
                    text=f"Copied {len(picked):,} gene(s) to clipboard"
                )
            except Exception as exc:
                messagebox.showerror("Clipboard error", str(exc), parent=win)

        def _save_current():
            picked, key = _selected_or_all_rows()
            if not picked:
                messagebox.showinfo("Empty", "No genes to save.", parent=win)
                return
            default_name = (key or "gene_list").split(' (')[0].replace(' ', '_') + "_genes.csv"
            path = filedialog.asksaveasfilename(
                parent=win,
                title="Save Gene List",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")]
            )
            if not path:
                return
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("gene_symbol,coverage,platforms\n")
                    for r in picked:
                        # quote platform list (may contain commas)
                        f.write(f"{r['symbol']},{r['coverage']},"
                                f"\"{r['platforms']}\"\n")
                status_lbl.config(text=f"Saved {len(picked):,} genes -> {path}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc), parent=win)

        btns = ttk.Frame(win, padding=6)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Copy to Clipboard", command=_copy_current,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save CSV...", command=_save_current,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Close", command=win.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=4)

        # Focus the filter box for quick typing
        try:
            search_entry.focus()
        except Exception:
            pass

    def _await_worker(self, fn, *args, **kwargs):
        """Run *fn* off the Tk thread, keeping the window repainting.

        A platform matrix is hundreds of MB, and reading or scanning one on
        the Tk thread stops the event loop for tens of seconds - the window
        greys out and the window manager reports it as not responding, so a
        load that is working looks like a hang.

        Only *idle* events are pumped while we wait: redraws happen, but user
        input stays queued, so a second click cannot re-enter the load
        half-way through. The call is still synchronous - several callers
        load platforms in sequence and rely on the data (and on any
        exception) being there when this returns.
        """
        box = {}

        def _run():
            try:
                box['value'] = fn(*args, **kwargs)
            except BaseException as exc:          # re-raised on the Tk thread
                box['error'] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while t.is_alive():
            self.update_idletasks()
            t.join(0.05)
        if 'error' in box:
            raise box['error']
        return box.get('value')

    def _load_gpl_data(self, gpl_name, file_path, metadata_path=None, gsm_filter=None):
        """
        Load GPL platform data.

        Args:
            gpl_name: Platform identifier (e.g., "GPL570")
            file_path: Path to expression data CSV
            metadata_path: Optional path to metadata CSV (for GPL570)
            gsm_filter: Optional set/list of GSM IDs - if provided, only keep these samples
        """
        full_file_path = Path(file_path)
        full_meta_path = Path(metadata_path) if metadata_path else None
        
        try:
            self.enqueue_log(f"[{gpl_name}] Loading platform data from {full_file_path.name}...")
            
            if not full_file_path.exists():
                raise FileNotFoundError(f"Data file not found: {full_file_path}")
            
            if gpl_name in self.gpl_datasets:
                existing_n = len(self.gpl_datasets[gpl_name])
                # 3-option dialog: Replace, Keep Both (for comparison), Cancel
                response = messagebox.askyesnocancel(
                    "Platform Already Loaded", 
                    f"{gpl_name} is already loaded with {existing_n:,} samples.\n\n"
                    f"  YES = Replace (overwrite existing)\n"
                    f"  NO = Keep Both (load as '{gpl_name}_2' for comparison)\n"
                    f"  CANCEL = Cancel", 
                    parent=self
                )
                if response is None:
                    # Cancel
                    self.enqueue_log(f"[{gpl_name}] Load cancelled by user")
                    return
                elif response is False:
                    # Keep Both - find next available suffix
                    suffix = 2
                    while f"{gpl_name}_{suffix}" in self.gpl_datasets:
                        suffix += 1
                    gpl_name = f"{gpl_name}_{suffix}"
                    self.enqueue_log(f"[{gpl_name}] Loading alongside existing (for comparison)")
            
            self.update_progress(value=0)
            self.status_label.config(text=f"Loading {gpl_name}...", foreground=AERO["accent"])
            self.update_idletasks()
            
            self.enqueue_log(f"[{gpl_name}] Reading expression data (this may take a moment)...")
            _is_gz = str(full_file_path).endswith('.gz')
            data_df = self._await_worker(
                pd.read_csv, full_file_path,
                compression="gzip" if _is_gz else "infer", low_memory=False)
            self.update_progress(value=30, text=f"Parsing {gpl_name}…")
            self.update_idletasks()
            
            if gpl_name == "GPL570" and full_meta_path:
                if full_meta_path.exists():
                    self.enqueue_log(f"[{gpl_name}] Loading metadata...")
                    _meta_gz = str(full_meta_path).endswith('.gz')
                    meta_df = self._await_worker(
                        pd.read_csv, full_meta_path,
                        compression="gzip" if _meta_gz else "infer", low_memory=False)
                    
                    if len(meta_df) == len(data_df):
                        data_df = pd.concat([meta_df.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1)
                        self.enqueue_log(f"[{gpl_name}] OK Metadata merged ({len(meta_df.columns)} columns)")
                    else:
                        self.enqueue_log(f"[{gpl_name}] [!] Metadata row count mismatch ({len(meta_df)} vs {len(data_df)}), skipping merge")
                else:
                    self.enqueue_log(f"[{gpl_name}] [!] Metadata file not found at {full_meta_path}")
            
            self.update_progress(value=50, text=f"Identifying samples in {gpl_name}…")
            self.update_idletasks()

            gsm_col = None
            if "gsm" in data_df.columns:
                data_df.rename(columns={"gsm": "GSM"}, inplace=True)
                gsm_col = "GSM"
            elif "GSM" in data_df.columns:
                gsm_col = "GSM"
            else:
                candidates = [col for col in data_df.columns if "GSM" in col.upper()]
                if candidates:
                    data_df.rename(columns={candidates[0]: "GSM"}, inplace=True)
                    gsm_col = "GSM"
                    self.enqueue_log(f"[{gpl_name}] Found GSM column: {candidates[0]} -> GSM")
            
            if gsm_col:
                data_df["GSM"] = data_df["GSM"].astype(str).str.upper()
                
                # ── Apply subset filter if requested ──
                if gsm_filter is not None:
                    filter_set = {g.upper() for g in gsm_filter}
                    before_n = len(data_df)
                    data_df = data_df[data_df["GSM"].isin(filter_set)].copy()
                    self.enqueue_log(
                        f"[{gpl_name}] Subset filter: {before_n:,} → {len(data_df):,} samples "
                        f"(matched {len(data_df)} of {len(filter_set)} requested)")
                    if data_df.empty:
                        messagebox.showwarning(
                            "No Matching Samples",
                            f"None of the {len(filter_set)} labeled samples were found in "
                            f"the {gpl_name} expression data file.\n\n"
                            f"The label file and expression file may be for different platforms.",
                            parent=self)
                        self.update_progress(value=0)
                        self.status_label.config(text="Ready", foreground="gray")
                        return
                
                if self._gsm_lookup is not None:
                    self.enqueue_log(f"[{gpl_name}] Fetching GSE metadata for {len(data_df['GSM'].unique()):,} unique samples...")
                    
                    unique_gsms = data_df['GSM'].unique().tolist()
                    lookup_result = self._await_worker(self._fast_gsm_lookup, unique_gsms)
                    
                    if 'series_id' in data_df.columns:
                        data_df = data_df.merge(lookup_result, on="GSM", how="left", suffixes=('', '_new'))
                        data_df['series_id'] = data_df['series_id'].fillna(data_df['series_id_new'])
                        data_df.drop(columns=['series_id_new'], inplace=True, errors='ignore')
                    else:
                        data_df = data_df.merge(lookup_result, on="GSM", how="left")
                    
                    num_found = data_df['series_id'].notna().sum()
                    self.enqueue_log(f"[{gpl_name}] OK Found GSE info for {num_found:,} / {len(data_df):,} samples")
                else:
                    self.enqueue_log(f"[{gpl_name}] [!] GSM lookup table unavailable (database not loaded)")
            else:
                self.enqueue_log(f"[{gpl_name}] [!] Could not identify GSM column - some features may be limited")
            
            self.update_progress(value=70, text=f"Indexing genes for {gpl_name}…")
            self.update_idletasks()

            self.gpl_datasets[gpl_name] = data_df
            self.gpl_source_paths[gpl_name] = str(full_file_path)

            # Tens of thousands of columns, each possibly coerced - off the Tk
            # thread with the rest of the heavy work.
            gene_map, coerced_cols = self._await_worker(
                lambda: index_gene_columns(data_df))

            if coerced_cols > 0:
                self.enqueue_log(f"[{gpl_name}] Converted {coerced_cols} columns from string -> numeric")
                # Update stored df with coerced types
                self.gpl_datasets[gpl_name] = data_df
            
            self.gpl_gene_mappings[gpl_name] = gene_map
            
            self.update_progress(value=100, text=f"{gpl_name} loaded")
            self.update_idletasks()

            # deep=True walks every object cell - seconds on a wide matrix.
            mem_usage_mb = self._await_worker(
                lambda: data_df.memory_usage(deep=True).sum() / 1024**2)
            
            _load_type = "SUBSET for labeled samples" if gsm_filter else "ALL SAMPLES LOADED"
            self.enqueue_log(
                f"[{gpl_name}] OK Successfully loaded:\n"
                f"  - {len(data_df):,} samples ({_load_type})\n"
                f"  - {len(gene_map):,} gene expression columns identified\n"
                f"  - Memory usage: ~{mem_usage_mb:.1f} MB"
            )
            
            load_mode = ""
            if gsm_filter:
                load_mode = f" (subset: {len(data_df):,} labeled samples)"
            messagebox.showinfo(
                f"{gpl_name} Loaded", 
                f"Successfully loaded {gpl_name} {load_mode}:\n\n"
                f"- {len(data_df):,} samples\n"
                f"- {len(gene_map):,} genes\n\n"
                f"Expression data is now available for analysis\n"
                f"(Gene Explorer, Compare Distributions).",
                parent=self
            )
            
            self._update_platform_status()

            self.update_progress(value=0)
            self.status_label.config(text="Ready", foreground="gray")

        except FileNotFoundError as e:
            self.enqueue_log(f"[{gpl_name}] X ERROR: {e}")
            messagebox.showerror(f"{gpl_name} Error", str(e), parent=self)
            self.update_progress(value=0)
            self.status_label.config(text="Ready", foreground="gray")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.enqueue_log(f"[{gpl_name}] X ERROR: {e}\n{error_details}")
            messagebox.showerror(
                f"{gpl_name} Error",
                f"An error occurred while loading {gpl_name}:\n\n{e}\n\nSee log for details.",
                parent=self
            )
            self.update_progress(value=0)
            self.status_label.config(text="Ready", foreground="gray")

    def start_extraction(self):
        """Starts search thread across the selected data sources."""
        if self.current_extraction_thread and self.current_extraction_thread.is_alive():
            messagebox.showwarning("Busy", "Search is already running.\n\nPlease wait for it to complete.", parent=self)
            return

        # Which sources did the user enable?
        use_geo = bool(self.src_geo_var.get())
        extra_sources = set()
        if getattr(self, "src_archs4_var", None) and self.src_archs4_var.get():
            extra_sources.add("archs4")
        if getattr(self, "src_cellxgene_var", None) and self.src_cellxgene_var.get():
            extra_sources.add("cellxgene")
        if getattr(self, "src_atlas_var", None) and self.src_atlas_var.get():
            extra_sources.add("atlas")

        if not use_geo and not extra_sources:
            messagebox.showerror("No source selected",
                "Please tick at least one data source in section 1.",
                parent=self)
            return

        gz_path = CONFIG['paths']['geo_db']
        if use_geo and (not gz_path or not os.path.exists(gz_path)):
            messagebox.showerror(
                "GEOmetadb Required",
                "GEOmetadb.sqlite.gz is required for GEOmetadb search.\n\n"
                "Please download from:\n"
                "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz\n\n"
                f"Expected at: {gz_path}\n\n"
                "Or un-tick 'GEOmetadb' in section 1 to skip it.",
                parent=self
            )
            return

        plat_filter = self.platform_entry.get().strip() if use_geo else ""
        tokens = self.filter_entry.get().strip()

        if not tokens:
            messagebox.showerror("Input Required", "Please enter at least one keyword to search for.\n\nExample: cancer, breast, treatment", parent=self)
            return

        # Collect per-source sub-filters
        subfilters = {}
        if "archs4" in extra_sources:
            subfilters["archs4_organism"] = self.archs4_org_var.get() or "human"
        if "atlas" in extra_sources:
            subfilters["atlas_species"] = self.atlas_species_entry.get().strip()
        
        self.step1_results_df = None
        self.step2_data_df = None
        self.step1_gse_keywords = {}
        self.step1_gse_descriptions = {}
        self.gse_to_keep_for_step2 = []
        self.gse_listbox.delete(0, tk.END)
        self.gse_frame.pack_forget()
        self.step2_status_label.config(text="Searching GEO database...", foreground=AERO["accent"])
        # Mark Step 1 as running (search kicking off)
        try:
            self._set_step_status(self.step1_frame, self._step1_title, "running")
        except Exception:
            pass
        
        self.enqueue_log("[Step 1] Starting GEO database search...")
        self.enqueue_log(f"[Step 1] Keywords: {tokens}")
        if plat_filter:
            self.enqueue_log(f"[Step 1] Platform filter: {plat_filter}")
        else:
            self.enqueue_log("[Step 1] No platform filter - searching all platforms")
        
        self.update_progress(value=0)
        self.status_label.config(text="Searching GEO database...", foreground=AERO["accent"])
        
        if extra_sources:
            self.enqueue_log(f"[Step 1] Extra sources: {sorted(extra_sources)}")

        active_sources = set()
        if use_geo:
            active_sources.add("geo")
        active_sources |= extra_sources

        self.current_extraction_thread = ExtractionThread(
            gz_path=CONFIG['paths']['geo_db'],
            plat_filter=plat_filter,
            search_tokens=tokens,
            log_func=self.enqueue_log,
            on_finish=self.on_extraction_finish,
            gui_ref=self,
            search_sources=active_sources,
            subfilters=subfilters,
        )
        # Kick off shimmer immediately so the bar feels active even before the
        # worker's first progress message arrives. update_progress() handles
        # subsequent transitions automatically.
        try:
            self.update_progress(value=1, text="Starting search…")
        except Exception:
            pass
        self.current_extraction_thread.start()
    

    
    
    def on_extraction_finish(self):
        """Callback when extraction finishes - opens review window."""
        if not self.current_extraction_thread:
            return
        
        thread = self.current_extraction_thread
        self.step1_results_df = thread.final_df
        self.step1_gse_keywords = thread.gse_keywords
        self.step1_gse_descriptions = thread.gse_descriptions
        self._step1_gsm_descriptions = thread.gsm_descriptions
        search_tokens = thread.search_tokens
        gsm_descriptions = thread.gsm_descriptions
        
        self.current_extraction_thread = None

        # Route through update_progress so the shimmer stops cleanly
        self.update_progress(value=100, text="Search complete")
        self.status_label.config(text="Search complete", foreground="green")
        
        if self.step1_results_df is None or self.step1_results_df.empty:
            self.enqueue_log("[Step 1] ✗ No experiments found matching your criteria")
            messagebox.showinfo(
                "No Results", 
                "No experiments were found matching your search criteria.\n\n"
                "Try:\n"
                "- Different keywords\n"
                "- Broader search terms\n"
                "- Removing platform filter\n"
                "- Checking spelling", 
                parent=self
            )
            self.step2_status_label.config(text="No experiments found. Try different search terms.", foreground="orange")
            self.update_progress(value=0)
            return
        
        num_gses = self.step1_results_df['series_id'].nunique() if 'series_id' in self.step1_results_df.columns else 0
        num_samples = len(self.step1_results_df)
        
        self.enqueue_log(f"[Step 1] OK Found {num_gses} experiment(s) with {num_samples:,} samples total")
        
        # ── Open the interactive review window ──
        GSEReviewWindow(
            parent=self,
            app_ref=self,
            results_df=self.step1_results_df,
            gse_descriptions=self.step1_gse_descriptions,
            gse_keywords=self.step1_gse_keywords,
            gsm_descriptions=gsm_descriptions,
            search_tokens=search_tokens
        )
        
        self.step2_status_label.config(
            text=f"Found {num_gses} experiment(s). Review and select in the review window.",
            foreground="green"
        )
        self.update_progress(value=0)

    def _save_selected_gses(self):
        """Saves selected GSEs for Step 2 analysis - COMPLETE VERSION."""
        selected_indices = self.gse_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one experiment from the list.", parent=self)
            return
        
        selected_gses = []
        for idx in selected_indices:
            line = self.gse_listbox.get(idx)
            gse = line.split(' ')[0]
            selected_gses.append(gse)
        
        self.gse_to_keep_for_step2 = selected_gses
        
        total_samples = len(self.step1_results_df[self.step1_results_df['series_id'].isin(selected_gses)])
        
        self.enqueue_log(
            f"[Step 1.5] OK Saved {len(selected_gses)} experiment(s) ({total_samples:,} samples) for Step 2"
        )
        
        gse_list_str = "\n".join(f"  - {gse}" for gse in selected_gses[:10])
        if len(selected_gses) > 10:
            gse_list_str += f"\n  ... and {len(selected_gses) - 10} more"
        
        messagebox.showinfo(
            "Experiments Saved", 
            f"Saved {len(selected_gses)} experiment(s) with {total_samples:,} samples.\n\n"
            f"Selected experiments:\n{gse_list_str}\n\n"
            f"You can now use:\n"
            f"- LLM Classification\n"
            f"- Manual Labeling\n"
            f"- Auto expression data integration",
            parent=self
        )
        
        self.step2_status_label.config(
            text=f"OK Ready: {len(selected_gses)} experiment(s) ({total_samples:,} samples) loaded for extraction", 
            foreground="green"
        )
    
    def _review_gse_details(self):
        """Opens the interactive GSE review window with keyword highlighting."""
        if self.step1_results_df is None or self.step1_results_df.empty:
            messagebox.showinfo("No Data",
                                "No experiments to review.\n\nPerform a search in Step 1 first.",
                                parent=self)
            return

        # Recover search tokens (they're stored as a set on the thread output)
        tokens = set()
        filter_text = self.filter_entry.get().strip()
        if filter_text:
            tokens = {t.strip().lower() for t in filter_text.split(',') if t.strip()}

        GSEReviewWindow(
            parent=self,
            app_ref=self,
            results_df=self.step1_results_df,
            gse_descriptions=self.step1_gse_descriptions,
            gse_keywords=self.step1_gse_keywords,
            gsm_descriptions=getattr(self, '_step1_gsm_descriptions', {}),
            search_tokens=tokens
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Step 1.5 - Download Expression Data for Selected Experiments
    # ═══════════════════════════════════════════════════════════════════

    def _download_selected_expression(self):
        """Download expression data for platforms used by selected experiments.

        Workflow:
            1. Detect which GPLs the selected experiments use
            2. Check which are already loaded in gpl_datasets
            3. Download missing ones using GPLDownloader (only selected GSEs)
            4. Load into gpl_datasets
            5. Verify expression data integrity
        """
        from genevariate.core import gpl_downloader as _gpl_dl

        # Get selected GSEs
        if not self.gse_to_keep_for_step2:
            messagebox.showinfo("No Experiments",
                "Save experiments first using 'Save Selected for Step 2'.",
                parent=self)
            return

        if self.step1_results_df is None or self.step1_results_df.empty:
            messagebox.showinfo("No Data", "No Step 1 results available.", parent=self)
            return

        # Find platforms needed
        selected_gses = set(self.gse_to_keep_for_step2)
        sub = self.step1_results_df[
            self.step1_results_df['series_id'].isin(selected_gses)]

        if 'gpl' not in sub.columns:
            messagebox.showwarning("No Platform Info",
                "Platform information not available in search results.\n"
                "Use the GPL Downloader window instead.", parent=self)
            return

        needed_gpls = sorted(
            set(sub['gpl'].dropna().astype(str).str.strip().str.upper().unique()))
        if not needed_gpls:
            messagebox.showinfo("No Platforms",
                "Could not determine platforms for selected experiments.", parent=self)
            return

        # Check which are already loaded OR have existing full data on disk
        loaded = set(k.upper() for k in self.gpl_datasets.keys())
        # Also check for existing CSV files (not loaded but downloadable)
        for gpl in needed_gpls:
            if gpl not in loaded:
                candidates = [
                    _gpl_dl.normalized_csv_path(
                        gpl, os.path.join(self.data_dir, gpl)),
                    _gpl_dl.raw_csv_path(
                        gpl, os.path.join(self.data_dir, gpl)),
                    os.path.join(self.data_dir, f"{gpl}_data.csv.gz"),
                ]
                for c in candidates:
                    if os.path.exists(c) and os.path.getsize(c) > 1_000_000:
                        # File exists and is >1MB - offer to load it instead of re-downloading
                        loaded.add(gpl)
                        # Auto-load into app
                        try:
                            self._load_gpl_data(gpl, c)
                            self.enqueue_log(f"[Step 1.5] Auto-loaded {gpl} from {os.path.basename(c)}")
                        except Exception:
                            loaded.discard(gpl)
                        break
        missing = [g for g in needed_gpls if g not in loaded]
        already = [g for g in needed_gpls if g in loaded]

        # Count samples per platform
        plat_gse_map = {}
        for _, row in sub.iterrows():
            gpl = str(row.get('gpl', '')).strip().upper()
            gse = str(row.get('series_id', '')).strip()
            if gpl and gse:
                if gpl not in plat_gse_map:
                    plat_gse_map[gpl] = set()
                plat_gse_map[gpl].add(gse)

        # Build summary message
        lines = []
        lines.append(f"Selected experiments use {len(needed_gpls)} platform(s):\n")
        for gpl in needed_gpls:
            n_gse = len(plat_gse_map.get(gpl, set()))
            status = "LOADED" if gpl in loaded else "NEEDS DOWNLOAD"
            lines.append(f"  {gpl}: {n_gse} experiments [{status}]")

        if not missing:
            messagebox.showinfo("All Loaded",
                "All platforms are already loaded!\n\n" + "\n".join(lines),
                parent=self)
            return

        lines.append(f"\nDownload {len(missing)} platform(s)?")
        lines.append(f"This will download expression matrices from NCBI GEO,")
        lines.append(f"normalize them, and load into the application.")

        if not messagebox.askyesno("Download Expression Data",
                                    "\n".join(lines), parent=self):
            return

        # Disable button, show progress
        self._download_expr_btn.config(state=tk.DISABLED, text="Downloading...")
        self._dl_progress_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        self._dl_progress_bar['value'] = 0
        self._dl_progress_label.config(text="Initializing...")
        self.update_idletasks()

        def _download_thread():
            """Background download thread."""
            try:
                from genevariate.core.db_loader import open_geometadb
                from genevariate.core.gpl_downloader import GPLDownloader
                from genevariate.core import gpl_downloader as _gpl_dl
                from genevariate.config import CONFIG

                output_dir = str(CONFIG['paths']['data'])
                geo_db = str(CONFIG['paths']['geo_db'])

                # Load GEOmetadb
                self.after(0, lambda: self._dl_progress_label.config(
                    text="Loading GEOmetadb..."))
                conn = open_geometadb(geo_db, log_fn=self.enqueue_log)
                if conn is None:
                    self.after(0, lambda: messagebox.showerror("Error",
                        "Failed to load GEOmetadb", parent=self))
                    return

                downloader = GPLDownloader(
                    conn, output_dir, max_workers=3, download_timeout=180)

                total_platforms = len(missing)
                results = {}

                for pi, gpl_id in enumerate(missing):
                    gse_list = sorted(plat_gse_map.get(gpl_id, set()))
                    n_gse = len(gse_list)

                    def _cb(pct, stage, msg, _gpl=gpl_id, _pi=pi):
                        overall_pct = int((_pi / total_platforms +
                                          (pct or 0) / 100 / total_platforms) * 100)
                        self.after(0, lambda p=overall_pct, m=msg, g=_gpl: (
                            self._dl_progress_bar.configure(value=p),
                            self._dl_progress_label.config(
                                text=f"[{g}] {m}")))
                        self.enqueue_log(f"[Download {_gpl}] [{stage}] {msg}")

                    self.after(0, lambda g=gpl_id, n=n_gse: (
                        self._dl_progress_label.config(
                            text=f"Downloading {g} ({n} experiments)...")))

                    try:
                        info = downloader.get_platform_info(gpl_id)

                        # Filter GSE list to only our selected experiments
                        all_gses = info['gse_list']
                        filtered_gses = [g for g in all_gses if g in gse_list]
                        if not filtered_gses:
                            # Platform exists but none of our GSEs match - download a sample
                            filtered_gses = gse_list[:min(n_gse, 50)]
                            self.enqueue_log(
                                f"[Download {gpl_id}] "
                                f"GSE IDs not in GEOmetadb gse_gpl - "
                                f"downloading {len(filtered_gses)} directly")

                        # Override the info's gse_list with our filtered list
                        info['gse_list'] = filtered_gses
                        info['total_series'] = len(filtered_gses)

                        # Check if a full platform file exists - protect it.
                        # The download writes the RAW matrix, so that is the
                        # one a subset run would clobber.
                        full_csv = _gpl_dl.raw_csv_path(
                            gpl_id, os.path.join(output_dir, gpl_id))
                        had_full_file = (os.path.exists(full_csv) and
                                         os.path.getsize(full_csv) > 1_000_000)

                        result = downloader.run_with_info(
                            info, max_gse=0, callback=_cb)
                        results[gpl_id] = result

                        # If we downloaded a subset and a full file existed,
                        # rename the subset so we don't overwrite the full data
                        if had_full_file and os.path.exists(full_csv):
                            subset_csv = full_csv.replace(
                                '.csv.gz', '_selected_experiments.csv.gz')
                            os.rename(full_csv, subset_csv)
                            result['filepath'] = subset_csv
                            self.enqueue_log(
                                f"[Download {gpl_id}] Saved subset as "
                                f"{os.path.basename(subset_csv)} "
                                f"(full platform file preserved)")

                        self.enqueue_log(
                            f"[Download {gpl_id}] OK: "
                            f"{result['n_samples']:,} samples x "
                            f"{result['n_genes']:,} genes")

                    except Exception as e:
                        self.enqueue_log(f"[Download {gpl_id}] FAILED: {e}")
                        results[gpl_id] = {'error': str(e)}

                conn.close()

                # Load downloaded platforms into app
                self.after(0, lambda: self._dl_progress_label.config(
                    text="Loading downloaded platforms..."))

                loaded_count = 0
                for gpl_id, result in results.items():
                    if 'error' in result:
                        continue
                    filepath = result.get('filepath', '')
                    if filepath and os.path.exists(filepath):
                        # Verify data quality before loading
                        try:
                            import gzip as _gz
                            with _gz.open(filepath, 'rt') as f:
                                hdr = f.readline().strip().split(',')
                            test_df = pd.read_csv(
                                filepath, usecols=[hdr[2]], nrows=10)
                            non_null = test_df.iloc[:, 0].notna().sum()

                            if non_null > 0:
                                # Load into app on main thread
                                self.after(0, lambda g=gpl_id, f=filepath: (
                                    self._load_gpl_data(g, f)))
                                loaded_count += 1
                                self.enqueue_log(
                                    f"[Download {gpl_id}] Verified: "
                                    f"{non_null}/10 test values non-NaN")
                            else:
                                self.enqueue_log(
                                    f"[Download {gpl_id}] WARNING: "
                                    f"all test values NaN - skipping load")
                        except Exception as e:
                            self.enqueue_log(
                                f"[Download {gpl_id}] Load error: {e}")

                # Summary
                n_ok = sum(1 for r in results.values() if 'error' not in r)
                n_fail = sum(1 for r in results.values() if 'error' in r)
                summary = (
                    f"Download complete: {n_ok}/{total_platforms} platforms OK"
                    f"{f', {n_fail} failed' if n_fail else ''}")

                self.after(0, lambda s=summary: (
                    self._dl_progress_label.config(text=s),
                    self._dl_progress_bar.configure(value=100),
                    messagebox.showinfo("Download Complete", s, parent=self)))

            except Exception as e:
                import traceback
                self.enqueue_log(f"[Download] ERROR: {traceback.format_exc()}")
                # `e` is gone by the time Tk runs this, so bind it here.
                self.after(0, lambda msg=str(e): messagebox.showerror("Error",
                    f"Download failed: {msg}", parent=self))
            finally:
                self.after(0, lambda: (
                    self._download_expr_btn.config(
                        state=tk.NORMAL,
                        text="Download Expression Data for Selected Experiments"),
                ))

        import threading
        threading.Thread(target=_download_thread, daemon=True).start()

    # ═══════════════════════════════════════════════════════════════════
    #  LLM Extraction Window
    # ═══════════════════════════════════════════════════════════════════
    def _open_llm_extraction_window(self):
        """Open the LLM Extraction window for extracting labels from samples."""
        win = tk.Toplevel(self)
        style_window(win)
        win.title("LLM Label Extraction")
        win.geometry("850x850")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"850x850+{(_sw-850)//2}+{(_sh-850)//2}")
            win.minsize(500, 400)
        except Exception: pass
        win.resizable(True, True)
        win.transient(self)

        canvas = tk.Canvas(win, highlightthickness=0)
        vsb = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        main = scroll_frame
        PAD = 8

        # ── EXTRACTION OVERVIEW ──
        info_frame = labelframe(main, text=" Extraction Overview", padding=6)
        info_frame.pack(fill=tk.X, padx=PAD, pady=(PAD, 4))

        info_text = (
            "geo_label_extractor pipeline (the extractor's own entry point)\n"
            "  Phase 1 reads each of the five fields (Sex, Age, Tissue, Condition,\n"
            "  Treatment) verbatim from the sample's GEO metadata; phase 1b recovers\n"
            "  what the sample is silent about from its study; phase 1c takes the\n"
            "  consensus. Phase 2 then normalizes Tissue/Condition/Treatment against\n"
            "  MeSH, Cellosaurus and BioLORD and attaches their ontology ids -- it\n"
            "  runs when the reference artifacts are on this machine, and the log\n"
            "  says which stage the labels came from. No evidence -> 'Not Specified'."
        )
        info_lbl = tk.Label(info_frame, text=info_text, font=(MONO_FONT, 8),
                            justify=tk.LEFT, anchor='nw', fg=AERO["text"], bg=AERO["panel"],
                            padx=8, pady=6, relief=tk.FLAT,
                            highlightthickness=1, highlightbackground=AERO["border"])
        info_lbl.pack(fill=tk.X)

        # ── 1. DATA SOURCE ──
        src_frame = labelframe(main, text=" Data Source", padding=PAD)
        src_frame.pack(fill=tk.X, padx=PAD, pady=(PAD, 4))

        src_var = tk.StringVar(value="platform")
        ttk.Radiobutton(src_frame, text="Entire GPL Platform", variable=src_var, value="platform",
                         command=lambda: self._llm_ext_toggle_source(src_var, plat_row, exp_row, file_row)).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(src_frame, text="Specific Experiments (GSE search)", variable=src_var, value="experiments",
                         command=lambda: self._llm_ext_toggle_source(src_var, plat_row, exp_row, file_row)).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(src_frame, text="Step 1/1.5 Selected Experiments", variable=src_var, value="step1",
                         command=lambda: self._llm_ext_toggle_source(src_var, plat_row, exp_row, file_row)).grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(src_frame, text="External File (CSV/TXT with GSM list)", variable=src_var, value="external",
                         command=lambda: self._llm_ext_toggle_source(src_var, plat_row, exp_row, file_row)).grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(src_frame, text="Loaded Dataset (from 'Load Metadata')", variable=src_var, value="loaded_dataset",
                         command=lambda: self._llm_ext_toggle_source(src_var, plat_row, exp_row, file_row)).grid(row=4, column=0, sticky=tk.W, pady=2)

        # Platform selection row
        plat_row = ttk.Frame(src_frame)
        plat_row.grid(row=4, column=0, sticky=tk.EW, pady=4, columnspan=2)
        ttk.Label(plat_row, text="Platform:").pack(side=tk.LEFT, padx=(20, 4))
        plat_combo = ttk.Combobox(plat_row, width=20, state="readonly")
        plat_combo.pack(side=tk.LEFT, padx=4)
        # Populate with loaded platforms
        loaded = sorted(self.gpl_datasets.keys()) if self.gpl_datasets else []
        plat_combo['values'] = loaded
        if loaded:
            plat_combo.set(loaded[0])
        # Also allow typing a GPL ID for GEOmetadb lookup
        ttk.Label(plat_row, text="  or GPL ID:").pack(side=tk.LEFT, padx=4)
        gpl_id_entry = ttk.Entry(plat_row, width=12)
        gpl_id_entry.pack(side=tk.LEFT, padx=4)

        def _update_sample_count(*args):
            src = src_var.get()
            count = 0
            if src == "platform":
                sel = plat_combo.get()
                if sel in self.gpl_datasets:
                    count = len(self.gpl_datasets[sel])
            elif src == "experiments":
                # Count from experiment search results
                count = getattr(win, '_exp_sample_count', 0)
            elif src == "step1":
                if self.step1_results_df is not None and self.gse_to_keep_for_step2:
                    sub = self.step1_results_df[
                        self.step1_results_df['series_id'].isin(self.gse_to_keep_for_step2)]
                    # Count UNIQUE GSMs (same sample can appear in multiple series)
                    gsm_col = 'GSM' if 'GSM' in sub.columns else 'gsm'
                    if gsm_col in sub.columns:
                        count = sub[gsm_col].nunique()
                    else:
                        count = len(sub)
            elif src == "external":
                ext_df = getattr(win, '_ext_file_df', None)
                if ext_df is not None:
                    count = len(ext_df)
            elif src == "loaded_dataset":
                if self.full_dataset is not None:
                    count = len(self.full_dataset)
            est_lbl.config(text=self._llm_ext_time_estimate(count))
        # Store reference so radio button toggle can call it
        self._llm_update_count_fn = _update_sample_count
        plat_combo.bind("<<ComboboxSelected>>", _update_sample_count)

        # Experiment search row
        exp_row = ttk.Frame(src_frame)
        # Row 1: Keywords + Platform + Species
        exp_r1 = ttk.Frame(exp_row)
        exp_r1.pack(fill=tk.X, pady=2)
        ttk.Label(exp_r1, text="Keywords:").pack(side=tk.LEFT, padx=(20, 4))
        exp_kw_entry = ttk.Entry(exp_r1, width=25)
        exp_kw_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(exp_r1, text="Platform:").pack(side=tk.LEFT, padx=4)
        exp_plat_entry = ttk.Entry(exp_r1, width=10)
        exp_plat_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(exp_r1, text="Species:").pack(side=tk.LEFT, padx=4)
        exp_species_entry = ttk.Entry(exp_r1, width=18)
        exp_species_entry.pack(side=tk.LEFT, padx=4)
        exp_species_entry.insert(0, "")
        # Row 2: Search button + status
        exp_r2 = ttk.Frame(exp_row)
        exp_r2.pack(fill=tk.X, pady=2)

        def _search_experiments():
            kw = exp_kw_entry.get().strip()
            pf = exp_plat_entry.get().strip()
            sp = exp_species_entry.get().strip()
            if not kw:
                messagebox.showwarning("No Keywords", "Enter keywords to search.", parent=win)
                return
            if not self.gds_conn:
                messagebox.showerror("No Database", "GEOmetadb not loaded.", parent=win)
                return
            keywords = [k.strip() for k in kw.split(',') if k.strip()]
            platforms = [p.strip().upper() for p in pf.split(',') if p.strip()] if pf else []
            species_terms = [s.strip() for s in sp.split(',') if s.strip()] if sp else []
            try:
                conditions = []
                params = []
                for keyword in keywords:
                    conditions.append("(LOWER(gsm.title) LIKE ? OR LOWER(gsm.source_name_ch1) LIKE ? OR LOWER(gsm.characteristics_ch1) LIKE ?)")
                    pat = f"%{keyword.lower()}%"
                    params.extend([pat, pat, pat])
                where = " OR ".join(conditions)
                # Join with gpl table if species filter is used
                if species_terms:
                    query = (f"SELECT gsm.gsm, gsm.title, gsm.source_name_ch1, gsm.characteristics_ch1, gsm.series_id "
                             f"FROM gsm INNER JOIN gpl ON gsm.gpl = gpl.gpl WHERE ({where})")
                else:
                    query = f"SELECT gsm.gsm, gsm.title, gsm.source_name_ch1, gsm.characteristics_ch1, gsm.series_id FROM gsm WHERE ({where})"
                if platforms:
                    plat_placeholders = ','.join(['?' for _ in platforms])
                    query += f" AND UPPER(gsm.gpl) IN ({plat_placeholders})"
                    params.extend(platforms)
                if species_terms:
                    species_conditions = []
                    for st in species_terms:
                        species_conditions.append("LOWER(gpl.organism) LIKE ?")
                        params.append(f"%{st.lower()}%")
                    query += f" AND ({' OR '.join(species_conditions)})"
                query += " LIMIT 50000"
                df = pd.read_sql_query(query, self.gds_conn, params=params)
                win._exp_search_df = df
                win._exp_sample_count = len(df)
                species_msg = f" [{sp}]" if sp else ""
                exp_status.config(text=f"Found {len(df):,} samples from {df['series_id'].nunique() if 'series_id' in df.columns else '?'} experiments{species_msg}")
                _update_sample_count()
            except Exception as e:
                exp_status.config(text=f"Search error: {e}")
                win._exp_sample_count = 0

        ttk.Button(exp_r2, text="Search", command=_search_experiments,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=(20, 6))
        exp_status = ttk.Label(exp_r2, text="", foreground=AERO["muted"], font=('Segoe UI', 8))
        exp_status.pack(side=tk.LEFT, padx=4)
        ttk.Label(exp_r2, text="(e.g., Homo sapiens, Mus musculus)",
                  foreground="gray", font=('Segoe UI', 7, 'italic')).pack(side=tk.LEFT, padx=4)
        # Hide experiment row initially
        exp_row.grid(row=5, column=0, sticky=tk.EW, pady=4, columnspan=2)
        exp_row.grid_remove()

        # External file row
        file_row = ttk.Frame(src_frame)
        file_row.grid(row=6, column=0, sticky=tk.EW, pady=4, columnspan=2)
        file_row.grid_remove()
        ttk.Label(file_row, text="File:").pack(side=tk.LEFT, padx=(20, 4))
        ext_file_var = tk.StringVar(value="")
        ext_file_entry = ttk.Entry(file_row, textvariable=ext_file_var, width=50)
        ext_file_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        def _browse_ext_file():
            fp = filedialog.askopenfilename(
                title="Select file with GSM IDs",
                filetypes=[("CSV", "*.csv"), ("GZip CSV", "*.csv.gz"),
                           ("Text", "*.txt"), ("All", "*.*")],
                parent=win)
            if fp:
                ext_file_var.set(fp)
                # Count samples
                try:
                    if fp.endswith('.txt'):
                        with open(fp) as f:
                            lines = [l.strip() for l in f if l.strip()]
                        n = sum(1 for l in lines if l.upper().startswith('GSM'))
                    else:
                        df_peek = pd.read_csv(fp, nrows=0,
                            compression='gzip' if fp.endswith('.gz') else None)
                        df_full = pd.read_csv(fp,
                            compression='gzip' if fp.endswith('.gz') else None,
                            low_memory=False)
                        n = len(df_full)
                        win._ext_file_df = df_full
                    ext_count_lbl.config(text=f"{n:,} samples")
                    _update_sample_count()
                except Exception as e:
                    ext_count_lbl.config(text=f"Error: {e}")
        ttk.Button(file_row, text="Browse...", command=_browse_ext_file).pack(side=tk.LEFT, padx=4)
        ext_count_lbl = ttk.Label(file_row, text="", foreground=AERO["accent"])
        ext_count_lbl.pack(side=tk.LEFT, padx=4)
        win._ext_file_df = None

        # ── 2. LABELS TO EXTRACT ──
        lbl_frame = labelframe(main, text=" Labels to Extract", padding=PAD)
        lbl_frame.pack(fill=tk.X, padx=PAD, pady=4)

        field_vars = {}
        # Only the labels the vendored geo_label_extractor actually emits
        # (geo_extract_driver.ALL_FIELDS). Treatment_Time is not one of them
        # (it always returned "Not Specified") so it is gone.
        standard_fields = [
            ("Sex", "Biological sex (e.g., Male, Female)"),
            ("Tissue", "Tissue type (e.g., Brain, Blood, Liver)"),
            ("Condition", "Disease condition (e.g., Alzheimer, Cancer, Control)"),
            ("Treatment", "Treatment applied (e.g., LPS, Vehicle, Chemotherapy)"),
            ("Age", "Age of subject (e.g., 35 years, postnatal day 7)"),
        ]
        for i, (fname, fdesc) in enumerate(standard_fields):
            var = tk.BooleanVar(value=True)
            field_vars[fname] = var
            ttk.Checkbutton(lbl_frame, text=f"{fname}", variable=var,
                            command=_update_sample_count).grid(row=i, column=0, sticky=tk.W, padx=4)
            ttk.Label(lbl_frame, text=fdesc, foreground=AERO["muted"],
                      font=('Segoe UI', 8, 'italic')).grid(row=i, column=1, sticky=tk.W, padx=8)

        # ── 2b. EXTRACTION SETTINGS ──
        ollama_frame = labelframe(main, text=" Extraction Settings", padding=PAD)
        ollama_frame.pack(fill=tk.X, padx=PAD, pady=4)

        ttk.Label(ollama_frame,
                  text="Backend: geo_label_extractor (remote HTTP).",
                  foreground=AERO["muted"], font=('Segoe UI', 8, 'italic')).pack(anchor=tk.W, pady=(0, 4))

        req_row = ttk.Frame(ollama_frame)
        req_row.pack(fill=tk.X, pady=2)
        ttk.Label(req_row, text="Parallel requests:").pack(side=tk.LEFT)
        worker_var = tk.IntVar(value=getattr(self, '_llm_workers', 0) or 0)
        ttk.Spinbox(req_row, from_=0, to=32, textvariable=worker_var, width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(req_row, text="(0 = auto)",
                  foreground=AERO["muted"], font=('Segoe UI', 8, 'italic')).pack(side=tk.LEFT)

        # ── 3. SAVE DIRECTORY ──
        save_frame = labelframe(main, text=" Save Directory", padding=PAD)
        save_frame.pack(fill=tk.X, padx=PAD, pady=4)

        save_dir_var = tk.StringVar(value=os.path.join(self.data_dir, "labels"))
        ttk.Label(save_frame, text="Labels will be saved to:").pack(anchor=tk.W)
        dir_row = ttk.Frame(save_frame)
        dir_row.pack(fill=tk.X, pady=4)
        dir_entry = ttk.Entry(dir_row, textvariable=save_dir_var, width=60)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        def _browse_dir():
            d = filedialog.askdirectory(title="Select Labels Directory", parent=win)
            if d:
                save_dir_var.set(d)
        ttk.Button(dir_row, text="Browse...", command=_browse_dir).pack(side=tk.LEFT)
        ttk.Label(save_frame, text="File will be named: {GPL_ID}_labels.csv or {GPL_ID}_{keyword}_labels.csv",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).pack(anchor=tk.W)

        # ── 4. TIME ESTIMATION ──
        est_frame = labelframe(main, text=" Time Estimation", padding=PAD)
        est_frame.pack(fill=tk.X, padx=PAD, pady=4)
        est_lbl = ttk.Label(est_frame, text="Select a data source to see estimation",
                            font=('Segoe UI', 10), foreground=AERO["accent"])
        est_lbl.pack(anchor=tk.W)
        ttk.Label(est_frame, text="Based on ~0.52 samples/second (NVIDIA RTX 3060). Speed varies by hardware.",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).pack(anchor=tk.W)
        _update_sample_count()

        # ── 5. PROGRESS ──
        prog_frame = labelframe(main, text=" Progress", padding=PAD)
        prog_frame.pack(fill=tk.X, padx=PAD, pady=4)
        prog_bar = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate")
        prog_bar.pack(fill=tk.X, pady=4)
        prog_lbl = ttk.Label(prog_frame, text="Ready", font=('Segoe UI', 9))
        prog_lbl.pack(anchor=tk.W)

        # ── 6. ACTION BUTTONS ──
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, padx=PAD, pady=PAD)

        win._extraction_thread = None

        def _start_extraction():
            # ── Apply parallel-request setting (shared with Region Analysis) ──
            n_req = worker_var.get()
            self.ai_agent.MAX_WORKERS = n_req
            self._llm_workers = n_req

            # Gather fields
            selected_fields = [f for f, v in field_vars.items() if v.get()]
            if not selected_fields:
                messagebox.showwarning("No Labels", "Select at least one label to extract.", parent=win)
                return

            self._extraction_fields = selected_fields

            # Get samples DataFrame
            src = src_var.get()
            samples_df = None
            source_name = ""

            if src == "platform":
                sel = plat_combo.get()
                gpl_raw = gpl_id_entry.get().strip().upper()
                # Full set of metadata columns for LLM
                _meta_cols = ("gsm, title, source_name_ch1, characteristics_ch1, "
                              "description, extract_protocol_ch1, treatment_protocol_ch1, "
                              "growth_protocol_ch1, molecule_ch1, label_ch1, "
                              "organism_ch1, series_id, gpl")
                if gpl_raw and gpl_raw not in self.gpl_datasets:
                    # Try to look up samples from GEOmetadb
                    if self.gds_conn:
                        gpl_val = gpl_raw if gpl_raw.startswith("GPL") else f"GPL{gpl_raw}"
                        try:
                            samples_df = pd.read_sql_query(
                                f"SELECT {_meta_cols} FROM gsm WHERE UPPER(gpl) = ?",
                                self.gds_conn, params=[gpl_val])
                            source_name = gpl_val
                        except Exception as e:
                            messagebox.showerror("DB Error", str(e), parent=win)
                            return
                    else:
                        messagebox.showerror("No Database", "GEOmetadb not loaded. Load a platform first or provide the database.", parent=win)
                        return
                elif sel in self.gpl_datasets:
                    # Use loaded platform - need GSM metadata from GEOmetadb.
                    # Not every platform has GEO sample IDs: a CELLxGENE
                    # pseudo-bulk platform is keyed by donor/cell-type group,
                    # and a user CSV need not carry GSM either. Indexing blind
                    # raised a KeyError straight out of the button callback.
                    _pcols = self.gpl_datasets[sel].columns
                    gsm_col = ('GSM' if 'GSM' in _pcols
                               else ('gsm' if 'gsm' in _pcols else None))
                    if gsm_col is None:
                        messagebox.showerror(
                            "No GEO sample IDs",
                            f"Platform '{sel}' has no GSM/gsm column, so its "
                            "samples cannot be looked up in GEO.\n\n"
                            "Label extraction reads GEO sample metadata; pick a "
                            "GEO platform, or enter a GPL ID above.",
                            parent=win)
                        return
                    gsm_ids = list(self.gpl_datasets[sel][gsm_col].astype(str).str.upper().values)
                    if gsm_ids and self.gds_conn:
                        # Query in chunks to avoid SQL parameter limit
                        chunk_sz = 500
                        meta_dfs = []
                        for ci in range(0, len(gsm_ids), chunk_sz):
                            chunk = gsm_ids[ci:ci+chunk_sz]
                            ph = ','.join(['?'] * len(chunk))
                            try:
                                cdf = pd.read_sql_query(
                                    f"SELECT {_meta_cols} FROM gsm WHERE UPPER(gsm) IN ({ph})",
                                    self.gds_conn, params=chunk)
                                meta_dfs.append(cdf)
                            except Exception:
                                pass
                        if meta_dfs:
                            samples_df = pd.concat(meta_dfs, ignore_index=True)
                        else:
                            # Fallback: no metadata available
                            samples_df = pd.DataFrame({'gsm': gsm_ids})
                    elif gsm_ids:
                        # No GEOmetadb - create minimal df
                        samples_df = pd.DataFrame({'gsm': gsm_ids})
                    else:
                        samples_df = pd.DataFrame()
                    # GEOmetadb is a dated snapshot. Sources that ship their own
                    # harmonised sample text (ARCHS4) leave a sidecar next to the
                    # matrix; use it for whatever the snapshot did not cover,
                    # otherwise those samples reach the extractor as bare
                    # accessions and come back unlabelled for no visible reason.
                    samples_df = self._fill_meta_from_sidecar(sel, gsm_ids, samples_df)
                    source_name = sel
                else:
                    messagebox.showwarning("No Platform", "Select a loaded platform or enter a GPL ID.", parent=win)
                    return

            elif src == "experiments":
                if hasattr(win, '_exp_search_df') and win._exp_search_df is not None:
                    samples_df = win._exp_search_df.copy()
                    kw = exp_kw_entry.get().strip()
                    source_name = f"search_{kw.replace(' ','_').replace(',','_')[:30]}"
                else:
                    messagebox.showwarning("No Results", "Search for experiments first.", parent=win)
                    return

            elif src == "step1":
                if self.step1_results_df is not None and self.gse_to_keep_for_step2:
                    sub = self.step1_results_df[
                        self.step1_results_df['series_id'].isin(self.gse_to_keep_for_step2)].copy()
                    # CRITICAL: Deduplicate by GSM - same sample can appear in multiple series
                    gsm_col_name = 'GSM' if 'GSM' in sub.columns else 'gsm'
                    before = len(sub)
                    sub = sub.drop_duplicates(subset=[gsm_col_name])
                    if before != len(sub):
                        self.enqueue_log(
                            f"[LLM] Deduplicated: {before:,} → {len(sub):,} unique samples")
                    samples_df = sub
                    # Build source name including GPL for auto-registration
                    gpls = set()
                    if 'gpl' in sub.columns:
                        gpls = set(sub['gpl'].dropna().astype(str).str.upper().unique())
                    if len(gpls) == 1:
                        source_name = f"{gpls.pop()}_step1"
                    else:
                        source_name = f"step1_{len(self.gse_to_keep_for_step2)}gse"
                else:
                    messagebox.showwarning("No Data", "Complete Step 1 and save GSEs first.", parent=win)
                    return

            elif src == "external":
                ext_path = ext_file_var.get().strip()
                if not ext_path:
                    messagebox.showwarning("No File", "Browse for a CSV or TXT file first.", parent=win)
                    return

                # Load file
                try:
                    if ext_path.endswith('.txt'):
                        with open(ext_path) as f:
                            lines = [l.strip() for l in f if l.strip()]
                        gsm_lines = [l for l in lines if l.upper().startswith('GSM')]
                        if not gsm_lines:
                            messagebox.showerror("Error", "No GSM IDs found in text file.", parent=win)
                            return
                        samples_df = pd.DataFrame({'gsm': gsm_lines})
                    else:
                        samples_df = pd.read_csv(ext_path,
                            compression='gzip' if ext_path.endswith('.gz') else None,
                            low_memory=False)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read file:\n{e}", parent=win)
                    return

                # Detect GSM column
                cols_map = {c.upper(): c for c in samples_df.columns}
                gsm_key = cols_map.get('GSM') or cols_map.get('ID') or cols_map.get('SAMPLE')
                if not gsm_key:
                    for c in samples_df.columns:
                        if str(c).upper().startswith("GSM"):
                            gsm_key = c; break
                    # Single-column file with GSM values
                    if not gsm_key and len(samples_df.columns) == 1:
                        first_col = samples_df.columns[0]
                        if samples_df[first_col].astype(str).str.upper().str.startswith('GSM').mean() > 0.5:
                            gsm_key = first_col
                if gsm_key:
                    samples_df.rename(columns={gsm_key: 'gsm'}, inplace=True)
                else:
                    messagebox.showerror("Error",
                        "Could not detect GSM column.\n"
                        "File should have a column named GSM, ID, or Sample.", parent=win)
                    return

                samples_df['gsm'] = samples_df['gsm'].astype(str).str.strip().str.upper()

                # Resolve metadata from GEOmetadb if missing
                if 'title' not in samples_df.columns and self.gds_conn:
                    self.enqueue_log(f"[LLM] Resolving metadata for {len(samples_df):,} GSMs from GEOmetadb...")
                    _meta_cols = ("gsm, title, source_name_ch1, characteristics_ch1, "
                                  "description, treatment_protocol_ch1, organism_ch1, series_id, gpl")
                    meta_dfs = []
                    gsm_list = samples_df['gsm'].unique().tolist()
                    for ci in range(0, len(gsm_list), 500):
                        chunk = gsm_list[ci:ci+500]
                        ph = ','.join(['?'] * len(chunk))
                        try:
                            cdf = pd.read_sql_query(
                                f"SELECT {_meta_cols} FROM gsm WHERE UPPER(gsm) IN ({ph})",
                                self.gds_conn, params=chunk)
                            meta_dfs.append(cdf)
                        except Exception:
                            pass
                    if meta_dfs:
                        meta_df = pd.concat(meta_dfs, ignore_index=True)
                        meta_df['gsm'] = meta_df['gsm'].astype(str).str.strip().str.upper()
                        meta_df = meta_df.drop_duplicates(subset='gsm', keep='first')
                        # Merge metadata into samples_df
                        merge_cols = [c for c in meta_df.columns if c != 'gsm' and c not in samples_df.columns]
                        if merge_cols:
                            samples_df = samples_df.merge(meta_df[['gsm'] + merge_cols],
                                                          on='gsm', how='left')
                        n_matched = samples_df['title'].notna().sum() if 'title' in samples_df.columns else 0
                        self.enqueue_log(f"[LLM] Metadata resolved: {n_matched:,}/{len(samples_df):,} GSMs matched")

                # Build source name from filename
                fname = os.path.basename(ext_path)
                m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
                source_name = m.group(1).upper() if m else os.path.splitext(fname)[0][:30]

            # Also add "Loaded Dataset" option
            elif src == "loaded_dataset":
                if self.full_dataset is not None and not self.full_dataset.empty:
                    samples_df = self.full_dataset.copy()
                    # Normalize GSM column
                    for c in samples_df.columns:
                        if c.lower() in ('gsm', 'id', 'sample'):
                            samples_df.rename(columns={c: 'gsm'}, inplace=True)
                            break
                    if 'gsm' not in samples_df.columns:
                        messagebox.showerror("Error", "No GSM column in loaded dataset.", parent=win)
                        return
                    samples_df['gsm'] = samples_df['gsm'].astype(str).str.strip().str.upper()
                    # Resolve metadata if needed
                    if self.gds_conn and 'title' not in samples_df.columns:
                        self.enqueue_log(f"[LLM] Resolving metadata for {len(samples_df):,} GSMs...")
                        gsm_list = samples_df['gsm'].unique().tolist()
                        _meta_cols = ("gsm, title, source_name_ch1, characteristics_ch1, "
                                      "description, treatment_protocol_ch1, organism_ch1, series_id, gpl")
                        meta_dfs = []
                        for ci in range(0, len(gsm_list), 500):
                            chunk = gsm_list[ci:ci+500]
                            ph = ','.join(['?'] * len(chunk))
                            try:
                                cdf = pd.read_sql_query(
                                    f"SELECT {_meta_cols} FROM gsm WHERE UPPER(gsm) IN ({ph})",
                                    self.gds_conn, params=chunk)
                                meta_dfs.append(cdf)
                            except Exception: pass
                        if meta_dfs:
                            meta = pd.concat(meta_dfs, ignore_index=True)
                            meta['gsm'] = meta['gsm'].astype(str).str.strip().str.upper()
                            add_cols = [c for c in meta.columns if c != 'gsm' and c not in samples_df.columns]
                            if add_cols:
                                samples_df = samples_df.merge(
                                    meta[['gsm'] + add_cols].drop_duplicates('gsm'),
                                    on='gsm', how='left')
                    source_name = "loaded_dataset"
                    if 'gpl' in samples_df.columns:
                        gpls = samples_df['gpl'].dropna().astype(str).str.upper().value_counts()
                        if len(gpls) == 1:
                            source_name = gpls.index[0]
                else:
                    messagebox.showwarning("No Dataset", "Load a dataset first via 'Load Metadata (CSV)'.", parent=win)
                    return

            if samples_df is None or samples_df.empty:
                messagebox.showwarning("No Samples", "No samples found for the selected source.", parent=win)
                return

            # Normalize GSM column
            for col in ['gsm', 'GSM']:
                if col in samples_df.columns:
                    samples_df.rename(columns={col: 'gsm'}, inplace=True)
                    break
            if 'gsm' not in samples_df.columns:
                messagebox.showerror("Error", "No GSM column found in data.", parent=win)
                return

            n = len(samples_df)
            speed_est = 0.52
            eta_est = n / speed_est
            response = messagebox.askyesno(
                "Confirm LLM Extraction",
                f"Extract {len(selected_fields)} label(s) for {n:,} samples\n\n"
                f"Labels: {', '.join(selected_fields)}\n"
                f"Source: {source_name}\n"
                f"Estimated time: {self._format_eta(eta_est)}\n"
                f"Speed: ~{speed_est:.2f} samples/sec\n\n"
                f"Results will be saved to:\n{save_dir_var.get()}\n\n"
                f"Continue?",
                parent=win
            )
            if not response:
                return

            # Disable start, enable stop
            start_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
            prog_bar["value"] = 0
            prog_bar["maximum"] = n
            prog_lbl.config(text=f"Starting extraction of {n:,} samples...")
            self.enqueue_log(f"[LLM] Starting Phase 1 extraction: {n:,} samples on {source_name}...")

            def _on_progress(done, total, speed, eta):
                def _tick():
                    # The window is user-closable while the extraction keeps
                    # running. Without this guard the already-queued callbacks
                    # fire against destroyed widgets and raise TclError inside
                    # Tk's callback loop (the try/except below only covers the
                    # scheduling call, not the queued lambda).
                    if not win.winfo_exists():
                        return
                    prog_bar.config(value=done)
                    prog_lbl.config(
                        text=f"{done:,}/{total:,} | {speed:.2f} smp/s "
                             f"| ETA: {self._format_eta(eta)}")
                try:
                    win.after(0, _tick)
                    # Update main progress bar + log every 50 samples
                    self.update_progress(
                        value=done * 100 // max(1, total),
                        text=f"LLM: {done:,}/{total:,} | {speed:.1f} smp/s | ETA {int(eta)}s")
                    if done % 50 == 0 or done == total:
                        self.enqueue_log(
                            f"[LLM] {done:,}/{total:,} samples extracted "
                            f"({speed:.1f} smp/s, ETA: {self._format_eta(eta)})")
                except Exception:
                    pass

            def _on_finish():
                try:
                    thread = win._extraction_thread
                    result_df = thread.result_df if thread else pd.DataFrame()
                    win._extraction_thread = None

                    if result_df is not None and not result_df.empty:
                        # ═══════════════════════════════════════════════════
                        # EXTRACTION COMPLETE - save & load immediately
                        # User can start using labels RIGHT NOW
                        # ═══════════════════════════════════════════════════
                        save_dir = save_dir_var.get()
                        os.makedirs(save_dir, exist_ok=True)

                        # Save extraction results. Which stage produced them is
                        # part of what they are -- phase-2 labels carry ontology
                        # ids and phase-1b ones do not -- so it names the file.
                        stage = str(result_df.attrs.get("stage") or "labels")
                        phase1_df = result_df.copy()

                        # ── Merge series_id and metadata back from samples_df ──
                        # The pipeline returns GSM + labels (+ gse/gpl); the
                        # source text and downstream per-experiment grouping
                        # come from the sample table.
                        gc = 'gsm' if 'gsm' in phase1_df.columns else 'GSM'
                        phase1_df[gc] = phase1_df[gc].astype(str).str.strip().str.upper()
                        for meta_col in ['series_id', 'title', 'source_name_ch1',
                                         'characteristics_ch1', 'gpl']:
                            if meta_col not in phase1_df.columns and meta_col in samples_df.columns:
                                try:
                                    gsm_col_src = 'gsm' if 'gsm' in samples_df.columns else 'GSM'
                                    mapping = samples_df.set_index(
                                        samples_df[gsm_col_src].astype(str).str.strip().str.upper()
                                    )[meta_col]
                                    phase1_df[meta_col] = phase1_df[gc].map(mapping)
                                except Exception:
                                    pass
                        if 'series_id' in phase1_df.columns:
                            n_sid = phase1_df['series_id'].notna().sum()
                            n_gse = phase1_df['series_id'].nunique()
                            self.enqueue_log(
                                f"[LLM] Metadata merged: series_id for {n_sid:,}/{len(phase1_df):,} "
                                f"samples ({n_gse} experiments)")
                        else:
                            self.enqueue_log("[LLM] series_id not in source data - looking up from GEOmetadb...")

                        # Ensure ALL samples have series_id (critical for Phase 2)
                        phase1_df = self._ensure_series_id(phase1_df)
                        fname_raw = f"{source_name}_labels_{stage.split()[0]}.csv"
                        fpath_raw = os.path.join(save_dir, fname_raw)
                        phase1_df.to_csv(fpath_raw, index=False)
                        self.enqueue_log(f"[LLM] {stage} labels saved: {fpath_raw} ({len(phase1_df):,} samples)")

                        # Also save as clean _labels.csv
                        fname_clean = f"{source_name}_labels.csv"
                        fpath_clean = os.path.join(save_dir, fname_clean)
                        phase1_df.to_csv(fpath_clean, index=False)
                        self.enqueue_log(f"[LLM] Labels also saved as: {fpath_clean}")

                        # Load into platform_labels immediately so user can work
                        plat_id = source_name.split('_')[0] if '_' in source_name else source_name
                        if plat_id.upper().startswith('GPL'):
                            self.platform_labels[plat_id.upper()] = phase1_df.copy()
                            self._rebuild_merged_labels()
                            self.label_source_var.set("file")
                            self._toggle_main_label_source()
                            self._refresh_labels_display()
                            self.after(500, lambda p=plat_id.upper(): self._ensure_expression_data_for_labels(p))

                        # Summary of Phase 1+1.5
                        _summary_parts = []
                        _NS_CURATE = {'Condition', 'Tissue', 'Treatment'}
                        ns_count = 0
                        for _c in phase1_df.columns:
                            if _c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform') and phase1_df[_c].dtype.kind in ('O','U','S'):
                                _nu = phase1_df[_c].nunique()
                                _summary_parts.append(f"{_c}: {_nu} unique")
                                if _c in _NS_CURATE:
                                    ns_count += int(phase1_df[_c].astype(str).str.strip().isin(
                                        _NOT_SPECIFIED_VALUES).sum())

                        self.enqueue_log(
                            f"[LLM] Extraction COMPLETE: {len(phase1_df):,} samples, "
                            f"NS in Condition/Tissue/Treatment: {ns_count:,}. Labels loaded.")

                        self.update_progress(
                            value=100,
                            text="Extraction complete - labels loaded")

                        win.after(0, lambda: prog_lbl.config(
                            text=f"Done! {len(phase1_df):,} labels ready"))
                        win.after(0, lambda: prog_bar.config(value=prog_bar["maximum"]))
                        win.after(0, lambda: start_btn.config(state=tk.NORMAL))
                        win.after(0, lambda: stop_btn.config(state=tk.DISABLED))

                        # Extraction finished - labels are already loaded.
                        self._release_progress()
                        win.after(0, lambda: messagebox.showinfo(
                            "Extraction Complete",
                            f"Extracted labels for {len(phase1_df):,} samples!\n\n"
                            f"Saved to: {fpath_raw}\n"
                            + (f"Labels: {', '.join(_summary_parts)}\n\n" if _summary_parts else "\n")
                            + f"'Not Specified' in Condition/Tissue/Treatment: {ns_count:,}\n\n"
                            f"Labels are LOADED - you can start analysis now.",
                            parent=win))
                        return  # Don't fall through to error handler

                    else:
                        self._release_progress()  # extraction failed
                        win.after(0, lambda: prog_lbl.config(text="Extraction failed - check log"))
                        win.after(0, lambda: messagebox.showerror(
                            "Extraction Failed", "No results produced. Check the Activity Log.", parent=win))

                    win.after(0, lambda: start_btn.config(state=tk.NORMAL))
                    win.after(0, lambda: stop_btn.config(state=tk.DISABLED))
                except Exception as e:
                    self._release_progress()  # release on error
                    self.enqueue_log(f"[LLM] Finish callback error: {e}")

            self.enqueue_log(
                "[Extraction] Labels via the geo_label_extractor pipeline "
                "(extract -> normalize -> assemble)")

            self._acquire_progress()  # register extraction as progress bar owner

            # One run directory per source, kept beside the labels: it holds the
            # pipeline's checkpoint, so an interrupted run resumes there instead
            # of sending every sample to the model a second time.
            run_dir = os.path.join(save_dir_var.get(), f"{source_name}_pipeline")

            win._extraction_thread = LabelingThread(
                input_dataframe=samples_df,
                ai_agent=self.ai_agent,
                gui_log_func=self.enqueue_log,
                on_finish=_on_finish,
                fields=selected_fields,
                on_progress=_on_progress,
                gui_ref=win,
                out_dir=run_dir,
                metadata_columns=getattr(self, '_extraction_columns', None),
            )
            win._extraction_thread.start()

        def _stop_extraction():
            if win._extraction_thread:
                win._extraction_thread.stop()
                prog_lbl.config(text="Stopping... (waiting for current samples to finish)")
                stop_btn.config(state=tk.DISABLED)

        start_btn = ttk.Button(btn_frame, text="Start Extraction",
                               style="Primary.TButton", command=_start_extraction)
        start_btn.pack(side=tk.LEFT, padx=8)

        stop_btn = ttk.Button(btn_frame, text="Stop", style="Destructive.TButton",
                              command=_stop_extraction, state=tk.DISABLED)
        stop_btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(btn_frame, text="Close", command=win.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=8)

        # ── Row 2: Load existing labels and process ──
        load_frame = ttk.Frame(main)
        load_frame.pack(fill=tk.X, pady=(5, 8), padx=10)
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=3)
        ttk.Label(load_frame, text="Or load existing labels:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        def _load_and_run_curator():
            """Load a labels CSV and open LLM Curator (Phase 3)."""
            fpath = filedialog.askopenfilename(
                title="Load Labels CSV for LLM Curator (Phase 3)",
                filetypes=[("CSV files", "*.csv"), ("GZip CSV", "*.csv.gz"),
                           ("All files", "*.*")],
                parent=win)
            if not fpath:
                return
            try:
                df = pd.read_csv(fpath,
                    compression='gzip' if fpath.endswith('.gz') else None,
                    low_memory=False)
                if 'GSM' not in df.columns and 'gsm' not in df.columns:
                    for c in df.columns:
                        if c.lower() == 'gsm':
                            df = df.rename(columns={c: 'GSM'})
                            break
                if 'GSM' not in df.columns:
                    messagebox.showerror("Error", "CSV must have a GSM column.", parent=win)
                    return

                df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()

                # Detect platform
                plat = "Unknown"
                if 'gpl' in df.columns:
                    plat = df['gpl'].dropna().iloc[0] if not df['gpl'].dropna().empty else plat
                m = re.search(r'(GPL\d+)', os.path.basename(fpath), re.IGNORECASE)
                if m:
                    plat = m.group(1).upper()
                pid = plat.upper() if plat.upper().startswith('GPL') else plat

                # Load into platform_labels
                self.platform_labels[pid] = df.copy()
                self._rebuild_merged_labels()
                self.label_source_var.set("file")
                self._toggle_main_label_source()
                self._refresh_labels_display()
                self.enqueue_log(f"[LLM] Loaded {len(df):,} labels from {os.path.basename(fpath)} as {pid}")

                # Open LLM Curator window
                self._open_llm_curator()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV:\n{e}", parent=win)

        ttk.Button(load_frame, text="Load Labels → LLM Curator (Phase 3)",
                   command=_load_and_run_curator,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=6)

        load_desc = ttk.Frame(main)
        load_desc.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Label(load_desc,
            text="Phase 3 (LLM Curator): Upload labels CSV → LLM reviews cross-experiment labels, "
                 "proposes merges (AML → Acute Myeloid Leukemia), user reviews before applying.",
            font=('Segoe UI', 8), fg=AERO["muted"], justify=tk.LEFT, anchor='nw',
            wraplength=750).pack(fill=tk.X)

        # Store refs
        win._src_var = src_var
        win._plat_combo = plat_combo
        win._field_vars = field_vars

    def _llm_ext_toggle_source(self, src_var, plat_row, exp_row, file_row=None):
        """Toggle visibility of platform vs experiment vs file source controls."""
        src = src_var.get()
        if src == "platform":
            plat_row.grid()
            exp_row.grid_remove()
            if file_row: file_row.grid_remove()
        elif src == "experiments":
            plat_row.grid_remove()
            exp_row.grid()
            if file_row: file_row.grid_remove()
        elif src == "external":
            plat_row.grid_remove()
            exp_row.grid_remove()
            if file_row: file_row.grid()
        else:  # step1
            plat_row.grid_remove()
            exp_row.grid_remove()
            if file_row: file_row.grid_remove()
        # CRITICAL: update sample count when radio changes
        if hasattr(self, '_llm_update_count_fn') and self._llm_update_count_fn:
            try:
                self._llm_update_count_fn()
            except Exception:
                pass

    def _llm_ext_time_estimate(self, count):
        """Return formatted time estimate for LLM extraction."""
        if count <= 0:
            return "No samples selected"
        speed = 0.52  # samples/sec baseline
        eta = count / speed
        return f"{count:,} samples | ~{speed:.2f} smp/s | Estimated: {self._format_eta(eta)}"

    @staticmethod
    def _format_eta(seconds):
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def run_manual_labeling(self):
        """Runs manual labeling workflow - COMPLETE VERSION."""
        df_to_label = None
        source = ""
        
        if self.step2_data_df is not None:
            df_to_label = self.step2_data_df.copy()
            source = "external file"
        elif self.step1_results_df is not None and self.gse_to_keep_for_step2:
            df_to_label = self.step1_results_df[
                self.step1_results_df['series_id'].isin(self.gse_to_keep_for_step2)
            ].copy()
            source = f"{len(self.gse_to_keep_for_step2)} selected GSE(s)"
        else:
            messagebox.showerror(
                "No Data", 
                "No data available for labeling.\n\n"
                "Please either:\n"
                "- Complete Step 1 and save GSEs in Step 1.5, OR\n"
                "- Load an external CSV file",
                parent=self
            )
            return
        
        if df_to_label.empty:
            messagebox.showwarning("Empty Data", f"Data source ({source}) is empty.", parent=self)
            return
        
        label_win = tk.Toplevel(self)
        style_window(label_win)
        label_win.title("Manual Sample Labeling")
        label_win.geometry("800x600")
        try:
            _sw, _sh = label_win.winfo_screenwidth(), label_win.winfo_screenheight()
            label_win.geometry(f"800x600+{(_sw-800)//2}+{(_sh-600)//2}")
            label_win.minsize(500, 400)
        except Exception: pass
        label_win.transient(self)
        label_win.grab_set()
        
        ttk.Label(label_win, text="Manual Sample Labeling", font=('Segoe UI', 14, 'bold')).pack(pady=10)
        
        ttk.Label(label_win, text=f"Labeling {len(df_to_label):,} samples from {source}", font=('Segoe UI', 10)).pack(pady=5)
        
        inst_frame = ttk.Frame(label_win, relief="solid", borderwidth=1)
        inst_frame.pack(fill=tk.X, padx=20, pady=10)
        
        inst_text = (
            "Instructions:\n"
            "1. Select a category to label (Condition, Tissue, or Treatment)\n"
            "2. Enter the label value for ALL samples\n"
            "3. Click 'Apply Label' to add the classification\n\n"
            "Note: This will apply the same label to all samples.\n"
            "For per-sample labeling, use a spreadsheet editor."
        )
        
        ttk.Label(inst_frame, text=inst_text, justify=tk.LEFT, font=('Segoe UI', 9), foreground='gray').pack(padx=10, pady=10)
        
        cat_frame = labelframe(label_win, text="Select Category", padding=10)
        cat_frame.pack(fill=tk.X, padx=20, pady=10)
        
        category_var = tk.StringVar(value="Condition")
        
        ttk.Radiobutton(cat_frame, text="Condition (e.g., cancer, healthy, disease)", variable=category_var, value="Condition").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cat_frame, text="Tissue (e.g., liver, brain, blood)", variable=category_var, value="Tissue").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cat_frame, text="Treatment (e.g., drug A, control, untreated)", variable=category_var, value="Treatment").pack(anchor=tk.W, pady=2)
        
        input_frame = labelframe(label_win, text="Enter Label", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(input_frame, text="Label for ALL samples:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=2)
        
        label_entry = ttk.Entry(input_frame, width=50, font=('Segoe UI', 10))
        label_entry.pack(fill=tk.X, pady=5)
        label_entry.focus()
        
        preview_var = tk.StringVar(value="Preview: Condition = [your label here]")
        preview_label = ttk.Label(input_frame, textvariable=preview_var, font=('Segoe UI', 9, 'italic'), foreground='blue')
        preview_label.pack(pady=5)
        
        def update_preview(*args):
            cat = category_var.get()
            val = label_entry.get()
            if val:
                preview_var.set(f"Preview: {cat} = '{val}'")
            else:
                preview_var.set(f"Preview: {cat} = [your label here]")
        
        category_var.trace('w', update_preview)
        label_entry.bind('<KeyRelease>', update_preview)
        
        def apply_label():
            category = category_var.get()
            label_text = label_entry.get().strip()
            
            if not label_text:
                messagebox.showwarning("Input Required", "Please enter a label value.", parent=label_win)
                return
            
            col_name = f"{category}"
            df_to_label[col_name] = label_text
            
            self.step2_data_df = df_to_label
            
            self.enqueue_log(f"[Manual] Applied label '{label_text}' to column '{col_name}' for {len(df_to_label):,} samples")
            
            messagebox.showinfo(
                "Label Applied", 
                f"Successfully labeled {len(df_to_label):,} samples:\n\n"
                f"{col_name} = '{label_text}'\n\n"
                f"You can now:\n"
                f"- Apply additional labels (different categories)\n"
                f"- Use Gene Explorer or Compare Distributions",
                parent=label_win
            )
            
            self.step2_status_label.config(
                text=f"OK {len(df_to_label):,} samples labeled manually - Ready for analysis",
                foreground="green"
            )
            try:
                self._set_step_status(self._step2_frame, self._step2_title, "done")
            except Exception:
                pass
            
            if messagebox.askyesno("Continue?", "Add another label category?", parent=label_win):
                label_entry.delete(0, tk.END)
                label_entry.focus()
            else:
                label_win.destroy()
        
        btn_frame = ttk.Frame(label_win)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Apply Label", command=apply_label,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=10)

        ttk.Button(btn_frame, text="Cancel", command=label_win.destroy,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=10)
    
    def apply_semantic_clustering(self, df, threshold=0.4):
        """Semantic clustering disabled - returns df unchanged."""
        self.enqueue_log("[LLM] Semantic clustering skipped (disabled)")
        return df
        
    def show_gene_distribution_popup(self):
        """Opens the Gene Distribution Explorer popup - COMPLETE VERSION."""
        available = self._discover_available_platforms()
        if not self.gpl_datasets and not available:
            messagebox.showinfo(
                "No Platforms Available", 
                "No GPL platform data found.\n\n"
                "Either load a platform from the main window, or ensure\n"
                "platform data files (.csv.gz) are in your data directory.",
                parent=self
            )
            return
        
        if hasattr(self, 'gene_dist_popup_root') and self.gene_dist_popup_root is not None:
            try:
                if self.gene_dist_popup_root.winfo_exists():
                    self.gene_dist_popup_root.lift()
                    self.gene_dist_popup_root.focus_force()
                    return
            except (tk.TclError, Exception):
                pass
            self.gene_dist_popup_root = None
        
        self.gene_dist_popup_root = tk.Toplevel(self)
        style_window(self.gene_dist_popup_root)
        popup = self.gene_dist_popup_root
        popup.title("Gene Distribution Explorer")
        popup.geometry("1100x800")
        try:
            _sw, _sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
            popup.geometry(f"1100x800+{(_sw-1100)//2}+{(_sh-800)//2}")
            popup.minsize(500, 400)
        except Exception: pass
        popup.transient(self)
        
        popup._axis_map_dist_plot = {}
        popup._current_popup_figs = {}
        popup.rect_selectors = []
        popup.active_selections = {}
        popup.selector_colors = Plotter.get_distinct_colors(25)
        
        top_frame = ttk.Frame(popup, padding=10)
        top_frame.pack(fill=tk.X)
        
        inst_frame = ttk.Frame(top_frame)
        inst_frame.pack(fill=tk.X, pady=(0, 10))
        
        inst_label = ttk.Label(
            inst_frame,
            text="Instructions: Select genes and platforms below, then click 'Plot'. "
                 "DRAG rectangles on histograms to select expression ranges. "
                 "Multiple regions can be selected. Click legend items to change colors.",
            wraplength=850,
            font=('Segoe UI', 9),
            foreground=AERO['accent_dark'],
            background=AERO['accent_light'],
            padding=8,
            relief='flat'
        )
        inst_label.pack(fill=tk.X)
        
        plat_label_frame = labelframe(top_frame, text="Select Platforms", padding=5)
        plat_label_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            plat_label_frame, 
            text="Choose one or more platforms to compare:", 
            font=('Segoe UI', 9, 'italic'), 
            foreground='gray'
        ).pack(anchor=tk.W, pady=2)
        
        plat_check_frame = ttk.Frame(plat_label_frame)
        plat_check_frame.pack(fill=tk.X)
        
        # Collect all platforms and organize by species
        gpls_loaded = sorted(self.gpl_datasets.keys())
        available = self._discover_available_platforms()
        gpls_available = sorted(k for k in available.keys() if k not in self.gpl_datasets)
        popup.gpl_selection_vars = {}
        
        # Group by species
        from collections import OrderedDict
        species_groups = OrderedDict()  # {species: [(plat, is_loaded, info_text)]}
        
        for plat in gpls_loaded:
            sp = GPL_SPECIES.get(plat, 'other').title()
            sample_count = len(self.gpl_datasets[plat])
            if sp not in species_groups:
                species_groups[sp] = []
            species_groups[sp].append((plat, True, f"{plat} ({sample_count:,} samples)"))
        
        for plat in gpls_available:
            sp = GPL_SPECIES.get(plat, 'other').title()
            if sp not in species_groups:
                species_groups[sp] = []
            species_groups[sp].append((plat, False, f"{plat} (gene-only load)"))
        
        # Sort species: Human first, Mouse second, then alphabetical
        priority = {'Human': 0, 'Mouse': 1, 'Rat': 2}
        sorted_species = sorted(species_groups.keys(), key=lambda s: (priority.get(s, 99), s))
        
        row_idx = 0
        for species in sorted_species:
            platforms = species_groups[species]
            # Species header
            ttk.Label(plat_check_frame,
                      text=f"── {species} Platforms ──",
                      font=('Segoe UI', 9, 'bold'), foreground='#1565C0'
                      ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(6, 2))
            row_idx += 1
            col_idx = 0
            for plat, is_loaded, info_text in platforms:
                var = tk.BooleanVar(master=popup, value=False)
                cb = ttk.Checkbutton(plat_check_frame, text=info_text, variable=var)
                cb.grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=2)
                popup.gpl_selection_vars[plat] = var
                col_idx += 1
                if col_idx >= 3:
                    col_idx = 0; row_idx += 1
            if col_idx > 0:
                row_idx += 1
        
        # "Add Data Directory" button
        dir_row = ttk.Frame(plat_label_frame)
        dir_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(dir_row, text="+ Add Data Directory...",
                   command=lambda: self._add_data_dir_and_refresh(popup),
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Label(dir_row, text="Point to a folder with GPL .csv.gz files",
                  font=('Segoe UI', 8, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=5)

        # Batch correction option (for multi-platform comparisons)
        batch_row = ttk.Frame(plat_label_frame)
        batch_row.pack(fill=tk.X, pady=(4, 0))
        popup.batch_correct_var = tk.BooleanVar(master=popup, value=False)
        ttk.Checkbutton(batch_row, text="Apply batch correction (median centering) when comparing across platforms",
                        variable=popup.batch_correct_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(batch_row, text="recommended for cross-platform gene comparison",
                  font=('Segoe UI', 8, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=5)
        
        gene_label_frame = labelframe(top_frame, text="Enter Gene Symbols", padding=5)
        gene_label_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            gene_label_frame, 
            text="Gene symbols (comma-separated, e.g., TP53, BRCA1, EGFR, MYC):", 
            font=('Segoe UI', 9, 'italic'), 
            foreground='gray'
        ).pack(anchor=tk.W, pady=2)
        
        popup.current_gene_entry = ttk.Entry(gene_label_frame, width=70, font=('Segoe UI', 10))
        popup.current_gene_entry.pack(fill=tk.X, pady=2)
        popup.current_gene_entry.focus()

        # Overlay mode: all genes on same plot per platform
        popup.overlay_mode = tk.BooleanVar(master=popup, value=False)
        ttk.Checkbutton(gene_label_frame, text="Overlay all genes on same plot (per platform)",
                        variable=popup.overlay_mode).pack(anchor=tk.W, pady=(2, 0))
        
        # ── Label Source status (reads from main window) ──────────────
        label_info = ttk.Frame(top_frame)
        label_info.pack(fill=tk.X, pady=(2, 5))
        popup.label_status_indicator = ttk.Label(label_info, text="", font=("Segoe UI", 9))
        popup.label_status_indicator.pack(side=tk.LEFT, padx=6, fill=tk.X,
                                          expand=True)
        # Update indicator based on main window state
        if self.label_source_var.get() == "file" and self.platform_labels:
            # The platforms are already named one per row in the selector
            # above; spelling them out again makes a line whose length is set
            # by the data, and that line is cut off mid-name rather than
            # wrapped as soon as there are more than a few platforms.
            total = sum(len(df) for df in self.platform_labels.values())
            cols = [c for c, v in self.labels_col_vars.items() if v.get()]
            popup.label_status_indicator.config(
                text=f"Per-platform labels: {len(self.platform_labels)} "
                     f"platform(s), {total:,} samples, {len(cols)} label "
                     f"column(s) - LLM disabled",
                foreground=AERO["green_dark"], background=AERO["green_light"])
        elif self.label_source_var.get() == "file" and self.default_labels_df is not None:
            n = len(self.default_labels_df)
            cols = [c for c, v in self.labels_col_vars.items() if v.get()]
            popup.label_status_indicator.config(
                text=f"Labels loaded ({n:,} samples, {len(cols)} columns) - LLM disabled",
                foreground=AERO["green_dark"], background=AERO["green_light"])
        else:
            popup.label_status_indicator.config(
                text="Labels: LLM Extraction - change in main window",
                foreground=AERO["muted"], background=AERO["panel"])
        
        btn_frame = ttk.Frame(popup, padding=5)
        btn_frame.pack(fill=tk.X)
        
        plot_btn = ttk.Button(
            btn_frame,
            text="Plot Distributions",
            command=lambda: self._plot_histograms(popup),
            style="Add.TButton", cursor="hand2",
        )
        plot_btn.pack(side=tk.LEFT, padx=5)

        gene_list_btn = ttk.Button(
            btn_frame,
            text="Show Gene List",
            command=lambda: self._show_gene_list_for_selected_platforms(popup),
            style="Action.TButton", cursor="hand2",
        )
        gene_list_btn.pack(side=tk.LEFT, padx=5)

        popup.analyze_selection_btn = ttk.Button(
            btn_frame,
            text="Analyze Selected Range(s)",
            command=lambda: self._pre_analyze_dialog(popup),
            style="Warn.TButton", cursor="hand2",
            state=tk.DISABLED,
        )
        popup.analyze_selection_btn.pack(side=tk.LEFT, padx=5)

        popup.compare_btn = ttk.Button(
            btn_frame,
            text="Compare Regions",
            command=lambda: self._compare_regions_logic(popup),
            style="Primary.TButton", cursor="hand2",
            state=tk.DISABLED,
        )
        popup.compare_btn.pack(side=tk.LEFT, padx=5)

        # ── Highlight Region button (opens specification dialog) ──
        highlight_btn = ttk.Button(
            btn_frame,
            text="Highlight Region...",
            command=lambda: self._open_highlight_region_dialog(popup),
            style="Destructive.TButton", cursor="hand2",
        )
        highlight_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = ttk.Button(
            btn_frame,
            text="Clear Selections",
            command=lambda: self._clear_selections_logic(popup),
            style="Secondary.TButton", cursor="hand2",
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        export_btn = ttk.Button(
            btn_frame,
            text="\u2913 Export Plots",
            command=lambda: self._export_distribution_plots(popup),
            style="Secondary.TButton", cursor="hand2",
        )
        export_btn.pack(side=tk.LEFT, padx=5)

        popup.selection_label = ttk.Label(
            btn_frame,
            text="No regions selected",
            font=('Segoe UI', 9, 'italic'),
            foreground=AERO['muted']
        )
        popup.selection_label.pack(side=tk.RIGHT, padx=10)
        
        separator = ttk.Separator(popup, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=5)
        
        popup.gene_dist_out_frame = ttk.Frame(popup)
        popup.gene_dist_out_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        placeholder = ttk.Label(
            popup.gene_dist_out_frame,
            text="Enter gene symbols above and click 'Plot Distributions' to begin.\n\n"
                 "After plotting, drag rectangles on the histograms to select expression ranges.\n"
                 "Selected regions will be highlighted in color.\n\n"
                 "Click 'Analyze Selected Range(s)' to classify samples using LLM.",
            font=('Segoe UI', 10),
            foreground='gray',
            justify=tk.CENTER
        )
        placeholder.pack(expand=True)
        
        def _on_close_popup_handler():
            """Enhanced cleanup on popup close."""
            import matplotlib.pyplot as plt
            import gc
            
            try:
                for key in list(popup._current_popup_figs.keys()):
                    try:
                        fig, canv_widget, tool = popup._current_popup_figs.pop(key)
                        if canv_widget and canv_widget.winfo_exists():
                            canv_widget.destroy()
                        if tool and tool.winfo_exists():
                            tool.destroy()
                        plt.close(fig)
                    except:
                        pass
                        
                popup.rect_selectors.clear()
                popup.active_selections.clear()
                gc.collect()
            except:
                pass
            finally:
                # ALWAYS clear the reference so window can reopen
                try:
                    popup.destroy()
                except:
                    pass
                self.gene_dist_popup_root = None
                
        popup.protocol("WM_DELETE_WINDOW", _on_close_popup_handler)
        popup.current_gene_entry.bind('<Return>', lambda e: self._plot_histograms(popup))
        self._fit_window(popup, 1100, 800)
        
    # ── Main Window Label Source helpers ───────────────────────────────
    def _toggle_main_label_source(self):
        """Show/hide label controls based on label source radio."""
        if self.label_source_var.get() == "file":
            self.labels_file_row.pack(fill=tk.X, pady=3)
            self.labels_plat_frame.pack(fill=tk.X, pady=2)
            self.labels_col_frame.pack(fill=tk.X, pady=2)
            self._refresh_labels_display()
        else:
            self.labels_file_row.pack_forget()
            self.labels_plat_frame.pack_forget()
            self.labels_col_frame.pack_forget()
            self.labels_status_lbl.config(
                text="LLM mode: samples labeled via geo_label_extractor during analysis.",
                foreground="gray")

    def _add_label_file(self):
        """Browse for a single label file, auto-detect GPL from filename, add to platform_labels."""
        if self._dialog_active:
            return
        self._dialog_active = True
        try:
            if not self.winfo_exists():
                return
            paths = filedialog.askopenfilenames(
                title="Select Label File(s) - GPL ID will be detected from filename",
                filetypes=[("CSV files", "*.csv *.csv.gz"), ("All files", "*.*")],
                parent=self
            )
        except tk.TclError:
            return
        finally:
            self._dialog_active = False
        if not paths:
            return
        for p in paths:
            self._load_single_label_file(p)

    def _load_single_label_file(self, fpath, skip_auto_check=False):
        """Load one label CSV, auto-detect GPL from filename, add to platform_labels.
        
        skip_auto_check: if True, don't schedule expression data check dialog
                         and suppress messageboxes. Used when loading from folder.
        """
        fname = os.path.basename(fpath)
        # Detect GPL ID from filename
        m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
        if not m:
            if skip_auto_check:
                self.enqueue_log(f"[Labels] SKIP {fname}: No GPL ID in filename")
            else:
                messagebox.showwarning(
                    "No GPL ID Found",
                    f"Could not detect a GPL ID in the filename:\n{fname}\n\n"
                    f"Please name files with the platform ID, e.g.:\n"
                    f"  GPL570_labels.csv\n"
                    f"  conditions_GPL96.csv.gz\n"
                    f"  matrix_condition_unannotated_GPL10558.csv.gz",
                    parent=self)
            return

        plat_id = m.group(1).upper()
        self.enqueue_log(f"[Labels] Loading {fname} -> {plat_id}")

        try:
            comp = 'gzip' if fpath.lower().endswith('.gz') else None
            df = pd.read_csv(fpath, compression=comp, low_memory=False)
            self.enqueue_log(f"[Labels]   Read OK: {len(df):,} rows, {len(df.columns)} columns")

            # Detect GSM column
            gsm_col = None
            for c in df.columns:
                if c.lower().strip() in ('gsm', 'sample', 'sample_id', 'geo_accession'):
                    gsm_col = c
                    break
            if gsm_col is None:
                first = df.iloc[:, 0].astype(str)
                if first.str.upper().str.startswith('GSM').mean() > 0.5:
                    gsm_col = df.columns[0]
            if gsm_col is None:
                self.enqueue_log(f"[Labels]   FAIL {fname}: No GSM/sample column found "
                                 f"(columns: {list(df.columns[:10])})")
                if not skip_auto_check:
                    messagebox.showerror("No GSM Column",
                                         f"{fname}: No GSM/sample column found.", parent=self)
                return

            df = df.rename(columns={gsm_col: 'GSM'})
            df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()

            # Backward compat: strip Classified_ prefix from old label files
            rename_strip = {}
            for c in df.columns:
                if c.startswith('Classified_'):
                    rename_strip[c] = c.replace('Classified_', '', 1)
            if rename_strip:
                df = df.rename(columns=rename_strip)
                self.enqueue_log(f"[Labels]   Stripped 'Classified_' prefix from {len(rename_strip)} columns")

            # Detect label columns: non-numeric OR low-cardinality columns
            label_cols = []
            for c in df.columns:
                if c == 'GSM':
                    continue
                # Accept: object/string columns with >1 unique value
                if df[c].dtype == 'object':
                    if df[c].nunique() > 1:
                        label_cols.append(c)
                    elif df[c].nunique() == 1:
                        # Single-value label column - still useful for identification
                        label_cols.append(c)
                # Accept: numeric columns with very low cardinality (likely category codes)
                elif df[c].nunique() < 50 and df[c].nunique() > 1:
                    label_cols.append(c)

            if not label_cols:
                self.enqueue_log(f"[Labels]   FAIL {fname}: No label columns detected "
                                 f"(dtypes: {dict(df.dtypes.value_counts())})")
                if not skip_auto_check:
                    messagebox.showerror("No Label Columns",
                                         f"{fname}: No label columns found.", parent=self)
                return

            # User-provided labels are kept AS-IS - no harmonization.
            # Harmonization is only applied to LLM-extracted labels.

            # Store per-platform
            self.platform_labels[plat_id] = df
            self.enqueue_log(f"[Labels]   STORED {plat_id}: {len(df):,} samples, "
                             f"{len(label_cols)} label columns: {label_cols}")

            # When loading from folder (skip_auto_check=True), defer all UI
            # updates to _browse_labels_folder which does them once at the end.
            if not skip_auto_check:
                self._rebuild_merged_labels()
                self._refresh_labels_display()
                self.label_source_var.set("file")
                self._toggle_main_label_source()
                self.after(200, lambda p=plat_id: self._ensure_expression_data_for_labels(p))

        except Exception as e:
            self.enqueue_log(f"[Labels]   EXCEPTION {fname}: {e}")
            if not skip_auto_check:
                try:
                    messagebox.showerror("Load Error", f"{fname}:\n{e}", parent=self)
                except tk.TclError:
                    pass

    def _browse_labels_folder(self):
        """Browse for folder containing multiple per-platform label files."""
        import glob as _glob

        if self._dialog_active:
            return
        self._dialog_active = True
        try:
            if not self.winfo_exists():
                return
            d = filedialog.askdirectory(
                title="Select Folder Containing Per-Platform Label Files",
                parent=self
            )
        except tk.TclError:
            return
        finally:
            self._dialog_active = False
        if not d:
            return

        self.enqueue_log(f"[Labels] Scanning folder: {d}")

        files = []
        for ext in ('*.csv', '*.csv.gz', '*.CSV', '*.CSV.GZ'):
            files.extend(_glob.glob(os.path.join(d, ext)))
        files = sorted(set(files))

        if not files:
            messagebox.showinfo("No Files",
                                f"No CSV files found in:\n{d}", parent=self)
            return

        self.enqueue_log(f"[Labels] Found {len(files)} CSV file(s) in folder")

        # Load each file, track results
        loaded_plats = []
        skipped = []
        failed = []
        plats_before = set(self.platform_labels.keys())

        for fpath in files:
            fname = os.path.basename(fpath)
            m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
            if not m:
                skipped.append(fname)
                self.enqueue_log(f"[Labels]   SKIP {fname} (no GPL ID in name)")
                continue

            plat_id = m.group(1).upper()
            self.enqueue_log(f"[Labels]   Loading {fname} -> {plat_id} ...")
            try:
                self._load_single_label_file(fpath, skip_auto_check=True)
            except Exception as e:
                failed.append(f"{fname}: {e}")
                self.enqueue_log(f"[Labels]   FAIL {fname}: {e}")
                continue

            if plat_id in self.platform_labels:
                if plat_id not in loaded_plats:
                    loaded_plats.append(plat_id)
                self.enqueue_log(f"[Labels]   OK {plat_id} stored "
                                 f"({len(self.platform_labels[plat_id]):,} samples)")
            else:
                failed.append(f"{fname}: stored but not found (label/GSM detection failed)")
                self.enqueue_log(f"[Labels]   FAIL {fname}: not stored in platform_labels")

        # Summary
        new_plats = set(self.platform_labels.keys()) - plats_before
        self.enqueue_log(f"[Labels] Folder results: {len(loaded_plats)} loaded, "
                         f"{len(skipped)} skipped, {len(failed)} failed")
        self.enqueue_log(f"[Labels] platform_labels now has: "
                         f"{sorted(self.platform_labels.keys())}")

        if not loaded_plats:
            msg = "No label files were successfully loaded.\n\n"
            if skipped:
                msg += f"Skipped (no GPL in name): {len(skipped)}\n"
            if failed:
                msg += f"\nFailed:\n" + "\n".join(f"  - {f}" for f in failed[:10])
            msg += "\n\nName files like: GPL570_labels.csv, GPL96_conditions.csv.gz"
            messagebox.showinfo("No Labels Loaded", msg, parent=self)
            return

        # Rebuild UI once after all files loaded
        self._rebuild_merged_labels()
        self.label_source_var.set("file")
        self._toggle_main_label_source()
        self._refresh_labels_display()

        # Success message with details
        msg = f"Loaded {len(loaded_plats)} platform(s):\n"
        for p in loaded_plats:
            n = len(self.platform_labels.get(p, []))
            msg += f"  {p}: {n:,} samples\n"
        if skipped:
            msg += f"\nSkipped (no GPL in name): {len(skipped)}"
        if failed:
            msg += f"\nFailed: {len(failed)}"
            for f in failed[:5]:
                msg += f"\n  - {f}"

        messagebox.showinfo("Labels Loaded", msg, parent=self)

        # Batch auto-check for expression data
        missing_expr = [p for p in loaded_plats if p not in self.gpl_datasets]
        if missing_expr:
            self.after(300, lambda plats=missing_expr: self._batch_ensure_expression(plats))

    def _auto_load_labels(self):
        """Auto-scan {data_dir}/labels/ on startup and load any label files found.
        This ensures labels from previous sessions are available immediately.
        """
        labels_dir = os.path.join(self.data_dir, "labels")
        if not os.path.isdir(labels_dir):
            return

        found = 0
        for fname in sorted(os.listdir(labels_dir)):
            fpath = os.path.join(labels_dir, fname)
            if not os.path.isfile(fpath):
                continue
            fn_lower = fname.lower()
            # Only load clean label files (not _phase1, not _raw)
            if not (fn_lower.endswith('.csv') or fn_lower.endswith('.csv.gz')):
                continue
            if '_phase1' in fn_lower or '_raw' in fn_lower:
                continue
            if 'label' not in fn_lower and 'classified' not in fn_lower:
                continue
            # Extract GPL ID from filename
            m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
            if not m:
                continue
            gpl_id = m.group(1).upper()
            if gpl_id in self.platform_labels:
                continue  # already loaded

            try:
                df = pd.read_csv(fpath, low_memory=False)
                if 'GSM' not in df.columns and 'gsm' in df.columns:
                    df.rename(columns={'gsm': 'GSM'}, inplace=True)
                if 'GSM' in df.columns and len(df) > 0:
                    self.platform_labels[gpl_id] = df
                    found += 1
            except Exception as e:
                print(f"[Labels] Failed to auto-load {fname}: {e}")

        if found > 0:
            self._rebuild_merged_labels()
            self.label_source_var.set("file")
            self._toggle_main_label_source()
            self._refresh_labels_display()
            self.enqueue_log(f"[Labels] Auto-loaded {found} label file(s) from {labels_dir}: "
                             f"{', '.join(sorted(self.platform_labels.keys()))}")

    def _set_labels_directory(self):
        """Set a persistent labels directory and load all label files from it."""
        if self._dialog_active:
            return
        self._dialog_active = True
        try:
            if not self.winfo_exists():
                return
            d = filedialog.askdirectory(
                title="Select Labels Directory (contains *_labels.csv files)",
                initialdir=os.path.join(self.data_dir, "labels"))
        except tk.TclError:
            return
        finally:
            self._dialog_active = False
        if not d:
            return
        # Store as preferred labels directory
        self._labels_directory = d
        self.enqueue_log(f"[Labels] Labels directory set: {d}")

        # Scan for label files
        found = 0
        loaded_plats = []
        for fname in sorted(os.listdir(d)):
            fpath = os.path.join(d, fname)
            if not os.path.isfile(fpath):
                continue
            fn_lower = fname.lower()
            if fn_lower.endswith('.csv') or fn_lower.endswith('.csv.gz'):
                if 'label' in fn_lower or 'classified' in fn_lower:
                    try:
                        self._load_single_label_file(fpath, skip_auto_check=True)
                        m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
                        if m:
                            loaded_plats.append(m.group(1).upper())
                        found += 1
                    except Exception as e:
                        self.enqueue_log(f"[Labels] Failed to load {fname}: {e}")

        if found > 0:
            # Deferred UI rebuild: do ONCE after all files loaded
            self._rebuild_merged_labels()
            self.label_source_var.set("file")
            self._toggle_main_label_source()
            self._refresh_labels_display()
            messagebox.showinfo(
                "Labels Loaded",
                f"Loaded {found} label file(s) from:\n{d}\n\n"
                f"Platforms with labels: {', '.join(sorted(self.platform_labels.keys()))}",
                parent=self)
            # One batch auto-check for all missing expression platforms
            missing_expr = [p for p in loaded_plats if p in self.platform_labels and p not in self.gpl_datasets]
            if missing_expr:
                self.after(300, lambda plats=missing_expr: self._batch_ensure_expression(plats))
        else:
            messagebox.showinfo(
                "No Label Files Found",
                f"No label files found in:\n{d}\n\n"
                f"Expected files like: GPL570_labels.csv, GPL96_labels.csv\n"
                f"Use 'LLM Extraction' to generate labels first.",
                parent=self)

    def _clear_all_labels(self):
        """Clear all loaded per-platform labels."""
        if not self.platform_labels:
            return
        n = len(self.platform_labels)
        if not messagebox.askyesno("Clear Labels",
                                    f"Remove all {n} loaded label file(s)?", parent=self):
            return
        self.platform_labels.clear()
        self.default_labels_df = None
        self._refresh_labels_display()
        self.enqueue_log("[Labels] All labels cleared.")

    def _open_llm_curator(self):
        """Cross-experiment label curation is now geo_label_extractor Stage-2.

        GeneVariate's own LLM curator was removed. Cross-experiment label
        harmonization/normalization is provided by the vendored
        geo_label_extractor normalization stage (MeSH / Cellosaurus / BioLORD),
        which needs prebuilt reference artifacts. Inform the user and return.
        """
        from genevariate.core import geo_extract_driver as _drv
        ref = _drv.phase2_reference_status(
            vocab=os.environ.get("GEO_VOCAB", ""),
            index=os.environ.get("GEO_INDEX", ""),
            cellosaurus=os.environ.get("GEO_CELLOSAURUS", ""))
        messagebox.showinfo(
            "Label Curation → geo_label_extractor Stage-2",
            "Cross-experiment label harmonization is now handled by the vendored "
            "geo_label_extractor normalization stage (MeSH / Cellosaurus / BioLORD).\n\n"
            + ("Reference artifacts are available - run `genevariate-llm-extract` "
               "with --vocab/--index/--cellosaurus to normalize.\n"
               if ref["available"] else
               "The Stage-2 reference artifacts (vocab / index / cellosaurus) are "
               "not built on this machine, so this step is unavailable here. The "
               "Stage-1 labels from geo_label_extractor are already loaded.\n")
            + "\nGeneVariate's former built-in LLM curator has been removed.",
            parent=self)
        return

    def _open_label_entities(self):
        """Show what each loaded label was resolved to, and by which vocabulary."""
        from genevariate.gui.windows.label_entities_window import (
            LabelEntitiesWindow)

        if not self.platform_labels:
            messagebox.showinfo(
                "Label Entities",
                "No labels are loaded. Add a label file, or run the extractor, "
                "and the entities its normalization pass resolved will be "
                "listed here.",
                parent=self)
            return
        win = LabelEntitiesWindow(self, self.platform_labels)
        win.focus_set()
        return win

    def _rebuild_merged_labels(self):
        """Rebuild self.default_labels_df from all platform_labels."""
        if not self.platform_labels:
            self.default_labels_df = None
            return
        all_dfs = []
        for plat_id, df in self.platform_labels.items():
            tagged = df.copy()
            tagged['_platform'] = plat_id
            all_dfs.append(tagged)
        self.default_labels_df = pd.concat(all_dfs, ignore_index=True)

    def _curated_sample_labels(self):
        """``{GSM: 'Field: value; Field: value'}`` from the live label store.

        Labels live in ``platform_labels`` (per-platform frames, merged into
        ``default_labels_df``). The enrichment and pseudo-cohort windows used
        to read a ``sample_labels`` attribute that nothing has ever assigned,
        so they always saw zero labels no matter what the user had loaded.
        """
        if self.default_labels_df is None or self.default_labels_df.empty:
            self._rebuild_merged_labels()
        df = self.default_labels_df
        if df is None or df.empty or 'GSM' not in df.columns:
            return {}
        skip = {'GSM', '_platform', 'series_id', 'gpl', 'GPL', 'title'}
        cols = [c for c in df.columns if c not in skip]
        if not cols:
            return {}
        blank = {'', 'nan', 'none', 'n/a', 'na', 'ns', 'not specified'}
        out = {}
        for row in df[['GSM'] + cols].itertuples(index=False, name=None):
            gsm = str(row[0]).strip().upper()
            if not gsm or gsm in blank:
                continue
            parts = [f"{c}: {str(v).strip()}"
                     for c, v in zip(cols, row[1:])
                     if v is not None and str(v).strip().lower() not in blank]
            if parts:
                out[gsm] = "; ".join(parts)
        return out

    def _refresh_labels_display(self):
        """Update the per-platform label status display and column checkboxes."""
        # Clear old platform status
        self.labels_plat_frame.clear()

        # A normalized label file records, beside each Tissue, whether the
        # value resolved to a piece of anatomy or to a catalogued cell line.
        # That is a distinction no analysis can recover from the value alone,
        # so it is turned into a column here -- once, wherever the labels came
        # from -- rather than left in an accession no window reads.
        for _df in self.platform_labels.values():
            try:
                label_entities.add_kind_columns(_df)
            except Exception as e:
                self.enqueue_log(f"[Labels] entity links unreadable: {e}")

        if not self.platform_labels:
            self.labels_status_lbl.config(
                text="No label files loaded. Click '+ Add Label File...' to add per-platform labels.",
                foreground="gray")
            for w in self.labels_col_frame.winfo_children():
                w.destroy()
            self.labels_col_vars.clear()
            return

        # Show per-platform status with remove buttons. One row each would put
        # a dozen platforms' worth of blank space between this panel and the
        # controls under it, so they wrap into columns instead.
        for plat_id in sorted(self.platform_labels.keys()):
            row = self.labels_plat_frame.add(
                ttk.Frame(self.labels_plat_frame))
            n = len(self.platform_labels[plat_id])
            loaded_as_expr = plat_id in (self.gpl_datasets or {})
            fg = AERO["green_dark"] if loaded_as_expr else AERO["muted"]
            marker = "\u2611" if loaded_as_expr else "\u2610"
            cols = [c for c in self.platform_labels[plat_id].columns
                    if c not in ('GSM', '_platform')]
            ttk.Label(row, text=f"  {marker} {plat_id}: {n:,} samples, "
                      f"{len(cols)} columns",
                      font=("Segoe UI", 9), foreground=fg).pack(side=tk.LEFT)
            ttk.Button(row, text="\u2715", command=lambda p=plat_id: self._remove_platform_label(p),
                       width=3, style="Destructive.TButton").pack(side=tk.LEFT, padx=4)

        # Every column of every label file is a candidate - a file that names
        # its tissue column something the program has never seen must still be
        # usable, so nothing is dropped here. Which of them are labels is the
        # user's decision, taken in the picker below.
        all_label_cols = set()
        for df in self.platform_labels.values():
            for c in df.columns:
                if c not in ('GSM', '_platform'):
                    all_label_cols.add(c)
        all_label_cols = sorted(all_label_cols)

        # Distinct non-empty values per column, so the picker can tell a label
        # (a few repeated values) from an identifier (one value per sample).
        self._labels_col_levels = {}
        n_rows = 0
        for c in all_label_cols:
            seen = set()
            for df in self.platform_labels.values():
                if c not in df.columns:
                    continue
                try:
                    vals = (df[c].dropna().astype(str).str.strip()
                            .replace("", pd.NA).dropna().unique())
                    seen.update(vals.tolist())
                except Exception:
                    pass
            self._labels_col_levels[c] = len(seen)
        n_rows = sum(len(df) for df in self.platform_labels.values())

        # Populate column selection (keep existing selections)
        old_selections = {c: v.get() for c, v in self.labels_col_vars.items()}
        for w in self.labels_col_frame.winfo_children():
            w.destroy()
        self.labels_col_vars.clear()

        if all_label_cols:
            detected = set(semantic_label_columns(all_label_cols))
            if not detected:
                detected = {c for c in all_label_cols
                            if 2 <= self._labels_col_levels[c]
                            <= max(2, min(50, n_rows - 1))}
            for c in all_label_cols:
                val = old_selections.get(c, c in detected)
                self.labels_col_vars[c] = tk.BooleanVar(value=val)
            # One line plus a dialog, never one checkbutton per column: with a
            # dozen columns a single row runs off the edge of the window and
            # the ones past the edge cannot be reached at all.
            ttk.Label(self.labels_col_frame, text="Use columns:",
                      font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=2)
            ttk.Label(self.labels_col_frame,
                      textvariable=self._labels_cols_summary,
                      foreground=AERO["text"],
                      font=("Segoe UI", 9)).pack(side=tk.LEFT)
            ttk.Button(self.labels_col_frame, text="Choose\u2026",
                       command=self._choose_labels_columns,
                       style="Secondary.TButton").pack(side=tk.LEFT, padx=(8, 0))
            self._refresh_labels_cols_summary()

        # Update status. The platform names are not repeated here: every one
        # of them is already listed on its own row above, and spelling them
        # out again makes a line that runs past the edge of the window and is
        # cut off mid-name as soon as there are more than a handful.
        total = sum(len(df) for df in self.platform_labels.values())
        self.labels_status_lbl.config(
            text=f"OK {len(self.platform_labels)} platform(s)  |  "
                 f"{total:,} total samples  |  LLM labeling disabled",
            foreground="#1B5E20")

    def _refresh_labels_cols_summary(self):
        """One-line description of the current label-column selection."""
        chosen = [c for c, v in self.labels_col_vars.items() if v.get()]
        total = len(self.labels_col_vars)
        if not chosen:
            self._labels_cols_summary.set(f"none of {total} selected")
            return
        shown = ", ".join(chosen[:3])
        if len(chosen) > 3:
            shown += f", +{len(chosen) - 3} more"
        self._labels_cols_summary.set(f"{len(chosen)} of {total}: {shown}")

    def _choose_labels_columns(self):
        """Open the scrollable checklist of every column in the label files."""
        if not self.labels_col_vars:
            return
        self._labels_cols_dialog = column_picker_dialog(
            self, self.labels_col_vars, self._labels_col_levels,
            n_rows=sum(len(df) for df in self.platform_labels.values()),
            title="Label columns to use",
            on_close=self._refresh_labels_cols_summary)

    def _remove_platform_label(self, plat_id):
        """Remove a single platform's labels."""
        if plat_id in self.platform_labels:
            del self.platform_labels[plat_id]
            self._rebuild_merged_labels()
            self._refresh_labels_display()
            self.enqueue_log(f"[Labels] Removed {plat_id}")

    def _get_labels_for_gsms(self, gsms, platform=None):
        """Extract labels from loaded label files for a set of GSMs.
        If platform is specified, uses that platform's label file first.
        Falls back to merged default_labels_df.
        Returns DataFrame with GSM + label columns, or empty DF."""
        sel_cols = [c for c, v in self.labels_col_vars.items() if v.get()]
        if not sel_cols:
            return pd.DataFrame()

        gsm_set = set(str(g).strip().upper() for g in gsms)

        # Try platform-specific labels first
        df = None
        if platform and platform in self.platform_labels:
            df = self.platform_labels[platform]
        elif self.default_labels_df is not None:
            df = self.default_labels_df

        if df is None:
            return pd.DataFrame()

        # Only keep columns that exist in this specific df
        avail_cols = [c for c in sel_cols if c in df.columns]
        if not avail_cols:
            return pd.DataFrame()

        sub = df[df['GSM'].isin(gsm_set)][['GSM'] + avail_cols].copy()
        return sub

    # ── Popup-level wrappers (no longer needed - main window handles labels) ──
    def _ensure_expression_data_for_labels(self, gpl_id):
        """Auto-check: when labels arrive, verify expression data exists.
        Uses simple messagebox (YES/NO/CANCEL) instead of custom Toplevel.
        Protected by _dialog_active guard to prevent grab conflicts.
        """
        try:
            if self._dialog_active:
                # Another dialog is open; retry later
                self.after(500, lambda p=gpl_id: self._ensure_expression_data_for_labels(p))
                return
            if not self.winfo_exists():
                return
            if not gpl_id or not str(gpl_id).upper().startswith('GPL'):
                return

            gpl_id = str(gpl_id).upper()

            # Already loaded?
            if gpl_id in self.gpl_datasets:
                n_expr = len(self.gpl_datasets[gpl_id])
                n_labels = len(self.platform_labels.get(gpl_id, []))
                self.enqueue_log(
                    f"[Auto] {gpl_id}: labels ({n_labels:,}) + expression ({n_expr:,}) -> Ready.")
                return

            label_df = self.platform_labels.get(gpl_id)
            if label_df is None or label_df.empty:
                return

            n_labeled = len(label_df)
            label_cols = [c for c in label_df.columns if c not in ('GSM', '_platform')]

            self.enqueue_log(
                f"[Auto] Labels for {gpl_id}: {n_labeled:,} samples. "
                f"Expression data NOT loaded.")

            self._dialog_active = True
            try:
                choice = messagebox.askyesno(
                    f"Load Expression Data? - {gpl_id}",
                    f"Labels loaded for {gpl_id}: {n_labeled:,} samples\n"
                    f"Columns: {', '.join(label_cols[:5])}\n\n"
                    f"Expression data is needed for gene-level analysis\n"
                    f"(Gene Explorer, Compare Distributions).\n\n"
                    f"Load expression data for {gpl_id} now?",
                    parent=self)
            except tk.TclError:
                return
            finally:
                self._dialog_active = False

            if choice:
                self.enqueue_log(f"[Auto] User chose: Load {gpl_id}")
                self._smart_load_gpl(gpl_id)
            else:
                self.enqueue_log(f"[Auto] User skipped expression loading for {gpl_id}.")

        except Exception as e:
            self.enqueue_log(f"[Auto] Warning: expression check failed: {e}")
            import traceback
            self.enqueue_log(traceback.format_exc())

    def _batch_ensure_expression(self, platform_list):
        """Ask about expression data for MULTIPLE platforms in ONE dialog.
        Used when loading labels from a folder (avoids per-file dialog cascade).
        Protected by _dialog_active guard to prevent grab conflicts.
        """
        try:
            if self._dialog_active:
                # Another dialog is open; retry later
                self.after(500, lambda pl=platform_list: self._batch_ensure_expression(pl))
                return
            if not self.winfo_exists():
                return
            # Filter to platforms that still need expression data
            missing = [p for p in platform_list
                       if p in self.platform_labels and p not in self.gpl_datasets]
            if not missing:
                return

            plat_info = []
            for p in missing:
                n = len(self.platform_labels.get(p, []))
                plat_info.append(f"  {p}: {n:,} samples")
            info_str = "\n".join(plat_info)

            self._dialog_active = True
            try:
                choice = messagebox.askyesno(
                    f"Load Expression Data? - {len(missing)} Platform(s)",
                    f"Labels loaded but expression data missing for:\n\n"
                    f"{info_str}\n\n"
                    f"Expression data is needed for gene-level analysis.\n\n"
                    f"Load expression data for these platforms now?",
                    parent=self)
            except tk.TclError:
                return
            finally:
                self._dialog_active = False

            if choice:
                for p in missing:
                    self.enqueue_log(f"[Auto] Loading {p}...")
                    self._smart_load_gpl(p)
            else:
                self.enqueue_log(f"[Auto] Skipped expression loading for: {', '.join(missing)}")
        except tk.TclError:
            pass  # window destroyed
        except Exception as e:
            self.enqueue_log(f"[Auto] Batch expression check failed: {e}")

    def _export_distribution_plots(self, popup, out_dir=None):
        """Write every distribution plot the explorer currently shows to disk.

        The plots live in their own windows, one per gene and platform, so
        there is no single frame an "Export All" could hang off; this walks
        the popup's own figure registry instead.
        """
        figs = getattr(popup, "_current_popup_figs", {}) or {}
        if not figs:
            messagebox.showinfo("Nothing to export",
                                "Plot at least one distribution first.",
                                parent=popup)
            return []
        if out_dir is None:
            out_dir = filedialog.askdirectory(title="Export folder", parent=popup)
            if not out_dir:
                return []
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        # The drag selector is a live editing widget drawn on the axes, not a
        # result; an exported figure must show the analysis, not the handles.
        selectors = [s for s in getattr(popup, "rect_selectors", []) or []
                     if getattr(s, "artists", None)]
        hidden = []
        for sel in selectors:
            try:
                for art in sel.artists:
                    if art.get_visible():
                        art.set_visible(False)
                        hidden.append(art)
            except Exception:
                pass
        try:
            for key, entry in figs.items():
                fig = entry[0] if isinstance(entry, tuple) else entry
                path = out / f"distribution_{key}.png"
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    written.append(path)
                except Exception as exc:
                    self.enqueue_log(f"[Gene Explorer] Export failed for {key}: {exc}")
        finally:
            for art in hidden:
                try:
                    art.set_visible(True)
                except Exception:
                    pass
        messagebox.showinfo("Exported",
                            f"Saved {len(written)} plot(s) to:\n{out}",
                            parent=popup)
        return written

    def _plot_histograms(self, popup):
        """Plots histograms for selected genes - SEPARATE WINDOWS or OVERLAY mode."""
        if not popup or not popup.winfo_exists():
            return
        
        gene_input_str = popup.current_gene_entry.get().strip()
        if not gene_input_str:
            messagebox.showerror("Input Required", "Please enter at least one gene symbol.", parent=popup)
            return
        
        popup.current_genes = [g.strip().upper() for g in gene_input_str.split(',') if g.strip()]
        
        sel_plats = [p for p, v in popup.gpl_selection_vars.items() if v.get()]
        
        if not sel_plats:
            messagebox.showerror("Platform Required", "Please select at least one platform to plot.", parent=popup)
            return

        # ── Quick Gene Load: for platforms not fully loaded, load only requested genes ──
        genes = popup.current_genes
        for plat in sel_plats:
            if plat not in self.gpl_datasets:
                # Not fully loaded - try gene-only quick load
                self.enqueue_log(f"[Gene Explorer] {plat} not fully loaded - trying quick gene load...")
                self.status_label.config(text=f"Quick-loading {plat} genes...", foreground=AERO["accent"])
                self.update_idletasks()
                ok = self._quick_load_genes(plat, genes)
                if not ok:
                    self.enqueue_log(f"[Gene Explorer] {plat}: quick load failed - genes may not be found")
                self.status_label.config(text="Ready", foreground="gray")

        # Check overlay mode
        if popup.overlay_mode.get() and len(popup.current_genes) > 1:
            self._plot_histograms_overlay(popup, sel_plats)
            return
        
        num_genes = len(popup.current_genes)
        num_plats = len(sel_plats)
        
        # ── Batch correction pre-computation (median centering) ────
        batch_offsets = {}   # {plat: offset_to_subtract}
        batch_corrected = getattr(popup, 'batch_correct_var', None)
        use_batch_correction = (batch_corrected and batch_corrected.get()
                                and len(sel_plats) > 1)
        if use_batch_correction:
            plat_medians = {}
            for plat in sel_plats:
                df = self.gpl_datasets.get(plat)
                gmap = self.gpl_gene_mappings.get(plat, {})
                # Fallback to gene cache
                if df is None and plat in self.gpl_gene_cache:
                    df = self.gpl_gene_cache[plat]
                    gmap = self.gpl_gene_mappings.get(f"_cache_{plat}", {})
                if df is None:
                    continue
                vals = []
                for gene in popup.current_genes:
                    col = gmap.get(gene)
                    if col and col in df.columns:
                        v = pd.to_numeric(df[col], errors='coerce').dropna().values
                        vals.extend(v.tolist())
                plat_medians[plat] = np.nanmedian(vals) if vals else 0
            global_median = np.nanmedian(list(plat_medians.values())) if plat_medians else 0
            for plat in sel_plats:
                batch_offsets[plat] = plat_medians.get(plat, 0) - global_median
            self.enqueue_log(f"[Gene Explorer] Batch correction (median centering) "
                             f"offsets: {', '.join(f'{p}: {batch_offsets[p]:+.3f}' for p in sel_plats)}")
        
        self.enqueue_log(f"[Gene Explorer] Creating {num_genes * num_plats} separate distribution window(s)..."
                         + (" [BATCH CORRECTED]" if use_batch_correction else ""))
        
        # Create separate window for each gene-platform combination
        window_count = 0
        failed_genes = []
        for gene in popup.current_genes:
            for plat in sel_plats:
                # Create new toplevel window for this gene-platform
                plot_win = tk.Toplevel(popup)
                style_window(plot_win)
                plot_win.title(f"{gene} - {plat} Distribution")
                plot_win.geometry("1100x800")
                try:
                    _sw, _sh = plot_win.winfo_screenwidth(), plot_win.winfo_screenheight()
                    plot_win.geometry(f"1100x800+{(_sw-1100)//2}+{(_sh-800)//2}")
                    plot_win.minsize(500, 400)
                except Exception: pass
                
                # Fixed margins (no constrained_layout) so the single Density axis
                # position is stable across Tk canvas resizes and leaves room for
                # the suptitle.
                fig = Figure(figsize=(10, 6))
                ax = fig.subplots()
                fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.10)

                bc_label = " [BATCH CORRECTED]" if use_batch_correction else ""
                fig.suptitle(f"Gene Expression Distribution: {gene} on {plat}{bc_label}", 
                           fontsize=14, fontweight='bold')
                
                # Plot the histogram
                details = self._plot_single_histogram(ax, gene, plat, popup,
                                                       batch_offset=batch_offsets.get(plat, 0))
                
                if not details:
                    plt.close(fig)
                    plot_win.destroy()
                    failed_genes.append(f"{gene} on {plat}")
                    self.enqueue_log(f"[Gene Explorer] X {gene} not found on {plat}")
                    continue
                
                window_count += 1
                
                # Create canvas. Pack + realize the widget to its FINAL size BEFORE
                # the first draw, so every artist is only ever rendered at one canvas
                # size. Drawing before the expand-to-fill left stale ghost artists
                # (a second axis/legend) composited from the smaller initial size.
                canvas = FigureCanvasTkAgg(fig, master=plot_win)
                canvas_widget = canvas.get_tk_widget()

                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, plot_win)
                toolbar.update()
                style_toolbar(toolbar)
                toolbar.pack(side=tk.TOP, fill=tk.X)

                canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Draw once, at the size the widget ends up with. Tk assigns
                # that size on its next idle pass, so the draw is queued behind
                # it rather than forcing a synchronous flush: update_idletasks
                # here would make each new window's construction wait on the
                # idle work of every other window the program has open.
                plot_win.after_idle(canvas.draw)
                viz_make_interactive(fig)
                # DO NOT plt.close(fig) - canvas needs figure alive
                
                # Store in popup's figure dict
                key = f"{gene}_{plat}_{window_count}"
                popup._current_popup_figs[key] = (fig, canvas_widget, toolbar)
                
                # Setup selector for this single plot
                selector_props = dict(
                    facecolor='red',
                    edgecolor='black',
                    alpha=0.3,
                    fill=True
                )
                
                rs = RectangleSelector(
                    ax,
                    lambda eclick, erelease, ax=ax, p=popup: self._on_select(eclick, erelease, ax, p),
                    useblit=False,
                    props=selector_props,
                    button=[1],
                    minspanx=0.01,
                    minspany=0.01,
                    spancoords='data',
                    interactive=True
                )
                popup.rect_selectors.append(rs)
                
                # Add info label at bottom
                info_label = ttk.Label(
                    plot_win,
                    text=f"OK {gene} on {plat} | DRAG on histogram to select expression ranges",
                    font=('Segoe UI', 9),
                    foreground='green',
                    background='#E8F5E9',
                    padding=5
                )
                info_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.enqueue_log(f"[Gene Explorer] OK Created {window_count} distribution window(s)")
        
        if window_count == 0:
            fail_text = "\n".join(f"  - {f}" for f in failed_genes[:10])
            messagebox.showerror(
                "No Plots Created",
                f"Could not find any of the specified genes on the selected platform(s).\n\n"
                f"Failed:\n{fail_text}\n\n"
                f"Possible causes:\n"
                f"- Gene symbol not present on this platform\n"
                f"- Expression data not numeric (check CSV)\n"
                f"- Column not detected as gene expression\n\n"
                f"Tip: Check the log for gene mapping details.",
                parent=popup
            )
            return
        
        msg = f"Created {window_count} distribution window(s)!\n\n"
        msg += f"- Each gene-platform combination has its own window\n"
        msg += f"- DRAG rectangles on histograms to select ranges\n"
        msg += f"- Click 'Analyze Selected Range(s)' when ready"
        if failed_genes:
            msg += f"\n\n[!] {len(failed_genes)} gene(s) not found:\n"
            msg += "\n".join(f"  - {f}" for f in failed_genes[:5])
        
        messagebox.showinfo("Plots Created", msg, parent=popup)
    
    def _plot_histograms_overlay(self, popup, sel_plats):
        """Overlay all selected genes on one plot per platform."""
        genes = popup.current_genes
        gene_colors = ['#1976D2', '#C62828', '#2E7D32', '#17BECF', '#7B1FA2',
                        '#00838F', '#AD1457', '#4E342E', '#37474F', '#827717']

        if not hasattr(popup, '_overlay_genes'):
            popup._overlay_genes = {}

        window_count = 0
        for plat in sel_plats:
            dfg = self.gpl_datasets.get(plat)
            gmap = self.gpl_gene_mappings.get(plat, {})
            # Fallback to gene cache
            if dfg is None and plat in self.gpl_gene_cache:
                dfg = self.gpl_gene_cache[plat]
                gmap = self.gpl_gene_mappings.get(f"_cache_{plat}", {})
            if dfg is None:
                continue

            # Resolve all genes to columns
            gene_cols = []
            for gene in genes:
                col = gmap.get(gene)
                if col is None:
                    for c in dfg.columns:
                        if c.upper() == gene.upper():
                            col = c
                            break
                if col and col in dfg.columns:
                    expr = pd.to_numeric(dfg[col], errors='coerce').dropna()
                    if not expr.empty:
                        gene_cols.append((gene, col, expr))

            if not gene_cols:
                continue

            # Create window
            plot_win = tk.Toplevel(popup)
            style_window(plot_win)
            plot_win.title(f"{plat} - {len(gene_cols)} genes overlaid")
            plot_win.geometry("1100x800")
            try:
                _sw, _sh = plot_win.winfo_screenwidth(), plot_win.winfo_screenheight()
                plot_win.geometry(f"1100x800+{(_sw-1100)//2}+{(_sh-800)//2}")
                plot_win.minsize(500, 400)
            except Exception: pass

            fig = Figure(figsize=(12, 7), constrained_layout=True)
            ax = fig.subplots()

            # Track overlay info for this axis
            overlay_entries = []
            all_bins_list = []

            for gi, (gene, col, expr) in enumerate(gene_cols):
                clr = gene_colors[gi % len(gene_colors)]
                num_bins = Plotter.get_optimal_bins(expr, method='auto')
                counts, bins, patches = ax.hist(
                    expr, bins=num_bins,
                    edgecolor='black', alpha=0.35, color=clr,
                    linewidth=0.4, label=f"{gene} (n={len(expr):,})"
                )
                all_bins_list.append(bins)

                # Register each gene in the axis map
                details = (ax, bins, patches, dfg, plat, col, gene)
                ax_key = (id(ax), gi)
                popup._axis_map_dist_plot[ax_key] = details
                overlay_entries.append((dfg, plat, col, gene, bins, patches))

            # Store overlay info for selection propagation
            popup._overlay_genes[ax] = overlay_entries

            ax.set_xlabel(self.platform_measurement_label(plat), fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            stats_parts = [f"{g}: u={e.mean():.2f} SD={e.std():.2f}"
                           for g, c, e in gene_cols]
            ax.set_title(f"{plat} - {len(gene_cols)} genes overlaid\n"
                         f"{' | '.join(stats_parts)}", fontsize=11, weight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')

            # Canvas
            canvas = FigureCanvasTkAgg(fig, master=plot_win)

            # Build artist_groups: each group = all patches from one hist call
            hist_patch_groups = []
            for _, _, _, _, _, patches in overlay_entries:
                hist_patch_groups.append(list(patches))

            self._setup_interactive_legend(fig, ax, canvas, outside=True,
                                            fontsize=9, artist_groups=hist_patch_groups)
            canvas.draw()
            viz_make_interactive(fig)
            canvas_widget = canvas.get_tk_widget()
            toolbar = NavigationToolbar2Tk(canvas, plot_win)
            toolbar.update()
            style_toolbar(toolbar)
            toolbar.pack(side=tk.TOP, fill=tk.X)
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            key = f"overlay_{plat}_{window_count}"
            popup._current_popup_figs[key] = (fig, canvas_widget, toolbar)

            # Setup selector - selection applies to ALL genes on this axis
            selector_props = dict(facecolor='red', edgecolor='black', alpha=0.3, fill=True)
            rs = RectangleSelector(
                ax,
                lambda eclick, erelease, ax=ax, p=popup: self._on_select_overlay(
                    eclick, erelease, ax, p),
                useblit=False, props=selector_props,
                button=[1], minspanx=0.01, minspany=0.01,
                spancoords='data', interactive=True
            )
            popup.rect_selectors.append(rs)

            info = ttk.Label(plot_win,
                             text=f"OK {len(gene_cols)} genes on {plat} | DRAG to select - "
                                  f"applies to ALL overlapping genes",
                             font=('Segoe UI', 9), foreground='green', background='#E8F5E9',
                             padding=5)
            info.pack(side=tk.BOTTOM, fill=tk.X)
            window_count += 1

        if window_count > 0:
            messagebox.showinfo("Overlay Plots",
                                f"Created {window_count} overlay window(s).\n\n"
                                f"DRAG a rectangle to select an expression range.\n"
                                f"Selection will apply to ALL genes that overlap.",
                                parent=popup)
        else:
            messagebox.showerror("No Plots", "No genes found on selected platforms.", parent=popup)

    def _on_select_overlay(self, eclick, erelease, ax, popup):
        """Handle selection on overlay plot - registers regions for ALL genes."""
        if not popup or not popup.winfo_exists():
            return
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        if abs(x2 - x1) < 0.01:
            return

        if not hasattr(popup, 'active_selections'):
            popup.active_selections = {}
        if not hasattr(popup, '_overlay_genes'):
            return

        overlay = popup._overlay_genes.get(ax, [])
        if not overlay:
            return

        chosen_color = '#7B1FA2'
        count = 0

        for dfg, plat, col, gene, bins, patches in overlay:
            expr = pd.to_numeric(dfg[col], errors='coerce').dropna()
            n_in_range = ((expr >= x1) & (expr <= x2)).sum()
            if n_in_range == 0:
                continue

            # Find the axis_map key for this gene
            for k, v in popup._axis_map_dist_plot.items():
                if isinstance(k, tuple) and v[6] == gene and v[4] == plat:
                    ax_key = k
                    break
            else:
                continue

            if ax_key not in popup.active_selections:
                popup.active_selections[ax_key] = []
            popup.active_selections[ax_key].append((x1, x2, chosen_color))

            # Recolor bins in range
            for i, p in enumerate(patches):
                bin_mid = (bins[i] + bins[i + 1]) / 2
                if x1 <= bin_mid <= x2:
                    p.set_alpha(0.85)
                    p.set_edgecolor('red')
                    p.set_linewidth(1.5)

            self.enqueue_log(f"[Overlay] {gene}/{plat}: {n_in_range} samples in [{x1:.2f}, {x2:.2f}]")
            count += 1

        try:
            ax.figure.canvas.draw_idle()
        except:
            pass

        total = sum(len(sels) for sels in popup.active_selections.values())
        popup.analyze_selection_btn.config(state=tk.NORMAL)
        popup.compare_btn.config(state=tk.NORMAL if total > 1 else tk.DISABLED)
        popup.selection_label.config(
            text=f"[*] {count} gene(s) x {total} region(s) selected (overlay)",
            foreground='#7B1FA2'
        )

    def _plot_single_histogram(self, ax, gene, plat, popup, batch_offset=0):
        """Plots a single histogram with enhanced styling and statistics - COMPLETE VERSION.
        
        batch_offset: value to subtract from expression data for batch correction.
                      When non-zero, adds '[BATCH CORRECTED]' to the plot title.
        Checks gpl_datasets first, then gpl_gene_cache for gene-only loads.
        """
        dfg = self.gpl_datasets.get(plat)
        gmap = self.gpl_gene_mappings.get(plat, {})
        col = gmap.get(gene)
        
        # Fallback 1: if gene not in map, try direct column name match (case-insensitive)
        if col is None and dfg is not None:
            for c in dfg.columns:
                if c.upper() == gene.upper():
                    col = c
                    if not pd.api.types.is_numeric_dtype(dfg[c]):
                        test = pd.to_numeric(dfg[c], errors='coerce')
                        if test.notna().sum() > len(test) * 0.3:
                            dfg[c] = test
                            self.gpl_datasets[plat] = dfg
                            gmap[gene] = c
                            self.gpl_gene_mappings[plat] = gmap
                            self.enqueue_log(f"[{plat}] Fallback: '{c}' coerced to numeric for {gene}")
                    break

        # Fallback 2: check gene cache (gene-only quick loads)
        if (dfg is None or col is None) and plat in self.gpl_gene_cache:
            cache_df = self.gpl_gene_cache[plat]
            cache_gmap = self.gpl_gene_mappings.get(f"_cache_{plat}", {})
            cache_col = cache_gmap.get(gene)
            if cache_col is None:
                for c in cache_df.columns:
                    if c.upper() == gene.upper() and c != 'GSM':
                        cache_col = c
                        break
            if cache_col and cache_col in cache_df.columns:
                dfg = cache_df
                col = cache_col
                self.enqueue_log(f"[{plat}] Using gene cache for {gene}")
        
        if dfg is None or col is None or col not in dfg.columns:
            ax.set_title(f"{plat} - {gene}\n[X] Not found", color='red', fontsize=10, weight='bold')
            ax.axis("off")
            return None
        
        # Use pd.to_numeric for safe conversion (handles string expression values)
        expr = pd.to_numeric(dfg[col], errors='coerce').dropna()
        
        # Apply batch correction offset if provided
        if batch_offset != 0:
            expr = expr - batch_offset
        
        if expr.empty:
            ax.set_title(f"{plat} - {col}\n[!] No data", color='#C0392B', fontsize=10, weight='bold')
            ax.axis("off")
            return None
        
        dist_class = BioAI_Engine.analyze_gene_distribution(expr)
        
        # The drawing is `draw_gene_histogram` - the same routine the
        # assistant's chart builder calls. This window keeps what is its own:
        # the axis, the span selector and the bars it makes interactive. It no
        # longer keeps a second copy of the bars, the KDE curve and the mode
        # rules that someone would have to hold in step with the assistant's
        # by hand, which is how the two came to differ in the first place.
        from genevariate.core.analysis.figures import draw_gene_histogram
        desc = draw_gene_histogram(
            ax, np.asarray(expr, dtype=np.float64),
            gene=col, label=plat, dist_class=dist_class,
            x_label=self.platform_measurement_label(plat))
        bins, patches = desc.get("bins"), desc.get("patches")
        
        details = (ax, bins, patches, dfg, plat, col, gene)
        popup._axis_map_dist_plot[ax] = details
        
        return details
    
    def _on_select(self, eclick, erelease, selected_ax, popup_ref):
        """Enhanced selection handler with visual feedback - COMPLETE VERSION."""
        popup = popup_ref
        if not popup or not popup.winfo_exists():
            return
        
        if eclick.xdata is None or erelease.xdata is None:
            return
        
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        
        if selected_ax in popup._axis_map_dist_plot:
            _, bins, _, _, _, _, _ = popup._axis_map_dist_plot[selected_ax]
            min_width = (bins[-1] - bins[0]) * 0.01
            
            if abs(x2 - x1) < min_width:
                return
        
        if not hasattr(popup, 'active_selections'):
            popup.active_selections = {}
        if selected_ax not in popup.active_selections:
            popup.active_selections[selected_ax] = []
        
        num_selections = sum(len(sels) for sels in popup.active_selections.values())
        color_idx = num_selections % len(popup.selector_colors)
        chosen_color = popup.selector_colors[color_idx]
        
        popup.active_selections[selected_ax].append((x1, x2, chosen_color))
        
        if selected_ax in popup._axis_map_dist_plot:
            _, bins, patches, _, _, _, _ = popup._axis_map_dist_plot[selected_ax]
            
            for i, p in enumerate(patches):
                bin_start, bin_end = bins[i], bins[i+1]
                bin_mid = (bin_start + bin_end) / 2
                
                matched = False
                for low, high, col in popup.active_selections[selected_ax]:
                    if low <= bin_mid <= high:
                        p.set_facecolor(col)
                        p.set_alpha(0.8)
                        p.set_edgecolor('black')
                        p.set_linewidth(0.8)
                        matched = True
                        break
                
                if not matched:
                    plot_cfg = CONFIG['plotting']['histogram']
                    p.set_facecolor(plot_cfg['default_color'])
                    p.set_alpha(plot_cfg['alpha'])
                    p.set_linewidth(0.5)
            
            selected_ax.figure.canvas.draw_idle()
        
        total_selections = sum(len(sels) for sels in popup.active_selections.values())
        popup.analyze_selection_btn.config(state=tk.NORMAL)
        popup.compare_btn.config(state=tk.NORMAL if total_selections > 1 else tk.DISABLED)
        
        popup.selection_label.config(
            text=f"{total_selections} region(s) selected", 
            foreground='green'
        )
        
        if selected_ax in popup._axis_map_dist_plot:
            _, _, _, _, plat, col, gene = popup._axis_map_dist_plot[selected_ax]
            self.enqueue_log(f"[Selection] {gene} on {plat}: range [{x1:.2f}, {x2:.2f}]")
    
    def _recolor_tail_bins(self, ax_obj, bins, patches, selections):
        """Recolor histogram bins based on ALL active tail selections on this axis."""
        plot_cfg = CONFIG['plotting']['histogram']
        for i, p in enumerate(patches):
            bin_start, bin_end = bins[i], bins[i + 1]
            bin_mid = (bin_start + bin_end) / 2
            matched_color = None
            for sel_lo, sel_hi, sel_clr in selections:
                if sel_lo <= bin_mid <= sel_hi:
                    matched_color = sel_clr
                    break
            if matched_color:
                p.set_facecolor(matched_color)
                p.set_alpha(0.85)
                p.set_edgecolor('black')
                p.set_linewidth(0.8)
            else:
                p.set_facecolor(plot_cfg['default_color'])
                p.set_alpha(plot_cfg['alpha'])
                p.set_linewidth(0.5)

    def _open_highlight_region_dialog(self, popup):
        """Open dialog to specify a region to highlight on all open histograms."""
        if not popup or not popup.winfo_exists():
            return
        if not hasattr(popup, '_axis_map_dist_plot') or not popup._axis_map_dist_plot:
            messagebox.showinfo("No Plots", "Plot gene distributions first.", parent=popup)
            return

        dlg = tk.Toplevel(popup)
        style_window(dlg)
        dlg.title("Highlight Region - Specify Criteria")
        dlg.transient(popup)

        # Header
        tk.Label(dlg, text="Define Region to Highlight",
                 font=('Segoe UI', 13, 'bold'), bg=AERO["accent_dark"], fg="white",
                 pady=8).pack(fill=tk.X)

        main = ttk.Frame(dlg, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Method selection ──
        method_frame = labelframe(main, text="Method", padding=8)
        method_frame.pack(fill=tk.X, pady=5)

        method_var = tk.StringVar(value="sd_mean")
        methods = [
            ("sd_mean",   "Standard Deviations from Mean"),
            ("sd_median", "Standard Deviations from Median"),
            ("sd_mode",   "Standard Deviations from Mode"),
            ("percentile","Percentile Range"),
            ("custom",    "Custom Value Range"),
        ]
        for val, label in methods:
            ttk.Radiobutton(method_frame, text=label, variable=method_var,
                            value=val, command=lambda: _update_preview()).pack(anchor=tk.W, pady=1)

        # ── Direction ──
        dir_frame = labelframe(main, text="Direction", padding=8)
        dir_frame.pack(fill=tk.X, pady=5)

        dir_var = tk.StringVar(value="above")
        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill=tk.X)
        for val, label in [("above", "Above threshold"), ("below", "Below threshold"),
                           ("between", "Between two values"), ("outside", "Outside (both tails)")]:
            ttk.Radiobutton(dir_row, text=label, variable=dir_var, value=val,
                            command=lambda: _update_preview()).pack(side=tk.LEFT, padx=8)

        # ── Value inputs ──
        val_frame = labelframe(main, text="Value(s)", padding=8)
        val_frame.pack(fill=tk.X, pady=5)

        v_row1 = ttk.Frame(val_frame)
        v_row1.pack(fill=tk.X, pady=2)
        ttk.Label(v_row1, text="Value 1:").pack(side=tk.LEFT)
        val1_var = tk.StringVar(value="2.0")
        val1_entry = ttk.Entry(v_row1, textvariable=val1_var, width=10)
        val1_entry.pack(side=tk.LEFT, padx=5)
        val1_hint = ttk.Label(v_row1, text="(e.g., 2.0 = 2 SDs, or 95 = 95th percentile)",
                               font=('Segoe UI', 8, 'italic'), foreground='gray')
        val1_hint.pack(side=tk.LEFT, padx=5)

        v_row2 = ttk.Frame(val_frame)
        v_row2.pack(fill=tk.X, pady=2)
        ttk.Label(v_row2, text="Value 2:").pack(side=tk.LEFT)
        val2_var = tk.StringVar(value="")
        val2_entry = ttk.Entry(v_row2, textvariable=val2_var, width=10)
        val2_entry.pack(side=tk.LEFT, padx=5)
        val2_hint = ttk.Label(v_row2, text="(only for 'between' / 'outside' / percentile range)",
                               font=('Segoe UI', 8, 'italic'), foreground='gray')
        val2_hint.pack(side=tk.LEFT, padx=5)

        # ── Color ──
        clr_frame = ttk.Frame(main)
        clr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(clr_frame, text="Highlight color:").pack(side=tk.LEFT)
        color_var = tk.StringVar(value="#C62828")
        colors = [("#C62828", "Red"), ("#1565C0", "Blue"), ("#2E7D32", "Green"),
                  ("#F57C00", "Orange"), ("#7B1FA2", "Purple"), ("#00838F", "Teal")]
        for cval, cname in colors:
            tk.Radiobutton(clr_frame, text=cname, variable=color_var, value=cval,
                           fg=cval, selectcolor='white', font=('Segoe UI', 9, 'bold'),
                           indicatoron=1).pack(side=tk.LEFT, padx=4)

        # ── Live preview ──
        preview_frame = labelframe(main, text="Preview (computed thresholds per gene)", padding=6)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        preview_text = tk.Text(preview_frame, font=(MONO_FONT, 9), height=6,
                                wrap=tk.WORD, state=tk.DISABLED, bg=AERO["panel"])
        preview_sb = ttk.Scrollbar(preview_frame, command=preview_text.yview)
        preview_text.configure(yscrollcommand=preview_sb.set)
        preview_sb.pack(side=tk.RIGHT, fill=tk.Y)
        preview_text.pack(fill=tk.BOTH, expand=True)

        def _compute_thresholds():
            """Compute highlight bounds for each gene/platform axis."""
            results = []
            method = method_var.get()
            direction = dir_var.get()
            try:
                v1 = float(val1_var.get()) if val1_var.get().strip() else 2.0
            except ValueError:
                v1 = 2.0
            try:
                v2 = float(val2_var.get()) if val2_var.get().strip() else v1
            except ValueError:
                v2 = v1

            for map_key, details in popup._axis_map_dist_plot.items():
                _, _, _, dfg, plat, col, gene = details
                vals = pd.to_numeric(dfg[col], errors='coerce').dropna()
                if vals.empty:
                    continue

                mean = vals.mean()
                median = vals.median()
                std = vals.std()
                vmin, vmax = vals.min(), vals.max()

                # Compute mode (most common bin center)
                try:
                    from scipy.stats import mode as _scipy_mode
                    mode_result = _scipy_mode(vals, keepdims=False)
                    mode_val = float(mode_result.mode)
                except Exception:
                    mode_val = float(vals.mode().iloc[0]) if not vals.mode().empty else mean

                if method == 'sd_mean':
                    center = mean
                    lo = center - v1 * std
                    hi = center + v1 * std
                elif method == 'sd_median':
                    center = median
                    lo = center - v1 * std
                    hi = center + v1 * std
                elif method == 'sd_mode':
                    center = mode_val
                    lo = center - v1 * std
                    hi = center + v1 * std
                elif method == 'percentile':
                    lo = float(np.percentile(vals, min(v1, v2)))
                    hi = float(np.percentile(vals, max(v1, v2)))
                elif method == 'custom':
                    lo, hi = min(v1, v2), max(v1, v2)
                else:
                    lo, hi = mean - 2*std, mean + 2*std

                # Compute actual bounds based on direction
                if direction == 'above':
                    bound_lo, bound_hi = hi, vmax
                    n_samples = int((vals > hi).sum())
                    desc = f"> {hi:.2f}"
                elif direction == 'below':
                    bound_lo, bound_hi = vmin, lo
                    n_samples = int((vals < lo).sum())
                    desc = f"< {lo:.2f}"
                elif direction == 'between':
                    bound_lo, bound_hi = lo, hi
                    n_samples = int(vals.between(lo, hi).sum())
                    desc = f"[{lo:.2f}, {hi:.2f}]"
                elif direction == 'outside':
                    # Both tails
                    n_samples = int((vals < lo).sum() + (vals > hi).sum())
                    bound_lo, bound_hi = lo, hi  # stored as "outside" pair
                    desc = f"< {lo:.2f} or > {hi:.2f}"
                else:
                    bound_lo, bound_hi = hi, vmax
                    n_samples = int((vals > hi).sum())
                    desc = f"> {hi:.2f}"

                results.append({
                    'map_key': map_key, 'gene': gene, 'plat': plat,
                    'bound_lo': bound_lo, 'bound_hi': bound_hi,
                    'direction': direction, 'n_samples': n_samples,
                    'desc': desc, 'mean': mean, 'median': median,
                    'mode': mode_val, 'std': std, 'lo': lo, 'hi': hi,
                })
            return results

        def _update_preview(*args):
            results = _compute_thresholds()
            preview_text.config(state=tk.NORMAL)
            preview_text.delete('1.0', tk.END)
            if not results:
                preview_text.insert('1.0', "No plots available.")
            else:
                for r in results:
                    line = (f"{r['gene']} / {r['plat']}:  {r['desc']}  "
                            f"({r['n_samples']} samples)  "
                            f"[mean={r['mean']:.2f}, med={r['median']:.2f}, "
                            f"mode={r['mode']:.2f}, SD={r['std']:.2f}]\n")
                    preview_text.insert(tk.END, line)
            preview_text.config(state=tk.DISABLED)

        # Bind live preview updates
        for var in (val1_var, val2_var):
            var.trace_add('write', _update_preview)
        _update_preview()

        def _apply():
            results = _compute_thresholds()
            color = color_var.get()
            direction = dir_var.get()
            if not results:
                messagebox.showinfo("Nothing to highlight", "No plots available.", parent=dlg)
                return

            if not hasattr(popup, 'active_selections'):
                popup.active_selections = {}

            count = 0
            for r in results:
                mk = r['map_key']
                details = popup._axis_map_dist_plot.get(mk)
                if not details:
                    continue
                ax_obj, bins, patches, dfg, plat, col, gene = details

                existing = popup.active_selections.get(mk, [])
                # Remove previous highlight of same color
                existing = [(lo, hi, c) for lo, hi, c in existing if c != color]

                if direction == 'outside':
                    # Two regions: left tail + right tail
                    existing.append((r['bound_lo'] - 999999, r['lo'], color))
                    existing.append((r['hi'], r['bound_hi'] + 999999, color))
                else:
                    existing.append((r['bound_lo'], r['bound_hi'], color))

                popup.active_selections[mk] = existing
                self._recolor_tail_bins(ax_obj, bins, patches, existing)

                # Remove old markers of this color
                for artist in list(ax_obj.lines) + list(ax_obj.texts):
                    if hasattr(artist, '_highlight_color') and artist._highlight_color == color:
                        artist.remove()

                # Draw threshold lines
                ylim = ax_obj.get_ylim()
                if direction in ('above', 'outside'):
                    vl = ax_obj.axvline(r['hi'], color=color, ls='--', lw=2, alpha=0.8, zorder=10)
                    vl._highlight_color = color
                if direction in ('below', 'outside'):
                    vl = ax_obj.axvline(r['lo'], color=color, ls='--', lw=2, alpha=0.8, zorder=10)
                    vl._highlight_color = color
                if direction == 'between':
                    for threshold in (r['bound_lo'], r['bound_hi']):
                        vl = ax_obj.axvline(threshold, color=color, ls='--', lw=2, alpha=0.8, zorder=10)
                        vl._highlight_color = color

                # Annotation
                txt_x = r['hi'] if direction in ('above', 'outside') else r['lo']
                txt_ha = 'left' if direction in ('above', 'between') else 'right'
                ann = ax_obj.text(txt_x, ylim[1] * 0.90,
                    f"  {r['desc']}\n  {r['n_samples']} samples",
                    fontsize=8, color=color, fontweight='bold',
                    va='top', ha=txt_ha, zorder=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.9))
                ann._highlight_color = color

                try:
                    ax_obj.figure.canvas.draw_idle()
                except Exception:
                    pass
                count += 1

            # Update button states
            total = sum(len(sels) for sels in popup.active_selections.values())
            if total > 0:
                popup.analyze_selection_btn.config(state=tk.NORMAL)
                popup.compare_btn.config(state=tk.NORMAL if total > 1 else tk.DISABLED)
                popup.selection_label.config(
                    text=f"[*] {count} region(s) highlighted ({total} total selections)",
                    foreground=color)

            self.enqueue_log(f"[Highlight] Applied to {count} plots: {results[0]['desc'] if results else ''}")
            dlg.destroy()

        # Buttons
        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=8)
        ttk.Button(btn_row, text="Apply Highlight", command=_apply,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        self._fit_window(dlg, 680, 680)
        dlg.grab_set()


    def _clear_selections_logic(self, popup):
        """Clears all selections with confirmation - COMPLETE VERSION."""
        if not popup:
            return
        
        if not popup.active_selections:
            return
        
        total = sum(len(sels) for sels in popup.active_selections.values())
        
        if total > 3:
            if not messagebox.askyesno(
                "Clear Selections", 
                f"Clear all {total} selected region(s)?", 
                parent=popup
            ):
                return
        
        popup.active_selections.clear()
        
        plot_cfg = CONFIG['plotting']['histogram']
        for key, details in popup._axis_map_dist_plot.items():
            actual_ax, _, patches, _, _, _, _ = details
            for p in patches:
                p.set_facecolor(plot_cfg['default_color'])
                p.set_alpha(plot_cfg['alpha'])
                p.set_edgecolor(plot_cfg['edge_color'])
                p.set_linewidth(0.5)
            # Remove tail threshold lines and annotations
            for artist in list(actual_ax.lines) + list(actual_ax.texts):
                if hasattr(artist, '_tail_marker'):
                    artist.remove()
            try:
                actual_ax.figure.canvas.draw_idle()
            except:
                pass

        # The drag rectangle is a widget artist, so clearing the selections
        # without hiding it leaves the user looking at a region that is gone.
        for sel in getattr(popup, "rect_selectors", []) or []:
            try:
                sel.set_visible(False)
                sel.update()
            except Exception:
                pass

        popup.analyze_selection_btn.config(state=tk.DISABLED)
        popup.compare_btn.config(state=tk.DISABLED)
        popup.selection_label.config(text="No regions selected", foreground='gray')
        
        self.enqueue_log("[UI] All selections cleared")

    def _pre_analyze_dialog(self, popup):
        """Show extraction settings dialog before running region analysis.
        If labels from file → skip dialog, run directly.
        If LLM enabled → ask which fields to extract + Phase 2 Re-extraction.
        """
        if not popup or not popup.active_selections:
            messagebox.showwarning(
                "No Selection",
                "Please select at least one range on the distribution plots first.\n\n"
                "Drag a rectangle on any histogram to make a selection.",
                parent=popup
            )
            return

        # If using file labels, skip dialog - no LLM extraction needed
        if self.label_source_var.get() == "file":
            self._analyze_selected_range(popup)
            return

        # ── LLM Extraction Settings Dialog ──
        dlg = tk.Toplevel(popup)
        style_window(dlg)
        dlg.title("LLM Extraction Settings")
        dlg.transient(popup)
        dlg.grab_set()

        ttk.Label(dlg, text="Select labels to extract for the selected region(s):",
                  font=('Segoe UI', 11, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="The LLM agent will classify samples in your selected range.",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        # Field checkboxes -- only the labels the vendored geo_label_extractor
        # actually produces (geo_extract_driver.ALL_FIELDS). The old build also
        # offered Treatment_Time and free-text "custom fields", but the remote
        # extractor supports neither (they always came back "Not Specified"), so
        # they are gone.
        from genevariate.core.geo_extract_driver import ALL_FIELDS as _EXTRACT_FIELDS
        fields_frame = labelframe(dlg, text="Labels to extract", padding=10)
        fields_frame.pack(fill=tk.X, padx=15, pady=5)

        field_vars = {}
        for f in _EXTRACT_FIELDS:
            var = tk.BooleanVar(value=f in self._extraction_fields)
            field_vars[f] = var
            ttk.Checkbutton(fields_frame, text=f, variable=var).pack(
                side=tk.LEFT, padx=8)

        # ── Metadata the model reads ──
        # Which GEO columns go in front of the model is a real choice: an
        # RNA-seq submission carries text in different fields than an array
        # one. The five ticked by default are the set the extractor paper's
        # numbers came from, so departing from them is opt-in and is said out
        # loud (geo_label_extractor.metadata_fields).
        from genevariate.core import geo_extract_driver as _fdrv
        _default_cols = _fdrv.default_metadata_columns()
        _chosen_cols = list(getattr(self, '_extraction_columns', _default_cols))
        cols_frame = labelframe(dlg, text="Metadata the model reads", padding=8)
        cols_frame.pack(fill=tk.X, padx=15, pady=5)
        col_note = ttk.Label(
            cols_frame, text="", foreground=AERO["muted"],
            font=('Segoe UI', 8, 'italic'), wraplength=500, justify=tk.LEFT)

        col_vars = {}
        col_grid = ttk.Frame(cols_frame)
        col_grid.pack(fill=tk.X)

        def _sync_col_note(*_a):
            picked = [c for c, v in col_vars.items() if v.get()]
            if picked == _default_cols:
                col_note.config(
                    text="Published default: these five columns reproduce the "
                         "extractor paper's numbers.")
            elif not picked:
                col_note.config(text="Pick at least one column.")
            else:
                col_note.config(
                    text="Non-default selection: labels will not be comparable "
                         "with the published run.")

        try:
            _offer = _fdrv.metadata_columns(
                getattr(self, '_geometadb_path', '') or '')
        except Exception:
            _offer = [{"column": c, "prompt": c, "description": "",
                       "default": True} for c in _default_cols]
        for i, spec in enumerate(_offer):
            col = spec["column"]
            var = tk.BooleanVar(value=col in _chosen_cols)
            var.trace_add('write', _sync_col_note)
            col_vars[col] = var
            cb = ttk.Checkbutton(col_grid, text=col, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=6, pady=1)
            desc = str(spec.get("description") or "")
            if desc:
                # Hovering explains what the model would see in that column.
                cb.bind("<Enter>",
                        lambda _e, d=desc, c=col: col_note.config(text=f"{c} - {d}"))
                cb.bind("<Leave>", _sync_col_note)
        col_note.pack(anchor=tk.W, pady=(6, 0))
        _sync_col_note()

        # The sample's own metadata is sometimes silent where the study
        # abstract is not, so phase 1b can fall back to the GSE description.
        # Fetching it means one NCBI request per study, so it is the user's
        # call and it is off unless they ask (geo_extract_driver.set_gse_scrape).
        scrape_var = tk.BooleanVar(
            value=bool(getattr(self, '_extraction_scrape_gse',
                               _fdrv.gse_scrape_enabled())))
        ttk.Checkbutton(
            cols_frame,
            text="Also read the study (GSE) description - fetched from NCBI",
            variable=scrape_var).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(
            cols_frame,
            text="Off by default: it is one network request per study and the "
                 "fetched text is cached for later runs.",
            foreground=AERO["muted"], font=('Segoe UI', 8, 'italic'),
            wraplength=500, justify=tk.LEFT).pack(anchor=tk.W)

        # Third-party curation for the same study, from whichever project
        # covers the platform's assay (core.external_enrichment). Different
        # modalities have different curators, and two modalities have none, so
        # what this adds is decided per region from its platform's technology.
        enrich_var = tk.BooleanVar(
            value=bool(getattr(self, '_extraction_enrich', False)))
        ttk.Checkbutton(
            cols_frame,
            text="Also read third-party curation of the study",
            variable=enrich_var).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(
            cols_frame,
            text="Source depends on the platform: Expression Atlas for arrays "
                 "and bulk RNA-seq, CELLxGENE for single cell. Methylation and "
                 "peak assays (ChIP/ATAC/Hi-C) have no such curator and are "
                 "left alone. Values are study-wide, not per sample.",
            foreground=AERO["muted"], font=('Segoe UI', 8, 'italic'),
            wraplength=500, justify=tk.LEFT).pack(anchor=tk.W)

        # ── Extraction Settings ──
        # The only real knob is how many samples are in flight at once (one
        # worker thread per request). The old GPU/CPU-worker + VRAM panel and
        # the Fast/Full "mode" split were residue -- Fast and Full produced
        # identical labels.
        set_frame = labelframe(dlg, text="Extraction Settings", padding=8)
        set_frame.pack(fill=tk.X, padx=15, pady=5)
        # Name the endpoint and models that will actually be called, so a run
        # against a remote GPU is never mistaken for a local one.
        _picked = _fdrv.resolve_backend()
        ttk.Label(set_frame,
                  text=(f"Endpoint: {_picked['url']}\n"
                        f"Phase 1: {_picked['model']}\n"
                        f"Age / Phase 2: {_picked['age_model']}"),
                  foreground=AERO["muted"], font=('Segoe UI', 8, 'italic'),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))
        req_row = ttk.Frame(set_frame)
        req_row.pack(fill=tk.X, pady=2)
        ttk.Label(req_row, text="Parallel requests:").pack(side=tk.LEFT)
        req_var = tk.IntVar(value=getattr(self, '_llm_workers', 0) or 0)
        ttk.Spinbox(req_row, from_=0, to=32, textvariable=req_var,
                    width=4).pack(side=tk.LEFT, padx=4)
        ttk.Label(req_row, text="(0 = auto)", foreground=AERO["muted"],
                  font=('Segoe UI', 8, 'italic')).pack(side=tk.LEFT)

        # Buttons
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _ok():
            selected = [f for f, v in field_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("No Labels", "Select at least one label to extract.", parent=dlg)
                return

            picked_cols = [c for c, v in col_vars.items() if v.get()]
            if not picked_cols:
                messagebox.showwarning(
                    "No Metadata", "Select at least one metadata column for "
                    "the model to read.", parent=dlg)
                return

            self._extraction_fields = selected
            # Publish the column choice so every phase-1 call in this process
            # renders the same prompt (geo_label_extractor.metadata_fields).
            self._extraction_columns = picked_cols
            _fdrv.set_metadata_columns(
                None if picked_cols == _default_cols else picked_cols)
            self._extraction_scrape_gse = bool(scrape_var.get())
            _fdrv.set_gse_scrape(self._extraction_scrape_gse)
            self._extraction_enrich = bool(enrich_var.get())
            # One knob: parallel in-flight HTTP requests (0 = auto). Stored at
            # app level so _analyze_selected_range picks it up.
            n_req = req_var.get()
            self._llm_workers = n_req
            self.ai_agent.MAX_WORKERS = n_req

            dlg.destroy()
            self._analyze_selected_range(popup)

        def _cancel():
            dlg.destroy()

        ttk.Button(btn_frame, text="Run Analysis", style="Primary.TButton",
                   command=_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", style="Secondary.TButton",
                   command=_cancel).pack(side=tk.RIGHT, padx=5)

        # Center dialog
        dlg.update_idletasks()
        w = max(550, dlg.winfo_reqwidth())
        h = dlg.winfo_reqheight()
        x = popup.winfo_x() + (popup.winfo_width() - w) // 2
        y = popup.winfo_y() + (popup.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")

    def _analyze_selected_range(self, popup):
        """
        Comprehensive region analysis: fetches metadata, runs AI classification,
        computes experiment enrichment, and launches RegionAnalysisWindow.
        """
        if not popup or not popup.active_selections:
            messagebox.showwarning(
                "No Selection",
                "Please select at least one range on the distribution plots first.\n\n"
                "Drag a rectangle on any histogram to make a selection.",
                parent=popup
            )
            return

        # ── 1. Collect region data ──────────────────────────────────────
        region_specs = []
        for ax, range_list in popup.active_selections.items():
            details = popup._axis_map_dist_plot.get(ax)
            if not details:
                self.enqueue_log("[Analysis] Warning: axis not found in map - skipping")
                continue
            _, _, _, dfg, plat, col, gene = details

            for i, (low, high, color) in enumerate(range_list):
                # Same builder the assistant uses, so the window is fed the
                # same dict whichever way the region was defined.
                from genevariate.gui.region_analysis import build_region_spec
                spec = build_region_spec(
                    dfg, gene=gene, platform=plat, column=col,
                    low=low, high=high,
                    label=f"{gene}_R{len(region_specs)+1}_{plat}",
                    color=color, log=self.enqueue_log)
                if spec is None:
                    continue
                region_specs.append(spec)

        if not region_specs:
            # Diagnostics
            n_axes = len(popup._axis_map_dist_plot)
            n_sels = sum(len(v) for v in popup.active_selections.values())
            diag = f"Axes mapped: {n_axes}, Selections: {n_sels}"
            
            if n_sels == 0:
                msg = ("No ranges selected!\n\n"
                       "Drag a rectangle on the histogram to select an expression range, "
                       "then click 'Analyze Selected Range(s)'.")
            elif n_axes == 0:
                msg = ("No histogram data available.\n\n"
                       "The gene might not have been found on this platform.\n"
                       "Check the log for details.")
            else:
                msg = (f"No samples found in the selected range(s).\n\n"
                       f"Debug: {diag}\n"
                       f"This can happen if:\n"
                       f"- Expression values are non-numeric\n"
                       f"- The selected range is too narrow\n"
                       f"- The platform data has no GSM column")
            
            messagebox.showinfo("No Data", msg, parent=popup)
            return

        total_gsms = sum(len(r['gsm_list']) for r in region_specs)
        self.enqueue_log(f"[Analysis] Processing {total_gsms} samples across {len(region_specs)} region(s)...")

        # ── 2. Progress window ──────────────────────────────────────────
        progress_win = tk.Toplevel(popup)
        style_window(progress_win)
        progress_win.title("Region Analysis Pipeline")
        progress_win.geometry("520x230")
        progress_win.resizable(False, False)
        progress_win.transient(popup)
        progress_win.grab_set()

        _pbody = tk.Frame(progress_win, bg=AERO["bg_top"], padx=26, pady=22)
        _pbody.pack(fill=tk.BOTH, expand=True)

        tk.Label(_pbody, text=f"Analysing {len(region_specs)} region"
                              f"{'s' if len(region_specs) != 1 else ''}"
                              f"  ·  {total_gsms:,} samples",
                 font=('Segoe UI', 12, 'bold'), bg=AERO["bg_top"],
                 fg=AERO["accent_dark"], anchor='w').pack(fill=tk.X)

        status_lbl = tk.Label(_pbody, text="Initializing…", bg=AERO["bg_top"],
                              fg=AERO["muted"], font=('Segoe UI', 9),
                              anchor='w', justify=tk.LEFT, wraplength=460)
        status_lbl.pack(fill=tk.X, pady=(4, 14))

        prog_bar = ttk.Progressbar(_pbody, mode='determinate', maximum=100,
                                   style='Accent.Horizontal.TProgressbar')
        prog_bar.pack(fill=tk.X)

        detail_lbl = tk.Label(_pbody, text="", bg=AERO["bg_top"],
                              fg=AERO["accent"], font=('Segoe UI', 9),
                              anchor='w', justify=tk.LEFT, wraplength=460)
        detail_lbl.pack(fill=tk.X, pady=(12, 0))

        # Thread-safe progress queue
        _progress_queue = queue.Queue()

        def _update_status(text, pct=None, detail=""):
            """Thread-safe: puts update into queue, main thread polls it."""
            _progress_queue.put((text, pct, detail))

        def _poll_progress():
            """Main-thread poller: applies queued progress updates."""
            try:
                while not _progress_queue.empty():
                    text, pct, detail = _progress_queue.get_nowait()
                    try:
                        status_lbl.config(text=text)
                        if pct is not None:
                            prog_bar['value'] = pct
                        if detail:
                            detail_lbl.config(text=detail)
                    except tk.TclError:
                        return  # window was destroyed
                progress_win.after(50, _poll_progress)
            except tk.TclError:
                pass  # window was destroyed

        _poll_progress()  # start polling

        # ── 3. Capture label source settings (from main window) ────────
        use_default_labels = (self.label_source_var.get() == "file" and
                              (bool(self.platform_labels) or self.default_labels_df is not None))

        default_labels_for_regions = {}
        if use_default_labels:
            self.enqueue_log("[Labels] Using pre-computed labels (skipping LLM)")
            for r_idx, region in enumerate(region_specs):
                gsms = region['gsm_list']
                plat = region.get('platform', '')
                ldf = self._get_labels_for_gsms(gsms, platform=plat)
                default_labels_for_regions[r_idx] = ldf
                self.enqueue_log(f"[Labels] Region {region['label']} ({plat}): {len(ldf)}/{len(gsms)} GSMs matched")
        else:
            self.enqueue_log("[Labels] Using LLM extraction (geo_label_extractor)")

        # ── 4. Background processing thread ─────────────────────────────
        def process_regions():
            try:
                n_regions = len(region_specs)

                # Step A: Load GEOmetadb once for all regions
                _update_status("Loading GEO database...", 5)
                gz_path = str(CONFIG['paths']['geo_db'])

                from genevariate.core.db_loader import open_geometadb
                thread_conn = open_geometadb(gz_path)
                if thread_conn is None:
                    _update_status("ERROR: Could not open GEOmetadb", 0)
                    return

                # Step B: Fetch metadata & classify each region
                for r_idx, region in enumerate(region_specs):
                    base_pct = 10 + int((r_idx / n_regions) * 70)
                    gsms = region['gsm_list']
                    _update_status(
                        f"Region {r_idx+1}/{n_regions}: Fetching metadata...",
                        base_pct,
                        f"{region['label']} - {len(gsms)} samples"
                    )

                    # Fetch metadata in chunks
                    chunk_size = CONFIG['database']['sql_chunk_size']
                    meta_chunks = []
                    for ci in range(0, len(gsms), chunk_size):
                        chunk = gsms[ci:ci + chunk_size]
                        ph = ','.join(['?'] * len(chunk))
                        meta_chunks.append(
                            pd.read_sql_query(f"SELECT * FROM gsm WHERE UPPER(gsm) IN ({ph})", thread_conn, params=[g.upper() for g in chunk])
                        )

                    meta_df = pd.concat(meta_chunks, ignore_index=True) if meta_chunks else pd.DataFrame()
                    region['meta_df'] = meta_df

                    # Step C: Labels (AI Classification OR Default Labels File)
                    ai_pct = base_pct + int(35 / n_regions)

                    if use_default_labels:
                        _update_status(
                            f"Region {r_idx+1}/{n_regions}: Applying default labels...",
                            ai_pct,
                            f"Matching {len(gsms)} samples against loaded labels"
                        )
                        ai_labels = default_labels_for_regions.get(r_idx, pd.DataFrame())
                        if not ai_labels.empty:
                            self.enqueue_log(f"[Labels] Region {region['label']}: matched {len(ai_labels)} samples")
                        else:
                            self.enqueue_log(f"[Labels] Region {region['label']}: no matches in labels file")
                    else:
                        _update_status(
                            f"Region {r_idx+1}/{n_regions}: LLM classification...",
                            ai_pct,
                            f"Classifying {len(meta_df)} samples with LLM agent"
                        )
                        ai_labels = pd.DataFrame()
                        if not meta_df.empty:
                            try:
                                n_samples = len(meta_df)

                                # Progress callback for real-time updates
                                def _extraction_progress(done, total, speed, eta):
                                    _update_status(
                                        f"Region {r_idx+1}/{n_regions}: LLM extraction {done}/{total}",
                                        ai_pct + int((done / max(1, total)) * 30 / n_regions),
                                        f"{done}/{total} samples | {speed:.1f} smp/s | "
                                        f"ETA: {int(eta//60)}m {int(eta%60)}s"
                                    )
                                    # Also update main progress bar
                                    self.update_progress(
                                        value=done * 100 // max(1, total),
                                        text=f"LLM: {done}/{total} | {speed:.1f} smp/s | ETA {int(eta)}s")

                                self.ai_agent.MAX_WORKERS = getattr(self, '_llm_workers', 0)

                                ai_labels = self.ai_agent.process_samples(
                                    meta_df,
                                    fields=self._extraction_fields,
                                    progress_fn=_extraction_progress,
                                    out_dir=os.path.join(
                                        self.results_dir, "extraction",
                                        f"region_{r_idx+1}"),
                                    metadata_columns=getattr(
                                        self, '_extraction_columns', None),
                                    enrich=getattr(
                                        self, '_extraction_enrich', False),
                                    category=self._platform_facts(
                                        region.get('platform', '')
                                    ).get('category', ''))

                                if not ai_labels.empty:
                                    ai_labels = self.apply_semantic_clustering(ai_labels)

                                self.update_progress(value=0)  # safe: won't reset if another extraction owns the bar
                            except Exception as e:
                                self.enqueue_log(f"[LLM Warning] Region {region['label']}: {e}")

                    region['ai_labels_df'] = ai_labels

                # Only the connection is ours to close: open_geometadb keeps the
                # decompressed database and hands it to every later caller.
                thread_conn.close()

                # Step D: Launch window on main thread
                _update_status("Rendering analysis...", 95)

                # Attach platform-wide labels for enrichment analysis
                # Always pass if available - needed for "Total Platform" scope
                platform_labels_df = None
                if self.default_labels_df is not None:
                    platform_labels_df = self.default_labels_df
                elif self.platform_labels:
                    platform_labels_df = pd.concat(
                        list(self.platform_labels.values()), ignore_index=True)

                def launch():
                    try:
                        progress_win.destroy()
                    except tk.TclError:
                        pass

                    mode = "compare" if len(region_specs) > 1 else "analyze"
                    RegionAnalysisWindow(
                        parent=self,
                        app_ref=self,
                        regions_data=region_specs,
                        mode=mode,
                        platform_labels_df=platform_labels_df
                    )
                    self.enqueue_log(f"[Analysis] OK Launched analysis for {len(region_specs)} region(s)")

                self.after(0, launch)

            except Exception as e:
                import traceback
                self.enqueue_log(f"[Analysis Error] {traceback.format_exc()}")

                # `msg` defaults now: `e` no longer exists once the except
                # block ends, and this runs later on the Tk thread.
                def show_err(msg=str(e)):
                    try:
                        progress_win.destroy()
                    except tk.TclError:
                        pass
                    messagebox.showerror("Analysis Error", f"Error:\n\n{msg}", parent=popup)

                self.after(0, show_err)

        threading.Thread(target=process_regions, daemon=True).start()
    
    def _compare_regions_logic(self, popup):
        """Compares multiple selected regions using the unified analysis pipeline."""
        if not popup or not popup.active_selections:
            return

        total_regions = sum(len(ranges) for ranges in popup.active_selections.values())

        if total_regions < 2:
            messagebox.showinfo(
                "More Regions Needed",
                "Please select at least 2 regions to compare.\n\n"
                f"Current selections: {total_regions} region",
                parent=popup
            )
            return

        # Delegate to unified pipeline - it auto-detects compare mode when >1 region
        self._analyze_selected_range(popup)
    
    def _open_distribution_analysis(self):
        """Unified 'Distribution Analysis' entry point.

        Presents the two distribution tools - Compare Distributions and
        Distribution Classification - from a single button via a small popup
        menu next to the button.
        """
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Compare Distributions",
                         command=self.open_compare_window)
        menu.add_command(label="Distribution Classification",
                         command=self._open_dist_classification)
        btn = getattr(self, "dist_analysis_btn", None)
        try:
            if btn is not None:
                x = btn.winfo_rootx()
                y = btn.winfo_rooty() + btn.winfo_height()
                menu.tk_popup(x, y)
            else:
                menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def open_compare_window(self):
        """Opens Compare Distributions setup dialog - similar to Gene Explorer.
        User picks genes, platforms, batch correction, and label options.
        """
        available = self._discover_available_platforms()
        if not self.gpl_datasets and not available:
            messagebox.showinfo(
                "No Platforms Available",
                "No GPL platform data found.\n\n"
                "Either load a platform from the main window, or ensure\n"
                "platform data files (.csv.gz) are in your data directory.",
                parent=self)
            return

        # Check if window already open
        if hasattr(self, 'compare_window') and self.compare_window is not None:
            try:
                if self.compare_window.winfo_exists():
                    self.compare_window.lift()
                    self.compare_window.focus_force()
                    return
            except:
                self.compare_window = None

        # ── Setup Dialog ──
        dlg = tk.Toplevel(self)
        style_window(dlg)
        dlg.title("Compare Distributions - Setup")
        dlg.transient(self)

        # Instructions
        ttk.Label(dlg, text="Compare gene distributions across platforms and conditions",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 5))

        top_frame = ttk.Frame(dlg, padding=10)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # ── Platform Selection ──
        plat_frame = labelframe(top_frame, text="Select Platforms", padding=5)
        plat_frame.pack(fill=tk.X, pady=5)

        plat_check_frame = ttk.Frame(plat_frame)
        plat_check_frame.pack(fill=tk.X)

        gpls_loaded = sorted(self.gpl_datasets.keys())
        gpls_available = sorted(k for k in available.keys() if k not in self.gpl_datasets)
        plat_vars = {}

        row_idx = 0; col_idx = 0
        for plat in gpls_loaded:
            var = tk.BooleanVar(value=True)
            n = len(self.gpl_datasets[plat])
            ttk.Checkbutton(plat_check_frame, text=f"{plat} ({n:,} samples)",
                            variable=var).grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=2)
            plat_vars[plat] = var
            col_idx += 1
            if col_idx >= 3: col_idx = 0; row_idx += 1

        if gpls_available:
            row_idx += 1
            ttk.Label(plat_check_frame, text="── Quick Gene Load (not fully loaded) ──",
                      font=('Segoe UI', 8, 'italic'), foreground='#888'
                      ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, padx=10, pady=(4, 2))
            row_idx += 1; col_idx = 0
            for plat in gpls_available:
                var = tk.BooleanVar(value=False)
                ttk.Checkbutton(plat_check_frame, text=f"{plat} (gene-only load)",
                                variable=var).grid(row=row_idx, column=col_idx, sticky=tk.W, padx=10, pady=2)
                plat_vars[plat] = var
                col_idx += 1
                if col_idx >= 3: col_idx = 0; row_idx += 1

        # Add Data Directory button
        dir_row = ttk.Frame(plat_frame)
        dir_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(dir_row, text="+ Add Data Directory...",
                   command=lambda: self._add_data_directory(),
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)

        # ── Gene Input ──
        gene_frame = labelframe(top_frame, text="Enter Gene Symbols", padding=5)
        gene_frame.pack(fill=tk.X, pady=5)

        ttk.Label(gene_frame, text="Gene symbols (comma-separated, e.g., TP53, BRCA1, EGFR):",
                  font=('Segoe UI', 9, 'italic')).pack(fill=tk.X, padx=5)
        gene_entry = ttk.Entry(gene_frame, font=('Consolas', 11))
        gene_entry.pack(fill=tk.X, padx=5, pady=5)

        # ── Options ──
        opts_frame = labelframe(top_frame, text="Options", padding=5)
        opts_frame.pack(fill=tk.X, pady=5)

        batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts_frame,
                        text="Apply batch correction (median centering) for cross-platform comparison",
                        variable=batch_var).pack(anchor=tk.W, padx=5)

        # Label status
        label_status = ttk.Frame(opts_frame)
        label_status.pack(fill=tk.X, padx=5, pady=(8, 2))

        has_labels = bool(self.platform_labels) or self.default_labels_df is not None
        if has_labels and self.label_source_var.get() == "file":
            plats = ', '.join(sorted(self.platform_labels.keys())) if self.platform_labels else "default"
            ttk.Label(label_status,
                      text=f"Labels loaded: {plats} - full comparison with PCA, enrichment, etc.",
                      font=('Segoe UI', 9, 'bold'), foreground='green').pack(anchor=tk.W)
            compare_mode = "labels"
        else:
            ttk.Label(label_status,
                      text="No labels loaded - will compare expression distributions only\n"
                           "(distribution shape, statistics, classification, overlap)",
                      font=('Segoe UI', 9), foreground='#888').pack(anchor=tk.W)
            compare_mode = "expression"

        # ── Buttons ──
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _run():
            genes_text = gene_entry.get().strip()
            if not genes_text:
                messagebox.showwarning("No Genes", "Enter at least one gene symbol.", parent=dlg)
                return

            selected_plats = [p for p, v in plat_vars.items() if v.get()]
            if not selected_plats:
                messagebox.showwarning("No Platforms", "Select at least one platform.", parent=dlg)
                return

            genes = [g.strip().upper() for g in genes_text.replace(';', ',').split(',') if g.strip()]
            do_batch = batch_var.get()

            dlg.destroy()
            self._launch_compare_analysis(selected_plats, genes, do_batch, compare_mode)

        ttk.Button(btn_frame, text="Run Comparison", command=_run,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        self._fit_window(dlg, 700, 550)

    def _launch_compare_analysis(self, platforms, genes, batch_correct, mode):
        """Load gene data, build groups, and launch comparison with auto-run."""
        self.enqueue_log(f"[Compare] Platforms: {platforms}, Genes: {genes}, "
                         f"Batch: {batch_correct}, Mode: {mode}")

        # Quick-load genes for platforms not fully loaded
        available = self.gpl_available_files
        for plat in platforms:
            if plat not in self.gpl_datasets:
                fpath = available.get(plat)
                if fpath:
                    self.enqueue_log(f"[Compare] Quick-loading {plat} for genes: {genes}")
                    self._quick_load_genes(plat, genes, fpath)

        # ── Build expression data per gene/platform ──
        import numpy as np
        compare_data = {}  # {key: {gene, platform, values, col, df_ref}}
        for plat in platforms:
            if plat in self.gpl_datasets:
                df = self.gpl_datasets[plat]
            elif plat in self.gpl_gene_cache:
                df = self.gpl_gene_cache[plat]
            else:
                self.enqueue_log(f"[Compare] {plat}: no data available")
                continue

            gene_mapping = self.gpl_gene_mappings.get(plat, {})
            cache_mapping = self.gpl_gene_mappings.get(f"_cache_{plat}", {})
            all_mappings = {**gene_mapping, **cache_mapping}

            for gene in genes:
                col = all_mappings.get(gene.upper())
                if col is None:
                    for g, c in all_mappings.items():
                        if g.upper() == gene.upper():
                            col = c; break
                if col and col in df.columns:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    if not vals.empty:
                        gsm_col = 'GSM' if 'GSM' in df.columns else None
                        gsms = df.loc[vals.index, 'GSM'].tolist() if gsm_col else [f"S{i}" for i in range(len(vals))]
                        key = f"{gene} / {plat}"
                        compare_data[key] = {
                            'gene': gene, 'platform': plat, 'col': col,
                            'values': vals, 'gsms': gsms,
                            'n': len(vals), 'mean': vals.mean(),
                            'std': vals.std(), 'median': vals.median(),
                        }
                        self.enqueue_log(f"[Compare] {key}: {len(vals):,} samples")
                else:
                    self.enqueue_log(f"[Compare] {gene} not found on {plat}")

        if not compare_data:
            messagebox.showerror("No Data",
                                "No gene data found for the selected genes/platforms.\n"
                                "Check gene symbols and platform availability.",
                                parent=self)
            return

        # ── Batch correction ──
        if batch_correct and len(platforms) > 1:
            self.enqueue_log("[Compare] Applying batch correction (median centering)...")
            plat_medians = {}
            for key, data in compare_data.items():
                plat = data['platform']
                if plat not in plat_medians:
                    plat_medians[plat] = []
                plat_medians[plat].append(data['values'].median())
            for plat in plat_medians:
                plat_medians[plat] = np.median(plat_medians[plat])
            global_median = np.median(list(plat_medians.values()))
            for key, data in compare_data.items():
                plat = data['platform']
                shift = global_median - plat_medians[plat]
                data['values'] = data['values'] + shift
                data['mean'] = data['values'].mean()
                data['median'] = data['values'].median()
                if abs(shift) > 0.001:
                    self.enqueue_log(f"[Compare] {plat}: shifted by {shift:+.3f}")

        # ── Launch window ──
        self.enqueue_log("[Compare] Launching comparison window...")
        try:
            import matplotlib.pyplot as _plt
            _plt.close('all')

            self.compare_window = CompareDistributionsWindow(self, self, skip_autoload=True)
            win = self.compare_window

            # ── Populate groups directly from gene/platform combos ──
            # Each gene/platform becomes a group - this is what gets compared
            win.user_defined_groups = {}
            win.loaded_files_listbox.delete(0, tk.END)

            for key, data in compare_data.items():
                label = f"{key} (n={data['n']:,})"
                win.user_defined_groups[label] = {
                    'gsms': data['gsms'],
                    'platform': data['platform'],
                    'raw_val': key,
                }
                win.loaded_files_listbox.insert(tk.END, label)

            # Select all groups
            win.loaded_files_listbox.select_set(0, tk.END)

            # Set gene entry
            win.gene_entry.delete(0, tk.END)
            win.gene_entry.insert(0, ', '.join(genes))

            # Set platform checkboxes
            for p, v in win.platform_vars.items():
                v.set(p in platforms)

            # ── Build labels / GSE browser if labels available ──
            labels_df = None
            if mode == "labels":
                labels_df = self._build_labels_for_compare()
                if labels_df is not None and not labels_df.empty:
                    win.full_dataset = labels_df
                    for col in ['Condition', 'Tissue', 'Treatment', 'series_id']:
                        if col in labels_df.columns and 2 <= labels_df[col].nunique() <= 50:
                            win.grouping_column = col
                            win.lbl_grouping.config(text=col, foreground=AERO["green_dark"])
                            break
                    try:
                        win._refresh_data_table()
                    except Exception as e:
                        # Swallowed silently this left the Data & Grouping tab
                        # blank with no clue why. Match the GSE branch below
                        # and say so in the log.
                        self.enqueue_log(f"[Compare] Data table: {e}")

            # If no labels, build GSE data from GEOmetadb for the Data tab
            if win.full_dataset.empty and self.gds_conn:
                try:
                    self._build_gse_dataset_for_compare(win, platforms)
                except Exception as e:
                    self.enqueue_log(f"[Compare] GSE dataset: {e}")

            # Store data refs
            win._compare_data = compare_data
            win._compare_genes = genes
            win._compare_platforms = platforms
            win._compare_mode = mode

            # ── Auto-run analysis ──
            win.status_label.config(text=f"Ready: {len(compare_data)} distributions loaded")
            self.enqueue_log(f"[Compare] OK {len(compare_data)} distributions ready - click RUN ANALYSIS")
            win.lift()
            win.focus_force()

        except Exception as e:
            self.enqueue_log(f"[Compare] Error: {e}")
            import traceback
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    #  Cross-Platform Advanced Analysis - opener
    # ═══════════════════════════════════════════════════════════════
    def _open_cross_platform_analysis(self):
        """Launch the Cross-Platform Analysis window. Requires >=2 loaded platforms."""
        try:
            loaded = list(self.gpl_datasets.keys()) if getattr(self, "gpl_datasets", None) else []
            if len(loaded) < 2:
                messagebox.showinfo(
                    "Cross-Platform Analysis",
                    "Load at least two GPL platforms before running Cross-Platform Analysis.",
                    parent=self)
                return
            CrossPlatformAnalysisWindow(self)
        except Exception as e:
            self.enqueue_log(f"[CrossPlatform] Error opening window: {e}")
            import traceback; traceback.print_exc()
            messagebox.showerror("Cross-Platform Analysis",
                                 f"Failed to open window:\n{e}", parent=self)

    # ═══════════════════════════════════════════════════════════════
    #  ARCHS4 RNA-seq ingestion window
    # ═══════════════════════════════════════════════════════════════
    def _fill_meta_from_sidecar(self, ds_name, gsm_ids, samples_df):
        """Top up `samples_df` from a dataset's own sample-metadata sidecar.

        Returns `samples_df` unchanged when there is no sidecar, so platforms
        that rely wholly on GEOmetadb are unaffected.
        """
        path = (getattr(self, "gpl_source_paths", {}) or {}).get(ds_name)
        if not path:
            return samples_df
        sidecar = str(path).replace(".csv.gz", "_sample_meta.csv.gz")
        if not os.path.exists(sidecar):
            return samples_df
        try:
            extra = pd.read_csv(sidecar, compression="gzip", low_memory=False)
        except Exception as e:
            self.enqueue_log(f"[{ds_name}] Sample metadata sidecar unreadable: {e}")
            return samples_df
        if extra.empty or "gsm" not in extra.columns:
            return samples_df

        extra["gsm"] = extra["gsm"].astype(str).str.strip().str.upper()
        extra = extra[extra["gsm"].isin({str(g).upper() for g in gsm_ids})]

        have = set()
        if samples_df is not None and not samples_df.empty:
            gcol = "gsm" if "gsm" in samples_df.columns else (
                "GSM" if "GSM" in samples_df.columns else None)
            if gcol:
                have = set(samples_df[gcol].astype(str).str.upper())
        missing = extra[~extra["gsm"].isin(have)]
        if missing.empty:
            return samples_df

        merged = (missing if samples_df is None or samples_df.empty
                  else pd.concat([samples_df, missing], ignore_index=True))
        self.enqueue_log(
            f"[{ds_name}] {len(missing):,} sample(s) absent from GEOmetadb "
            f"described from the dataset's own metadata sidecar")
        return merged

    def _open_archs4_window(self):
        """Open the ARCHS4 RNA-seq ingestion dialog."""
        try:
            from genevariate.core.sources import Archs4Source
        except Exception:
            Archs4Source = None
        if Archs4Source is None:
            messagebox.showerror(
                "ARCHS4 unavailable",
                "archs4py is not installed.\n\n"
                "Install with:\n  pip install 'genevariate[analysis]'",
                parent=self)
            return

        dlg = tk.Toplevel(self); dlg.title("Load RNA-seq from ARCHS4"); dlg.transient(self)
        style_window(dlg)
        ttk.Label(dlg, text="Fetch bulk RNA-seq counts from the ARCHS4 mirror",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 4))
        ttk.Label(dlg, text="First download is ~62 GB for human, ~51 GB for mouse "
                            "(one-time, cached afterwards).",
                  foreground="gray").pack(padx=15, pady=(0, 10))

        form = ttk.Frame(dlg); form.pack(padx=15, pady=5, fill="x")
        ttk.Label(form, text="Query (GSE accession or comma-separated GSMs):")\
            .grid(row=0, column=0, sticky="w")
        q_var = tk.StringVar(value="GSE64016")
        ttk.Entry(form, textvariable=q_var, width=40).grid(row=0, column=1, padx=5, pady=3)
        ttk.Label(form, text="Species:").grid(row=1, column=0, sticky="w")
        sp_var = tk.StringVar(value="human")
        ttk.Combobox(form, textvariable=sp_var, values=["human", "mouse"],
                     state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(form, text="Dataset name:").grid(row=2, column=0, sticky="w")
        name_var = tk.StringVar(value="ARCHS4_RNA-seq")
        ttk.Entry(form, textvariable=name_var, width=40).grid(row=2, column=1, padx=5, pady=3)

        status = ttk.Label(dlg, text="", foreground="#0A5B9A"); status.pack(padx=15, pady=(4, 0))
        pbar = ttk.Progressbar(dlg, mode="determinate", length=420)
        pbar.pack(padx=15, pady=(2, 6))

        def _progress(msg, frac):
            self.after(0, lambda: (status.config(text=msg),
                                    pbar.configure(value=max(0, min(100, int(frac * 100))))))

        def _run():
            q = q_var.get().strip()
            if not q:
                messagebox.showwarning("Missing query", "Provide a GSE or GSM list.", parent=dlg); return
            ds_name = (name_var.get() or "").strip() or "ARCHS4_RNA-seq"
            if ds_name in self.gpl_datasets:
                messagebox.showerror("Name exists", f"'{ds_name}' is already loaded.", parent=dlg); return
            go_btn.config(state="disabled")

            def _worker():
                try:
                    src = Archs4Source(species=sp_var.get())
                    query = q if q.upper().startswith("GSE") else [s.strip() for s in q.split(",") if s.strip()]
                    df = src.fetch(query, progress=_progress)
                    out_dir = os.path.join(self.results_dir, "archs4")
                    fname = f"{ds_name.lower().replace(' ', '_')}.csv.gz"
                    prov = df.attrs.get("provenance")
                    path = src.save_csv(df, out_dir, fname, provenance=prov)

                    # Keep ARCHS4's harmonised submitter text next to the
                    # matrix under the same name the loader looks for, so
                    # label extraction has something to read even for samples
                    # newer than the local GEOmetadb snapshot.
                    smeta = df.attrs.get("sample_meta")
                    n_meta = 0
                    if smeta is not None and len(smeta):
                        smeta.to_csv(path.replace(".csv.gz", "_sample_meta.csv.gz"),
                                     index=False, compression="gzip")
                        n_meta = len(smeta)

                    self.after(0, lambda: (dlg.destroy(),
                                            self._load_gpl_data(ds_name, path)))
                    ver = (prov or {}).get("h5_version", "unknown")
                    self.enqueue_log(
                        f"[ARCHS4] Saved {df.shape[0]} samples × {df.shape[1] - 2} genes → {path} "
                        f"(ARCHS4 v{ver}; provenance .meta.json + "
                        f"{n_meta} rows of sample metadata written)"
                    )
                except Exception as e:
                    self.after(0, lambda err=e: (go_btn.config(state="normal"),
                                                  status.config(text=f"Error: {err}", foreground="red")))
                    import traceback; traceback.print_exc()

            import threading
            threading.Thread(target=_worker, daemon=True).start()

        btns = ttk.Frame(dlg); btns.pack(pady=(4, 12))
        go_btn = ttk.Button(btns, text="Download & Load", command=_run, style="Action.TButton")
        go_btn.pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)

    def _open_display_limits(self):
        """How much of each result the program draws, set by the user.

        The setting is global and persisted, so it is reachable from the menu
        and not only from the window that happens to be open. Windows mark
        their tabs stale and redraw when they are next looked at.
        """
        from genevariate.gui import display_limits
        display_limits.open_dialog(
            self,
            on_change=lambda: self.enqueue_log(
                "[Display limits] changed; open windows redraw when revisited"))

    # ═══════════════════════════════════════════════════════════════
    #  Enrichment analysis window
    # ═══════════════════════════════════════════════════════════════
    def _open_enrichment_window(self):
        """Open the enrichment dialog: mean-based and bimodality-gated."""
        from genevariate.core.analysis import (
            run_enrichr, run_prerank_gsea, rank_genes_by_condition,
            enrichment_report_markdown, DEFAULT_LIBRARIES,
            classify_distributions, filter_ranked_by_distribution,
        )

        loaded = list(self.gpl_datasets.keys()) if getattr(self, "gpl_datasets", None) else []
        if not loaded:
            messagebox.showinfo("Enrichment",
                                "Load a platform first (Tools → Download GPL / ARCHS4 / Add Custom).",
                                parent=self); return

        dlg = tk.Toplevel(self); dlg.title("Run Enrichment Analysis"); dlg.transient(self)
        style_window(dlg)
        ttk.Label(dlg, text="Pathway / Ontology Enrichment",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 2))
        ttk.Label(dlg,
                  text="Four ranking methods - see method dropdown for details.",
                  foreground="gray").pack(padx=15, pady=(0, 8))

        form = ttk.Frame(dlg); form.pack(padx=15, pady=5, fill="x")

        ttk.Label(form, text="Method:").grid(row=0, column=0, sticky="w")
        METHOD_OPTS = [
            "mean (standard GSEA, t-statistic)",
            "mean (moderated t, empirical Bayes)",
            "bimodality-gated mean GSEA",
        ]
        method_var = tk.StringVar(value=METHOD_OPTS[0])
        ttk.Combobox(form, textvariable=method_var, values=METHOD_OPTS,
                     state="readonly", width=48).grid(row=0, column=1, padx=5, pady=3, sticky="w")

        ttk.Label(form, text="Platform:").grid(row=1, column=0, sticky="w")
        plat_var = tk.StringVar(value=loaded[0])
        ttk.Combobox(form, textvariable=plat_var, values=loaded,
                     state="readonly", width=32).grid(row=1, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(form, text="  (meta methods use ALL loaded platforms)",
                  foreground="gray").grid(row=1, column=2, sticky="w")

        ttk.Label(form, text="Case label:").grid(row=2, column=0, sticky="w")
        case_var = tk.StringVar(value="case")
        ttk.Entry(form, textvariable=case_var, width=32).grid(row=2, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(form, text="Control label:").grid(row=3, column=0, sticky="w")
        ctrl_var = tk.StringVar(value="control")
        ttk.Entry(form, textvariable=ctrl_var, width=32).grid(row=3, column=1, padx=5, pady=3, sticky="w")

        ttk.Label(form, text="Gene-set libraries (comma-sep):")\
            .grid(row=4, column=0, sticky="w")
        libs_var = tk.StringVar(value=",".join(DEFAULT_LIBRARIES))
        ttk.Entry(form, textvariable=libs_var, width=60).grid(row=4, column=1, padx=5, pady=3, sticky="w", columnspan=2)

        ttk.Label(form, text="Top genes for ORA:").grid(row=5, column=0, sticky="w")
        topn_var = tk.IntVar(value=250)
        ttk.Entry(form, textvariable=topn_var, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=3)

        status = ttk.Label(dlg, text="", foreground="#0A5B9A"); status.pack(padx=15, pady=(4, 0))

        def _sample_labels_for(df):
            labels = self._curated_sample_labels()
            if not labels and "condition" in df.columns:
                labels = {str(g).upper(): str(c)
                          for g, c in zip(df["GSM"], df["condition"])
                          if isinstance(c, str)}
            return labels

        def _run():
            method_txt = method_var.get()
            libs = [s.strip() for s in libs_var.get().split(",") if s.strip()]
            case_lbl, ctrl_lbl = case_var.get().strip(), ctrl_var.get().strip()
            case_fn, ctrl_fn = _safe_filename(case_lbl), _safe_filename(ctrl_lbl)
            topn = max(10, int(topn_var.get() or 250))
            go_btn.config(state="disabled")

            def _worker():
                try:
                    out_dir = os.path.join(self.results_dir, "enrichment")
                    os.makedirs(out_dir, exist_ok=True)

                    # ─── Single-platform paths ──────────────────────
                    plat = plat_var.get()
                    df = self.gpl_datasets.get(plat)
                    if df is None or df.empty:
                        raise RuntimeError(f"Platform '{plat}' has no data.")
                    labels = _sample_labels_for(df)
                    if not labels:
                        raise RuntimeError("No sample labels found. Run Tools → Curate Labels first.")

                    # Bimodality-gated
                    if method_txt.startswith("bimodality"):
                        self.after(0, lambda: status.config(text="Classifying distributions..."))
                        tags = classify_distributions(df)
                        ranked = rank_genes_by_condition(df, labels, case_lbl, ctrl_lbl)
                        gated  = filter_ranked_by_distribution(ranked, tags,
                                                                keep=("Bimodal", "Multimodal"))
                        if len(gated) < 15:
                            raise RuntimeError(
                                f"Too few bimodal genes ({len(gated)}) on {plat}. "
                                "Use a larger platform or relax keep=() tags.")
                        self.after(0, lambda: status.config(text=f"GSEA on {len(gated)} bimodal genes..."))
                        gsea = run_prerank_gsea(gated, gene_sets=libs)
                        comp = f"{plat}__{case_fn}_vs_{ctrl_fn}__bimodal"
                        md_path = os.path.join(out_dir, f"bimodality_{comp}.md")
                        enrichment_report_markdown(pd.DataFrame(), gsea, comp,
                                                     out_path=md_path,
                                                     ranked=ranked)
                        self.enqueue_log(
                            f"[Bimodality-gated] {ranked.attrs.get('note', '')}")
                        tags.to_csv(os.path.join(out_dir, f"bimodality_{comp}_tags.csv"),
                                    header=["distribution"])
                        if gsea is not None and not gsea.empty:
                            gsea.to_csv(os.path.join(out_dir, f"bimodality_{comp}_gsea.csv"), index=False)
                        self.enqueue_log(f"[Bimodality-gated] Report → {md_path}")
                        self.after(0, lambda: (status.config(text=f"Saved → {md_path}", foreground="#2E7D32"),
                                                go_btn.config(state="normal")))
                        return

                    # Mean-based (default standard GSEA + ORA)
                    use_moderated = "moderated" in method_txt
                    self.after(0, lambda: status.config(
                        text=f"Ranking by mean ({'moderated t' if use_moderated else 't-stat'})..."))
                    ranked = rank_genes_by_condition(df, labels, case_lbl, ctrl_lbl,
                                                      moderated=use_moderated)
                    top_up = list(ranked.head(topn).index)
                    self.after(0, lambda: status.config(text="Running Enrichr ORA..."))
                    ora = run_enrichr(top_up, gene_sets=libs)
                    self.after(0, lambda: status.config(text="Running GSEA prerank..."))
                    gsea = run_prerank_gsea(ranked, gene_sets=libs)
                    comp = f"{plat}__{case_fn}_vs_{ctrl_fn}"
                    md_path = os.path.join(out_dir, f"{comp}.md")
                    enrichment_report_markdown(ora, gsea, comp, out_path=md_path,
                                                ranked=ranked)
                    # Whether n counted studies or samples decides what the
                    # p-values mean, so it goes in the log too, not only the file.
                    self.enqueue_log(f"[Enrichment] {ranked.attrs.get('note', '')}")
                    ora.to_csv(os.path.join(out_dir, f"{comp}_enrichr.csv"), index=False)
                    if gsea is not None and not gsea.empty:
                        gsea.to_csv(os.path.join(out_dir, f"{comp}_gsea.csv"), index=False)
                    self.enqueue_log(f"[Enrichment] Report → {md_path}")
                    self.after(0, lambda: (status.config(text=f"Saved → {md_path}", foreground="#2E7D32"),
                                            go_btn.config(state="normal")))
                except Exception as e:
                    self.after(0, lambda err=e: (status.config(text=f"Error: {err}", foreground="red"),
                                                  go_btn.config(state="normal")))
                    import traceback; traceback.print_exc()

            import threading
            threading.Thread(target=_worker, daemon=True).start()

        btns = ttk.Frame(dlg); btns.pack(pady=(10, 12))
        go_btn = ttk.Button(btns, text="Run Enrichment", command=_run, style="Action.TButton")
        go_btn.pack(side="left", padx=5)
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="left", padx=5)

    # ═══════════════════════════════════════════════════════════════
    #  Pseudo-cohort discovery window (embedding-clustered labels)
    # ═══════════════════════════════════════════════════════════════
    def _open_label_enrichment(self):
        """Open the Label Enrichment analysis window.

        Tests whether LLM-extracted sample labels (tissue, condition, treatment,
        age-bin, …) are over- or under-represented in a chosen foreground
        (a gene's high-expression region, a subset of samples, or a platform).
        """
        try:
            LabelEnrichmentWindow(self)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Label Enrichment",
                f"Could not open Label Enrichment window:\n{exc}",
                parent=self,
            )

    # ═══════════════════════════════════════════════════════════════
    #  Single-cell (CELLxGENE) browser - real scRNA-seq ingest
    # ═══════════════════════════════════════════════════════════════
    def _open_cellxgene_browser(self):
        """Open the CELLxGENE Census browser to fetch single-cell data.

        Fetched data can be (a) pseudo-bulked and registered as a new
        platform in ``self.gpl_datasets`` so every existing analysis
        window sees it, or (b) explored cell-level via composition /
        UMAP / dot-plot / QC. All values are real measurements - see
        ``utils.pseudobulk`` for the aggregation semantics.
        """
        try:
            from genevariate.gui.windows.cellxgene_browser import CellxGeneBrowserWindow
            CellxGeneBrowserWindow(self)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Single-cell (CELLxGENE)",
                f"Could not open CELLxGENE browser:\n{exc}",
                parent=self,
            )

    def _build_agent_icon_frames(self, size=26, n=24, bob=3.0, wobble=6.0):
        """Pre-render the animation frames for the AI-Assistant launcher icon.

        Loads the branded chatbot glyph (``assets/chat-filled-256.png`` - a
        speech bubble + DNA double-helix + AI sparkle) once and bakes ``n``
        frames of a gentle sine bob (±``bob`` px) plus a slight rotation
        wobble (±``wobble``°). Frames are ``PhotoImage``s kept on ``self`` so
        Tk can't garbage-collect them. Degrades to no image (text-only button)
        if Pillow or the asset is unavailable - never raises.
        """
        import math
        self._agent_icon_frames = []
        self._agent_icon_i = 0
        try:
            import os
            from PIL import Image, ImageTk
            here = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(here, "assets", "chat-filled-256.png")
            base = Image.open(path).convert("RGBA").resize(
                (size, size), Image.LANCZOS)
            pad = int(math.ceil(bob)) + 2
            canvas_h = size + 2 * pad
            for k in range(n):
                t = 2 * math.pi * k / n
                dy = int(round(bob * math.sin(t)))
                ang = wobble * math.sin(t)
                rot = base.rotate(ang, resample=Image.BICUBIC, expand=False)
                frame = Image.new("RGBA", (size, canvas_h), (0, 0, 0, 0))
                frame.paste(rot, (0, pad + dy), rot)
                self._agent_icon_frames.append(ImageTk.PhotoImage(frame))
        except Exception:
            self._agent_icon_frames = []

    def _place_agent_launcher(self, w):
        """Pin the frameless AI-Assistant launcher (label + bobbing icon) to
        the header's right edge; the icon sits just left of the label."""
        cv = getattr(self, "_agent_header_canvas", None)
        txt = getattr(self, "_agent_text_item", None)
        ico = getattr(self, "_agent_icon_item", None)
        if cv is None or txt is None:
            return
        try:
            cy = int(cv.coords(txt)[1]) if cv.coords(txt) else 20
            cv.coords(txt, w - 18, cy)          # label right-aligned to the edge
            if ico is not None:
                x0 = cv.bbox(txt)[0]            # left x of the label
                cv.coords(ico, x0 - 8, cy)     # icon abuts the label, right-anchored
        except Exception:
            pass

    def _start_agent_icon_anim(self):
        """Cycle the AI-Assistant icon frames on a repeating ``after`` timer."""
        frames = getattr(self, "_agent_icon_frames", None)
        cv = getattr(self, "_agent_header_canvas", None)
        ico = getattr(self, "_agent_icon_item", None)
        if not frames or cv is None or ico is None:
            return

        def _tick():
            c = getattr(self, "_agent_header_canvas", None)
            it = getattr(self, "_agent_icon_item", None)
            fr = getattr(self, "_agent_icon_frames", None)
            if not fr or c is None or it is None or not c.winfo_exists():
                return
            self._agent_icon_i = (self._agent_icon_i + 1) % len(fr)
            try:
                c.itemconfig(it, image=fr[self._agent_icon_i])
            except Exception:
                return
            self._agent_icon_after = self.after(90, _tick)

        self._agent_icon_after = self.after(90, _tick)

    def _toggle_chat_sidebar(self):
        """Show/hide the conversational assistant sidebar (Ctrl+/).

        Lazily creates a :class:`ChatSidebar` inside ``self._content_row``
        (the wrapper around the main scroll area). The assistant proposes an
        analysis tool + params from a typed request and only runs it after the
        user confirms. Degrades to keyword routing when ollama is unavailable.
        """
        try:
            frame = getattr(self, "_chat_frame", None)
            if frame is None:
                return
            if getattr(self, "_chat_visible", False):
                frame.pack_forget()
                self._chat_visible = False
                return
            if getattr(self, "_chat_sidebar", None) is None:
                from genevariate.gui.windows.chat_sidebar import ChatSidebar
                self._chat_sidebar = ChatSidebar(frame, self)
                self._chat_sidebar.pack(fill=tk.BOTH, expand=True)
            frame.pack(side=tk.RIGHT, fill=tk.Y, before=self._main_vsb)
            self._chat_visible = True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                "Assistant",
                f"Could not open the assistant sidebar:\n{exc}",
                parent=self,
            )

    # ═══════════════════════════════════════════════════════════════
    #  Distribution Classification - per-gene statistics
    # ═══════════════════════════════════════════════════════════════
    def _open_dist_classification(self):
        """Open a window to classify distributions of every gene on a platform.
        Computes: normality, skewness, kurtosis, modality, mean, median, std, IQR.
        """
        available = self._discover_available_platforms()
        all_plats = sorted(set(list(self.gpl_datasets.keys()) + list(available.keys())))
        if not all_plats:
            messagebox.showinfo("No Platforms", "No platforms available.", parent=self)
            return

        # Setup dialog - pick platform
        dlg = tk.Toplevel(self)
        style_window(dlg)
        dlg.title("Distribution Classification - Setup")
        dlg.transient(self)

        ttk.Label(dlg, text="Classify gene distributions on a platform",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="Computes normality, skewness, kurtosis, modality,\n"
                            "and descriptive statistics for every gene.",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        plat_frame = labelframe(dlg, text="Select Platform", padding=5)
        plat_frame.pack(fill=tk.X, padx=15, pady=5)
        plat_var = tk.StringVar(value=all_plats[0] if all_plats else "")
        for p in all_plats:
            loaded = p in self.gpl_datasets
            n = len(self.gpl_datasets[p]) if loaded else 0
            text = f"{p} ({n:,} samples)" if loaded else f"{p} (will load)"
            ttk.Radiobutton(plat_frame, text=text, variable=plat_var, value=p).pack(
                anchor=tk.W, padx=10, pady=1)

        # Max genes
        opt_frame = ttk.Frame(dlg)
        opt_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(opt_frame, text="Max genes to analyze:", font=('Segoe UI', 9)).pack(side=tk.LEFT)
        max_genes_var = tk.IntVar(value=500)
        ttk.Entry(opt_frame, textvariable=max_genes_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(opt_frame, text="(set 0 for all)", font=('Segoe UI', 8, 'italic'),
                  foreground='#888').pack(side=tk.LEFT)

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _run():
            plat = plat_var.get()
            max_g = max_genes_var.get()
            dlg.destroy()
            self._run_dist_classification(plat, max_g)

        ttk.Button(btn_frame, text="Run Classification", command=_run,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=5)

        self._fit_window(dlg, 500, 400)

    def _run_dist_classification(self, plat_id, max_genes=500):
        """Compute distribution statistics for every gene on a platform.
        Uses BioAI_Engine.analyze_gene_distribution for classification
        (same as Gene Distribution Explorer).
        """
        # Ensure platform is loaded
        if plat_id not in self.gpl_datasets:
            available = self._discover_available_platforms()
            fpath = available.get(plat_id)
            if fpath:
                self.enqueue_log(f"[DistClass] Loading {plat_id}...")
                self._load_gpl_data(plat_id, fpath)
            else:
                messagebox.showerror("Not Found", f"Platform {plat_id} data not found.", parent=self)
                return

        df = self.gpl_datasets.get(plat_id)
        if df is None or df.empty:
            return

        gene_mapping = self.gpl_gene_mappings.get(plat_id, {})

        # ── Filter: ONLY genes, not metadata ──
        # Metadata columns to exclude (case-insensitive)
        _META_UPPER = {
            'GSM', 'GENE', 'SERIES_ID', 'GPL', '_PLATFORM', 'PLATFORM',
            'PLATFORMID', 'EXPERIMENTID', 'AGE', 'SEX', 'TISSUEID',
            'SAMPLEID', 'SUBJECTID', 'PATIENTID', 'BATCHID', 'GROUPID',
            'CONDITION', 'TISSUE', 'TREATMENT', 'TREATMENT_TIME',
            'ID_REF', 'IDENTIFIER', 'DESCRIPTION', 'CHROMOSOME',
            'UNNAMED: 0', 'INDEX', 'STATUS', 'TYPE',
        }

        if gene_mapping:
            # Use known gene->column mapping (most reliable)
            gene_cols = [(gene, col) for gene, col in gene_mapping.items()
                         if col in df.columns and gene.upper() not in _META_UPPER
                         and col.upper() not in _META_UPPER]
        else:
            # Fallback: numeric columns not in metadata
            gene_cols = []
            for c in df.columns:
                if c.upper() in _META_UPPER:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    gene_cols.append((c, c))

        if max_genes > 0 and len(gene_cols) > max_genes:
            gene_cols = gene_cols[:max_genes]

        self.enqueue_log(f"[DistClass] Analyzing {len(gene_cols)} genes on {plat_id} "
                         f"({len(df):,} samples, {len(gene_mapping)} in mapping)")

        # ── Compute stats using BioAI_Engine ──
        from scipy import stats as sp_stats
        results = []

        for i, (gene, col) in enumerate(gene_cols):
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) < 20:
                continue

            row = {'Gene': gene, 'N': len(vals)}
            row['Mean'] = round(vals.mean(), 4)
            row['Median'] = round(vals.median(), 4)
            row['Std'] = round(vals.std(), 4)
            row['Min'] = round(vals.min(), 4)
            row['Max'] = round(vals.max(), 4)
            row['IQR'] = round(vals.quantile(0.75) - vals.quantile(0.25), 4)
            row['Skewness'] = round(vals.skew(), 4)
            row['Kurtosis'] = round(vals.kurtosis(), 4)

            # ── Use BioAI_Engine for classification ──
            # Same logic as Gene Distribution Explorer
            classification = BioAI_Engine.analyze_gene_distribution(vals.values)
            row['Classification'] = classification

            results.append(row)

            if (i + 1) % 100 == 0:
                self.enqueue_log(f"[DistClass] {i+1}/{len(gene_cols)} genes classified...")

        if not results:
            messagebox.showinfo("No Results", "No genes with enough data to classify.", parent=self)
            return

        results_df = pd.DataFrame(results)
        self.enqueue_log(f"[DistClass] Classified {len(results)} genes on {plat_id}")

        # ── Count classifications ──
        class_counts = results_df['Classification'].value_counts()

        # ── Results Window ──
        win = tk.Toplevel(self)
        style_window(win)
        win.title(f"Distribution Classification - {plat_id} ({len(results)} genes)")
        win.geometry("1200x700")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"1200x700+{(_sw-1200)//2}+{(_sh-700)//2}")
        except: pass

        # Summary bar
        n_normal = class_counts.get('Normal', 0)
        n_bimodal = class_counts.get('Bimodal', 0)
        n_lognorm = class_counts.get('Lognormal', 0)
        n_gamma = class_counts.get('Gamma', 0)
        n_cauchy = class_counts.get('Cauchy', 0)
        n_multi = class_counts.get('Multimodal', 0)
        n_uniform = class_counts.get('Uniform', 0)

        summary = ttk.Frame(win)
        summary.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(summary, text=f"{plat_id}: {len(results)} genes analyzed  |  "
                                f"Normal: {n_normal} ({n_normal*100//max(1,len(results))}%)  |  "
                                f"Lognormal: {n_lognorm}  |  Gamma: {n_gamma}  |  "
                                f"Cauchy: {n_cauchy}  |  "
                                f"Bimodal: {n_bimodal}  |  Multimodal: {n_multi}  |  "
                                f"Uniform: {n_uniform}",
                  font=('Segoe UI', 10, 'bold')).pack()

        class_text = "  |  ".join(f"{k}: {v}" for k, v in class_counts.items())
        ttk.Label(summary, text=class_text, font=('Segoe UI', 9), foreground='#555').pack(pady=2)

        # ── 5 Representative Gene Distribution Plots ──
        try:
            # Pick 5 genes from different classification categories
            example_genes = []
            seen_classes = set()
            for _, row in results_df.iterrows():
                cls = row.get('Classification', '?')
                if cls not in seen_classes and cls not in ('?', 'Not Enough Data', 'Effectively Constant'):
                    example_genes.append(row)
                    seen_classes.add(cls)
                if len(example_genes) >= 5:
                    break
            # Fill remaining with highest skew
            if len(example_genes) < 5:
                for _, row in results_df.sort_values('Skewness', key=abs, ascending=False).iterrows():
                    if row['Gene'] not in [e['Gene'] for e in example_genes]:
                        example_genes.append(row)
                        if len(example_genes) >= 5:
                            break

            if example_genes:
                plot_frame = labelframe(win, text="Representative Gene Distributions", padding=5)
                plot_frame.pack(fill=tk.X, padx=10, pady=5)

                fig = Figure(figsize=(min(5, len(example_genes)) * 3.2, 4.5))
                axes = fig.subplots(1, min(5, len(example_genes)))
                if len(example_genes) == 1:
                    axes = [axes]

                colors = ['#1565C0', '#C62828', '#2E7D32', '#00838F', '#7B1FA2']
                for idx, (gene_row, ax_i) in enumerate(zip(example_genes, axes)):
                    gene_name = gene_row['Gene']
                    gene_col = gene_mapping.get(gene_name, gene_name)
                    if gene_col in df.columns:
                        vals = pd.to_numeric(df[gene_col], errors='coerce').dropna()
                        if len(vals) > 10:
                            try:
                                kde = robust_kde(vals)
                                xs = np.linspace(vals.min(), vals.max(), 300)
                                ys = kde(xs)
                                ys = ys / ys.max()
                                clr = colors[idx % len(colors)]
                                ax_i.fill_between(xs, ys, alpha=0.3, color=clr)
                                ax_i.plot(xs, ys, color=clr, lw=2)
                                ax_i.axvline(vals.mean(), color='black', ls='--', lw=1, alpha=0.5)
                                ax_i.axvline(vals.median(), color='gray', ls=':', lw=1, alpha=0.5)
                            except:
                                ax_i.hist(vals, bins=50, density=True, alpha=0.5,
                                          color=colors[idx % len(colors)])

                    cls = gene_row.get('Classification', '?')
                    skew = gene_row.get('Skewness', 0)
                    ax_i.set_title(f"{gene_name}\n{cls}", fontsize=8, weight='bold')
                    ax_i.set_xlabel(f"sk={skew:.1f}", fontsize=7)
                    ax_i.set_yticks([])
                    ax_i.tick_params(axis='x', labelsize=7)

                fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.35, hspace=0.3)
                fig.suptitle(f"{plat_id} - Example Distributions by Classification Type",
                             fontsize=10, weight='bold', y=0.92)

                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, plot_frame)
                canvas.draw()
                viz_make_interactive(fig)
                canvas.get_tk_widget().pack(fill=tk.X, padx=5, pady=(5, 10))
                win._dist_fig = fig
                win._dist_canvas = canvas
        except Exception as e:
            self.enqueue_log(f"[DistClass] Plot error: {e}")

        # Treeview with results
        tv_frame = ttk.Frame(win)
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cols = ['Gene', 'N', 'Mean', 'Median', 'Std', 'IQR', 'Skewness', 'Kurtosis', 'Classification']
        tree = ttk.Treeview(tv_frame, columns=cols, show="headings", height=25)
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tv_frame.grid_rowconfigure(0, weight=1)
        tv_frame.grid_columnconfigure(0, weight=1)

        col_widths = {'Gene': 110, 'N': 70, 'Mean': 85, 'Median': 85, 'Std': 75,
                      'IQR': 75, 'Skewness': 85, 'Kurtosis': 85, 'Classification': 140}
        for c in cols:
            tree.heading(c, text=c, command=lambda _c=c: self._sort_dist_tree(tree, results_df, _c))
            tree.column(c, width=col_widths.get(c, 80), anchor='center' if c != 'Gene' else 'w')

        # Enable multi-select
        tree.configure(selectmode='extended')

        # A platform can carry tens of thousands of genes; a Treeview stalls if
        # every one is inserted. Show the first _DIST_TREE_CAP (sorting reorders
        # the full frame first, so the visible slice is always the top by the
        # sorted column) and note the remainder.
        _DIST_TREE_CAP = 2000
        for _, row in results_df.head(_DIST_TREE_CAP).iterrows():
            vals = [str(row.get(c, '')) for c in cols]
            tree.insert("", tk.END, values=vals)
        if len(results_df) > _DIST_TREE_CAP:
            note = [f"… {len(results_df) - _DIST_TREE_CAP:,} more genes "
                    f"(table capped at {_DIST_TREE_CAP:,}; sort a column to rank)"] \
                   + [""] * (len(cols) - 1)
            tree.insert("", tk.END, values=note)

        # Double-click gene → open in Gene Distribution Explorer
        def _on_gene_dblclick(event):
            selected = tree.selection()
            if not selected:
                return
            genes = []
            for item in selected:
                vals = tree.item(item, 'values')
                if vals:
                    genes.append(vals[0])  # Gene column
            if genes:
                self._plot_genes_in_explorer(plat_id, genes)

        tree.bind("<Double-1>", _on_gene_dblclick)
        ttk.Label(tv_frame, text="Double-click gene(s) to plot in Gene Distribution Explorer  |  "
                                  "Ctrl+click to select multiple, then double-click to plot all",
                  font=('Segoe UI', 8, 'italic'), foreground='#888').grid(
            row=2, column=0, sticky='w', padx=5, pady=2)

        # Buttons
        btn_frame = ttk.Frame(win, padding=5)
        btn_frame.pack(fill=tk.X)

        def _plot_selected():
            selected = tree.selection()
            genes = [tree.item(item, 'values')[0] for item in selected if tree.item(item, 'values')]
            if genes:
                self._plot_genes_in_explorer(plat_id, genes)
            else:
                messagebox.showinfo("No Selection", "Select one or more genes first.", parent=win)

        ttk.Button(btn_frame, text="Plot Selected Genes", command=_plot_selected,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)

        def _save_csv():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                initialfile=f"{plat_id}_distribution_classification.csv", parent=win)
            if path:
                results_df.to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved {len(results_df)} genes to:\n{path}", parent=win)

        def _save_xlsx():
            path = filedialog.asksaveasfilename(
                defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
                initialfile=f"{plat_id}_distribution_classification.xlsx", parent=win)
            if path:
                try:
                    results_df.to_excel(path, index=False, sheet_name=plat_id[:30])
                    messagebox.showinfo("Saved", f"Saved {len(results_df)} genes to:\n{path}", parent=win)
                except ImportError:
                    messagebox.showwarning("openpyxl Required",
                                           "Install openpyxl for Excel export:\npip install openpyxl\n\n"
                                           "Use 'Save as CSV' instead.", parent=win)

        ttk.Button(btn_frame, text="Save as CSV", command=_save_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save as Excel (.xlsx)", command=_save_xlsx).pack(side=tk.LEFT, padx=5)
        def _close_dist():
            if hasattr(win, '_dist_fig'):
                plt.close(win._dist_fig)
            win.destroy()
        ttk.Button(btn_frame, text="Close", command=_close_dist).pack(side=tk.RIGHT, padx=5)

    def _plot_genes_in_explorer(self, plat_id, genes):
        """Open Gene Distribution Explorer with the specified genes pre-filled and auto-plot."""
        # Open or bring to front
        self.show_gene_distribution_popup()
        popup = self.gene_dist_popup_root
        if popup is None or not popup.winfo_exists():
            return

        # Check the platform checkbox
        if hasattr(popup, 'gpl_selection_vars') and plat_id in popup.gpl_selection_vars:
            popup.gpl_selection_vars[plat_id].set(True)

        # Set gene entry
        gene_text = ', '.join(genes[:10])  # Cap at 10 genes for readability
        if hasattr(popup, 'current_gene_entry'):
            popup.current_gene_entry.delete(0, tk.END)
            popup.current_gene_entry.insert(0, gene_text)

        # Auto-plot after a short delay (let the window render)
        popup.after(300, lambda: self._plot_histograms(popup))

    def _sort_dist_tree(self, tree, df, col):
        """Sort distribution classification treeview by column."""
        cols = list(df.columns)
        if col not in cols:
            return
        try:
            sorted_df = df.sort_values(col, ascending=not getattr(self, '_dist_sort_asc', True))
            self._dist_sort_asc = not getattr(self, '_dist_sort_asc', True)
        except:
            return
        tree.delete(*tree.get_children())
        display_cols = ['Gene', 'N', 'Mean', 'Median', 'Std', 'IQR', 'Skewness', 'Kurtosis', 'Classification']
        _DIST_TREE_CAP = 2000
        for _, row in sorted_df.head(_DIST_TREE_CAP).iterrows():
            vals = [str(row.get(c, '')) for c in display_cols]
            tree.insert("", tk.END, values=vals)
        if len(sorted_df) > _DIST_TREE_CAP:
            note = [f"… {len(sorted_df) - _DIST_TREE_CAP:,} more genes "
                    f"(table capped at {_DIST_TREE_CAP:,})"] \
                   + [""] * (len(display_cols) - 1)
            tree.insert("", tk.END, values=note)

    def _build_gse_dataset_for_compare(self, win, platforms):
        """Build a minimal GSE dataset from GEOmetadb for the Data tab (no labels mode)."""
        all_gsms = set()
        for plat in platforms:
            df = self.gpl_datasets.get(plat) or self.gpl_gene_cache.get(plat)
            if df is not None and 'GSM' in df.columns:
                all_gsms.update(df['GSM'].astype(str).str.upper().tolist())

        if not all_gsms or not self.gds_conn:
            return

        # Query GEOmetadb for GSM -> series_id + platform
        rows_all = []
        gsm_list = list(all_gsms)
        for i in range(0, len(gsm_list), 500):
            chunk = gsm_list[i:i+500]
            ph = ','.join(['?'] * len(chunk))
            try:
                rows = self.gds_conn.execute(
                    f"SELECT gsm, series_id, gpl FROM gsm WHERE UPPER(gsm) IN ({ph})",
                    [g.upper() for g in chunk]).fetchall()
                for gsm, gse, gpl in rows:
                    rows_all.append({'GSM': str(gsm).strip(), 'series_id': str(gse).strip(),
                                     'platform': str(gpl).strip()})
            except:
                pass

        if not rows_all:
            return

        gse_df = pd.DataFrame(rows_all)
        gse_df = gse_df[gse_df['series_id'].notna() & (gse_df['series_id'] != 'nan')]
        win.full_dataset = gse_df
        win._refresh_data_table()
        self.enqueue_log(f"[Compare] GSE dataset: {len(gse_df):,} samples, "
                         f"{gse_df['series_id'].nunique()} experiments")

    def _build_labels_for_compare(self):
        """Build a unified labels DataFrame from all available label sources.
        
        Returns DataFrame with GSM column + all label columns, matched to loaded platforms.
        """
        frames = []
        
        # Source 1: platform_labels (per-platform label DataFrames)
        if self.platform_labels:
            for plat_name, ldf in self.platform_labels.items():
                if ldf is not None and not ldf.empty and 'GSM' in ldf.columns:
                    df = ldf.copy()
                    if '_platform' not in df.columns:
                        df['_platform'] = plat_name
                    frames.append(df)
        
        # Source 2: default_labels_df (merged/legacy)
        if not frames and self.default_labels_df is not None and not self.default_labels_df.empty:
            frames.append(self.default_labels_df.copy())
        
        # Source 3: Build from platform data's metadata columns
        if not frames:
            for plat_name, plat_df in self.gpl_datasets.items():
                if 'GSM' not in plat_df.columns:
                    continue
                # Find string/object columns that look like labels
                meta_cols = ['GSM']
                for col in plat_df.columns:
                    if col == 'GSM':
                        continue
                    if plat_df[col].dtype == 'object':
                        n_unique = plat_df[col].nunique()
                        if 2 <= n_unique <= 100:
                            meta_cols.append(col)
                
                if len(meta_cols) > 1:  # has at least one label column
                    df = plat_df[meta_cols].copy()
                    df['_platform'] = plat_name
                    frames.append(df)
        
        if not frames:
            return None
        
        # Merge all frames
        result = pd.concat(frames, ignore_index=True)
        
        # Ensure GSM is clean
        if 'GSM' in result.columns:
            result['GSM'] = result['GSM'].astype(str).str.strip().str.upper()
            result = result.drop_duplicates(subset=['GSM'])
        
        # Only keep samples that exist in loaded platforms
        all_platform_gsms = set()
        for plat_name, plat_df in self.gpl_datasets.items():
            if 'GSM' in plat_df.columns:
                all_platform_gsms.update(plat_df['GSM'].astype(str).str.upper())
        
        if all_platform_gsms and 'GSM' in result.columns:
            before = len(result)
            result = result[result['GSM'].isin(all_platform_gsms)]
            if len(result) < before:
                self.enqueue_log(f"[UI] Labels filtered: {before} → {len(result)} "
                                 f"(matched to loaded platforms)")
        
        # Drop internal columns from display
        drop_cols = [c for c in result.columns
                     if c.startswith('_') or c in ('data_processing', 'contact',
                         'supplementary_file', 'data_row_count', 'channel_count',
                         'status', 'submission_date', 'last_update_date')]
        result = result.drop(columns=drop_cols, errors='ignore')
        
        return result
    def load_external_file_for_step2(self):
        """Smart external file loader: auto-detects labels vs expression data.
        
        - If mostly string/category columns → treat as LABEL file
          → auto-detect GPL from filename/content → store in platform_labels
          → prompt to load expression data if needed
        - If mostly numeric columns → treat as EXPRESSION data
          → load as custom platform
        """
        filepath = filedialog.askopenfilename(
            title="Select CSV file (labels or expression data)",
            filetypes=[("CSV files", "*.csv"), ("Compressed CSV", "*.csv.gz"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            compression = 'gzip' if filepath.endswith('.gz') else None
            df = pd.read_csv(filepath, compression=compression, low_memory=False)
            fname = os.path.basename(filepath)
            
            # Normalize GSM column
            gsm_col = None
            for c in df.columns:
                if c.lower().strip() in ('gsm', 'sample', 'sample_id', 'geo_accession', 'id'):
                    gsm_col = c
                    break
            if gsm_col is None:
                first = df.iloc[:, 0].astype(str)
                if first.str.upper().str.startswith('GSM').mean() > 0.3:
                    gsm_col = df.columns[0]
            
            if gsm_col:
                df.rename(columns={gsm_col: 'GSM'}, inplace=True)
                df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()
            
            # ── Classify file type: labels vs expression ──
            non_gsm_cols = [c for c in df.columns if c != 'GSM']
            n_numeric = sum(1 for c in non_gsm_cols if pd.api.types.is_numeric_dtype(df[c]))
            n_string = sum(1 for c in non_gsm_cols if df[c].dtype == 'object')
            
            # Heuristic: if >80% of columns are numeric → expression data
            # otherwise → label file
            is_expression = len(non_gsm_cols) > 5 and n_numeric / max(1, len(non_gsm_cols)) > 0.8
            
            self.enqueue_log(
                f"[Load] {fname}: {len(df):,} rows, {n_numeric} numeric cols, "
                f"{n_string} string cols → {'EXPRESSION' if is_expression else 'LABELS'}")
            
            if is_expression:
                # ── Expression data: load as platform ──
                # Try to detect GPL from filename
                m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
                if m:
                    gpl_id = m.group(1).upper()
                else:
                    gpl_id = simpledialog.askstring(
                        "Platform Name",
                        f"Enter a name for this dataset\n(e.g., 'GPL570' or 'MyStudy'):",
                        parent=self)
                    if not gpl_id:
                        return
                
                self._load_gpl_data(gpl_id.strip(), filepath)
                # Auto-register parent directory for future quick gene loads
                parent_dir = str(Path(filepath).parent)
                if parent_dir not in self._user_data_dirs:
                    self._user_data_dirs.append(parent_dir)
                    self.gpl_available_files.clear()
                    self.enqueue_log(f"[DataDir] Auto-registered: {parent_dir}")
                
            else:
                # ── Label file: detect GPL and integrate ──
                # Try to detect GPL from filename
                m = re.search(r'(GPL\d+)', fname, re.IGNORECASE)
                gpl_from_file = m.group(1).upper() if m else None
                
                # Try to detect GPL from GSMs matching loaded platforms
                gpl_from_match = None
                if 'GSM' in df.columns and self.gpl_datasets:
                    file_gsms = set(df['GSM'].astype(str).str.upper())
                    best_overlap = 0
                    for plat_id, plat_df in self.gpl_datasets.items():
                        if 'GSM' in plat_df.columns:
                            plat_gsms = set(plat_df['GSM'].astype(str).str.upper())
                            overlap = len(file_gsms & plat_gsms)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                gpl_from_match = plat_id
                    if best_overlap > 0:
                        self.enqueue_log(
                            f"[Load] Matched {best_overlap:,} GSMs to loaded platform {gpl_from_match}")
                
                # Try to detect GPL from GEOmetadb
                gpl_from_db = None
                if not gpl_from_file and not gpl_from_match and 'GSM' in df.columns and self.gds_conn:
                    sample_gsms = df['GSM'].head(50).tolist()
                    ph = ','.join(['?'] * len(sample_gsms))
                    try:
                        result = self.gds_conn.execute(
                            f"SELECT UPPER(gpl) as gpl, COUNT(*) as n FROM gsm "
                            f"WHERE UPPER(gsm) IN ({ph}) GROUP BY UPPER(gpl) "
                            f"ORDER BY n DESC LIMIT 1",
                            [g.upper() for g in sample_gsms]).fetchone()
                        if result:
                            gpl_from_db = result[0]
                            self.enqueue_log(f"[Load] GEOmetadb detected platform: {gpl_from_db}")
                    except Exception:
                        pass
                
                # Determine final GPL ID
                gpl_id = gpl_from_file or gpl_from_match or gpl_from_db
                
                if not gpl_id:
                    # Ask user
                    gpl_id = simpledialog.askstring(
                        "Platform ID",
                        f"Could not auto-detect the GPL platform for:\n{fname}\n\n"
                        f"Enter the GPL ID (e.g., GPL570, GPL96):\n"
                        f"(This is needed to match labels with expression data)",
                        parent=self)
                    if not gpl_id:
                        # Fall back: store without GPL prefix
                        gpl_id = fname.replace('.csv', '').replace('.gz', '')
                
                gpl_id = gpl_id.strip().upper()
                
                # User-provided labels are kept AS-IS - no harmonization.

                # Store as platform labels
                self.platform_labels[gpl_id] = df
                self._rebuild_merged_labels()
                self._refresh_labels_display()
                self.label_source_var.set("file")
                self._toggle_main_label_source()
                
                # Detect label columns for summary
                label_cols = [c for c in df.columns
                              if c != 'GSM' and (df[c].dtype == 'object' or df[c].nunique() < 200)
                              and df[c].nunique() > 1]
                
                self.enqueue_log(
                    f"[Load] OK Labels stored for {gpl_id}: {len(df):,} samples, "
                    f"columns: {label_cols}")
                
                messagebox.showinfo(
                    "Labels Loaded",
                    f"Loaded {len(df):,} samples from:\n{fname}\n\n"
                    f"Platform: {gpl_id}\n"
                    f"Label columns: {', '.join(label_cols[:8])}\n\n"
                    f"Labels are now available for all analysis tools.",
                    parent=self)
                
                # Auto-check expression data
                self.after(200, lambda p=gpl_id: self._ensure_expression_data_for_labels(p))
            
        except Exception as e:
            self.enqueue_log(f"[UI ERROR] Failed to load file: {e}")
            import traceback
            self.enqueue_log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load file:\n\n{e}", parent=self)
    
    def _show_log_window(self):
        """Show activity log window."""
        if self.log_window:
            self.log_window.deiconify()
            self.log_window.lift()
            self.log_window.focus_force()

    # ══════════════════════════════════════════════════════════════════
    #  Compare two loaded platforms (e.g. local file vs downloaded)
    # ══════════════════════════════════════════════════════════════════
    def _open_gpl_downloader_window(self):
        if hasattr(self, '_gpl_dl_window') and self._gpl_dl_window.winfo_exists():
            self._gpl_dl_window.deiconify()
            self._gpl_dl_window.lift()
            return
        win = tk.Toplevel(self)
        style_window(win)
        win.title("GeneVariate - Platform Downloader (any species, any technology)")
        win.geometry("1050x800")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"1050x800+{(_sw-1050)//2}+{(_sh-800)//2}")
            win.minsize(500, 400)
        except Exception: pass
        self._gpl_dl_window = win

        ttk.Label(win,
                  text="Download and preprocess an expression platform from "
                       "NCBI GEO - microarray, bulk RNA-seq or single-cell. "
                       "Search by species or enter a GPL ID directly.",
                  foreground="gray", font=('Segoe UI', 9, 'italic'),
                  wraplength=900).pack(padx=15, pady=(10, 5))

        # ── Direct platform download row ────────────────────────────
        direct_frame = labelframe(win, text="Direct Platform Download (by GPL ID)", padding=8)
        direct_frame.pack(fill=tk.X, padx=15, pady=5)

        inp = ttk.Frame(direct_frame)
        inp.pack(fill=tk.X)
        ttk.Label(inp, text="GPL ID:", font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self.auto_gpl_entry = ttk.Entry(inp, width=12, font=('Segoe UI', 11))
        self.auto_gpl_entry.pack(side=tk.LEFT, padx=8)
        self.auto_gpl_entry.insert(0, "GPL1355")
        ttk.Label(inp, text="Max GSEs (0=all):").pack(side=tk.LEFT, padx=(15, 0))
        self.auto_max_gse_entry = ttk.Entry(inp, width=6, font=('Segoe UI', 11))
        self.auto_max_gse_entry.pack(side=tk.LEFT, padx=5)
        self.auto_max_gse_entry.insert(0, "0")
        ttk.Button(inp, text="Download & Process",
                   command=self._auto_download_gpl,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=15)

        # ── Species Browser ─────────────────────────────────────────
        species_frame = labelframe(win, text="Browse Platforms by Species", padding=8)
        species_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Search row
        search_row = ttk.Frame(species_frame)
        search_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_row, text="Species / GPL ID:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self._species_entry = ttk.Entry(search_row, width=30, font=('Segoe UI', 11))
        self._species_entry.pack(side=tk.LEFT, padx=8)
        self._species_entry.insert(0, "Mus musculus")
        ttk.Button(search_row, text="Search",
                   command=self._search_species_gpls,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(search_row, text="Select Species...",
                   command=self._open_all_species_picker,
                   style="Secondary.TButton").pack(side=tk.LEFT, padx=4)
        self._species_entry.bind('<Return>', lambda e: self._search_species_gpls())

        # Quick species buttons
        quick_row1 = ttk.Frame(species_frame)
        quick_row1.pack(fill=tk.X, pady=(3, 0))
        quick_row2 = ttk.Frame(species_frame)
        quick_row2.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(quick_row1, text="Quick:",
                  font=('Segoe UI', 9), foreground='gray').pack(side=tk.LEFT, padx=(0, 5))
        quick_species = [
            "Homo sapiens", "Mus musculus", "Rattus norvegicus",
            "Danio rerio", "Drosophila melanogaster", "Caenorhabditis elegans",
            "Arabidopsis thaliana", "Sus scrofa", "Canis lupus familiaris",
            "Saccharomyces cerevisiae", "Gallus gallus", "Bos taurus",
        ]
        for i, sp in enumerate(quick_species):
            row = quick_row1 if i < 6 else quick_row2
            short = sp.split()[0][:3] + ". " + sp.split()[-1] if ' ' in sp else sp
            ttk.Button(row, text=short, width=14,
                       command=lambda s=sp: self._quick_species_search(s)
                       ).pack(side=tk.LEFT, padx=2)

        # ── Technology category filter row ──────────────────────────
        from genevariate.core.gpl_downloader import (
            TECH_CATEGORIES, CATEGORY_LABELS, CATEGORY_COLORS,
        )
        # Saturated icon colours (CATEGORY_COLORS are pastels meant for tree
        # backgrounds - darken them so the badge initials read well in white).
        _TECH_ICON_COLORS = {
            "microarray":       "#1976D2",  # blue chip
            "bulk-rna-seq":     "#2E7D32",  # green
            "single-cell":      "#E65100",  # deep orange
            "methylation":      "#6A1B9A",  # purple
            "sequencing-other": "#F9A825",  # amber
            "other":            "#616161",  # grey
        }
        _TECH_INITIALS = {
            "microarray":       "µa",
            "bulk-rna-seq":     "Rb",
            "single-cell":      "sc",
            "methylation":      "Me",
            "sequencing-other": "Sq",
            "other":            "?",
        }
        _TECH_TOOLTIPS = {
            "microarray":       "Microarray platforms (Affymetrix, Agilent, Illumina BeadChip, …)",
            "bulk-rna-seq":     "Bulk RNA-seq platforms (HiSeq, NovaSeq, NextSeq, …)",
            "single-cell":      "Single-cell / single-nucleus RNA-seq (10x, Smart-seq, Drop-seq, …)",
            "methylation":      "DNA methylation arrays (Illumina 450K / EPIC, bisulfite-seq, …) - not analysable here: beta values are not expression",
            "sequencing-other": "Other sequencing (ChIP-seq, ATAC-seq, proteomics, …) - not analysable here: no gene-level matrix",
            "other":            "Uncategorised / other platforms",
        }
        # Keep PhotoImage references alive for the lifetime of the window
        self._tech_icons = {
            c: self._make_source_badge(_TECH_ICON_COLORS[c],
                                        _TECH_INITIALS[c], size=18)
            for c in TECH_CATEGORIES
        }
        # Generic "globe" icon for the All radio button
        self._tech_icons["all"] = self._make_source_badge("#455A64", "•", size=18)

        filter_row = ttk.Frame(species_frame)
        filter_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(filter_row, text="Filter by technology:",
                  font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        self._tech_filter_var = tk.StringVar(value="all")
        filter_options = [("All", "all", "Show every category - no filter")] + [
            (CATEGORY_LABELS[c], c, _TECH_TOOLTIPS[c]) for c in TECH_CATEGORIES
        ]
        for lbl, val, tip in filter_options:
            item = ttk.Frame(filter_row)
            item.pack(side=tk.LEFT, padx=3)
            icon = self._tech_icons.get(val)
            if icon is not None:
                ic_lbl = ttk.Label(item, image=icon)
                ic_lbl.pack(side=tk.LEFT, padx=(0, 2))
                self._set_tooltip(ic_lbl, tip)
            rb = ttk.Radiobutton(item, text=lbl, value=val,
                                  variable=self._tech_filter_var,
                                  command=self._apply_gpl_tech_filter)
            rb.pack(side=tk.LEFT)
            self._set_tooltip(rb, tip)

        # Results treeview
        self._species_status = ttk.Label(species_frame,
                                          text="Enter a species name and click Search, "
                                               "or use a quick button above.",
                                          font=('Segoe UI', 9, 'italic'),
                                          foreground='gray')
        self._species_status.pack(anchor=tk.W, pady=2)

        tree_frame = ttk.Frame(species_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("GPL", "Title", "Category", "Technology", "Samples", "Genes")
        self._species_tree = ttk.Treeview(tree_frame, columns=cols,
                                           show="headings", height=12)
        self._species_tree.heading("GPL", text="GPL ID")
        self._species_tree.heading("Title", text="Platform Title")
        self._species_tree.heading("Category", text="Category")
        self._species_tree.heading("Technology", text="GEO Technology")
        self._species_tree.heading("Samples", text="# Samples")
        self._species_tree.heading("Genes", text="# Genes/Probes")

        self._species_tree.column("GPL", width=70, anchor=tk.CENTER)
        self._species_tree.column("Title", width=320)
        self._species_tree.column("Category", width=120, anchor=tk.CENTER)
        self._species_tree.column("Technology", width=150)
        self._species_tree.column("Samples", width=80, anchor=tk.CENTER)
        self._species_tree.column("Genes", width=90, anchor=tk.CENTER)

        # Row coloring by category
        for cat, color in CATEGORY_COLORS.items():
            self._species_tree.tag_configure(cat, background=color)

        # Backing store so filter can hide/show rows without re-querying
        self._species_tree_rows = []

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self._species_tree.yview)
        self._species_tree.configure(yscrollcommand=vsb.set)
        self._species_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click or select + button to use
        self._species_tree.bind('<Double-1>', self._on_species_gpl_select)

        sel_row = ttk.Frame(species_frame)
        sel_row.pack(fill=tk.X, pady=5)
        ttk.Button(sel_row, text="Use Selected GPL",
                   command=self._use_selected_species_gpl,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Label(sel_row,
                  text="Double-click a row or select and click the button to "
                       "set it as the download target",
                  font=('Segoe UI', 8, 'italic'),
                  foreground='gray').pack(side=tk.LEFT, padx=8)

        # ── Progress bar ────────────────────────────────────────────
        self.auto_dl_progress_bar = ttk.Progressbar(win, orient="horizontal",
                                                     mode="determinate")
        self.auto_dl_progress_bar.pack(fill=tk.X, padx=15, pady=(10, 2))
        self.auto_dl_status = ttk.Label(win, text="Ready", foreground="gray",
                                         font=('Segoe UI', 9))
        self.auto_dl_status.pack(anchor=tk.W, padx=15, pady=(0, 10))

    def _apply_gpl_tech_filter(self):
        """Re-render the GPL treeview to show only the selected technology category."""
        selected = getattr(self, "_tech_filter_var", None)
        target = selected.get() if selected else "all"
        rows = getattr(self, "_species_tree_rows", []) or []
        for item in self._species_tree.get_children():
            self._species_tree.delete(item)

        # Tally per-category so the status line can show a breakdown -
        # this makes it obvious that "0 under Single-cell" isn't a bug,
        # it means this search turned up no single-cell platforms.
        from collections import Counter
        from genevariate.core.gpl_downloader import (
            category_label, TECH_CATEGORIES,
        )
        counts = Counter(cat for _, cat in rows)

        shown = 0
        for values, category in rows:
            if target != "all" and category != target:
                continue
            self._species_tree.insert("", tk.END, values=values, tags=(category,))
            shown += 1

        if not rows or not hasattr(self, "_species_status"):
            return

        breakdown = ", ".join(
            f"{category_label(c)}: {counts.get(c, 0)}"
            for c in TECH_CATEGORIES if counts.get(c)
        )
        if target == "all":
            text = (f"Showing all {len(rows)} platform(s)  |  {breakdown}  |  "
                    f"Double-click a row to download.")
            color = "green"
        else:
            text = (f"Showing {shown} of {len(rows)} platform(s) "
                    f"({category_label(target)})  |  {breakdown}  |  "
                    f"Double-click a row to download.")
            color = "green" if shown else "orange"
        self._species_status.config(text=text, foreground=color)

    def _quick_species_search(self, species):
        """Set species entry and search."""
        self._species_entry.delete(0, tk.END)
        self._species_entry.insert(0, species)
        self._search_species_gpls()

    def _load_all_species_from_geometadb(self):
        """Return a list of (species_name, platform_count) tuples for every
        organism known to GEOmetadb, sorted by descending platform count.
        Cached after first call.
        """
        cached = getattr(self, '_all_species_cache', None)
        if cached is not None:
            return cached
        if not self.gds_conn:
            return []
        try:
            cur = self.gds_conn.cursor()
            # Each platform may list several organisms separated by ';' or ','
            # We take the raw organism field as-is and split client-side so the
            # user sees exactly what GEO records.
            cur.execute(
                "SELECT organism FROM gpl WHERE organism IS NOT NULL "
                "AND TRIM(organism) <> ''"
            )
            from collections import Counter
            import re
            counts = Counter()
            for (org,) in cur.fetchall():
                # Split on ';' or newline, fall back to whole string. Do NOT
                # split on ',' because some species names contain commas.
                parts = [p.strip() for p in re.split(r'[;\n]+', org) if p.strip()]
                if not parts:
                    parts = [org.strip()]
                for p in parts:
                    counts[p] += 1
            items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
            self._all_species_cache = items
            return items
        except Exception as exc:
            try:
                self.enqueue_log(f"[Species Picker] Failed to query species: {exc}")
            except Exception:
                pass
            return []

    def _open_all_species_picker(self):
        """Show a filterable list of every species in GEOmetadb. Selecting one
        fills the species entry and triggers a search."""
        if not self.gds_conn:
            messagebox.showwarning(
                "GEOmetadb Not Loaded",
                "The GEOmetadb database is not loaded yet. "
                "Wait for startup to finish, then try again.",
                parent=self,
            )
            return

        win = tk.Toplevel(self)
        style_window(win)
        win.title("Select Species - All GEO organisms")
        win.geometry("620x620")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"620x620+{(_sw-620)//2}+{(_sh-620)//2}")
        except Exception:
            pass
        try:
            win.transient(self)
            win.lift()
            win.focus_force()
            win.attributes('-topmost', True)
            win.after(400, lambda w=win: (w.attributes('-topmost', False)
                                          if w.winfo_exists() else None))
        except Exception:
            pass

        header = ttk.Frame(win, padding=10)
        header.pack(fill=tk.X)
        ttk.Label(header,
                  text="All species in GEOmetadb",
                  font=('Segoe UI', 13, 'bold')).pack(anchor=tk.W)
        subtitle = ttk.Label(header,
                             text="Loading species list from GEOmetadb...",
                             foreground='gray', font=('Segoe UI', 9, 'italic'))
        subtitle.pack(anchor=tk.W, pady=(2, 0))

        filt_row = ttk.Frame(win, padding=(10, 0))
        filt_row.pack(fill=tk.X)
        ttk.Label(filt_row, text="Filter:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        filter_var = tk.StringVar(master=win)
        filter_entry = ttk.Entry(filt_row, textvariable=filter_var,
                                 font=('Segoe UI', 11))
        filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        count_lbl = ttk.Label(filt_row, text="", foreground='#555',
                              font=('Segoe UI', 9))
        count_lbl.pack(side=tk.LEFT, padx=10)

        list_frame = ttk.Frame(win, padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("rank", "organism", "count")
        tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                            selectmode="browse", height=22)
        tree.heading("rank", text="#")
        tree.heading("organism", text="Organism (scientific name)")
        tree.heading("count", text="# Platforms")
        tree.column("rank", width=60, anchor=tk.CENTER, stretch=False)
        tree.column("organism", width=380, anchor=tk.W, stretch=True)
        tree.column("count", width=110, anchor=tk.CENTER, stretch=False)

        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Row styling: alternating stripes + tier color per platform count
        # Combined so each row is visually separated AND coloured by importance.
        tree.tag_configure("row_even", background='#FFFFFF')
        tree.tag_configure("row_odd",  background='#F0F4F8')
        tree.tag_configure("tier_top",    foreground='#1B5E20')  # 100+
        tree.tag_configure("tier_high",   foreground='#2E7D32')  # 30+
        tree.tag_configure("tier_mid",    foreground='#EF6C00')  # 5+
        tree.tag_configure("tier_low",    foreground='#B71C1C')  # <5

        # Bigger row height so separators stand out like the GPL list
        _style = ttk.Style()
        try:
            _style.configure("SpeciesPicker.Treeview",
                             rowheight=26,
                             borderwidth=1,
                             relief="solid")
            _style.configure("SpeciesPicker.Treeview.Heading",
                             font=('Segoe UI', 10, 'bold'))
            tree.configure(style="SpeciesPicker.Treeview")
        except Exception:
            pass

        def _tier_tag(n):
            if n >= 100:
                return "tier_top"
            if n >= 30:
                return "tier_high"
            if n >= 5:
                return "tier_mid"
            return "tier_low"

        state = {'all': [], 'shown': []}

        def _render(items):
            tree.delete(*tree.get_children())
            for rank, (name, n) in enumerate(items, start=1):
                stripe = "row_odd" if rank % 2 else "row_even"
                tree.insert("", tk.END, iid=str(rank - 1),
                            values=(rank, name, f"{n:,}"),
                            tags=(_tier_tag(n), stripe))
            state['shown'] = list(items)
            count_lbl.config(text=f"{len(items):,} / {len(state['all']):,}")

        def _apply_filter(*_):
            q = filter_var.get().strip().lower()
            if not q:
                _render(state['all'])
                return
            filtered = [(n, c) for (n, c) in state['all']
                        if q in n.lower()]
            _render(filtered)

        filter_var.trace_add('write', _apply_filter)

        def _use_selected():
            sel = tree.selection()
            if not sel:
                return
            try:
                idx = int(sel[0])
            except ValueError:
                return
            if idx < 0 or idx >= len(state['shown']):
                return
            name, _ = state['shown'][idx]
            self._species_entry.delete(0, tk.END)
            self._species_entry.insert(0, name)
            win.destroy()
            self._search_species_gpls()

        tree.bind('<Double-1>', lambda e: _use_selected())
        tree.bind('<Return>',   lambda e: _use_selected())

        btns = ttk.Frame(win, padding=10)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Use Selected",
                   command=_use_selected,
                   style="Primary.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy,
                   style="Secondary.TButton").pack(side=tk.RIGHT, padx=4)

        # Load list in background so the dialog pops up instantly
        import threading

        def _load():
            try:
                anim = self._animator
                anim.spinner(subtitle,
                             "Querying GEOmetadb for all organisms...",
                             color='#6A1B9A')
            except Exception:
                pass

            def _do_query():
                items = self._load_all_species_from_geometadb()

                def _finish():
                    try:
                        self._animator.stop(subtitle)
                    except Exception:
                        pass
                    state['all'] = items
                    if not items:
                        subtitle.config(
                            text="No species found (GEOmetadb may be empty).",
                            foreground='#C62828')
                    else:
                        subtitle.config(
                            text=f"{len(items):,} distinct organisms - "
                                 f"type to filter, double-click to use.",
                            foreground='#1B5E20')
                    _render(items)
                    try:
                        filter_entry.focus()
                    except Exception:
                        pass

                try:
                    win.after(0, _finish)
                except Exception:
                    pass

            threading.Thread(target=_do_query, daemon=True).start()

        _load()

    def _search_species_gpls(self):
        """Query GEOmetadb for platforms matching species OR GPL ID. Runs in background thread."""
        species = self._species_entry.get().strip()
        if not species:
            self._species_status.config(text="Enter a species name or GPL ID.", foreground="orange")
            return

        if not self.gds_conn:
            self._species_status.config(
                text="GEOmetadb not loaded - cannot search.", foreground="red")
            return

        self.enqueue_log(f"[GPL Browser] Searching for '{species}'...")
        self.update_progress(value=10, text=f"Searching GPLs: {species}")
        # Also update the GPL downloader's own progress bar
        try:
            self.auto_dl_progress_bar["maximum"] = 100
            self.auto_dl_progress_bar["value"] = 10
        except Exception:
            pass

        # ── Liquid animations: spinner + pulsing bar while searching ──
        try:
            anim = self._animator
            anim.spinner(self._species_status,
                         f"Searching for '{species}'...",
                         color="#1565C0")
            anim.spinner(self.auto_dl_status,
                         f"Searching for '{species}'...",
                         color="#1565C0")
            anim.pulse_bar(self.auto_dl_progress_bar, palette=_UIAnimator.PULSE_BLUES)
            anim.pulse_bar(self.progressbar, palette=_UIAnimator.PULSE_BLUES)
        except Exception:
            pass

        # Clear old results
        for item in self._species_tree.get_children():
            self._species_tree.delete(item)
        self._species_tree_rows = []

        def _bg_search():
            try:
                search_upper = species.upper().strip()
                is_gpl_id = False
                if search_upper.startswith("GPL") and search_upper[3:].isdigit():
                    is_gpl_id = True
                elif species.isdigit():
                    is_gpl_id = True
                    search_upper = f"GPL{species}"

                if is_gpl_id:
                    query = """
                        SELECT gpl.gpl, gpl.title, gpl.technology, gpl.organism,
                               gpl.data_row_count
                        FROM gpl WHERE UPPER(gpl.gpl) = ? OR UPPER(gpl.gpl) LIKE ?
                        ORDER BY gpl.data_row_count DESC LIMIT 50
                    """
                    like_pattern = f"%{search_upper}%"
                    rows = self.gds_conn.execute(query, (search_upper, like_pattern)).fetchall()
                    search_desc = f"GPL ID '{search_upper}'"
                else:
                    # Fast query: get GPLs first WITHOUT sample count (avoids slow subquery).
                    # IMPORTANT: do NOT sort/limit by data_row_count - RNA-seq and
                    # single-cell GPLs have 0 probes in GEOmetadb and would otherwise
                    # be pushed off the end of the list, making them invisible under
                    # any non-microarray technology filter.
                    #
                    # NO LIMIT: previous LIMIT 600 hid thousands of platforms
                    # (Homo sapiens alone has ~5,900 GPLs with >=1 GSE). Return
                    # everything that matches the species/title filter.
                    query = """
                        SELECT gpl.gpl, gpl.title, gpl.technology, gpl.organism,
                               gpl.data_row_count
                        FROM gpl
                        WHERE LOWER(gpl.organism) LIKE ? OR LOWER(gpl.title) LIKE ?
                    """
                    pattern = f"%{species.lower()}%"
                    rows = self.gds_conn.execute(query, (pattern, pattern)).fetchall()
                    search_desc = f"'{species}'"

                def _update_status(text, pct=None):
                    try:
                        # Update the spinner's base text on both status labels so
                        # the glyph keeps cycling while the message changes.
                        self.after(0, lambda: self._animator.update_spinner_text(
                            self._species_status, text))
                        self.after(0, lambda: self._animator.update_spinner_text(
                            self.auto_dl_status, text))
                        if pct is not None:
                            self.after(0, lambda p=pct:
                                self._animator.smooth_to(
                                    self.auto_dl_progress_bar, p))
                    except: pass

                _update_status(f"Found {len(rows)} platforms, counting samples...", 20)

                # Now count samples per GPL in batches (much faster than subquery per row)
                gpl_ids = [r[0] for r in rows]
                sample_counts = {}
                if gpl_ids:
                    for i in range(0, len(gpl_ids), 50):
                        chunk = gpl_ids[i:i+50]
                        ph = ",".join(["?"] * len(chunk))
                        try:
                            cnt_rows = self.gds_conn.execute(
                                f"SELECT gpl, COUNT(*) FROM gsm WHERE gpl IN ({ph}) GROUP BY gpl",
                                chunk).fetchall()
                            for gpl, cnt in cnt_rows:
                                sample_counts[gpl] = cnt
                        except Exception:
                            pass
                        pct = min(90, 20 + int(70 * (i + len(chunk)) / len(gpl_ids)))
                        self.update_progress(value=pct, text=f"Counting samples: {i+len(chunk)}/{len(gpl_ids)} GPLs")
                        _update_status(f"Counting samples: {i+len(chunk)}/{len(gpl_ids)} platforms...", pct)

                # Sort by sample count descending
                rows_with_counts = []
                for gpl_id, title, tech, organism, n_probes in rows:
                    n_samples = sample_counts.get(gpl_id, 0)
                    rows_with_counts.append((gpl_id, title, tech, organism, n_probes, n_samples))
                rows_with_counts.sort(key=lambda x: x[5], reverse=True)

                # Populate treeview on main thread
                def _populate():
                    try:
                        from genevariate.core.gpl_downloader import (
                            classify_technology, category_label,
                        )
                        self._species_tree_rows = []
                        for gpl_id, title, tech, organism, n_probes, n_samples in rows_with_counts:
                            title_full = title or ""
                            title_disp = title_full[:80]
                            tech_full = tech or ""
                            tech_disp = tech_full[:30] or "Unknown"
                            n_probes = n_probes or 0
                            category = classify_technology(tech_full, title_full)
                            cat_disp = category_label(category)
                            values = (
                                gpl_id, title_disp, cat_disp, tech_disp,
                                f"{n_samples:,}" if n_samples else "?",
                                f"{n_probes:,}" if n_probes else "?",
                            )
                            self._species_tree_rows.append((values, category))
                        self._apply_gpl_tech_filter()
                        actual_org = rows[0][3] if rows and rows[0][3] else species
                        done_text = (f"Found {len(rows)} platform(s) for {search_desc} "
                                     f"({actual_org}). Double-click to download.")
                        # Stop spinner/pulse and set final text with a success flash
                        self._animator.stop(self._species_status)
                        self._animator.stop(self.auto_dl_status)
                        self._animator.stop_pulse(self.auto_dl_progress_bar)
                        self._animator.stop_pulse(self.progressbar)
                        self._species_status.config(text=done_text, foreground="green")
                        self.auto_dl_status.config(text=done_text, foreground="green")
                        self._animator.flash(self._species_status, "#2E7D32",
                                             revert_to="green", duration_ms=1200)
                        self._animator.smooth_to(self.auto_dl_progress_bar, 100)
                        self.enqueue_log(f"[GPL Browser] Found {len(rows)} platforms for {search_desc}")
                        self.update_progress(value=0)
                        self.after(2000, lambda: self.auto_dl_progress_bar.config(value=0))
                    except Exception as e:
                        self._animator.stop(self._species_status)
                        self._animator.stop(self.auto_dl_status)
                        self._animator.stop_pulse(self.auto_dl_progress_bar)
                        self._animator.stop_pulse(self.progressbar)
                        self._species_status.config(text=f"Error: {e}", foreground="red")
                        self.auto_dl_progress_bar["value"] = 0
                        self.update_progress(value=0)

                if not rows:
                    def _no_results():
                        no_text = (f"No platforms found for {search_desc}. "
                                   f"Try 'Homo sapiens' or 'GPL570'.")
                        self._animator.stop(self._species_status)
                        self._animator.stop(self.auto_dl_status)
                        self._animator.stop_pulse(self.auto_dl_progress_bar)
                        self._animator.stop_pulse(self.progressbar)
                        self._species_status.config(text=no_text, foreground="orange")
                        self.auto_dl_status.config(text=no_text, foreground="orange")
                        self.auto_dl_progress_bar["value"] = 0
                        self.update_progress(value=0)
                    self.after(0, _no_results)
                else:
                    self.after(0, _populate)

            except Exception as e:
                self.enqueue_log(f"[GPL Browser] Search error: {e}")
                # `msg` defaults now: `e` is deleted when the except block
                # ends, and this runs later on the Tk thread.
                def _err(msg=str(e)):
                    try:
                        self._animator.stop(self._species_status)
                        self._animator.stop(self.auto_dl_status)
                        self._animator.stop_pulse(self.auto_dl_progress_bar)
                        self._animator.stop_pulse(self.progressbar)
                    except Exception:
                        pass
                    self._species_status.config(text=f"Search error: {msg}", foreground="red")
                    try:
                        self.auto_dl_status.config(text=f"Search error: {msg}", foreground="red")
                        self.auto_dl_progress_bar["value"] = 0
                    except: pass
                    self.update_progress(value=0)
                self.after(0, _err)

        threading.Thread(target=_bg_search, daemon=True).start()

    def _on_species_gpl_select(self, event):
        """Handle double-click on species tree row."""
        self._use_selected_species_gpl()

    def _use_selected_species_gpl(self):
        """Copy selected GPL from species browser to the download entry."""
        sel = self._species_tree.selection()
        if not sel:
            return
        values = self._species_tree.item(sel[0], 'values')
        gpl_id = values[0]  # First column is GPL ID
        self.auto_gpl_entry.delete(0, tk.END)
        self.auto_gpl_entry.insert(0, gpl_id)
        self.auto_dl_status.config(
            text=f"Selected: {gpl_id} - {values[1][:60]}  |  "
                 f"Click 'Download & Process' to start",
            foreground="#1565C0")

    def _query_gpl_info_local(self, gpl_id):
        """
        Query platform info directly from self.gds_conn - same approach as
        the working species browser in _search_species_gpls.
        Returns dict or raises ValueError.
        """
        gpl_id = gpl_id.strip().upper()

        # ── Find the platform (case-insensitive, same style as species search) ──
        row = self.gds_conn.execute(
            "SELECT gpl, title, organism, technology "
            "FROM gpl WHERE UPPER(gpl) = ? LIMIT 1",
            (gpl_id,)
        ).fetchone()

        if not row:
            # Fallback: LIKE search
            row = self.gds_conn.execute(
                "SELECT gpl, title, organism, technology "
                "FROM gpl WHERE gpl LIKE ? LIMIT 1",
                (gpl_id,)
            ).fetchone()

        if not row:
            # Show what IS in the DB for debugging
            sample = self.gds_conn.execute(
                "SELECT gpl FROM gpl ORDER BY gpl LIMIT 10"
            ).fetchall()
            sample_ids = [r[0] for r in sample]
            raise ValueError(
                f"Platform {gpl_id} not found in GEOmetadb.\n\n"
                f"Sample GPL IDs in database: {sample_ids}\n\n"
                f"Check the GPL ID or update your GEOmetadb.sqlite.gz file."
            )

        db_gpl = str(row[0])  # actual value stored in DB

        # ── Get GSE list (same join style as species browser) ──
        gse_rows = self.gds_conn.execute(
            "SELECT DISTINCT gse FROM gse_gpl WHERE gpl = ?",
            (db_gpl,)
        ).fetchall()

        if not gse_rows:
            # Also try case-insensitive
            gse_rows = self.gds_conn.execute(
                "SELECT DISTINCT gse FROM gse_gpl WHERE UPPER(gpl) = ?",
                (gpl_id,)
            ).fetchall()

        # The SELECT order is (gpl, title, organism, technology). Reading
        # row[1] as the organism and row[2] as the title swapped the two: the
        # technology classifier then judged "Illumina NextSeq 500" by the
        # string "Homo sapiens", which names no assay, so every sequencing
        # platform fell through to the unclassified bucket.
        return {
            'gpl_id':       gpl_id,
            'title':        str(row[1] or 'Unknown'),
            'organism':     str(row[2] or 'Unknown'),
            'technology':   str(row[3] or 'Unknown'),
            'gse_list':     [r[0] for r in gse_rows],
            'total_series': len(gse_rows),
        }

    def _auto_download_gpl(self):
        gpl_id = self.auto_gpl_entry.get().strip().upper()
        if not gpl_id.startswith("GPL") or not gpl_id[3:].isdigit():
            messagebox.showerror("Invalid", "Enter valid GPL ID (e.g. GPL1355)", parent=self)
            return
        if not self.gds_conn:
            messagebox.showerror("Database Required", "GEOmetadb.sqlite.gz required", parent=self)
            return
        
        max_gse = int(self.auto_max_gse_entry.get() or 0)
        
        # ── Query platform info LOCALLY (same as species browser) ──
        try:
            info = self._query_gpl_info_local(gpl_id)
        except Exception as e:
            self.enqueue_log(f"[GPL-DL] Platform lookup failed: {e}")
            messagebox.showerror("Platform Not Found", str(e), parent=self)
            return

        if info['total_series'] == 0:
            messagebox.showwarning("No Series",
                f"{gpl_id} found but has 0 GSE series in GEOmetadb.\n"
                f"The database may be too old.", parent=self)
            return

        # ── Route by technology category ───────────────────────────
        from genevariate.core.gpl_downloader import (
            classify_technology, category_label, ingestion_route,
        )
        category = classify_technology(info.get('technology', ''),
                                        info.get('title', ''))
        cat_disp = category_label(category)
        route = ingestion_route(category)

        if route == "pseudobulk":
            messagebox.showwarning(
                "Single-cell platform",
                f"{gpl_id} is a single-cell platform ({info['title']}).\n\n"
                f"A GSM here is a library of thousands of cells, so there is "
                f"no per-sample expression column to download: the values live "
                f"in supplementary 10x / h5ad / mtx bundles whose layout is "
                f"chosen per submission, and they only become a sample-level "
                f"matrix after the cells are aggregated.\n\n"
                f"Two ways forward:\n"
                f"  • Single-cell (CELLxGENE) - browse and load curated "
                f"scRNA-seq, pseudo-bulked automatically. Not GEO, but it "
                f"feeds every other window.\n"
                f"  • Add Custom Platform - if you have already pseudo-bulked "
                f"this GSE yourself, load the sample x gene matrix directly.",
                parent=self)
            return

        if route == "none":
            messagebox.showwarning(
                f"{cat_disp} platform",
                f"{gpl_id} ({info['title']}) measures signal over genomic "
                f"intervals - peaks, coverage windows - not over genes.\n\n"
                f"GeneVariate's analyses all key on a gene x sample matrix, "
                f"and no uniform gene-level matrix exists for this assay: "
                f"NCBI's reprocessing covers RNA-seq only. Folding peak "
                f"coverage into gene bodies would produce a table that looks "
                f"like expression but answers a different question, so this "
                f"platform is not loadable.",
                parent=self)
            return

        if route == "ncbi-counts":
            if not messagebox.askyesno(
                "Bulk RNA-seq platform",
                f"{gpl_id} is a bulk RNA-seq platform ({info['title']}).\n\n"
                f"GEO series matrices for RNA-seq carry the sample metadata "
                f"but not the expression values, so GeneVariate loads the "
                f"counts NCBI produced by re-aligning the raw reads - one "
                f"uniform table per series, comparable across series.\n\n"
                f"Any series NCBI has not reprocessed will be listed at the "
                f"end so you can load those from ARCHS4 instead.\n\n"
                f"Proceed?",
                parent=self):
                return

        elif route == "unsupported":
            # Every analysis downstream - the log2/quantile normalisation, the
            # distribution fits, the region and enrichment statistics - is
            # written for expression values. A methylation beta is a bounded
            # proportion, so those results would be meaningless. Say so
            # instead of implying the platform is supported.
            messagebox.showwarning(
                "Methylation platform not supported",
                f"{gpl_id} is a methylation array ({info['title']}).\n\n"
                f"GeneVariate analyses expression values. Methylation beta "
                f"values are bounded proportions, not intensities or counts, "
                f"so the normalisation, distribution fitting and enrichment "
                f"statistics here do not apply to them.\n\n"
                f"Pick a microarray, bulk RNA-seq or single-cell platform "
                f"instead.",
                parent=self)
            return

        elif category == "other":
            if not messagebox.askyesno(
                f"{cat_disp} platform",
                f"{gpl_id} is a {cat_disp.lower()} platform "
                f"({info['title']}).\n\n"
                f"The values are read from the series matrix and mapped to "
                f"genes through the platform annotation, the same way arrays "
                f"are. Check the gene column the configurator picks, and be "
                f"sure the values really are expression measurements - "
                f"nothing else is supported.\n\n"
                f"Proceed?",
                parent=self):
                return

        # ── Import downloader for the actual download work ──
        try:
            from genevariate.core.gpl_downloader import GPLDownloader
            downloader = GPLDownloader(gds_conn=self.gds_conn, output_base_dir=self.data_dir)
            downloader.check_dependencies()
        except ImportError as e:
            messagebox.showerror("Missing Module",
                f"gpl_downloader.py not found:\n{e}\n\n"
                f"Place gpl_downloader.py in:\n"
                f"  genevariate/core/gpl_downloader.py\n\n"
                f"With __init__.py files in genevariate/ and genevariate/core/",
                parent=self)
            return
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        
        # Open the interactive configurator: choose metadata columns to scrape
        # and, optionally, a gene subset, before the (large) download starts.
        self._open_gpl_config_dialog(info, cat_disp, downloader, max_gse)

    # ==================================================================
    # Interactive GPL download configurator (metadata + scalable gene picker)
    # ==================================================================
    _GPL_GENE_RENDER_CAP = 500   # never render more than this many rows at once

    def _gpl_install_config_styles(self, win):
        """Register the named ttk styles this dialog uses, all from the AERO
        palette - the same roles region_analysis defines, so the configurator
        reads like the rest of the app instead of bare tkinter."""
        s = ttk.Style(win)
        s.configure('Section.TLabel', foreground=AERO['accent_dark'],
                    font=(UI_FONT, 12, 'bold'))
        s.configure('Field.TLabel', foreground=AERO['text'],
                    font=(UI_FONT, 9, 'bold'))
        s.configure('Hint.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 9, 'italic'))
        s.configure('Footnote.TLabel', foreground=AERO['muted'],
                    font=(UI_FONT, 8, 'italic'))
        s.configure('MetricStrong.TLabel', foreground=AERO['accent_dark'],
                    font=(MONO_FONT, 10, 'bold'))
        s.configure('Card.TLabelframe', background=AERO['panel'],
                    bordercolor=AERO['border'], relief='solid', borderwidth=1)
        s.configure('Card.TLabelframe.Label', background=AERO['panel'],
                    foreground=AERO['accent_dark'], font=(UI_FONT, 10, 'bold'))
        # A Treeview variant WITHOUT the vertical column gridline (the theme
        # draws a 1px rule down every cell's right edge; on the last, full-width
        # description column that rule sits flush against the scrollbar and the
        # text runs into it). Dropping the GV.Treedata.rule wrapper from the cell
        # layout removes those vertical lines; the horizontal row rules stay.
        s.layout('Clean.Treeview.Cell', [
            ('Treedata.padding', {'sticky': 'nswe', 'children': [
                ('Treeitem.text', {'sticky': 'nswe'})]})])

    def _open_gpl_config_dialog(self, info, cat_disp, downloader, max_gse):
        gpl_id = info['gpl_id']
        self._gplcfg_gpl_id = gpl_id
        self._gplcfg_info = info
        # per-dialog state -- gene picker
        self._gplcfg_pool = []            # full sorted gene list (once loaded)
        self._gplcfg_pool_upper = []      # parallel upper-cased for filtering
        self._gplcfg_selected = set()     # selected gene symbols
        self._gplcfg_matches = []         # current filter matches (full, uncapped)
        self._gplcfg_filter_after = None
        # per-dialog state -- GSE / experiment picker
        self._gplcfg_gse_pool = []        # list of {gse, desc, n_samples, ...}
        self._gplcfg_gse_pool_upper = []  # parallel "GSE  desc" upper for filter
        self._gplcfg_gse_selected = set() # selected GSE accessions
        self._gplcfg_gse_downloaded = set()  # already on disk for this platform
        self._gplcfg_gse_matches = []     # current filter matches (full, uncapped)
        self._gplcfg_gse_filter_after = None

        win = tk.Toplevel(self)
        ensure_theme(win)
        style_window(win)
        self._gpl_install_config_styles(win)
        win.title(f"Configure download - {gpl_id}")
        win.geometry("780x820")
        win.minsize(700, 660)
        win.transient(self)
        self._gplcfg_win = win

        ttk.Label(win, text=f"{gpl_id} - {info['title']}",
                  style="Section.TLabel").pack(anchor="w", padx=14, pady=(12, 0))
        ttk.Label(win,
                  text=f"{info['organism']}  ·  {cat_disp}  ·  "
                       f"{info['total_series']:,} GSE series",
                  style="Hint.TLabel").pack(anchor="w", padx=14, pady=(0, 8))

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 4))

        # ============ TAB 1: Metadata scraping ============
        mtab = ttk.Frame(nb, padding=10)
        nb.add(mtab, text="  Metadata  ")
        top = ttk.Frame(mtab); top.pack(fill=tk.X)
        ttk.Label(top, text="Max GSEs (0 = all):",
                  style="Field.TLabel").pack(side=tk.LEFT)
        self._gplcfg_maxgse = ttk.Entry(top, width=8)
        self._gplcfg_maxgse.pack(side=tk.LEFT, padx=6)
        self._gplcfg_maxgse.insert(0, str(max_gse or 0))
        ttk.Label(top, text="  applies only when downloading ALL experiments",
                  style="Footnote.TLabel").pack(side=tk.LEFT)

        mframe = labelframe(mtab, text="GSE / Metadata Scraping",
                            bg=AERO['panel'], font=(UI_FONT, 9, 'bold'),
                            padding=10)
        mframe.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(mframe,
                  text="Harvested from the same series-matrix files (no extra "
                       "download). Feeds the LLM label-extraction phase 2.",
                  style="Hint.TLabel").pack(anchor="w", pady=(0, 6))
        self._gplcfg_meta_vars = {}
        srow = ttk.Frame(mframe); srow.pack(fill=tk.X, anchor="w")
        ttk.Label(srow, text="Series", width=8,
                  style="Field.TLabel").pack(side=tk.LEFT)
        for field, lbl in (("title", "Title"), ("summary", "Summary"),
                           ("design", "Overall design")):
            v = tk.BooleanVar(value=True)
            self._gplcfg_meta_vars[("series", field)] = v
            ttk.Checkbutton(srow, text=lbl, variable=v).pack(side=tk.LEFT, padx=6)
        prow = ttk.Frame(mframe); prow.pack(fill=tk.X, anchor="w", pady=(6, 0))
        ttk.Label(prow, text="Sample", width=8,
                  style="Field.TLabel").pack(side=tk.LEFT)
        for field, lbl in (("title", "Title"), ("source_name", "Source name"),
                           ("characteristics", "Characteristics"),
                           ("treatment_protocol", "Treatment"),
                           ("description", "Description")):
            v = tk.BooleanVar(value=True)
            self._gplcfg_meta_vars[("sample", field)] = v
            ttk.Checkbutton(prow, text=lbl, variable=v).pack(side=tk.LEFT, padx=6)

        # ============ TAB 2: Experiments (GSE) ============
        etab = ttk.Frame(nb, padding=10)
        nb.add(etab, text="  Experiments  ")
        self._gplcfg_gse_mode = tk.StringVar(value="all")
        emrow = ttk.Frame(etab); emrow.pack(fill=tk.X, anchor="w")
        ttk.Radiobutton(emrow, text="All experiments", value="all",
                        variable=self._gplcfg_gse_mode,
                        command=self._gpl_gse_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(emrow, text="Selected experiments", value="selected",
                        variable=self._gplcfg_gse_mode,
                        command=self._gpl_gse_mode_changed).pack(side=tk.LEFT, padx=(12, 0))
        self._gplcfg_gse_load_btn = ttk.Button(
            emrow, text="Load experiment list",
            style="Secondary.TButton", command=self._gpl_load_gse_list)
        self._gplcfg_gse_load_btn.pack(side=tk.RIGHT)

        ebody = ttk.Frame(etab); ebody.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        esrch = ttk.Frame(ebody); esrch.pack(fill=tk.X)
        ttk.Label(esrch, text="Search",
                  style="Field.TLabel").pack(side=tk.LEFT)
        self._gplcfg_gse_search = ttk.Entry(esrch)
        self._gplcfg_gse_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._gplcfg_gse_search.bind("<KeyRelease>", self._gpl_gse_search_changed)
        self._gplcfg_gse_showing = ttk.Label(esrch, text="", style="Hint.TLabel")
        self._gplcfg_gse_showing.pack(side=tk.RIGHT)

        ttk.Label(etab,
                  text="Tip: click a GSE accession to open its full record on "
                       "the GEO website; click the checkbox or description to "
                       "pick it.", style="Footnote.TLabel").pack(anchor="w")
        etframe = ttk.Frame(ebody); etframe.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._gplcfg_gse_tree = ttk.Treeview(
            etframe, columns=("sel", "gse", "desc"), show="headings",
            height=12, selectmode="none", style="Clean.Treeview")
        self._gplcfg_gse_tree.heading("sel", text="\u2713")
        self._gplcfg_gse_tree.heading("gse", text="GSE \u2197")
        self._gplcfg_gse_tree.heading("desc", text="Experiment (n samples · description)")
        # sel + gse are fixed; desc stretches to fill the rest so the text
        # never runs under the scrollbar and no horizontal scroll is needed.
        self._gplcfg_gse_tree.column("sel", width=40, minwidth=40,
                                     anchor="center", stretch=False)
        self._gplcfg_gse_tree.column("gse", width=120, minwidth=100,
                                     anchor="center", stretch=False)
        self._gplcfg_gse_tree.column("desc", width=420, minwidth=200,
                                     anchor="w", stretch=True)
        evsb = ttk.Scrollbar(etframe, orient="vertical",
                             command=self._gplcfg_gse_tree.yview)
        self._gplcfg_gse_tree.configure(yscrollcommand=evsb.set)
        self._gplcfg_gse_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        evsb.pack(side=tk.LEFT, fill=tk.Y)
        self._gplcfg_gse_tree.bind("<Button-1>", self._gpl_gse_tree_click)
        self._gplcfg_gse_tree.bind("<Motion>", self._gpl_gse_tree_hover)

        ebrow = ttk.Frame(ebody); ebrow.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(ebrow, text="Select all matches", style="Secondary.TButton",
                   command=self._gpl_gse_select_matches).pack(side=tk.LEFT)
        ttk.Button(ebrow, text="Select all", style="Secondary.TButton",
                   command=self._gpl_gse_select_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(ebrow, text="Select downloaded", style="Secondary.TButton",
                   command=self._gpl_gse_select_downloaded).pack(side=tk.LEFT,
                                                                 padx=(0, 4))
        ttk.Button(ebrow, text="Clear", style="Destructive.TButton",
                   command=self._gpl_gse_clear).pack(side=tk.LEFT)
        self._gplcfg_gse_counter = ttk.Label(
            ebrow, text="Selected: 0 / 0 exp.", style="MetricStrong.TLabel")
        self._gplcfg_gse_counter.pack(side=tk.RIGHT)

        # ============ TAB 3: Genes ============
        gframe = ttk.Frame(nb, padding=10)
        nb.add(gframe, text="  Genes  ")
        self._gplcfg_gene_mode = tk.StringVar(value="all")
        mrow = ttk.Frame(gframe); mrow.pack(fill=tk.X, anchor="w")
        ttk.Radiobutton(mrow, text="All genes", value="all",
                        variable=self._gplcfg_gene_mode,
                        command=self._gpl_gene_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(mrow, text="Selected genes", value="selected",
                        variable=self._gplcfg_gene_mode,
                        command=self._gpl_gene_mode_changed).pack(side=tk.LEFT, padx=(12, 0))
        self._gplcfg_load_btn = ttk.Button(
            mrow, text="Load gene list from platform",
            style="Secondary.TButton",
            command=self._gpl_load_gene_list)
        self._gplcfg_load_btn.pack(side=tk.RIGHT)

        # picker body (disabled until "Selected genes" + list loaded)
        body = ttk.Frame(gframe); body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._gplcfg_picker_body = body

        srch = ttk.Frame(body); srch.pack(fill=tk.X)
        ttk.Label(srch, text="Search",
                  style="Field.TLabel").pack(side=tk.LEFT)
        self._gplcfg_search = ttk.Entry(srch)
        self._gplcfg_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._gplcfg_search.bind("<KeyRelease>", self._gpl_gene_search_changed)
        self._gplcfg_showing = ttk.Label(srch, text="", style="Hint.TLabel")
        self._gplcfg_showing.pack(side=tk.RIGHT)

        tframe = ttk.Frame(body); tframe.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._gplcfg_tree = ttk.Treeview(
            tframe, columns=("sel", "gene"), show="headings", height=12,
            selectmode="none")
        self._gplcfg_tree.heading("sel", text="\u2713")
        self._gplcfg_tree.heading("gene", text="Gene")
        self._gplcfg_tree.column("sel", width=40, anchor="center", stretch=False)
        self._gplcfg_tree.column("gene", width=260, anchor="center")
        vsb = ttk.Scrollbar(tframe, orient="vertical",
                            command=self._gplcfg_tree.yview)
        self._gplcfg_tree.configure(yscrollcommand=vsb.set)
        self._gplcfg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self._gplcfg_tree.bind("<Button-1>", self._gpl_gene_tree_click)

        brow = ttk.Frame(body); brow.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(brow, text="Select all matches", style="Secondary.TButton",
                   command=self._gpl_gene_select_matches).pack(side=tk.LEFT)
        ttk.Button(brow, text="Select all", style="Secondary.TButton",
                   command=self._gpl_gene_select_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(brow, text="Clear", style="Destructive.TButton",
                   command=self._gpl_gene_clear).pack(side=tk.LEFT)
        self._gplcfg_counter = ttk.Label(brow, text="Selected: 0 / 0 genes",
                                         style="MetricStrong.TLabel")
        self._gplcfg_counter.pack(side=tk.RIGHT)

        prow2 = ttk.Frame(body); prow2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(prow2, text="Paste / load a gene list",
                  style="Field.TLabel").pack(anchor="w")
        self._gplcfg_paste = tk.Text(
            prow2, height=3, wrap="word", relief="flat", bd=1,
            highlightthickness=1, highlightbackground=AERO["border"],
            highlightcolor=AERO["accent"], bg="#FFFFFF", fg=AERO["text"],
            insertbackground=AERO["accent_dark"], font=(MONO_FONT, 10))
        self._gplcfg_paste.pack(fill=tk.X, pady=(2, 2))
        pbtns = ttk.Frame(prow2); pbtns.pack(fill=tk.X)
        ttk.Button(pbtns, text="Add pasted", style="Secondary.TButton",
                   command=self._gpl_gene_add_pasted).pack(side=tk.LEFT)
        ttk.Button(pbtns, text="Load from file\u2026", style="Secondary.TButton",
                   command=self._gpl_gene_load_file).pack(side=tk.LEFT, padx=4)

        # ---- Bottom buttons ----
        bottom = ttk.Frame(win); bottom.pack(fill=tk.X, padx=14, pady=10)
        ttk.Button(bottom, text="Start Download", style="Primary.TButton",
                   command=lambda: self._gpl_start_configured_download(
                       info, downloader)).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Cancel", style="Secondary.TButton",
                   command=win.destroy).pack(side=tk.LEFT, padx=8)

        self._gpl_gene_mode_changed()   # set initial enabled/disabled state
        self._gpl_update_gene_counter()
        self._gpl_update_gse_counter()
        # Populate the experiment list up-front so the user can browse/pick
        # without hunting for a Load button.
        self._gpl_load_gse_list()

    # ---- GSE / experiment-picker helpers ----
    def _gpl_gse_mode_changed(self):
        # The experiment list is always browsable; the radio only decides
        # whether the whole platform or just the checked rows get downloaded.
        if not self._gplcfg_gse_pool:
            self._gplcfg_gse_showing.config(text="loading experiment list\u2026")

    def _gpl_load_gse_list(self):
        gpl_id = self._gplcfg_gpl_id
        if not getattr(self, "gds_conn", None):
            self._gplcfg_gse_showing.config(text="GEOmetadb not connected")
            return
        self._gplcfg_gse_load_btn.configure(state="disabled", text="Loading\u2026")
        self._gplcfg_gse_showing.config(text="querying GEOmetadb\u2026")

        def worker():
            try:
                from genevariate.core.gpl_downloader import list_platform_series
                res = list_platform_series(self.gds_conn, gpl_id)
                self.after(0, lambda: self._gpl_gse_list_loaded(res))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda _e=str(e), _tb=tb:
                           self._gpl_gse_list_failed(_e, _tb))

        threading.Thread(target=worker, daemon=True).start()

    def _gpl_gse_list_failed(self, err, tb=""):
        try:
            self._gplcfg_gse_load_btn.configure(
                state="normal", text="Load experiment list")
            self._gplcfg_gse_showing.config(text="load failed")
        except Exception:
            pass
        self.enqueue_log(f"[GPL-DL] Experiment list load failed: {err}")
        messagebox.showerror("Experiment list",
                             f"Could not load experiment list:\n{err}",
                             parent=getattr(self, "_gplcfg_win", self))

    def _gpl_gse_list_loaded(self, res):
        self._gplcfg_gse_pool = res['series']
        self._gplcfg_gse_pool_upper = [
            f"{r['gse']}  {r['desc']}".upper() for r in self._gplcfg_gse_pool]
        # A download rebuilds the platform from the checked rows alone, so the
        # experiments already on disk start checked. Otherwise picking a few
        # new ones would quietly throw the rest of the platform away.
        try:
            from genevariate.core.gpl_downloader import downloaded_series
            self._gplcfg_gse_downloaded = downloaded_series(
                self._gplcfg_gpl_id, self.data_dir)
        except Exception:
            self._gplcfg_gse_downloaded = set()
        if self._gplcfg_gse_downloaded:
            self._gplcfg_gse_selected.update(self._gplcfg_gse_downloaded)
            self._gplcfg_gse_mode.set("selected")
            self.enqueue_log(
                f"[GPL-DL] {len(self._gplcfg_gse_downloaded):,} experiments "
                f"are already downloaded and start selected; anything you add "
                f"extends the platform")
        try:
            self._gplcfg_gse_load_btn.configure(
                state="normal", text=f"Reload ({res['n_series']:,})")
        except Exception:
            pass
        self.enqueue_log(
            f"[GPL-DL] Loaded {res['n_series']:,} experiments for "
            f"{self._gplcfg_gpl_id}")
        self._gpl_render_gse_tree()
        self._gpl_update_gse_counter()

    def _gpl_gse_search_changed(self, _evt=None):
        if self._gplcfg_gse_filter_after is not None:
            try:
                self.after_cancel(self._gplcfg_gse_filter_after)
            except Exception:
                pass
        self._gplcfg_gse_filter_after = self.after(150, self._gpl_render_gse_tree)

    def _gpl_render_gse_tree(self):
        self._gplcfg_gse_filter_after = None
        tree = self._gplcfg_gse_tree
        q = self._gplcfg_gse_search.get().strip().upper()
        if q:
            matches = [r for r, u in zip(self._gplcfg_gse_pool,
                                         self._gplcfg_gse_pool_upper) if q in u]
        else:
            matches = list(self._gplcfg_gse_pool)
        self._gplcfg_gse_matches = matches
        tree.delete(*tree.get_children())
        cap = self._GPL_GENE_RENDER_CAP
        for r in matches[:cap]:
            gse = r['gse']
            mark = "\u2611" if gse in self._gplcfg_gse_selected else "\u2610"
            n = r.get('n_samples', 0)
            desc = f"({n}) {r['desc']}" if n else r['desc']
            if gse in self._gplcfg_gse_downloaded:
                desc = f"[downloaded] {desc}"
            tree.insert("", "end", iid=gse, values=(mark, gse, desc))
        shown = min(len(matches), cap)
        extra = "" if len(matches) <= cap else f" (first {cap})"
        self._gplcfg_gse_showing.config(
            text=f"Showing {shown:,} of {len(matches):,} matches{extra}")

    def _gpl_gse_tree_click(self, event):
        tree = self._gplcfg_gse_tree
        row = tree.identify_row(event.y)
        if not row:
            return
        # Clicking the GSE accession opens its GEO record in a browser; the
        # checkbox and description columns toggle selection.
        if tree.identify_column(event.x) == "#2":
            self._gpl_open_gse_page(row)
            return
        if row in self._gplcfg_gse_selected:
            self._gplcfg_gse_selected.discard(row)
            tree.set(row, "sel", "\u2610")
        else:
            self._gplcfg_gse_selected.add(row)
            tree.set(row, "sel", "\u2611")
        # checking a specific experiment means "download only these"
        if self._gplcfg_gse_selected:
            self._gplcfg_gse_mode.set("selected")
        self._gpl_update_gse_counter()

    def _gpl_gse_tree_hover(self, event):
        """Show a hand cursor over the clickable GSE-accession column."""
        tree = self._gplcfg_gse_tree
        over_link = (tree.identify_row(event.y)
                     and tree.identify_column(event.x) == "#2")
        try:
            tree.configure(cursor="hand2" if over_link else "")
        except Exception:
            pass

    def _gpl_open_gse_page(self, gse):
        """Open the GEO record for a series accession in the default browser."""
        import webbrowser
        url = ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
               + str(gse))
        try:
            webbrowser.open_new_tab(url)
            self.enqueue_log(f"[GPL-DL] Opening GEO record {gse}: {url}")
        except Exception as e:
            self.enqueue_log(f"[GPL-DL] Could not open {url}: {e}")

    def _gpl_gse_select_matches(self):
        if not self._gplcfg_gse_matches:
            return
        self._gplcfg_gse_selected.update(r['gse'] for r in self._gplcfg_gse_matches)
        self._gpl_render_gse_tree()
        self._gpl_update_gse_counter()

    def _gpl_gse_select_all(self):
        self._gplcfg_gse_selected = {r['gse'] for r in self._gplcfg_gse_pool}
        self._gpl_render_gse_tree()
        self._gpl_update_gse_counter()

    def _gpl_gse_select_downloaded(self):
        if not self._gplcfg_gse_downloaded:
            return
        self._gplcfg_gse_selected.update(self._gplcfg_gse_downloaded)
        self._gplcfg_gse_mode.set("selected")
        self._gpl_render_gse_tree()
        self._gpl_update_gse_counter()

    def _gpl_gse_clear(self):
        self._gplcfg_gse_selected.clear()
        self._gpl_render_gse_tree()
        self._gpl_update_gse_counter()

    def _gpl_update_gse_counter(self):
        n_sel = len(self._gplcfg_gse_selected)
        n_tot = len(self._gplcfg_gse_pool) or self._gplcfg_info['total_series']
        self._gplcfg_gse_counter.config(
            text=f"Selected: {n_sel:,} / {n_tot:,} exp.")

    # ---- gene-picker state helpers ----
    def _gpl_set_picker_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for w in ("_gplcfg_search", "_gplcfg_paste"):
            try:
                getattr(self, w).configure(state=state)
            except Exception:
                pass
        try:
            self._gplcfg_load_btn.configure(
                state="normal" if enabled else "disabled")
        except Exception:
            pass

    def _gpl_gene_mode_changed(self):
        selected = self._gplcfg_gene_mode.get() == "selected"
        self._gpl_set_picker_enabled(selected)
        if selected and not self._gplcfg_pool:
            self._gplcfg_showing.config(
                text="click 'Load gene list from platform'")

    def _gpl_load_gene_list(self):
        gpl_id = self._gplcfg_win.title().split("-")[-1].strip()
        self._gplcfg_load_btn.configure(state="disabled", text="Loading…")
        self._gplcfg_showing.config(text="downloading annotation…")

        def worker():
            try:
                from genevariate.core.gpl_downloader import list_platform_genes
                dest = os.path.join(self.data_dir, gpl_id, "annotation")
                res = list_platform_genes(gpl_id, dest)
                self.after(0, lambda: self._gpl_gene_list_loaded(res))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda _e=str(e), _tb=tb:
                           self._gpl_gene_list_failed(_e, _tb))

        threading.Thread(target=worker, daemon=True).start()

    def _gpl_gene_list_failed(self, err, tb=""):
        try:
            self._gplcfg_load_btn.configure(
                state="normal", text="Load gene list from platform")
            self._gplcfg_showing.config(text="load failed")
        except Exception:
            pass
        self.enqueue_log(f"[GPL-DL] Gene list load failed: {err}")
        messagebox.showerror("Gene list", f"Could not load gene list:\n{err}",
                             parent=getattr(self, "_gplcfg_win", self))

    def _gpl_gene_list_loaded(self, res):
        self._gplcfg_pool = res['genes']
        self._gplcfg_pool_upper = [g.upper() for g in self._gplcfg_pool]
        try:
            self._gplcfg_load_btn.configure(
                state="normal", text=f"Reload ({res['n_genes']:,} genes)")
        except Exception:
            pass
        self.enqueue_log(
            f"[GPL-DL] Loaded {res['n_genes']:,} genes "
            f"(col '{res['gene_col']}')")
        self._gpl_render_gene_tree()
        self._gpl_update_gene_counter()

    def _gpl_gene_search_changed(self, _evt=None):
        if self._gplcfg_filter_after is not None:
            try:
                self.after_cancel(self._gplcfg_filter_after)
            except Exception:
                pass
        self._gplcfg_filter_after = self.after(150, self._gpl_render_gene_tree)

    def _gpl_render_gene_tree(self):
        self._gplcfg_filter_after = None
        tree = self._gplcfg_tree
        q = self._gplcfg_search.get().strip().upper()
        if q:
            matches = [g for g, u in zip(self._gplcfg_pool,
                                         self._gplcfg_pool_upper) if q in u]
        else:
            matches = list(self._gplcfg_pool)
        self._gplcfg_matches = matches
        tree.delete(*tree.get_children())
        cap = self._GPL_GENE_RENDER_CAP
        for g in matches[:cap]:
            mark = "\u2611" if g in self._gplcfg_selected else "\u2610"
            tree.insert("", "end", iid=g, values=(mark, g))
        shown = min(len(matches), cap)
        extra = "" if len(matches) <= cap else f" (first {cap})"
        self._gplcfg_showing.config(
            text=f"Showing {shown:,} of {len(matches):,} matches{extra}")

    def _gpl_gene_tree_click(self, event):
        tree = self._gplcfg_tree
        row = tree.identify_row(event.y)
        if not row:
            return
        if row in self._gplcfg_selected:
            self._gplcfg_selected.discard(row)
            tree.set(row, "sel", "\u2610")
        else:
            self._gplcfg_selected.add(row)
            tree.set(row, "sel", "\u2611")
        self._gpl_update_gene_counter()

    def _gpl_gene_select_matches(self):
        if not self._gplcfg_matches:
            return
        self._gplcfg_selected.update(self._gplcfg_matches)
        self._gpl_render_gene_tree()
        self._gpl_update_gene_counter()

    def _gpl_gene_select_all(self):
        self._gplcfg_selected = set(self._gplcfg_pool)
        self._gpl_render_gene_tree()
        self._gpl_update_gene_counter()

    def _gpl_gene_clear(self):
        self._gplcfg_selected.clear()
        self._gpl_render_gene_tree()
        self._gpl_update_gene_counter()

    def _gpl_gene_add_pasted(self):
        raw = self._gplcfg_paste.get("1.0", "end")
        toks = [t.strip() for t in re.split(r"[\s,;]+", raw) if t.strip()]
        if not toks:
            return
        pool_upper = {u: g for g, u in zip(self._gplcfg_pool,
                                           self._gplcfg_pool_upper)}
        matched, unmatched = 0, []
        for t in toks:
            g = pool_upper.get(t.upper())
            if g:
                self._gplcfg_selected.add(g)
                matched += 1
            else:
                unmatched.append(t)
        self._gpl_render_gene_tree()
        self._gpl_update_gene_counter()
        msg = f"Added {matched} gene(s)."
        if unmatched:
            msg += (f" {len(unmatched)} not on platform: "
                    f"{', '.join(unmatched[:8])}"
                    f"{'…' if len(unmatched) > 8 else ''}")
        self.enqueue_log(f"[GPL-DL] {msg}")
        self._gplcfg_showing.config(text=msg)

    def _gpl_gene_load_file(self):
        path = filedialog.askopenfilename(
            title="Load gene list",
            filetypes=[("Text/CSV", "*.txt *.csv *.tsv"), ("All", "*.*")],
            parent=self._gplcfg_win)
        if not path:
            return
        try:
            with open(path, "r", errors="replace") as fh:
                content = fh.read()
        except Exception as e:
            messagebox.showerror("Load", str(e), parent=self._gplcfg_win)
            return
        self._gplcfg_paste.delete("1.0", "end")
        self._gplcfg_paste.insert("1.0", content)
        self._gpl_gene_add_pasted()

    def _gpl_update_gene_counter(self):
        n_sel = len(self._gplcfg_selected)
        n_tot = len(self._gplcfg_pool)
        self._gplcfg_counter.config(text=f"Selected: {n_sel:,} / {n_tot:,} genes")

    def _gpl_start_configured_download(self, info, downloader):
        gpl_id = info['gpl_id']
        try:
            max_gse = int(self._gplcfg_maxgse.get() or 0)
        except ValueError:
            max_gse = 0

        # metadata options
        series_fields = [f for (scope, f), v in self._gplcfg_meta_vars.items()
                         if scope == "series" and v.get()]
        sample_fields = [f for (scope, f), v in self._gplcfg_meta_vars.items()
                         if scope == "sample" and v.get()]
        metadata_opts = {"series": series_fields, "sample": sample_fields}

        # gene whitelist
        gene_whitelist = None
        if self._gplcfg_gene_mode.get() == "selected":
            if not self._gplcfg_selected:
                messagebox.showwarning(
                    "No genes selected",
                    "Select at least one gene, or choose 'All genes'.",
                    parent=self._gplcfg_win)
                return
            gene_whitelist = sorted(self._gplcfg_selected)

        # GSE / experiment whitelist
        gse_whitelist = None
        if self._gplcfg_gse_mode.get() == "selected":
            if not self._gplcfg_gse_selected:
                messagebox.showwarning(
                    "No experiments selected",
                    "Select at least one experiment, or choose "
                    "'All experiments'.",
                    parent=self._gplcfg_win)
                return
            gse_whitelist = sorted(self._gplcfg_gse_selected)

        if gse_whitelist:
            n_to_dl = len(gse_whitelist)
        else:
            n_to_dl = min(info['total_series'], max_gse) if max_gse else info['total_series']
        gene_desc = (f"{len(gene_whitelist):,} selected genes"
                     if gene_whitelist else "ALL genes")
        exp_desc = (f"{len(gse_whitelist):,} selected experiments"
                    if gse_whitelist else
                    (f"first {n_to_dl}" if max_gse else "ALL experiments"))
        meta_desc = (f"series[{','.join(series_fields) or '-'}], "
                     f"sample[{','.join(sample_fields) or '-'}]")
        # The run rebuilds the platform from this selection, so anything
        # already downloaded but left unchecked disappears from the matrix.
        dropped = (self._gplcfg_gse_downloaded - set(gse_whitelist)
                   if gse_whitelist else set())
        drop_line = (f"\nWARNING: {len(dropped):,} already-downloaded "
                     f"experiments are unchecked and will be dropped from the "
                     f"rebuilt platform.\n" if dropped else "")
        if not messagebox.askyesno(
                f"Download {gpl_id}?",
                f"Platform: {info['title']}\nOrganism: {info['organism']}\n"
                f"Experiments: {exp_desc} (of {info['total_series']})\n"
                f"Genes: {gene_desc}\nMetadata: {meta_desc}\n"
                f"{drop_line}\nProceed?",
                parent=self._gplcfg_win):
            return

        try:
            self._gplcfg_win.destroy()
        except Exception:
            pass

        self.auto_dl_status.config(text=f"Starting {gpl_id}...",
                                   foreground=AERO["accent"])
        if hasattr(self, 'auto_dl_progress_bar'):
            self.auto_dl_progress_bar["value"] = 0
        self.enqueue_log(
            f"[GPL-DL] Starting {gpl_id} ({info['organism']}) - "
            f"{exp_desc}, {gene_desc}, metadata {meta_desc}...")

        def worker():
            try:
                result = downloader.run_with_info(
                    info=info, max_gse=max_gse,
                    gene_whitelist=gene_whitelist, metadata_opts=metadata_opts,
                    gse_whitelist=gse_whitelist,
                    callback=lambda p, s, m: self.after(
                        0, self._gpl_dl_progress, p, s, m))
                self.after(0, lambda: self._gpl_dl_done(result))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda _e=str(e), _tb=tb:
                           self._gpl_dl_error(gpl_id, _e, _tb))

        threading.Thread(target=worker, daemon=True).start()

    def _gpl_dl_progress(self, pct, stage, msg):
        if pct is not None:
            self.progressbar["value"] = pct
            if hasattr(self, 'auto_dl_progress_bar'):
                self.auto_dl_progress_bar["value"] = pct
        self.auto_dl_status.config(text=f"[{stage}] {msg}", foreground=AERO["accent"])
        self.enqueue_log(f"[GPL-DL] {msg}")
        self.update_idletasks()
    
    def _gpl_dl_done(self, result):
        self.update_progress(value=0)
        if hasattr(self, 'auto_dl_progress_bar'):
            self.auto_dl_progress_bar["value"] = 100
        self.auto_dl_status.config(
            text=f"Done: {result['gpl_id']} - {result['n_samples']:,} samples, {result['n_genes']:,} genes",
            foreground="green")
        self.enqueue_log(
            f"[GPL-DL] Done {result['gpl_id']}: {result['n_samples']:,} samples, "
            f"{result['n_genes']:,} genes, {result['n_series']} series")
        gsm_filt = getattr(self, '_pending_gsm_filter', None)
        self._pending_gsm_filter = None
        self._load_gpl_data(result['gpl_id'], result['filepath'], gsm_filter=gsm_filt)
        self._update_platform_status()
        # Show download summary
        self._show_gpl_download_summary(result)

    def _show_gpl_download_summary(self, result):
        """Show summary window after GPL download with experiments, statistics."""
        gpl_id = result['gpl_id']
        win = tk.Toplevel(self)
        style_window(win)
        win.title(f"{gpl_id} - Download Summary")
        win.geometry("800x600")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"800x600+{(_sw-800)//2}+{(_sh-600)//2}")
        except: pass

        # Header
        hdr = ttk.Frame(win)
        hdr.pack(fill=tk.X, padx=15, pady=(15, 5))
        species = GPL_SPECIES.get(gpl_id, result.get('organism', '?'))
        ttk.Label(hdr, text=f"{gpl_id} - {species.title()}",
                  font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT)

        # Stats frame
        stats = labelframe(win, text="Platform Statistics", padding=10)
        stats.pack(fill=tk.X, padx=15, pady=5)

        # Where the values came from matters for RNA-seq: the counts are
        # NCBI's re-alignment of the raw reads, not the submitters' numbers.
        is_counts = result.get('source') == 'ncbi-counts'
        stats_data = [
            ("Total Samples (GSMs)", f"{result['n_samples']:,}"),
            ("Total Genes (probes)", f"{result['n_genes']:,}"),
            ("Experiments (GSEs)", f"{result.get('n_series', '?')}"),
            ("Species", species.title()),
            ("Values", "NCBI reprocessed RNA-seq counts (raw)"
                       if is_counts else "GEO series matrix"),
            ("File", str(result.get('filepath', '?'))[-60:]),
        ]
        for i, (label, value) in enumerate(stats_data):
            ttk.Label(stats, text=f"{label}:", font=('Segoe UI', 10, 'bold')).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(stats, text=value, font=('Segoe UI', 10)).grid(
                row=i, column=1, sticky=tk.W, padx=10, pady=2)

        # Series NCBI has not reprocessed are absent from the matrix entirely,
        # so say which ones and where to get them rather than letting the
        # sample count quietly come up short.
        missing = result.get('missing_gse') or []
        if missing:
            miss = labelframe(
                win, text=f"Not loaded - {len(missing)} series without NCBI counts",
                padding=8)
            miss.pack(fill=tk.X, padx=15, pady=5)
            note = ttk.Label(
                miss,
                text=("NCBI reprocesses human and mouse RNA-seq only, and not "
                      "every series yet. These have no uniform count table, so "
                      "their samples are not in the matrix - load them from "
                      "ARCHS4 (Sources ▸ ARCHS4) if you need them:\n"
                      + ", ".join(missing[:40])
                      + (f"  … and {len(missing) - 40} more" if len(missing) > 40 else "")),
                justify=tk.LEFT)
            note.pack(fill=tk.X)
            _wrap_to_parent(note)

        # Series NCBI refused to answer for are a different case: their counts
        # may well exist. Reporting them alongside the uncovered ones would
        # understate what NCBI has.
        unanswered = result.get('unanswered_gse') or []
        if unanswered:
            una = labelframe(
                win, text=f"Not checked - {len(unanswered)} series NCBI did "
                          f"not answer for", padding=8)
            una.pack(fill=tk.X, padx=15, pady=5)
            unote = ttk.Label(
                una,
                text=("The download endpoint dropped these requests, so "
                      "whether counts exist is unknown - not absent. Run the "
                      "download again to pick them up:\n"
                      + ", ".join(unanswered[:40])
                      + (f"  … and {len(unanswered) - 40} more"
                         if len(unanswered) > 40 else "")),
                justify=tk.LEFT)
            unote.pack(fill=tk.X)
            _wrap_to_parent(unote)

        # Experiment list
        gse_frame = labelframe(win, text="Experiments (GSEs)", padding=5)
        gse_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Get GSE list from loaded data
        df = self.gpl_datasets.get(gpl_id)
        gse_list = []
        if df is not None and 'GSM' in df.columns and self.gds_conn:
            try:
                gsms = df['GSM'].astype(str).str.upper().tolist()[:5000]
                ph = ','.join(['?'] * len(gsms))
                rows = self.gds_conn.execute(
                    f"SELECT series_id, COUNT(*) as cnt FROM gsm WHERE UPPER(gsm) IN ({ph}) "
                    f"GROUP BY series_id ORDER BY cnt DESC",
                    [g.upper() for g in gsms]).fetchall()
                for gse, cnt in rows:
                    gse = str(gse).strip()
                    if gse and gse != 'nan':
                        gse_list.append((gse, cnt))
            except Exception as e:
                self.enqueue_log(f"[GPL-DL] GSE list error: {e}")

        if gse_list:
            cols = ("GSE", "Samples")
            gse_tree = ttk.Treeview(gse_frame, columns=cols, show="headings", height=15)
            vsb = ttk.Scrollbar(gse_frame, orient="vertical", command=gse_tree.yview)
            gse_tree.configure(yscrollcommand=vsb.set)
            gse_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            gse_tree.heading("GSE", text="GSE Experiment")
            gse_tree.heading("Samples", text="Samples")
            gse_tree.column("GSE", width=150)
            gse_tree.column("Samples", width=80, anchor='center')

            # Store GSE→GSMs mapping for click handler
            gse_gsm_map = {}
            if self.gds_conn:
                try:
                    all_gsms = df['GSM'].astype(str).str.upper().tolist()
                    for i in range(0, len(all_gsms), 500):
                        chunk = all_gsms[i:i+500]
                        ph = ','.join(['?'] * len(chunk))
                        rows = self.gds_conn.execute(
                            f"SELECT gsm, series_id FROM gsm WHERE UPPER(gsm) IN ({ph})",
                            [g.upper() for g in chunk]).fetchall()
                        for gsm, gse in rows:
                            gse = str(gse).strip()
                            if gse and gse != 'nan':
                                if gse not in gse_gsm_map:
                                    gse_gsm_map[gse] = []
                                gse_gsm_map[gse].append(str(gsm).strip())
                except Exception:
                    pass

            for gse, cnt in gse_list:
                gse_tree.insert("", tk.END, values=(gse, cnt))

            def _on_gse_click(event):
                item = gse_tree.focus()
                if not item:
                    return
                vals = gse_tree.item(item, 'values')
                if not vals:
                    return
                gse_id = vals[0]

                # Open GEO website
                import webbrowser
                url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
                webbrowser.open(url)

                # Show samples window with raw metadata
                gsms = gse_gsm_map.get(gse_id, [])
                if gsms and self.gds_conn:
                    self._show_gse_raw_metadata(win, gse_id, gsms)

            gse_tree.bind("<Double-1>", _on_gse_click)
            ttk.Label(gse_frame,
                      text=f"{len(gse_list)} experiments - double-click to open GEO page & view sample metadata",
                      font=('Segoe UI', 8, 'italic'), foreground='#888').pack(pady=2)
        else:
            ttk.Label(gse_frame, text="Experiment list not available (GEOmetadb not loaded)",
                      font=('Segoe UI', 10), foreground=AERO["muted"]).pack(pady=30)

        # Close button
        ttk.Button(win, text="Close", command=win.destroy,
                   style="Secondary.TButton").pack(pady=10)

    def _show_gse_raw_metadata(self, parent, gse_id, gsms):
        """Show raw GEOmetadb metadata for all samples in a GSE experiment."""
        if not self.gds_conn or not gsms:
            return

        # Query all metadata columns from gsm table
        try:
            ph = ','.join(['?'] * len(gsms))
            meta_df = pd.read_sql_query(
                f"SELECT * FROM gsm WHERE UPPER(gsm) IN ({ph})",
                self.gds_conn, params=[g.upper() for g in gsms])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to query metadata:\n{e}", parent=parent)
            return

        if meta_df.empty:
            messagebox.showinfo("No Data", f"No metadata found for {gse_id} samples.", parent=parent)
            return

        top = tk.Toplevel(parent)
        style_window(top)
        top.title(f"{gse_id} - {len(meta_df)} samples (raw GEOmetadb metadata)")
        top.geometry("1300x700")
        try:
            _sw, _sh = top.winfo_screenwidth(), top.winfo_screenheight()
            top.geometry(f"1300x700+{(_sw-1300)//2}+{(_sh-700)//2}")
        except: pass

        # Summary
        ttk.Label(top, text=f"{gse_id}: {len(meta_df)} samples - all GEOmetadb fields shown",
                  font=('Segoe UI', 11, 'bold')).pack(fill=tk.X, padx=10, pady=(10, 5))

        # Treeview with all columns
        tv_frame = ttk.Frame(top)
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cols = list(meta_df.columns)
        tree = ttk.Treeview(tv_frame, columns=cols, show="headings", height=25)
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tv_frame.grid_rowconfigure(0, weight=1)
        tv_frame.grid_columnconfigure(0, weight=1)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=130, minwidth=80)

        _ROW_CAP = 2000
        for _, row in meta_df.head(_ROW_CAP).iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, ''))[:100] for c in cols])
        if len(meta_df) > _ROW_CAP:
            note = [f"… {len(meta_df) - _ROW_CAP:,} more samples "
                    f"(table capped at {_ROW_CAP:,}; use Save to CSV for all)"] \
                   + [""] * (len(cols) - 1)
            tree.insert("", tk.END, values=note)

        # Buttons
        btn_frame = ttk.Frame(top, padding=5)
        btn_frame.pack(fill=tk.X)

        def _save():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                initialfile=f"{gse_id}_raw_metadata.csv", parent=top)
            if path:
                meta_df.to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved {len(meta_df)} samples to:\n{path}", parent=top)

        ttk.Button(btn_frame, text="Save to CSV", command=_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=top.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _gpl_dl_error(self, gpl_id, err, tb=""):
        self.update_progress(value=0)
        if hasattr(self, 'auto_dl_progress_bar'):
            self.auto_dl_progress_bar["value"] = 0
        self.auto_dl_status.config(text=f"Failed: {err[:50]}", foreground="red")
        self.enqueue_log(f"[GPL-DL] ERROR {gpl_id}:\n{err}\n{tb}")
        messagebox.showerror(f"{gpl_id} Failed", f"Error: {err}\n\nSee log for details.", parent=self)
    


if __name__ == "__main__":
    import sys
    # Redirect stderr to file for crash debugging
    try:
        _crash_log = open(os.path.join(os.path.expanduser('~'), 'genevariate_crash.log'), 'w',
                          encoding='utf-8', errors='replace')
    except Exception:
        _crash_log = None
    try:
        app = GeoWorkflowGUI()
        app.mainloop()
    except Exception as e:
        msg = f"[MAIN] EXCEPTION: {e}"
        print(msg, flush=True)
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        if _crash_log:
            _crash_log.write(msg + "\n" + tb + "\n")
            _crash_log.close()
            print(f"Crash log saved to ~/genevariate_crash.log")
