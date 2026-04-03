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
import ollama
import uuid
import re
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import itertools

# ── Memory-Augmented Agent Extraction (from geo_ns_repair_v2_9_.py) ──
try:
    from .deterministic_extraction import (
        MemoryAgent, GSEContext, GSEWorker,
        build_gse_contexts, find_llm_memory_dir, phase15_collapse,
        is_ns, NS, LABEL_COLS,
        compute_ollama_parallel as compute_hybrid_parallel,
        start_ollama_cpu_server, stop_cpu_server,
        _pick_ollama_url, _CPU_OLLAMA_ACTIVE,
        detect_gpus as det_detect_gpus, MODEL_RAM_GB, DEFAULT_URL,
        CPU_OLLAMA_URL, vram_utilisation_pct
    )
    _HAS_DETERMINISTIC = True
except ImportError:
    try:
        from genevariate.gui.deterministic_extraction import (
            MemoryAgent, GSEContext, GSEWorker,
            build_gse_contexts, find_llm_memory_dir, phase15_collapse,
            is_ns, NS, LABEL_COLS,
            compute_ollama_parallel as compute_hybrid_parallel,
            start_ollama_cpu_server, stop_cpu_server,
            _pick_ollama_url, _CPU_OLLAMA_ACTIVE,
            detect_gpus as det_detect_gpus, MODEL_RAM_GB, DEFAULT_URL,
            CPU_OLLAMA_URL, vram_utilisation_pct
        )
        _HAS_DETERMINISTIC = True
    except ImportError:
        _HAS_DETERMINISTIC = False
        print("[WARN] deterministic_extraction.py not found — LLM-only mode")

# MUST set TkAgg BEFORE any pyplot import (default qtagg conflicts with Tk)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 20  # Warn early about too many figures

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import ranksums, wasserstein_distance, gaussian_kde
from scipy.signal import find_peaks
try:
    from .region_analysis import RegionAnalysisWindow
except ImportError:
    from genevariate.gui.region_analysis import RegionAnalysisWindow
try:
    from .compare_analysis import CompareRegionsWindow, CompareDistributionsWindow
except ImportError:
    try:
        from genevariate.gui.compare_analysis import CompareRegionsWindow, CompareDistributionsWindow
    except ImportError:
        CompareRegionsWindow = None
        CompareDistributionsWindow = None

# ═══════════════════════════════════════════════════════════════
#  GPU DETECTION + VRAM/RAM WATCHDOG
# ═══════════════════════════════════════════════════════════════
MODEL_RAM_GB = {
    'gemma2:2b': 2.0, 'gemma2:9b': 6.0, 'gemma2:27b': 18.0,
    'llama3:8b': 6.0, 'mistral:7b': 5.0,
}

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


def check_ollama_gpu(base_url="http://localhost:11434"):
    """Check if Ollama is using GPU. Returns ('gpu', vram_gb) | ('cpu', 0) | ('unknown', 0)."""
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(f"{base_url}/api/ps", method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode('utf-8'))
            models = data.get("models", [])
            if models:
                vram = models[0].get("size_vram", 0)
                total = models[0].get("size", 1)
                if vram > total * 0.5:
                    return "gpu", round(vram / 1e9, 1)
                return "cpu", 0
    except Exception:
        pass
    return "unknown", 0


def _get_vram_usage():
    """Returns (used_mb, total_mb, used_pct) for GPU 0 via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3)
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        used_mb, total_mb = int(parts[0]), int(parts[1])
        return used_mb, total_mb, 100.0 * used_mb / total_mb if total_mb else 0.0
    except Exception:
        return 0, 0, 0.0


class ResourceWatchdog:
    """Monitor RAM usage, pause LLM calls if RAM threshold exceeded.
    VRAM: no pausing — when VRAM is full, Ollama automatically falls back to CPU.
    Only RAM is dangerous (OOM crash), so only RAM triggers pause.
    """
    RAM_PAUSE_PCT = 90.0
    RAM_RESUME_PCT = 85.0
    CHECK_INTERVAL = 3

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda m: None)
        self._gate = threading.Event()
        self._gate.set()  # set = NOT paused
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._call_times = []
        self._pause_reason = None
        self._status_text = ""
        self._vram_warning_shown = False

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True, name="ResourceWatchdog")
        t.start()
        return self

    def stop(self):
        self._stop.set()
        self._gate.set()

    def record_call(self):
        with self._lock:
            now = time.time()
            self._call_times.append(now)
            self._call_times = [t for t in self._call_times if now - t <= 60]

    def wait_if_paused(self):
        self._gate.wait()

    @property
    def status(self):
        return self._status_text

    @property
    def is_paused(self):
        return not self._gate.is_set()

    def calls_per_min(self):
        with self._lock:
            now = time.time()
            return len([t for t in self._call_times if now - t <= 60])

    def _loop(self):
        try:
            import psutil
        except ImportError:
            self._log("[Watchdog] psutil not available — RAM monitoring disabled")
            return

        total_ram_mb = psutil.virtual_memory().total / 1e6
        while not self._stop.is_set():
            try:
                vm = psutil.virtual_memory()
                ram_pct, ram_mb = vm.percent, vm.used / 1e6
                vram_used, vram_total, vram_pct = _get_vram_usage()
                has_gpu = vram_total > 0
                cpm = self.calls_per_min()
                state = "running" if self._gate.is_set() else f"PAUSED ({self._pause_reason})"

                if has_gpu:
                    self._status_text = (
                        f"RAM: {ram_mb:.0f}/{total_ram_mb:.0f}MB ({ram_pct:.0f}%) | "
                        f"VRAM: {vram_used:,}/{vram_total:,}MB ({vram_pct:.0f}%) | "
                        f"LLM/min: {cpm} | {state}")
                else:
                    self._status_text = (
                        f"RAM: {ram_mb:.0f}/{total_ram_mb:.0f}MB ({ram_pct:.0f}%) | "
                        f"LLM/min: {cpm} | {state}")

                # Pause ONLY on RAM (prevents OOM crash)
                if ram_pct >= self.RAM_PAUSE_PCT and self._gate.is_set():
                    self._gate.clear()
                    self._pause_reason = "RAM"
                    self._log(f"[Watchdog] RAM at {ram_pct:.0f}% — PAUSING LLM calls")

                # VRAM full = just log once, Ollama falls back to CPU automatically
                elif has_gpu and vram_pct >= 90.0 and not self._vram_warning_shown:
                    self._vram_warning_shown = True
                    self._log(f"[Watchdog] VRAM at {vram_pct:.0f}% — Ollama using CPU fallback (no pause)")

                # Resume from RAM pause
                elif not self._gate.is_set():
                    if ram_pct < self.RAM_RESUME_PCT:
                        self._gate.set()
                        self._pause_reason = None
                        self._log(f"[Watchdog] RAM OK — RESUMING")

            except Exception:
                pass
            self._stop.wait(self.CHECK_INTERVAL)


def _find_geometadb():
    """Search common locations for GEOmetadb.sqlite.gz (case-insensitive filename).
    
    app.py lives in genevariate/gui/app.py
    GEOmetadb is at genevariate/data/GEOmetadb.sqlite.gz
    So the key path is: _PROG_DIR/../data/GEOmetadb.sqlite.gz
    """
    _TARGET = 'geometadb.sqlite.gz'  # lowercase for matching
    _PROG_DIR = os.path.dirname(os.path.abspath(__file__))  # genevariate/gui/
    _PROJ_ROOT = os.path.dirname(_PROG_DIR)                 # genevariate/
    _CWD = os.getcwd()

    candidates = [
        # PRIMARY: project structure (app.py in gui/, data in ../data/)
        os.path.join(_PROJ_ROOT, 'data', 'GEOmetadb.sqlite.gz'),
        os.path.join(_PROJ_ROOT, 'GEOmetadb.sqlite.gz'),
        # Fallbacks
        os.path.join(_PROG_DIR, 'GEOmetadb.sqlite.gz'),
        os.path.join(_CWD, 'GEOmetadb.sqlite.gz'),
        os.path.join(_PROG_DIR, 'data', 'GEOmetadb.sqlite.gz'),
        os.path.join(_CWD, 'data', 'GEOmetadb.sqlite.gz'),
        os.path.join(_PROG_DIR, '..', 'data', 'GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/.genevariate/GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/.genevariate/data/GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/Desktop/GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/Downloads/GEOmetadb.sqlite.gz'),
        os.path.expanduser('~/data/GEOmetadb.sqlite.gz'),
        './GEOmetadb.sqlite.gz',
        './data/GEOmetadb.sqlite.gz',
    ]

    # Case-insensitive search in common dirs
    search_dirs = [
        os.path.join(_PROJ_ROOT, 'data'),  # genevariate/data/ — PRIMARY
        _PROJ_ROOT,                          # genevariate/
        _PROG_DIR,                           # genevariate/gui/
        _CWD,
        os.path.join(_PROG_DIR, 'data'),
        os.path.join(_CWD, 'data'),
        os.path.expanduser('~'),
        os.path.expanduser('~/Desktop'),
        os.path.expanduser('~/Downloads'),
        os.path.expanduser('~/Documents'),
        os.path.expanduser('~/.genevariate'),
        os.path.expanduser('~/.genevariate/data'),
        '.',
        './data',
    ]
    dd = _find_data_dir()
    if dd:
        search_dirs.extend([dd, os.path.dirname(dd)])

    for d in search_dirs:
        if os.path.isdir(d):
            try:
                for f in os.listdir(d):
                    if f.lower() == _TARGET:
                        candidates.append(os.path.join(d, f))
            except Exception:
                pass

    seen = set()
    for p in candidates:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.exists(p):
            return p
        seen.add(rp)

    return candidates[0]  # fallback to default

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
    
    # Standard candidates — project root/data/ FIRST
    candidates = [
        os.path.join(_PROJ_ROOT, 'data'),       # genevariate/data/ — PRIMARY
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

CONFIG = {
    'threading': {'max_workers': 10},
    'paths': {
        'data': _find_data_dir(),
        'results': './results',
        'geo_db': _find_geometadb(),
        'embedding_cache': './cache/embeddings'
    },
    'database': {'sql_chunk_size': 500},
    'ai': {
        'model': 'gemma2:9b',
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

# ── AI Classification Engine (REAL implementation) ──────────────────────
class SampleClassificationAgent:
    """
    Agent that classifies GEO samples using Ollama LLM in parallel.
    Calls classify_sample() for each row via ThreadPoolExecutor.
    """
    def __init__(self, tools_list, gui_log_func, max_workers=15):
        self.tools = {func.__name__: func for func in tools_list}
        self.log = gui_log_func if gui_log_func else print
        self.MAX_WORKERS = max_workers

    def process_samples(self, gsm_df, fields=None, custom_fields=None,
                         stop_flag_fn=None, progress_fn=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tool_name = "classify_sample"
        if tool_name not in self.tools:
            self.log(f"[Agent ERROR] Tool '{tool_name}' not registered.")
            return pd.DataFrame()

        classify_fn = self.tools[tool_name]
        total = len(gsm_df)
        if total == 0:
            self.log("[Agent] No samples to extract.")
            return pd.DataFrame()

        # ── PRE-FLIGHT CHECK: test Ollama ──
        self.log("[Agent] Running pre-flight diagnostic...")
        try:
            import ollama as _ollama_test
            models_resp = _ollama_test.list()
            avail = [m.get('name', m.get('model', '')) for m in models_resp.get('models', [])]
            if not avail:
                self.log("[Agent ERROR] Ollama running but NO MODELS installed.")
                self.log("[Agent ERROR] Fix: run 'ollama pull gemma2:9b' in terminal")
                return pd.DataFrame()
            self.log(f"[Agent] OK Ollama running, models: {', '.join(avail[:4])}")
        except Exception as e:
            err = str(e).lower()
            if 'connection' in err or 'refused' in err or 'connect' in err:
                self.log("[Agent ERROR] Cannot connect to Ollama service!")
                self.log("[Agent ERROR] Fix: start Ollama with 'ollama serve'")
            else:
                self.log(f"[Agent ERROR] Ollama check failed: {type(e).__name__}: {e}")
            self.log("[Agent ERROR] LLM extraction requires Ollama (https://ollama.com)")
            return pd.DataFrame()

        # ── GPU DETECTION ──
        gpus = detect_gpus()
        if gpus:
            gpu_names = ", ".join(f"{g['name']} ({g['vram_gb']}GB, {g['free_vram_gb']}GB free)" for g in gpus)
            self.log(f"[Agent] GPU detected: {gpu_names}")
        else:
            self.log("[Agent] WARNING: No GPU detected via nvidia-smi/rocm-smi")

        # Check if Ollama is actually using GPU
        gpu_status, gpu_vram = check_ollama_gpu()
        if gpu_status == "gpu":
            self.log(f"[Agent] OK Ollama running on GPU ({gpu_vram}GB VRAM in use)")
        elif gpu_status == "cpu":
            self.log("[Agent] WARNING: Ollama is running on CPU — extraction will be SLOW!")
            self.log("[Agent] To fix: restart Ollama with GPU support:")
            self.log("[Agent]   CUDA_VISIBLE_DEVICES=0 OLLAMA_GPU_LAYERS=999 ollama serve")
            if gpus:
                self.log(f"[Agent] GPU is available ({gpus[0]['name']}) but Ollama is not using it!")
        else:
            self.log("[Agent] Could not determine Ollama GPU status (no model loaded yet)")

        # ── Start Resource Watchdog (RAM + VRAM) ──
        watchdog = ResourceWatchdog(log_fn=self.log)
        watchdog.start()
        self.log(f"[Agent] Resource watchdog started (RAM>{ResourceWatchdog.RAM_PAUSE_PCT}% → pause, "
                 f"VRAM full → Ollama CPU fallback)")

        # ── Pre-detect models (BEFORE threads to avoid race condition) ──
        global _OLLAMA_MODEL, _OLLAMA_EXTRACTION_MODEL
        if not _OLLAMA_MODEL:
            _OLLAMA_MODEL = _detect_ollama_model()
        if not _OLLAMA_MODEL:
            self.log("[Agent ERROR] No Ollama model found. Run: ollama pull gemma2:9b")
            return pd.DataFrame()
        self.log(f"[Agent] Using model: {_OLLAMA_MODEL}")

        # Detect fast extraction model (gemma2:2b)
        if not _OLLAMA_EXTRACTION_MODEL:
            _OLLAMA_EXTRACTION_MODEL = _detect_extraction_model() or _OLLAMA_MODEL
        if _OLLAMA_EXTRACTION_MODEL != _OLLAMA_MODEL:
            self.log(f"[Agent] Fast extraction model: {_OLLAMA_EXTRACTION_MODEL}")

        # ── Configure OLLAMA_NUM_PARALLEL for concurrent inference ──
        # Ollama shares model weights across parallel slots — only KV cache
        # is duplicated. This is the single biggest throughput multiplier.
        try:
            from genevariate.core.ollama_manager import compute_ollama_parallel as _core_parallel
            _total, _gpu_w, _cpu_w = _core_parallel(
                _OLLAMA_EXTRACTION_MODEL or _OLLAMA_MODEL)
            n_parallel = max(1, min(_gpu_w if _gpu_w > 0 else _total, 8))
            # Check current Ollama parallel config and restart if needed
            import os as _os
            current_parallel = _os.environ.get("OLLAMA_NUM_PARALLEL", "1")
            if int(current_parallel) < n_parallel:
                _os.environ["OLLAMA_NUM_PARALLEL"] = str(n_parallel)
                _os.environ["OLLAMA_KEEP_ALIVE"] = "-1"
                _os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
                self.log(f"[Agent] Restarting Ollama with NUM_PARALLEL={n_parallel} "
                         f"(was {current_parallel})...")
                try:
                    from genevariate.core.ollama_manager import (
                        kill_ollama, start_ollama_server_blocking)
                    kill_ollama()
                    proc = start_ollama_server_blocking(
                        log_fn=self.log, num_parallel=n_parallel)
                    if proc:
                        self.log(f"[Agent] Ollama restarted with {n_parallel} parallel slots")
                    else:
                        self.log("[Agent] Ollama restart failed — continuing with current config")
                except Exception as e:
                    self.log(f"[Agent] Ollama restart note: {e}")
            self.log(f"[Agent] OLLAMA_NUM_PARALLEL={n_parallel} "
                     f"(GPU={_gpu_w}, CPU={_cpu_w} inference slots)")
        except Exception:
            n_parallel = 1

        # ── Preload fast extraction model into VRAM ──
        try:
            if _OLLAMA_EXTRACTION_MODEL:
                ollama.chat(model=_OLLAMA_EXTRACTION_MODEL,
                            messages=[{"role": "user", "content": "1"}],
                            options={"num_predict": 1, "num_ctx": 512},
                            keep_alive=-1)
                self.log(f"[Agent] {_OLLAMA_EXTRACTION_MODEL} preloaded into VRAM")
        except Exception as e:
            self.log(f"[Agent] Model preload note: {e}")

        # ── TEST SINGLE SAMPLE (warm up model + verify) ──
        self.log("[Agent] Warming up model with test sample...")
        test_row = gsm_df.iloc[0]
        test_gsm = test_row.get('gsm', test_row.get('GSM', '?'))
        try:
            test_result = classify_fn(test_row, fields=fields, custom_fields=custom_fields)
            if test_result is None:
                self.log(f"[Agent ERROR] Test sample {test_gsm} returned None.")
                self.log("[Agent ERROR] Check Ollama logs: ollama serve")
                return pd.DataFrame()
            else:
                # Show first extracted value
                preview = {k: v for k, v in test_result.items() if k != 'gsm'}
                self.log(f"[Agent] OK Test: {test_gsm} -> {preview}")
                # Check if all NS — warning
                all_ns = all(str(v).strip() in _NOT_SPECIFIED_VALUES for v in preview.values())
                if all_ns:
                    self.log("[Agent WARNING] Test sample returned ALL 'Not Specified'!")
                    self.log(f"[Agent WARNING] Metadata: {get_comprehensive_gsm_text(test_row)[:200]}")
                    self.log("[Agent WARNING] This may indicate Ollama is not responding correctly.")
        except Exception as e:
            self.log(f"[Agent ERROR] Test sample {test_gsm} CRASHED: {type(e).__name__}: {e}")
            self.log("[Agent ERROR] Cannot proceed with batch extraction.")
            return pd.DataFrame()

        # ── Compute parallel workers from VRAM ──
        if self.MAX_WORKERS == 0:
            # Use the parallel slot count computed above (matches OLLAMA_NUM_PARALLEL)
            n_workers = n_parallel
            self.log(f"[Agent] VRAM-based auto workers: {n_workers}")
        else:
            n_workers = min(total, self.MAX_WORKERS)

        n_workers = max(1, min(n_workers, total))
        self.log(f"[Agent] Extracting {total} samples with {n_workers} workers...")
        if gpus:
            self.log(f"[Agent] GPU acceleration: {gpus[0]['name']} | "
                     f"VRAM: {gpus[0]['free_vram_gb']}GB free")

        results = []
        failures = 0
        first_error = None
        done = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(classify_fn, row, fields=fields, custom_fields=custom_fields): row
                for _, row in gsm_df.iterrows()
            }
            for fut in as_completed(futures):
                # Watchdog: wait if RAM/VRAM too high
                watchdog.wait_if_paused()

                if stop_flag_fn and stop_flag_fn():
                    self.log("[Agent] Extraction stopped by user.")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

                row = futures[fut]
                gsm_id = row.get('gsm', row.get('GSM', '?'))
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                    else:
                        failures += 1
                        if first_error is None:
                            first_error = f"{gsm_id}: returned None"
                except Exception as exc:
                    failures += 1
                    err_msg = f"{gsm_id}: {type(exc).__name__}: {exc}"
                    if first_error is None:
                        first_error = err_msg
                    if failures <= 3:
                        self.log(f"[Agent ERROR] {err_msg}")

                done += 1
                watchdog.record_call()
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t0
                    speed = done / max(0.01, elapsed)
                    eta = (total - done) / max(0.01, speed)
                    self.log(f"[Agent] {done}/{total} ({speed:.1f} smp/s, ETA {eta:.0f}s) "
                             f"ok={len(results)} fail={failures}")
                    if progress_fn:
                        progress_fn(done, total, speed, eta)
                # Log watchdog status every 100 samples
                if done % 100 == 0 and watchdog.status:
                    self.log(f"[Watchdog] {watchdog.status}")

        # Stop watchdog
        watchdog.stop()

        if failures:
            self.log(f"[Agent WARNING] {len(results)} ok, {failures} failed")
            if first_error:
                self.log(f"[Agent] First failure: {first_error}")

        if not results:
            self.log("[Agent ERROR] ALL EXTRACTIONS FAILED!")
            if first_error:
                self.log(f"[Agent ERROR] Reason: {first_error}")
            self.log("[Agent ERROR]   1. Is Ollama running? -> 'ollama serve'")
            self.log("[Agent ERROR]   2. Is a model installed? -> 'ollama list'")
            self.log("[Agent ERROR]   3. Pull a model: 'ollama pull gemma2:9b'")
            return pd.DataFrame()

        self.log(f"[Agent] OK Extraction complete: {len(results)} samples")
        ai_df = pd.DataFrame(results)
        # Normalize GSM column name
        rename_map = {'gsm': 'GSM'}
        ai_df.rename(columns=rename_map, inplace=True)

        # ═══════════════════════════════════════════════════════════
        # PER-GSE CONTEXT-AWARE LABEL COLLAPSING (Phase 1.5)
        #
        # STRICT rules — NEVER change the meaning of a label:
        #   1. Exact match after case/space/hyphen normalization → merge
        #   2. Abbreviation: if short label (≤5 chars) matches INITIALS
        #      of longer label in same GSE → merge to longer form
        #      (e.g., AD → Alzheimer Disease, but HSV ≠ HIV)
        #   3. Numbers must match exactly (Mut12 ≠ Mut10)
        #   4. NO substring matching, NO fuzzy/overlap matching
        #   5. NO synonym dictionaries
        # ═══════════════════════════════════════════════════════════
        _SKIP_NORMALIZE = {'Age', 'Treatment_Time', 'age', 'treatment_time'}
        if 'series_id' in ai_df.columns:
            label_cols = [c for c in ai_df.columns
                          if c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform')
                          and c not in _SKIP_NORMALIZE
                          and ai_df[c].dtype == 'object']

            if label_cols:
                from collections import Counter
                n_normalized = 0
                gse_groups = ai_df.groupby('series_id')

                def _get_initials(text):
                    """Get initials from a multi-word label.
                    'Alzheimer Disease' → 'AD', 'Acute Myeloid Leukemia' → 'AML'
                    """
                    words = re.split(r'[\s\-_/]+', text.strip())
                    # Filter out very short words like 'of', 'the', 'and'
                    meaningful = [w for w in words if len(w) > 1 or w[0].isupper()]
                    return ''.join(w[0].upper() for w in meaningful if w)

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

                        # Canonical = most common form
                        canonical = counter.most_common(1)[0][0]
                        canonical_norm = canonical.lower().replace(' ', '').replace('-', '').replace('_', '')

                        merge_map = {}
                        for val, cnt in counter.items():
                            if val == canonical:
                                continue
                            val_norm = val.lower().replace(' ', '').replace('-', '').replace('_', '')

                            # GUARD: different numbers = different entities, ALWAYS
                            val_nums = re.findall(r'\d+', val)
                            can_nums = re.findall(r'\d+', canonical)
                            if val_nums != can_nums:
                                continue

                            # Rule 1: Exact match after case/space/hyphen normalization
                            if val_norm == canonical_norm:
                                merge_map[val] = canonical
                                continue

                            # Rule 2: Abbreviation check (strict initials only)
                            # Short label (≤5 chars, all uppercase) might be initials
                            # of the longer label in the same GSE
                            shorter, longer = (val, canonical) if len(val) <= len(canonical) else (canonical, val)
                            if (len(shorter) <= 5 and shorter.replace('-','').replace(' ','').isupper()
                                    and len(longer) > len(shorter) * 2):
                                initials = _get_initials(longer)
                                short_clean = shorter.upper().replace('-','').replace(' ','').replace('(','').replace(')','')
                                if short_clean == initials:
                                    # Confirmed abbreviation → merge to longer form
                                    target = longer if len(canonical) >= len(val) else canonical
                                    merge_map[val] = target
                                    self.log(f"[Phase1.5] {gse_id}/{col}: "
                                             f"'{val}' → '{target}' (abbreviation: {short_clean}={initials})")

                            # NO OTHER MATCHING — no substring, no fuzzy, no overlap

                        if merge_map:
                            idx = group.index
                            for old_val, new_val in merge_map.items():
                                mask = ai_df.loc[idx, col] == old_val
                                if mask.any():
                                    ai_df.loc[idx[mask], col] = new_val
                                    n_normalized += mask.sum()

                if n_normalized > 0:
                    self.log(f"[Agent] Phase 1.5: collapsed {n_normalized} labels "
                             f"(exact+abbreviation only, no fuzzy matching)")

        return ai_df

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
                 on_finish, gui_ref):
        super().__init__(daemon=True)
        self.gz_path = gz_path
        self.plat_filter = plat_filter
        self.search_tokens_raw = search_tokens
        self.log_func = log_func
        self.on_finish_cb = on_finish
        self.gui_ref = gui_ref
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
        try:
            self._log("PROGRESS: 0")
            self._log("[Step 1] Loading GEOmetadb into thread-local memory...")

            if not os.path.exists(self.gz_path):
                self._log(f"[Step 1] ERROR: GEOmetadb not found at {self.gz_path}")
                self.final_df = pd.DataFrame()
                return

            # ── Open our own thread-local copy (resource-aware) ──
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
                tmp_path = tmp.name
                with gzip.open(self.gz_path, "rb") as gzf:
                    shutil.copyfileobj(gzf, tmp)

            # Decide: load into RAM or use disk-based access
            try:
                free_gb = psutil.virtual_memory().available / (1024 ** 3)
                file_mb = os.path.getsize(tmp_path) / (1024 * 1024)
                need_gb = (file_mb / 1024) * 2 + 2
                use_memory = (free_gb > need_gb) and (free_gb > 6)
            except Exception:
                use_memory = False

            if use_memory:
                self._log("[Step 1] Loading GEOmetadb into RAM...")
                disk_conn = sqlite3.connect(tmp_path)
                disk_conn.text_factory = lambda b: b.decode('utf-8', 'replace')
                mem_conn = sqlite3.connect(":memory:")
                mem_conn.text_factory = disk_conn.text_factory
                disk_conn.backup(mem_conn)
                disk_conn.close()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                db_conn = mem_conn
            else:
                self._log("[Step 1] Opening GEOmetadb from disk (low-RAM mode)...")
                db_conn = sqlite3.connect(tmp_path, timeout=60)
                db_conn.text_factory = lambda b: b.decode('utf-8', 'replace')
                db_conn.execute("PRAGMA journal_mode=WAL")
                db_conn.execute("PRAGMA cache_size=-65536")
                db_conn.execute("PRAGMA mmap_size=268435456")
                db_conn.execute("PRAGMA temp_store=MEMORY")
                # Create indexes for faster searches
                for idx, tbl, col in [
                    ("idx_gsm_gsm", "gsm", "gsm"),
                    ("idx_gsm_series", "gsm", "series_id"),
                    ("idx_gse_gse", "gse", "gse"),
                ]:
                    try:
                        db_conn.execute(
                            f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}({col})")
                    except Exception:
                        pass
                mem_conn = None  # track for cleanup

            self._log("PROGRESS: 10")
            self._do_search(db_conn)

        except Exception as e:
            import traceback
            self._log(f"[Step 1] THREAD EXCEPTION: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            self.final_df = pd.DataFrame()
        finally:
            # Close whichever connection was used
            for c in (mem_conn, db_conn if 'db_conn' in dir() else None):
                if c is not None:
                    try:
                        c.close()
                    except Exception:
                        pass
            if self.gui_ref:
                self.gui_ref.after(0, self.on_finish_cb)

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
        self._log(f"[Step 1] OK Final: {len(all_samples):,} samples, "
                  f"{n_gse} experiments, {n_gpl} platform(s), "
                  f"{n_matched:,} samples with direct keyword match")
        self._log("PROGRESS: 70")

        self.final_df = all_samples


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

        self.title("Step 1 — Review Experiments")
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

        # ── PanedWindow (left = tree, right = detail) ──
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # LEFT: Treeview
        left = ttk.Frame(pw)
        pw.add(left, weight=2)

        cols = ("gse", "samples", "platforms", "matched_kw")
        self.tree = ttk.Treeview(left, columns=cols, show='tree headings',
                                  selectmode='browse', height=25)
        self.tree.heading('#0', text='✓')
        self.tree.column('#0', width=35, stretch=False)
        self.tree.heading('gse', text='GSE ID')
        self.tree.column('gse', width=100)
        self.tree.heading('samples', text='Samples')
        self.tree.column('samples', width=70, anchor='e')
        self.tree.heading('platforms', text='Platform(s)')
        self.tree.column('platforms', width=90)
        self.tree.heading('matched_kw', text='Matched Keywords')
        self.tree.column('matched_kw', width=200)

        sb = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.config(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.tree.bind('<Button-1>', self._on_tree_click)

        # Style tags
        self.tree.tag_configure('checked', foreground='#1B5E20')
        self.tree.tag_configure('unchecked', foreground='#999999')

        # RIGHT: Detail text
        right = ttk.Frame(pw)
        pw.add(right, weight=3)

        self.detail_text = tk.Text(right, wrap=tk.WORD, font=('Consolas', 9),
                                    state='disabled', padx=8, pady=8)
        dsb = ttk.Scrollbar(right, command=self.detail_text.yview)
        self.detail_text.config(yscrollcommand=dsb.set)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tag for keyword highlighting
        self.detail_text.tag_configure('keyword', foreground='red', font=('Consolas', 9, 'bold'))
        self.detail_text.tag_configure('header', font=('Consolas', 10, 'bold'), foreground='#1565C0')
        self.detail_text.tag_configure('sample_header', font=('Consolas', 9, 'bold'),
                                        foreground='#6A1B9A')

        # ── Bottom buttons ──
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(btn_frame, text="✓ Select All", command=self._select_all,
                  bg='#4CAF50', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="✗ Deselect All", command=self._deselect_all,
                  bg='#F44336', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=10).pack(side=tk.LEFT, padx=4)

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._count_label = ttk.Label(btn_frame, text="", font=('Segoe UI', 10))
        self._count_label.pack(side=tk.LEFT, padx=8)

        tk.Button(btn_frame, text="💾 Save Selected for Step 2", command=self._save_and_close,
                  bg='#1976D2', fg='white', font=('Segoe UI', 10, 'bold'),
                  padx=15, pady=4).pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_frame, text="📊 Analyze on Platform →", command=self._analyze_on_platform,
                  bg='#6A1B9A', fg='white', font=('Segoe UI', 10, 'bold'),
                  padx=15, pady=4).pack(side=tk.RIGHT, padx=4)

    # ── Populate ───────────────────────────────────────────────────
    def _populate(self):
        df = self.results_df
        if 'series_id' not in df.columns:
            return

        for gse_id in sorted(df['series_id'].unique()):
            sub = df[df['series_id'] == gse_id]
            n = len(sub)
            plats = ', '.join(sorted(sub['gpl'].dropna().unique())) if 'gpl' in sub.columns else '?'
            kws = self.gse_keywords.get(gse_id, [])
            kw_str = ', '.join(kws[:5])
            if len(kws) > 5:
                kw_str += f' (+{len(kws)-5})'

            self._checks[gse_id] = True  # all checked initially
            iid = self.tree.insert('', tk.END, iid=gse_id,
                                    text='[✓]',
                                    values=(gse_id, f"{n:,}", plats, kw_str),
                                    tags=('checked',))

        self._update_count()

    # ── Tree interactions ──────────────────────────────────────────
    def _on_tree_click(self, event):
        """Toggle checkbox on click in the #0 (checkbox) column."""
        region = self.tree.identify_region(event.x, event.y)
        if region == 'tree':  # clicked on the #0 column (tree text area)
            iid = self.tree.identify_row(event.y)
            if iid and iid in self._checks:
                self._checks[iid] = not self._checks[iid]
                if self._checks[iid]:
                    self.tree.item(iid, text='[✓]', tags=('checked',))
                else:
                    self.tree.item(iid, text='[ ]', tags=('unchecked',))
                self._update_count()

    def _on_tree_select(self, event):
        """Show detail for selected experiment."""
        sel = self.tree.selection()
        if not sel:
            return
        gse_id = sel[0]
        self._show_detail(gse_id)

    def _show_detail(self, gse_id):
        """Render GSE + sample details with red keyword highlighting."""
        self.detail_text.config(state='normal')
        self.detail_text.delete('1.0', tk.END)

        # ── GSE-level description ──
        self.detail_text.insert(tk.END, f"═══ {gse_id} ═══\n", 'header')

        desc = self.gse_descriptions.get(gse_id, "No description available")
        self._insert_highlighted(desc + "\n\n")

        # ── Sample summary ──
        df = self.results_df
        sub = df[df['series_id'] == gse_id] if 'series_id' in df.columns else pd.DataFrame()
        if sub.empty:
            self.detail_text.config(state='disabled')
            return

        n_total = len(sub)
        n_matched = sub['Token_Match'].sum() if 'Token_Match' in sub.columns else 0

        self.detail_text.insert(tk.END,
            f"Samples: {n_total:,} total, {n_matched:,} with direct keyword match\n",
            'sample_header')

        if 'gpl' in sub.columns:
            plat_counts = sub['gpl'].value_counts()
            for plat, cnt in plat_counts.items():
                self.detail_text.insert(tk.END, f"  Platform {plat}: {cnt:,} samples\n")

        # ── Show sample-level details for MATCHED samples (up to 30) ──
        if 'Token_Match' in sub.columns:
            matched_samples = sub[sub['Token_Match'] == 1]
        else:
            matched_samples = sub.head(10)

        if not matched_samples.empty:
            self.detail_text.insert(tk.END,
                f"\n─── Matched Samples (showing {min(30, len(matched_samples))} "
                f"of {len(matched_samples):,}) ───\n", 'sample_header')

            gsm_col = 'GSM' if 'GSM' in matched_samples.columns else 'gsm'
            for _, row in matched_samples.head(30).iterrows():
                gsm_id = row.get(gsm_col, '?')
                self.detail_text.insert(tk.END, f"\n  {gsm_id}:\n", 'sample_header')

                # Use pre-built GSM description if available
                gsm_desc = self.gsm_descriptions.get(gsm_id, None)
                if gsm_desc is None:
                    gsm_desc = self.gsm_descriptions.get(str(gsm_id).upper(), None)
                if gsm_desc:
                    for line in gsm_desc.split('\n'):
                        self.detail_text.insert(tk.END, f"    ")
                        self._insert_highlighted(line + "\n")
                else:
                    # Fallback: show available columns
                    for c in ['title', 'source_name_ch1', 'characteristics_ch1']:
                        val = row.get(c, None)
                        if pd.notna(val) and str(val).strip():
                            self.detail_text.insert(tk.END, f"    {c}: ")
                            self._insert_highlighted(str(val).strip() + "\n")

        # Show a few NON-matched samples too for context
        if 'Token_Match' in sub.columns:
            non_matched = sub[sub['Token_Match'] == 0]
            if not non_matched.empty:
                self.detail_text.insert(tk.END,
                    f"\n─── Other Samples (showing {min(5, len(non_matched))} "
                    f"of {len(non_matched):,}) ───\n", 'sample_header')
                for _, row in non_matched.head(5).iterrows():
                    gsm_id = row.get('GSM', row.get('gsm', '?'))
                    title = row.get('title', 'N/A')
                    src = row.get('source_name_ch1', '')
                    line = f"  {gsm_id}: {title}"
                    if src:
                        line += f" | {src}"
                    self.detail_text.insert(tk.END, line + "\n")

        self.detail_text.config(state='disabled')

    def _insert_highlighted(self, text):
        """Insert text with search tokens highlighted in red.
        Matching is case-insensitive and symbol-tolerant."""
        if not self.search_tokens:
            self.detail_text.insert(tk.END, text)
            return

        text_lower = text.lower()
        # Also build a cleaned version for fuzzy matching
        # But we need to map cleaned positions back to original positions
        # Simpler approach: for each token, try matching in both raw and cleaned text

        highlights = []  # [(start, end), ...]
        for tok in self.search_tokens:
            # Direct substring match (handles "alzheimer" in "Alzheimer's Disease")
            start = 0
            while True:
                idx = text_lower.find(tok, start)
                if idx == -1:
                    break
                highlights.append((idx, idx + len(tok)))
                start = idx + 1

            # Fuzzy match: strip symbols from text and find token, then map back
            # This handles "alzheimers" matching "Alzheimer's"
            cleaned_tok = re.sub(r"[^a-z0-9\s]", "", tok)
            if cleaned_tok and cleaned_tok != tok:
                # Build position map: cleaned_pos -> original_pos
                orig_positions = []  # for each char in cleaned text, its position in original
                for i, ch in enumerate(text_lower):
                    if re.match(r"[a-z0-9\s]", ch):
                        orig_positions.append(i)
                cleaned_text = re.sub(r"[^a-z0-9\s]", "", text_lower)

                start = 0
                while True:
                    idx = cleaned_text.find(cleaned_tok, start)
                    if idx == -1:
                        break
                    if idx < len(orig_positions) and idx + len(cleaned_tok) - 1 < len(orig_positions):
                        orig_start = orig_positions[idx]
                        orig_end = orig_positions[idx + len(cleaned_tok) - 1] + 1
                        highlights.append((orig_start, orig_end))
                    start = idx + 1

        if not highlights:
            self.detail_text.insert(tk.END, text)
            return

        # Sort and merge overlapping ranges
        highlights.sort()
        merged = [highlights[0]]
        for s, e in highlights[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Insert text with tags
        pos = 0
        for s, e in merged:
            if pos < s:
                self.detail_text.insert(tk.END, text[pos:s])
            self.detail_text.insert(tk.END, text[s:e], 'keyword')
            pos = e
        if pos < len(text):
            self.detail_text.insert(tk.END, text[pos:])

    # ── Button actions ─────────────────────────────────────────────
    def _select_all(self):
        for iid in self._checks:
            self._checks[iid] = True
            self.tree.item(iid, text='[✓]', tags=('checked',))
        self._update_count()

    def _deselect_all(self):
        for iid in self._checks:
            self._checks[iid] = False
            self.tree.item(iid, text='[ ]', tags=('unchecked',))
        self._update_count()

    def _update_count(self):
        checked = [gid for gid, v in self._checks.items() if v]
        n_sel = len(checked)
        n_tot = len(self._checks)
        n_samples = 0
        if 'series_id' in self.results_df.columns:
            n_samples = len(self.results_df[
                self.results_df['series_id'].isin(checked)])
        self._count_label.config(
            text=f"{n_sel}/{n_tot} experiments selected  •  {n_samples:,} samples")

    def _save_and_close(self):
        selected = [gid for gid, v in self._checks.items() if v]
        if not selected:
            messagebox.showwarning("No Selection",
                                   "Please select at least one experiment.",
                                   parent=self)
            return

        self.app.gse_to_keep_for_step2 = selected

        # Filter results to selected GSEs only
        if 'series_id' in self.results_df.columns:
            kept = self.results_df[self.results_df['series_id'].isin(selected)]
        else:
            kept = self.results_df
        total_samples = len(kept)

        self.app.enqueue_log(
            f"[Step 1.5] OK Saved {len(selected)} experiment(s) "
            f"({total_samples:,} samples) for Step 2")

        # Also populate the listbox in Step 1.5
        self.app.gse_listbox.delete(0, tk.END)
        for gse_id in sorted(selected):
            count = len(kept[kept['series_id'] == gse_id]) if 'series_id' in kept.columns else 0
            desc = self.app.step1_gse_descriptions.get(gse_id, "")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            kws = self.app.step1_gse_keywords.get(gse_id, [])
            kw_str = ', '.join(kws[:3]) or 'N/A'
            self.app.gse_listbox.insert(
                tk.END, f"{gse_id} ({count:,} samples) - {desc} | Keywords: {kw_str}")

        self.app.gse_listbox.select_set(0, tk.END)
        self.app.gse_frame.pack(fill=tk.X, padx=5, pady=5,
                                 after=self.app.step1_frame)

        self.app.step2_status_label.config(
            text=f"OK Ready: {len(selected)} experiment(s) ({total_samples:,} samples)",
            foreground="green")

        messagebox.showinfo(
            "Saved",
            f"Saved {len(selected)} experiment(s) with {total_samples:,} samples.\n\n"
            f"You can now proceed to Step 2.",
            parent=self)

        self.destroy()

    def _analyze_on_platform(self):
        """Check selected experiments' platforms, guide user to load & analyze."""
        selected = [gid for gid, v in self._checks.items() if v]
        if not selected:
            messagebox.showwarning("No Selection",
                                   "Please select at least one experiment first.",
                                   parent=self)
            return

        # Find which platforms the selected experiments use
        df = self.results_df
        if 'series_id' not in df.columns or 'gpl' not in df.columns:
            messagebox.showerror("Data Error",
                                 "Cannot determine platforms from the results.",
                                 parent=self)
            return

        sub = df[df['series_id'].isin(selected)]
        needed_gpls = set(sub['gpl'].dropna().astype(str).str.strip().str.upper().unique())
        if not needed_gpls:
            messagebox.showwarning("No Platforms",
                                   "Could not determine platforms for selected experiments.",
                                   parent=self)
            return

        # Check which are loaded in the app
        missing = set()
        for gpl in needed_gpls:
            found = any(gpl.upper() in lk.upper() for lk in self.app.gpl_datasets.keys())
            if not found:
                missing.add(gpl)

        if missing:
            missing_str = ', '.join(sorted(missing))
            loaded_str = ', '.join(sorted(self.app.gpl_datasets.keys())) or '(none)'
            n_samples_missing = len(sub[sub['gpl'].astype(str).str.upper().isin(
                {m.upper() for m in missing})])

            messagebox.showwarning(
                "Platform(s) Not Loaded",
                f"The selected experiments require platform(s) not yet loaded:\n\n"
                f"  NEEDED:  {missing_str}\n"
                f"  LOADED:  {loaded_str}\n\n"
                f"  {n_samples_missing:,} samples on unloaded platform(s).\n\n"
                f"Please load the required platform(s) first:\n"
                f"  1. Use the platform buttons in the main window\n"
                f"     or 'Download from GEO' to fetch the data\n"
                f"  2. Then come back here and click this button again\n\n"
                f"Platform files contain the gene expression values\n"
                f"needed for distribution analysis.",
                parent=self
            )
            return

        # All platforms loaded — save selection and notify
        self._save_and_close()

        self.app.enqueue_log(
            f"[Step 1] All platforms ready ({', '.join(sorted(needed_gpls))}). "
            f"Use Gene Explorer (Ctrl+G).")

        messagebox.showinfo(
            "Ready to Analyze",
            f"All required platforms are loaded!\n\n"
            f"  Platforms: {', '.join(sorted(needed_gpls))}\n"
            f"  Experiments: {len(selected)}\n"
            f"  Samples: {len(sub):,}\n\n"
            f"Next steps:\n"
            f"  • Labels auto-integrate with expression data\n"
            f"  • Gene Distribution Explorer (Ctrl+G)\n"
            f"  • Compare Regions after selecting ranges",
            parent=self.app
        )

class LabelingThread(threading.Thread):
    """Thread that runs LLM extraction for a batch of samples."""
    def __init__(self, input_dataframe, ai_agent, gui_log_func, on_finish,
                 fields=None, custom_fields=None, on_progress=None, gui_ref=None):
        super().__init__(daemon=True)
        self.input_df = input_dataframe
        self.agent = ai_agent
        self.log = gui_log_func or print
        self.on_finish = on_finish
        self.on_progress = on_progress
        self.fields = fields  # list of field names to extract
        self.custom_fields = custom_fields  # list of {'name': ..., 'prompt': ...}
        self.gui_ref = gui_ref  # tkinter widget for after() scheduling
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
        """
        Classify a gene expression distribution into:
          Bimodal, Multimodal, Normal, Lognormal, Gamma, Cauchy, Uniform.

        Algorithm:
          1. Clean data (remove NaN, Inf)
          2. KDE + find_peaks for modality detection (bimodal/multimodal)
          3. MLE fitting via scipy log-likelihood: Normal, Lognormal,
             Gamma, Cauchy, Uniform — best fit wins
        """
        from scipy.stats import (gaussian_kde, norm, lognorm, gamma as gamma_dist,
                                  cauchy, uniform as uniform_dist)
        from scipy.signal import find_peaks

        # 1. Clean
        vals = np.asarray(expr, dtype=np.float64)
        vals = vals[np.isfinite(vals)]

        if vals.size < 20:
            return "Not Enough Data"
        if np.std(vals) < 1e-6:
            return "Effectively Constant"

        # 2. Modality check via KDE + find_peaks
        try:
            kde = gaussian_kde(vals)
            grid = np.linspace(vals.min(), vals.max(), min(300, vals.size))
            pdf = kde(grid)
            peaks, props = find_peaks(pdf, prominence=0.15 * pdf.max())
            if len(peaks) >= 3:
                return "Multimodal"
            if len(peaks) >= 2:
                p1, p2 = peaks[0], peaks[1]
                valley = pdf[p1:p2+1].min()
                peak_min = min(pdf[p1], pdf[p2])
                # Valley must drop to at least 75% of the shorter peak
                if valley < 0.75 * peak_min:
                    return "Bimodal"
        except Exception:
            pass

        # 3. MLE distribution fitting via scipy log-likelihood
        scores = {}

        # Normal
        try:
            mu, sigma = norm.fit(vals)
            scores["Normal"] = np.sum(norm.logpdf(vals, mu, sigma))
        except Exception:
            pass

        # Lognormal (only for positive data)
        if vals.min() > 0:
            try:
                shape, loc, scale = lognorm.fit(vals, floc=0)
                scores["Lognormal"] = np.sum(lognorm.logpdf(vals, shape, loc, scale))
            except Exception:
                pass

        # Gamma (only for positive data)
        if vals.min() > 0:
            try:
                a, loc, scale = gamma_dist.fit(vals, floc=0)
                scores["Gamma"] = np.sum(gamma_dist.logpdf(vals, a, loc, scale))
            except Exception:
                pass

        # Cauchy (heavy-tailed)
        try:
            loc_c, scale_c = cauchy.fit(vals)
            scores["Cauchy"] = np.sum(cauchy.logpdf(vals, loc_c, scale_c))
        except Exception:
            pass

        # Uniform
        try:
            loc_u, scale_u = uniform_dist.fit(vals)
            scores["Uniform"] = np.sum(uniform_dist.logpdf(vals, loc_u, scale_u))
        except Exception:
            pass

        # Pick best fit
        valid = {k: v for k, v in scores.items() if np.isfinite(v)}
        if valid:
            return max(valid, key=valid.get)

        return "Normal"

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
        dlg.title(title)
        dlg.transient(parent)
        dlg.grab_set()

        ttk.Label(dlg, text="Build a compound query — only samples matching ALL criteria will be selected",
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
                    val_combo['values'] = vals[:200]
                    if vals:
                        val_var.set(vals[0])

            col_combo.bind('<<ComboboxSelected>>', _on_col_change)
            _on_col_change()

            # Remove button
            def _remove():
                query_rows.remove((col_var, val_var, row_frame))
                row_frame.destroy()
                _update_preview()

            tk.Button(row_frame, text="✕", command=_remove, fg="red",
                      font=('Segoe UI', 8, 'bold'), padx=4, relief=tk.FLAT,
                      cursor="hand2").pack(side=tk.LEFT, padx=5)

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
        tk.Button(btn_row, text="+ Add Criterion", command=lambda: [_add_row(), _bind_updates()],
                  bg="#43A047", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.LEFT)

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

        tk.Button(btn_frame, text="  Apply Query  ", command=_ok,
                  bg="#1565C0", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                  font=('Segoe UI', 10), padx=15, pady=6).pack(side=tk.RIGHT, padx=5)

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


class BioStats:
    pass

class Plotter:
    @staticmethod
    def get_optimal_bins(data, method='auto'):
        """Freedman-Diaconis rule with higher floor for dense data."""
        import numpy as np
        arr = np.asarray(data)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n < 10:
            return max(10, n)
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        if iqr > 0 and n > 0:
            bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
            data_range = arr.max() - arr.min()
            if data_range > 0 and bin_width > 0:
                fd_bins = int(np.ceil(data_range / bin_width))
            else:
                fd_bins = 50
        else:
            fd_bins = 50
        # Clamp between 80 and 250 for high-resolution histograms
        return max(80, min(fd_bins, 250))
    @staticmethod
    def get_distinct_colors(n):
        import matplotlib.cm as cm
        if n <= 20:
            return [cm.tab20(i / 20) for i in range(n)]
        return [cm.gist_ncar(i / max(1, n - 1)) for i in range(n)]

def get_comprehensive_gsm_text(gsm_row):
    """Build a comprehensive text description of a GSM sample for LLM extraction.
    CRITICAL: characteristics_ch1 from GEOmetadb is tab/semicolon-separated.
    Each key-value pair must be on its own line for the LLM to parse correctly.
    """
    parts = []
    row_dict = dict(gsm_row) if hasattr(gsm_row, 'items') else {}
    lower_map = {k.lower(): k for k in row_dict.keys()}

    def _get(field):
        val = gsm_row.get(field, None)
        if val is None:
            actual_key = lower_map.get(field.lower())
            if actual_key:
                val = gsm_row.get(actual_key, None)
        if val and str(val).strip() and str(val).lower() not in ('nan', 'none', ''):
            return str(val).strip()
        return None

    # title — sample title
    v = _get('title')
    if v:
        parts.append(f"title: {v}")

    # source_name_ch1 — often contains tissue/cell line info
    v = _get('source_name_ch1')
    if v:
        parts.append(f"source_name: {v}")

    # characteristics_ch1 — CRITICAL: split tab/semicolon-separated pairs into lines
    v = _get('characteristics_ch1')
    if v:
        # GEOmetadb stores as: "tissue: cerebellum\tgender: female\tage (y): 25"
        # or sometimes semicolon-separated: "tissue: cerebellum; gender: female"
        # Split on tabs, semicolons, and literal \t
        import re as _re_chars
        items = _re_chars.split(r'[\t;]+|\\t', v)
        for item in items:
            item = item.strip()
            if item and len(item) > 1:
                parts.append(f"  {item}")

    # description
    v = _get('description')
    if v:
        parts.append(f"description: {v}")

    # treatment_protocol_ch1
    v = _get('treatment_protocol_ch1')
    if v:
        parts.append(f"treatment_protocol: {v}")

    # growth_protocol_ch1
    v = _get('growth_protocol_ch1')
    if v:
        parts.append(f"growth_protocol: {v}")

    # organism_ch1
    v = _get('organism_ch1')
    if v:
        parts.append(f"organism: {v}")

    # series_id (GSE experiment) + experiment context from cache
    sid = gsm_row.get('series_id', gsm_row.get('gse', gsm_row.get('Series_id', None)))
    if sid and str(sid).strip() and str(sid).lower() not in ('nan', 'none'):
        sid_str = str(sid).strip()
        parts.append(f"experiment: {sid_str}")
        
        # Add GSE title/summary — CRITICAL for decoding coded values
        # e.g. "smoking: 0" only makes sense if the study title says
        # "Gene expression in smokers vs non-smokers"
        gse_ctx = _GSE_CONTEXT.get(sid_str, {})
        gse_title = gse_ctx.get('title', '')
        gse_summary = gse_ctx.get('summary', '')
        if gse_title:
            parts.append(f"study_title: {gse_title}")
        if gse_summary:
            parts.append(f"study_summary: {gse_summary}")

    # Fallback: try ALL string columns if nothing found
    if not parts:
        excluded = {'gsm', 'GSM', 'gpl', 'status', 'submission_date',
                     'last_update_date', 'data_row_count', 'channel_count',
                     'supplementary_file', 'contact', 'Token_Match',
                     'Matched_Tokens', '_platform'}
        for k, v in row_dict.items():
            if k in excluded:
                continue
            if v and str(v).strip() and str(v).lower() not in ('nan', 'none', ''):
                val_str = str(v).strip()
                if len(val_str) > 3 and not val_str.replace('.', '').isdigit():
                    parts.append(f"{k}: {val_str}")

    return "\n".join(parts) if parts else "No metadata available"


def _detect_ollama_model():
    """Auto-detect best available Ollama model via ollama library."""
    preferred = [CONFIG['ai']['model'], 'gemma2:9b', 'gemma2', 'llama3:8b',
                 'llama3', 'llama2', 'mistral', 'phi3', 'qwen2']
    try:
        models_resp = ollama.list()
        available = [m.get('name', m.get('model', '')) for m in models_resp.get('models', [])]
        if not available:
            print("[LLM] Ollama running but no models installed")
            return None
        # strip :latest suffix for matching
        avail_clean = {m.split(':')[0].lower(): m for m in available}
        avail_clean.update({m.lower(): m for m in available})
        for p in preferred:
            if p.lower() in avail_clean:
                found = avail_clean[p.lower()]
                print(f"[LLM] Auto-detected model: {found}")
                return found
        # return first available
        print(f"[LLM] Using first available model: {available[0]}")
        return available[0]
    except Exception as e:
        err = str(e).lower()
        if 'connection' in err or 'refused' in err:
            print(f"[LLM] Cannot connect to Ollama — is it running? (ollama serve)")
        else:
            print(f"[LLM] Model detection failed: {e}")
        return None

_OLLAMA_MODEL = None  # cached (collapse model: gemma2:9b)
_OLLAMA_EXTRACTION_MODEL = None  # cached (fast extraction model: gemma2:2b)
_OLLAMA_URL = "http://localhost:11434"


def _detect_extraction_model():
    """Detect fast extraction model (gemma2:2b preferred, falls back to main model)."""
    preferred_fast = ['gemma2:2b', 'gemma2:2b-q4_0', 'gemma3:1b']
    try:
        models_resp = ollama.list()
        available = [m.get('name', m.get('model', '')) for m in models_resp.get('models', [])]
        avail_clean = {m.split(':')[0].lower() + ':' + m.split(':')[1].lower()
                       if ':' in m else m.lower(): m for m in available}
        avail_clean.update({m.lower(): m for m in available})
        for p in preferred_fast:
            if p.lower() in avail_clean:
                found = avail_clean[p.lower()]
                print(f"[LLM] Fast extraction model: {found}")
                return found
        # No fast model found — fall back to main model
        print("[LLM] No fast extraction model (gemma2:2b) — using main model")
        return None
    except Exception:
        return None


# ── Extraction system prompt (cached by Ollama for KV reuse) ──
_EXTRACTION_SYSTEM_PROMPT = (
    "TASK: Read the metadata below and extract exactly what is written.\n"
    "Do NOT normalise, generalise, or map to any vocabulary — copy the specific term.\n"
    "FIELDS:\n"
    "  Tissue    : anatomical tissue, organ, cell type, or cell line as written\n"
    "  Condition : disease, phenotype, or health status as written\n"
    "  Treatment : drug or stimulus as written. None/vehicle = Untreated.\n"
    "RULES:\n"
    "  - Copy the most specific term present (e.g. Alveolar Macrophages not Lung)\n"
    "  - If a cell type is named, use the cell type (e.g. NK cells not PBMC)\n"
    "  - Unknown or absent field = Not Specified\n"
    "  - Title Case. Output JSON only.\n"
    'JSON SCHEMA: {"Tissue":"", "Condition":"", "Treatment":""}'
)

# ── Global Memory Agent (deterministic extraction) ──
_MEMORY_AGENT = None  # MemoryAgent instance
_GSE_CONTEXTS = {}    # {gse_id: GSEContext}

# ── Phase enable flags (user-configurable) ──
_ENABLE_PHASE15 = True   # Phase 1.5: collapse agent
_ENABLE_PHASE2 = True    # Phase 2: GSE context rescue

_NOT_SPECIFIED_VALUES = {
    'Not Specified', 'not specified', 'Not specified',
    'N/A', 'n/a', 'NA', 'na', 'nan', 'NaN', 'None', 'none',
    'Unknown', 'unknown', 'UNKNOWN',
    '', 'Parse Error', 'parse error',
}

# ── GSE Experiment Context Cache (for Phase 1) ─────────────────────────
# Pre-fetched from GEOmetadb before Phase 1 starts.
# Keyed by GSE ID → {title, summary}
# The LLM needs this to decode numeric/coded values like "smoking: 0", "disease: 1"
_GSE_CONTEXT = {}   # {gse_id: {"title": str, "summary": str}}

def prefetch_gse_context(conn, series_ids, log_fn=None):
    """Pre-fetch GSE titles and summaries from GEOmetadb for Phase 1.
    This is FAST because GEOmetadb is already in memory.
    The LLM needs experiment context to decode coded values.
    """
    global _GSE_CONTEXT
    _log = log_fn or print
    
    unique_gses = list(set(
        str(g).strip() for g in series_ids 
        if g and str(g).strip() and str(g).strip().lower() not in ('nan', 'none', '')
    ))
    
    # Filter out already cached
    new_gses = [g for g in unique_gses if g not in _GSE_CONTEXT]
    
    if not new_gses:
        _log(f"[Phase1] GSE context: {len(_GSE_CONTEXT)} experiments already cached")
        return
    
    _log(f"[Phase1] Pre-fetching {len(new_gses)} GSE titles from GEOmetadb...")
    
    try:
        # Batch query in chunks of 500
        for i in range(0, len(new_gses), 500):
            chunk = new_gses[i:i+500]
            ph = ",".join("?" * len(chunk))
            df = pd.read_sql_query(
                f"SELECT gse, title, summary FROM gse WHERE gse IN ({ph})",
                conn, params=chunk)
            for _, row in df.iterrows():
                gse_id = str(row.get('gse', '')).strip()
                if gse_id:
                    title = str(row.get('title', '')).strip()
                    summary = str(row.get('summary', '')).strip()
                    # Keep summary short for Phase 1 (save context for metadata)
                    if len(summary) > 300:
                        summary = summary[:300]
                    _GSE_CONTEXT[gse_id] = {"title": title, "summary": summary}
        
        _log(f"[Phase1] GSE context ready: {len(_GSE_CONTEXT)} experiments cached")
    except Exception as e:
        _log(f"[Phase1] GSE prefetch warning: {e} — Phase 1 will work without GSE context")

# ── Thread-local session (for non-LLM HTTP calls only) ──
import threading as _thr_http
_tls = _thr_http.local()

class OllamaServerError(Exception):
    """Raised when Ollama is persistently unavailable."""
    pass

def _get_session():
    """Thread-local requests.Session for non-LLM HTTP calls."""
    if not hasattr(_tls, "s") or _tls.s is None:
        _tls.s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        _tls.s.mount("http://", a)
    return _tls.s

def _ollama_post(prompt, model=None, num_predict=None, num_ctx=None,
                 retries=3, timeout=120, system_prompt=None):
    """Call Ollama using the ollama Python library with KV cache optimisation.

    Performance optimisations vs naive approach:
        1. keep_alive=-1  : keeps model in VRAM permanently (no reload penalty)
        2. system_prompt   : sent as system role — Ollama caches its KV state
                             across calls, so only the user message is recomputed
                             (~40% latency reduction per call)
        3. num_predict cap : prevents runaway generation

    The ollama library handles connection pooling, queuing, and GPU scheduling.
    """
    global _OLLAMA_MODEL
    mdl = model or _OLLAMA_MODEL
    if not mdl:
        mdl = _detect_ollama_model()
        if mdl:
            _OLLAMA_MODEL = mdl
        else:
            raise OllamaServerError("No Ollama model detected. Run: ollama pull gemma2:9b")

    options = {'temperature': 0.0}
    if num_predict is not None:
        options['num_predict'] = num_predict
    if num_ctx is not None:
        options['num_ctx'] = num_ctx

    # Build messages with system/user split for KV cache reuse
    if system_prompt:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = [{'role': 'user', 'content': prompt}]

    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=mdl,
                messages=messages,
                options=options,
                keep_alive=-1,
            )
            # Handle both object (new ollama lib) and dict (old lib)
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = (response.message.content or '').strip()
            elif isinstance(response, dict):
                content = response.get('message', {}).get('content', '').strip()
            else:
                content = ''
            if not content and attempt < retries:
                time.sleep(2)
                continue
            return content
        except Exception as e:
            err = str(e).lower()
            if 'connection' in err or 'refused' in err or '503' in err:
                if attempt == retries:
                    raise OllamaServerError(
                        f"Cannot connect to Ollama after {retries} attempts: {e}\n"
                        f"Start Ollama: ollama serve")
                wait = min(3 * attempt, 15)
                print(f"[LLM] Ollama error (attempt {attempt}/{retries}): {e} — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[LLM ERROR] {type(e).__name__}: {e}")
                if attempt == retries:
                    return ""
                time.sleep(2 * attempt)
    return ""

def compute_ollama_parallel(model=None, base_url=None, reserve_gb=3.0):
    """Compute optimal number of parallel Ollama inference slots based on VRAM.
    Ollama shares model weights across slots — marginal cost per slot is only KV cache.
    """
    KV_CACHE_PER_SLOT = 1.0  # ~1GB per slot at num_ctx=1024 for 9B model
    url = base_url or _OLLAMA_URL
    mdl = model or _OLLAMA_MODEL or 'gemma2:9b'

    # Model size estimate
    param_match = re.search(r'(\d+)[bB]', mdl)
    model_gb = (int(param_match.group(1)) * 0.6) if param_match else 6.0

    try:
        # Strategy 1: query loaded model VRAM from Ollama
        try:
            r = requests.get(f"{url}/api/ps", timeout=3)
            if r.status_code == 200:
                models = r.json().get("models", [])
                if models:
                    size_vram = models[0].get("size_vram", 0)
                    gpus = detect_gpus()
                    if gpus and size_vram > 0:
                        total_vram = sum(g["vram_gb"] for g in gpus)
                        model_vram = size_vram / 1e9
                        free_for_kv = total_vram - model_vram - reserve_gb
                        return max(1, min(8, 1 + int(free_for_kv / KV_CACHE_PER_SLOT)))
        except Exception:
            pass

        # Strategy 2: nvidia-smi free VRAM
        gpus = detect_gpus()
        if gpus:
            free_vram = sum(g["free_vram_gb"] for g in gpus)
            if free_vram >= model_gb:
                remaining = free_vram - model_gb - reserve_gb
                return max(1, min(8, 1 + max(0, int(remaining / KV_CACHE_PER_SLOT))))
            return 1

        # Strategy 3: CPU / RAM fallback
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        return max(1, min(4, int((free_gb - reserve_gb - model_gb) / KV_CACHE_PER_SLOT) + 1))

    except Exception:
        return 1


def init_memory_agent(data_dir: str, log_fn=print) -> bool:
    """Initialize the global MemoryAgent (episodic memory + vocabulary).

    Loads cluster vocabulary from LLM_memory/ cluster .txt files if present,
    or from the existing SQLite DB. Without cluster vocabulary, every sample
    falls through to full LLM extraction (60-80% of samples can be resolved
    by O(1) cluster_map lookup when vocabulary is loaded).
    """
    global _MEMORY_AGENT
    if not _HAS_DETERMINISTIC:
        log_fn("[Memory] deterministic_extraction.py not available")
        return False

    db_path = os.path.join(data_dir, "biomedical_memory.db")
    _MEMORY_AGENT = MemoryAgent(db_path)

    # Check if cluster vocabulary is already in the DB
    stats = _MEMORY_AGENT.stats()
    has_clusters = sum(stats.get('clusters', {}).values()) > 0

    if not has_clusters:
        # Try to load cluster files from known locations
        cluster_dirs = [
            os.path.join(data_dir, "LLM_memory"),
            os.path.join(data_dir, "..", "memory", "clusters"),
            os.path.join(data_dir, "agent_memory"),
        ]
        for cdir in cluster_dirs:
            if os.path.isdir(cdir):
                log_fn(f"[Memory] Loading cluster vocabulary from {cdir}")
                try:
                    _MEMORY_AGENT.build_from_clusters(cdir, log_fn=log_fn)
                    stats = _MEMORY_AGENT.stats()
                    if sum(stats.get('clusters', {}).values()) > 0:
                        log_fn(f"[Memory] Cluster vocabulary loaded: {stats.get('clusters', {})}")
                        break
                except Exception as e:
                    log_fn(f"[Memory] Cluster load warning: {e}")
    else:
        log_fn(f"[Memory] Cluster vocabulary already in DB: {stats.get('clusters', {})}")

    # Load vector cache into RAM for fast semantic search
    try:
        _MEMORY_AGENT.load_cache_all(log_fn=log_fn)
    except Exception:
        pass

    log_fn(f"[Memory] Ready: episodic={stats.get('episodic', {})} "
           f"clusters={stats.get('clusters', {})}")
    return True


def init_gse_contexts(samples_df, gds_conn=None, log_fn=print):
    """Build GSEContext for each GSE in the dataset.
    Called before extraction to provide sibling label context.
    """
    global _GSE_CONTEXTS
    if not _HAS_DETERMINISTIC:
        return

    # Fetch GSE metadata from GEOmetadb
    gse_meta = {}
    if gds_conn and 'series_id' in samples_df.columns:
        try:
            gse_ids = samples_df['series_id'].dropna().unique().tolist()
            gse_ids = [str(g).strip() for g in gse_ids
                       if g and str(g).strip().lower() not in ('nan', 'none', '')]
            if gse_ids:
                for i in range(0, len(gse_ids), 500):
                    chunk = gse_ids[i:i+500]
                    ph = ",".join("?" * len(chunk))
                    df = pd.read_sql_query(
                        f"SELECT gse, title, summary FROM gse WHERE gse IN ({ph})",
                        gds_conn, params=chunk)
                    for _, row in df.iterrows():
                        gse_meta[str(row['gse']).strip()] = {
                            'title': str(row.get('title', '')).strip(),
                            'summary': str(row.get('summary', '')).strip()[:300]
                        }
                log_fn(f"[GSE Context] Fetched metadata for {len(gse_meta)} experiments")
        except Exception as e:
            log_fn(f"[GSE Context] Warning: {e}")

    _GSE_CONTEXTS = build_gse_contexts(samples_df, gse_meta, _MEMORY_AGENT)
    log_fn(f"[GSE Context] Built {len(_GSE_CONTEXTS)} experiment contexts")


# ── Global GSEWorker cache (one worker per GSE, reused across samples) ──
_GSE_WORKERS = {}   # {gse_id: GSEWorker}

def classify_sample(gsm_row, fields=None, custom_fields=None):
    """
    Extract labels using the full agent architecture from geo_ns_repair_v2.py:
      1. Raw LLM extraction (Tissue + Condition via ollama.chat)
      2. Fast-path: cluster_map O(1) or GSE dominant 70%
      3. Agent collapse loop: LLM with 3 tools + memory pre-load
         - tool_gse_context  (sibling labels)
         - tool_llm_memory   (cluster search)
         - tool_episodic     (past resolutions)
      4. Cluster gate: validates output against LLM_memory
      5. Age/Treatment/Treatment_Time: deterministic regex (no LLM)
    """
    gsm_id = gsm_row.get('gsm', gsm_row.get('GSM', f"Unknown_{uuid.uuid4().hex[:6]}"))
    
    if fields is None:
        fields = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']

    # ── Full agent pipeline (GSEWorker) ──────────────────────────────
    if _HAS_DETERMINISTIC and _MEMORY_AGENT is not None:
        gse_id = str(gsm_row.get('series_id', gsm_row.get('gse', ''))).strip()
        
        # Get or create GSEWorker for this experiment
        if gse_id and gse_id.lower() not in ('nan', 'none', ''):
            if gse_id not in _GSE_WORKERS:
                ctx = _GSE_CONTEXTS.get(gse_id, GSEContext(gse_id))
                model = _OLLAMA_MODEL or CONFIG.get('ai', {}).get('model', 'gemma2:9b')
                platform = str(gsm_row.get('gpl', gsm_row.get('platform', ''))).strip()
                _GSE_WORKERS[gse_id] = GSEWorker(
                    gse_id, ctx, mem_agent=_MEMORY_AGENT,
                    model=model, platform=platform,
                    enable_phase15=_ENABLE_PHASE15,
                    enable_phase2=_ENABLE_PHASE2,
                    ollama_url=_pick_ollama_url() if _HAS_DETERMINISTIC else None)
            
            worker = _GSE_WORKERS[gse_id]
            raw = dict(gsm_row)  # repair_one expects a plain dict
            result = worker.repair_one(gsm_id, raw, ns_cols=fields)
            
            # Debug first 3 samples
            _count = getattr(classify_sample, '_count', 0)
            classify_sample._count = _count + 1
            if _count < 3:
                ns_count = sum(1 for f in fields if result.get(f, 'Not Specified') in
                               ('Not Specified', 'Not specified', '', 'nan', 'None'))
                print(f"[Agent #{_count+1}] {gsm_id}: {len(fields)-ns_count}/{len(fields)} resolved")
                for f in fields:
                    print(f"  {f}: {result.get(f, 'NS')}")
            
            return result
    
    # ── Fallback: pure LLM extraction (no MemoryAgent available) ─────
    return _classify_sample_llm(gsm_row, fields, custom_fields)


def _classify_sample_llm(gsm_row, fields=None, custom_fields=None):
    """Pure LLM extraction fallback — used when MemoryAgent not available.

    Performance optimisations:
        1. Uses gemma2:2b (fast model) for extraction — 4-5x faster than 9b
        2. System prompt sent as system role — Ollama caches KV (~40% speedup)
        3. keep_alive=-1 prevents model unloading between calls
        4. Compact user message (only metadata) — minimises per-call tokens
    """
    global _OLLAMA_MODEL, _OLLAMA_EXTRACTION_MODEL
    if _OLLAMA_MODEL is None:
        _OLLAMA_MODEL = _detect_ollama_model()
        if not _OLLAMA_MODEL:
            gsm_id = gsm_row.get('gsm', gsm_row.get('GSM', '?'))
            return {'gsm': gsm_id, **{f: 'Not Specified' for f in (fields or [])}}
    if _OLLAMA_EXTRACTION_MODEL is None:
        _OLLAMA_EXTRACTION_MODEL = _detect_extraction_model() or _OLLAMA_MODEL

    gsm_id = gsm_row.get('gsm', gsm_row.get('GSM', f"Unknown_{uuid.uuid4().hex[:6]}"))
    sample_text = get_comprehensive_gsm_text(gsm_row)

    if fields is None:
        fields = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']

    if not sample_text or len(sample_text.strip()) < 10:
        return {'gsm': gsm_id, **{f: 'Not Specified' for f in fields}}

    import json as _json
    empty_template = {f: "" for f in fields}

    # System prompt is cached by Ollama KV — only user message changes per sample
    system = (
        "TASK: Extract biological metadata from GEO samples.\n"
        "RULES:\n"
        "1. Condition: specific disease/phenotype. Healthy/Control/WT = \"Control\".\n"
        "2. Tissue: organ from source_name/characteristics. \"Cell Line: X\" only if named.\n"
        "3. Age: numeric from characteristics.\n"
        "4. Treatment: drug/compound/stimulus.\n"
        "5. Treatment_Time: duration.\n"
        "6. Use study_title/study_summary to decode coded values.\n"
        "7. \"Not specified\" ONLY if genuinely absent.\n"
        "8. Title Case. Output JSON only."
    )

    # User message: only the variable part (metadata + template)
    user_msg = f"METADATA:\n{sample_text}\n\nFill in the JSON:\n{_json.dumps(empty_template)}"

    for attempt in range(2):
        try:
            raw = _ollama_post(user_msg, model=_OLLAMA_EXTRACTION_MODEL,
                               num_predict=200, system_prompt=system,
                               timeout=120)
            if not raw:
                if attempt == 0: continue
                return {'gsm': gsm_id, **{f: 'Not Specified' for f in fields}}
            cleaned = re.sub(r'```json\s*|\s*```', '', raw.strip(), flags=re.DOTALL).strip()
            s, e = cleaned.find('{'), cleaned.rfind('}')
            if s >= 0 and e > s:
                data = _json.loads(cleaned[s:e+1])
                for k, v in data.items():
                    if isinstance(v, str): data[k] = v.strip()
                data['gsm'] = gsm_id
                return data
        except Exception:
            if attempt == 0: continue
    return {'gsm': gsm_id, **{f: 'Not Specified' for f in fields}}


class ContextRecallExtractor:
    """Context-enriched re-extractor that re-processes 'Not Specified' labels using
    experiment context and sibling sample labels as GSE context cache.
    
    Phase 1 (standard extraction) processes each sample independently.
    Phase 2 (context-enriched re-extraction) uses:
      - GSE experiment descriptions from GEOmetadb
      - Labels already extracted for sibling samples in the same GSE
      - Consensus patterns (if 8/10 samples are 'Cancer', the 9th probably is too)
    """
    
    def __init__(self, log_func=None, saved_cache=None, saved_memory=None, **kwargs):
        self.log = log_func or print
        # Accept both parameter names for backward compatibility
        _cache = saved_cache or saved_memory
        # Memory stores
        self.gse_descriptions = {}   # {gse_id: {title, description, overall_design}}
        self.gse_consensus = {}      # {gse_id: {col: Counter({val: count})}}
        self.n_corrected = 0
        self.n_confirmed = 0
        self.n_failed = 0
        
        # Restore from saved cache (previous session)
        if _cache:
            from collections import Counter
            saved_descs = _cache.get('gse_descriptions', {})
            self.gse_descriptions.update(saved_descs)
            for gse, cols in _cache.get('gse_consensus', {}).items():
                self.gse_consensus[gse] = {
                    col: Counter(counts) for col, counts in cols.items()
                }
            info = _cache.get('_info', {})
            self.log(f"[Phase2] Restored {len(saved_descs)} GSE descriptions "
                     f"from previous session (saved: {info.get('created', '?')})")
    
    def build_context(self, result_df):
        """Phase 2a: Build GSE context cache from Phase 1 results."""
        self.log("[Phase2] Building GSE context cache from Phase 1 results...")
        
        if 'series_id' not in result_df.columns:
            self.log("[Phase2] No series_id column — context-enriched re-extraction requires GSE context")
            return False
        
        # Build consensus: what did Phase 1 extract for each GSE?
        label_cols = [c for c in result_df.columns
                      if c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform')
                      and result_df[c].dtype == 'object']
        
        for gse_id, group in result_df.groupby('series_id'):
            gse_id = str(gse_id).strip()
            if not gse_id or gse_id == 'nan':
                continue
            self.gse_consensus[gse_id] = {}
            for col in label_cols:
                vals = group[col].fillna("Not Specified").astype(str)
                real_vals = [v for v in vals if v.strip() not in _NOT_SPECIFIED_VALUES]
                if real_vals:
                    from collections import Counter
                    self.gse_consensus[gse_id][col] = Counter(real_vals)
        
        # Fetch GSE descriptions from GEO website (not from GEOmetadb)
        # Skip GSEs already in memory (from previous sessions)
        unique_gses = [g for g in result_df['series_id'].dropna().unique()
                      if str(g).strip() and str(g).strip() != 'nan'
                      and str(g).strip().upper().startswith('GSE')]
        new_gses = [g for g in unique_gses if str(g).strip() not in self.gse_descriptions]
        cached = len(unique_gses) - len(new_gses)
        if cached > 0:
            self.log(f"[Phase2] {cached} GSE descriptions already in memory (from previous session)")
        
        if new_gses:
            self.log(f"[Phase2] Fetching {len(new_gses)} NEW GSE descriptions from GEO website...")
            
            for i, gse_id in enumerate(new_gses):
                gse_id = str(gse_id).strip()
                if not gse_id.upper().startswith('GSE'):
                    continue
                try:
                    desc = self._fetch_gse_from_geo(gse_id)
                    if desc:
                        self.gse_descriptions[gse_id] = desc
                except Exception as e:
                    if i < 3:  # only log first few errors
                        self.log(f"[Phase2] GSE fetch error for {gse_id}: {e}")
                
                # Progress every 20
                if (i + 1) % 20 == 0 or i == len(new_gses) - 1:
                    self.log(f"[Phase2] Fetched {i+1}/{len(new_gses)} new GSE descriptions "
                             f"({len(self.gse_descriptions)} total in memory)")
                
                # Rate limit: ~3 requests/sec to be polite to NCBI
                time.sleep(0.35)
        
        n_with_desc = sum(1 for d in self.gse_descriptions.values() if d.get('title'))
        n_with_consensus = sum(1 for c in self.gse_consensus.values() if c)
        self.log(f"[Phase2] Memory built: {n_with_desc} GSE descriptions, "
                 f"{n_with_consensus} GSEs with consensus labels")
        return True
    
    @staticmethod
    def _fetch_gse_from_geo(gse_id):
        """Scrape GSE experiment description from NCBI GEO website.
        Uses the text format endpoint which returns structured metadata.
        """
        import urllib.request
        url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
               f"?acc={gse_id}&targ=self&form=text&view=quick")
        
        req = urllib.request.Request(url, headers={
            'User-Agent': 'GeneVariate/1.0 (bioinformatics research tool)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode('utf-8', errors='replace')
        
        result = {'title': '', 'summary': '', 'overall_design': ''}
        current_field = None
        current_lines = []
        
        for line in text.split('\n'):
            if line.startswith('!Series_title'):
                val = line.split('=', 1)[1].strip() if '=' in line else ''
                result['title'] = val
            elif line.startswith('!Series_summary'):
                val = line.split('=', 1)[1].strip() if '=' in line else ''
                if result['summary']:
                    result['summary'] += ' ' + val
                else:
                    result['summary'] = val
            elif line.startswith('!Series_overall_design'):
                val = line.split('=', 1)[1].strip() if '=' in line else ''
                if result['overall_design']:
                    result['overall_design'] += ' ' + val
                else:
                    result['overall_design'] = val
        
        # Store FULL description — no truncation in cache
        # Truncation only happens at prompt construction time based on num_ctx
        
        return result if result['title'] else None

    # Default fields for Not Specified curation (Phase 2)
    # Age and Treatment_Time are NOT curated by default — they are
    # legitimately missing from most GEO metadata and should stay "Not Specified".
    # Users can add extra fields via _ns_curate_extra.
    _NS_CURATE_DEFAULT = {'Condition', 'Tissue', 'Treatment'}

    def find_not_specified(self, result_df, extra_fields=None):
        """Find samples with 'Not Specified' labels that could be corrected.
        Only checks Condition, Tissue, Treatment (+ any extra_fields specified by user).
        Age and Treatment_Time are excluded — they are often genuinely absent.
        """
        curate_fields = set(self._NS_CURATE_DEFAULT)
        if extra_fields:
            curate_fields.update(extra_fields)

        label_cols = [c for c in result_df.columns
                      if c in curate_fields
                      and result_df[c].dtype == 'object']
        
        ns_indices = []
        for idx, row in result_df.iterrows():
            cols_to_fix = []
            for col in label_cols:
                val = str(row.get(col, '')).strip()
                if val in _NOT_SPECIFIED_VALUES:
                    cols_to_fix.append(col)
            if cols_to_fix:
                ns_indices.append((idx, cols_to_fix))
        
        return ns_indices, label_cols
    
    def recall_correct_sample(self, row, cols_to_fix, result_df):
        """Re-extract NS labels using experiment context.
        
        Single efficient LLM call with enriched context:
          - Sample metadata (title, chars, description)
          - GSE experiment description (from NCBI GEO cache)
          - Consensus labels from sibling samples
          - 5 sibling examples with their labels
        
        No history or multi-turn — just one focused prompt per sample.
        """
        import json as _json
        gsm_id = str(row.get('GSM', row.get('gsm', '?')))
        gse_id = str(row.get('series_id', '')).strip()

        sample_text = get_comprehensive_gsm_text(row)
        gse_desc = self.gse_descriptions.get(gse_id, {})
        consensus = self.gse_consensus.get(gse_id, {})

        # Consensus text
        cons_parts = []
        for col in cols_to_fix:
            if col in consensus:
                top = consensus[col].most_common(5)
                vals = ", ".join(f'"{v}" ({n})' for v, n in top)
                cons_parts.append(f"  {col}: {vals}")
        cons_text = "\n".join(cons_parts) or "None"

        # Sibling examples (same GSE, with real labels)
        sib_parts = []
        if gse_id and gse_id != 'nan' and 'series_id' in result_df.columns:
            sibs = result_df[result_df['series_id'].astype(str).str.strip() == gse_id]
            for _, sib_row in sibs.head(8).iterrows():
                sib_gsm = str(sib_row.get('GSM', sib_row.get('gsm', '')))
                if sib_gsm == gsm_id:
                    continue
                lbls = []
                for col in cols_to_fix:
                    if col in sib_row.index:
                        v = str(sib_row[col]).strip()
                        if v not in _NOT_SPECIFIED_VALUES:
                            lbls.append(f"{col}={v}")
                if lbls:
                    sib_parts.append(f"  {sib_gsm}: {', '.join(lbls)}")
        sib_text = "\n".join(sib_parts[:5]) or "None"

        json_schema = {col: "string" for col in cols_to_fix}
        cols_str = ", ".join(cols_to_fix)

        # Build GSE context — FULL descriptions, no truncation
        # num_ctx=4096 handles ~16K chars of prompt
        gse_title = gse_desc.get('title', 'N/A')
        gse_summary = gse_desc.get('summary', '')
        gse_design = gse_desc.get('overall_design', '')

        prompt = (
            f"You are recovering missing labels for a GEO sample using experiment context.\n\n"
            f"EXAMPLE:\n"
            f"Sample GSM123456 has Condition='Not Specified' but belongs to experiment GSE99999\n"
            f"titled 'Gene expression in Alzheimer Disease vs Control brains'.\n"
            f"Sibling samples show: Condition='Alzheimer Disease' (8), Condition='Control' (6).\n"
            f"Sample metadata says: source_name='AD brain cortex', tissue='cortex'\n"
            f"OUTPUT: {{\"Condition\": \"Alzheimer Disease\"}}\n\n"
            f"Now recover missing labels for {gsm_id}.\n"
            f"Missing fields: {cols_str}\n\n"
            f"Sample metadata:\n{sample_text}\n\n"
            f"EXPERIMENT ({gse_id}):\n"
            f"Title: {gse_title}\n"
            f"Summary: {gse_summary}\n"
            f"Design: {gse_design}\n\n"
            f"Consensus from other samples:\n{cons_text}\n\n"
            f"Sibling examples:\n{sib_text}\n\n"
            f"Return ONLY JSON with the missing fields. Use specific names, not generic.\n"
            f"Control = healthy/normal. Not Specified ONLY if truly unknowable.\n"
            f"{_json.dumps(json_schema, indent=2)}"
        )
        try:
            raw = _ollama_post(prompt, timeout=120)
            if not raw:
                return None
            raw = re.sub(r'```json\s*|\s*```', '', raw).strip()
            s, e = raw.find('{'), raw.rfind('}')
            if s >= 0 and e > s:
                data = _json.loads(raw[s:e+1])
                return {col: str(data.get(col, 'Not Specified')).strip() or 'Not Specified'
                        for col in cols_to_fix}
        except Exception:
            pass
        return None

    def run_recall_pass(self, result_df, stop_fn=None, progress_fn=None, extra_fields=None):
        """Phase 2: Process all Not Specified samples with context-enriched re-extraction.
        Only curates Condition, Tissue, Treatment (+ extra_fields) by default.
        """
        ns_indices, label_cols = self.find_not_specified(result_df, extra_fields=extra_fields)
        if not ns_indices:
            self.log("[Phase2] No 'Not Specified' labels to correct!")
            return result_df
        
        total = len(ns_indices)
        self.log(f"[Phase2] Phase 2: Processing {total:,} samples with Not Specified labels...")
        
        corrected_df = result_df.copy()
        t0 = time.time()
        
        for i, (idx, cols_to_fix) in enumerate(ns_indices):
            if stop_fn and stop_fn():
                self.log("[Phase2] Stopped by user.")
                break
            
            row = corrected_df.loc[idx]
            gsm_id = str(row.get('GSM', row.get('gsm', '?')))
            
            result = self.recall_correct_sample(row, cols_to_fix, corrected_df)
            
            if result:
                any_changed = False
                for col, new_val in result.items():
                    if new_val.strip() not in _NOT_SPECIFIED_VALUES:
                        corrected_df.at[idx, col] = new_val
                        any_changed = True
                if any_changed:
                    self.n_corrected += 1
                else:
                    self.n_confirmed += 1
            else:
                self.n_failed += 1
            
            # Progress
            if progress_fn and (i % 5 == 0 or i == total - 1):
                elapsed = time.time() - t0
                avg = elapsed / (i + 1)
                eta = avg * (total - i - 1)
                progress_fn(i + 1, total,
                            f"Phase 2 Re-extraction: {i+1}/{total} | "
                            f"Corrected: {self.n_corrected} | "
                            f"ETA: {timedelta(seconds=int(eta))}")
        
        elapsed = time.time() - t0
        self.log(f"[Phase2] Phase 2 complete: {self.n_corrected} corrected, "
                 f"{self.n_confirmed} confirmed NS, {self.n_failed} failed "
                 f"({elapsed:.0f}s)")
        
        return corrected_df

    def save_cache(self, path=None):
        """Save memory to JSON file on disk for inspection and reuse.
        Default location: ./gse_cache/ directory next to the labels file.
        """
        import json as _json
        
        if path is None:
            mem_dir = os.path.join(os.getcwd(), "gse_cache")
            os.makedirs(mem_dir, exist_ok=True)
            path = os.path.join(mem_dir, "gse_cache.json")
        
        data = {
            "_info": {
                "created": datetime.now().isoformat(),
                "n_experiments": len(self.gse_descriptions),
                "n_consensus": len(self.gse_consensus),
                "n_corrected": self.n_corrected,
                "n_confirmed_ns": self.n_confirmed,
                "n_failed": self.n_failed,
            },
            "gse_descriptions": self.gse_descriptions,
            "gse_consensus": {
                gse: {col: dict(counter) for col, counter in cols.items()}
                for gse, cols in self.gse_consensus.items()
            },
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
            self.log(f"[Phase2] Memory saved: {path} "
                     f"({len(self.gse_descriptions)} GSEs, "
                     f"{len(self.gse_consensus)} consensus entries)")
        except Exception as e:
            self.log(f"[Phase2] Memory save error: {e}")
        
        return path

    @staticmethod
    def load_memory(path):
        """Load memory from a previously saved JSON file."""
        import json as _json
        with open(path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        return data

    def get_memory_summary(self):
        """Return a human-readable summary of what's in memory."""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  GSE CONTEXT CACHE")
        lines.append(f"{'='*70}")
        lines.append(f"  Experiments (GSE): {len(self.gse_descriptions)}")
        lines.append(f"  Experiments with consensus: {len(self.gse_consensus)}")
        lines.append(f"  Corrections made: {self.n_corrected}")
        lines.append(f"  Confirmed Not Specified: {self.n_confirmed}")
        lines.append(f"  Failed: {self.n_failed}")
        lines.append(f"{'='*70}")
        
        # Top GSEs by sample count
        if self.gse_consensus:
            lines.append(f"\n  TOP EXPERIMENTS BY CONSENSUS LABELS:")
            gse_sizes = []
            for gse, cols in self.gse_consensus.items():
                total = sum(sum(c.values()) for c in cols.values())
                title = self.gse_descriptions.get(gse, {}).get('title', '?')[:60]
                gse_sizes.append((gse, total, title))
            gse_sizes.sort(key=lambda x: -x[1])
            for gse, n, title in gse_sizes[:15]:
                lines.append(f"  {gse} ({n} labeled): {title}")
                cdata = self.gse_consensus[gse]
                for col, counter in cdata.items():
                    if counter:
                        top3 = sorted(counter.items(), key=lambda x: -x[1])[:3]
                        vals = ", ".join(f'"{v}" ({c})' for v, c in top3)
                        lines.append(f"    {col}: {vals}")
        
        # Global label stats
        lines.append(f"\n  FETCHED GSE DESCRIPTIONS (sample):")
        for gse, desc in list(self.gse_descriptions.items())[:10]:
            lines.append(f"  {gse}: {desc.get('title', '?')[:70]}")
            summary = desc.get('summary', '')[:100]
            if summary:
                lines.append(f"    Summary: {summary}...")
        
        if len(self.gse_descriptions) > 10:
            lines.append(f"  ... and {len(self.gse_descriptions) - 10} more")
        
        lines.append(f"\n{'='*70}")
        return "\n".join(lines)

    # Backward-compatible method aliases
    save_memory = save_cache
    build_memory = build_context
#  Label Harmonization — Full Pipeline (inspired by merging_labels_faster)
#  Stages: Negation → Noise strip → Synonyms → Hierarchy → Concept group
# ═══════════════════════════════════════════════════════════════════

# ── Regex noise strippers ──
_CONC_RE = re.compile(
    r'\b\d+\.?\d*\s*(?:uM|µM|μM|nM|pM|mM|ug/?ml|µg/?ml|ng/?ml|pg/?ml|'
    r'mg/?ml|U/?ml|IU/?ml|mol/?[Ll]|%)\b', re.IGNORECASE)
_TIME_RE = re.compile(
    r'\b(?:\d+\.?\d*\s*(?:h(?:ours?|rs?)?|min(?:utes?|s)?|d(?:ays?)?|'
    r'wk|weeks?|months?)|(?:day|week|month|hr|hour|minute)\s*\d+)\b', re.IGNORECASE)
_DOSE_RE = re.compile(
    r'\b\d+\.?\d*\s*(?:Gy|mg/?kg|mg|ug|µg|ng|g/?kg|rad|cGy|mGy)\b', re.IGNORECASE)
_DOSE_CYCLE_RE = re.compile(
    r'\b(?:dose|cycle|passage|round|treatment)\s*#?\s*\d+\b', re.IGNORECASE)
_TEMP_RE = re.compile(r'\b\d+\.?\d*\s*°?\s*[CcFf]\b')
_VEHICLE_RE = re.compile(
    r'\b(?:DMSO|PBS|vehicle|saline|EtOH|ethanol|methanol|MeOH|'
    r'water|dH2O|H2O|placebo|sham|mock|carrier)\b', re.IGNORECASE)
_CELLLINE_RE = re.compile(
    r'\b(?:A549|MCF[- ]?7|HeLa|HEK[- ]?293T?|HepG2|K562|U2OS|'
    r'Jurkat|THP[- ]?1|U937|HL[- ]?60|MDA[- ]?MB[- ]?\d+|'
    r'SK[- ]?MEL[- ]?\d+|NCI[- ]?H\d+|PC[- ]?\d+|'
    r'BT[- ]?\d+|SW[- ]?\d+|HCT[- ]?\d+|Caco[- ]?\d+|'
    r'RAW\s*264\.?7|NIH[- ]?3T3|CHO|Vero|COS[- ]?\d)\b', re.IGNORECASE)
_SAMPLE_ID_RE = re.compile(
    r'\b(?:patient|sample|replicate|rep|batch|donor|subject|'
    r'biological\s*rep(?:licate)?|technical\s*rep(?:licate)?)\s*#?\s*\d+\b',
    re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r'\b\d+\.?\d*\b')
_EXTRA_WS_RE = re.compile(r'\s+')
_TRAIL_PUNCT_RE = re.compile(r'[\s,;:_\-+/]+$|^[\s,;:_\-+/]+')
_IGNORE_PFX_RE = re.compile(r"^(Human|Tissue)\s*[-:]?\s*", re.IGNORECASE)

# ── Negation detection ──
# ── Backward-compatible aliases (class was renamed) ──
MemoryRecallAgent = ContextRecallExtractor

_NEGATABLE_DISEASE_TERMS = {
    'tumor', 'tumour', 'cancer', 'carcinoma', 'malignancy', 'malignant',
    'neoplasm', 'neoplastic', 'metastasis', 'metastatic', 'lesion',
    'disease', 'disorder', 'syndrome', 'infection', 'infected',
    'inflammation', 'inflamed', 'fibrosis', 'fibrotic',
    'leukemia', 'leukaemia', 'lymphoma', 'melanoma', 'sarcoma',
    'glioblastoma', 'glioma', 'myeloma', 'adenocarcinoma',
    'dementia', 'alzheimer', 'parkinson', 'diabetes', 'diabetic',
    'cirrhosis', 'hepatitis', 'arthritis', 'asthma', 'copd',
    'hiv', 'aids', 'covid', 'pneumonia', 'sepsis',
    'obesity', 'obese', 'hypertension', 'anemia',
    'pathological', 'pathologic', 'diseased', 'affected',
    'positive', 'cancerous',
}

_KNOWN_NON_DISEASES = {
    'non-small cell lung cancer', 'non-small-cell lung cancer',
    'non-hodgkin lymphoma', "non-hodgkin's lymphoma",
    'non-alcoholic fatty liver', 'non-alcoholic steatohepatitis',
    'nafld', 'nash',
}

def _detect_negation(text):
    """Detect negated disease terms → True means label is actually Control."""
    tl = text.lower().strip()
    _terms_re = '|'.join(re.escape(t) for t in _NEGATABLE_DISEASE_TERMS)

    # "tumor-free", "disease-free", "cancer-free"
    if re.search(r'\b(' + _terms_re + r')[\s\-]*free\b', tl):
        return True
    # "free of <disease>"
    if re.search(r'\bfree[\s\-]+of[\s\-]+\w*(' + _terms_re + r')', tl):
        return True
    # "negative for <disease>"
    if re.search(r'\bnegative[\s\-]+(?:for[\s\-]+)?(' + _terms_re + r')', tl):
        return True
    # "without <disease>"
    if re.search(r'\bwithout[\s\-]+\w*\s*(' + _terms_re + r')', tl):
        return True
    # "absence of <disease>"
    if re.search(r'\babsence[\s\-]+of[\s\-]+\w*(' + _terms_re + r')', tl):
        return True
    # "no <disease>" (but NOT "non-small cell" etc.)
    m = re.search(r'\bno[\s\-]+(\w+)', tl)
    if m and m.group(1).lower() in _NEGATABLE_DISEASE_TERMS:
        return True
    # "non-<disease>" — exclude known disease names starting with "non-"
    for known in _KNOWN_NON_DISEASES:
        if known in tl:
            return False
    if re.search(r'\bnon[\s\-]?(' + _terms_re + r')', tl):
        return True
    # "not <disease>"
    if re.search(r'\bnot[\s\-]+\w*(' + _terms_re + r')', tl):
        return True
    # Explicit healthy/normal/unaffected combined with disease mention
    if re.search(r'\b(?:healthy|normal|unaffected|control)\b', tl):
        for term in _NEGATABLE_DISEASE_TERMS:
            if term in tl:
                return True
    return False


# ── Synonym patterns (regex-based, order matters) ──
_SYN_PATTERNS = [
    (re.compile(r'\b(?:non[\s\-]?small[\s\-]?cell[\s\-]?lung[\s\-]?(?:cancer|carcinoma)|nsclc)\b', re.I), 'non-small cell lung cancer'),
    (re.compile(r'\b(?:small[\s\-]?cell[\s\-]?lung[\s\-]?(?:cancer|carcinoma)|sclc)\b', re.I), 'small cell lung cancer'),
    (re.compile(r'\b(?:triple[\s\-]?negative[\s\-]?breast[\s\-]?(?:cancer|carcinoma)|tnbc)\b', re.I), 'triple negative breast cancer'),
    (re.compile(r'\b(?:acute[\s\-]?myeloid[\s\-]?leuk[ae]mia|aml)\b', re.I), 'acute myeloid leukemia'),
    (re.compile(r'\b(?:acute[\s\-]?lympho(?:blastic|cytic)[\s\-]?leuk[ae]mia|all)\b', re.I), 'acute lymphoblastic leukemia'),
    (re.compile(r'\b(?:chronic[\s\-]?myeloid[\s\-]?leuk[ae]mia|cml)\b', re.I), 'chronic myeloid leukemia'),
    (re.compile(r'\b(?:chronic[\s\-]?lymphocytic[\s\-]?leuk[ae]mia|cll)\b', re.I), 'chronic lymphocytic leukemia'),
    (re.compile(r'\b(?:hepatocellular[\s\-]?carcinoma|hcc)\b', re.I), 'liver cancer'),
    (re.compile(r'\b(?:renal[\s\-]?cell[\s\-]?carcinoma|rcc)\b', re.I), 'kidney cancer'),
    (re.compile(r'\b(?:glioblastoma[\s\-]?multiforme?|gbm)\b', re.I), 'glioblastoma'),
    (re.compile(r'\b(?:diffuse[\s\-]?large[\s\-]?b[\s\-]?cell[\s\-]?lymphoma|dlbcl)\b', re.I), 'diffuse large b-cell lymphoma'),
    (re.compile(r'\b(?:squamous[\s\-]?cell[\s\-]?carcinoma|scc)\b', re.I), 'squamous cell cancer'),
    (re.compile(r'\b(?:pancreatic[\s\-]?ductal[\s\-]?adenocarcinoma|pdac)\b', re.I), 'pancreatic cancer'),
    (re.compile(r'\bleuk[ae]mia\b', re.I), 'leukemia'),
    (re.compile(r'\b(?:tumou?r|neoplasm|malignancy|malignant)\b', re.I), 'tumor'),
    (re.compile(r'\badenocarcinoma\b', re.I), 'adenocarcinoma'),
    (re.compile(r'\bcarcinoma\b', re.I), 'carcinoma'),
    # Control synonyms
    (re.compile(r'\b(?:untreated|naive|naïve|uninfected|unstimulated|resting|'
                r'wildtype|wild[\s\-]?type|wt|ctrl)\b', re.I), 'control'),
    # Preserve disease names with numbers BEFORE bare number stripping
    # Use word form so BARE_NUMBER_RE doesn't strip them
    (re.compile(r'\btype[\s\-]?2[\s\-]?diabet\w*', re.I), 'type2diabetes'),
    (re.compile(r'\btype[\s\-]?1[\s\-]?diabet\w*', re.I), 'type1diabetes'),
    (re.compile(r'\bt2d\w*\b', re.I), 'type2diabetes'),
    (re.compile(r'\bt1d\w*\b', re.I), 'type1diabetes'),
    (re.compile(r"\balzheimer'?s?\b", re.I), 'alzheimer'),
    (re.compile(r"\bparkinson'?s?\b", re.I), 'parkinson'),
    (re.compile(r"\bhuntington'?s?\b", re.I), 'huntington'),
    (re.compile(r"\bcrohn'?s?\b", re.I), 'crohn'),
    (re.compile(r"\bhodgkin'?s?\b", re.I), 'hodgkin'),
    (re.compile(r'\bhealthy[\s\-]?(?:control|donor|volunteer|subject|individual)s?\b', re.I), 'control'),
    # Strip generic lab noise
    (re.compile(r'\bcell[\s\-]?line\b', re.I), ''),
    (re.compile(r'\bcells\b', re.I), ''),
    (re.compile(r'\btissue\b', re.I), ''),
    (re.compile(r'\bculture[ds]?\b', re.I), ''),
    (re.compile(r'\bin[\s\-]?vitro\b', re.I), ''),
    (re.compile(r'\bin[\s\-]?vivo\b', re.I), ''),
    (re.compile(r'\bprimary\b', re.I), ''),
    (re.compile(r'\bderived\b', re.I), ''),
    (re.compile(r'\bsamples?\b', re.I), ''),
    (re.compile(r'\bisolated?\b', re.I), ''),
    (re.compile(r'\bpurified\b', re.I), ''),
    (re.compile(r'\bsorted\b', re.I), ''),
]

# ── Category keyword sets (hierarchy enforcement) ──
_DISEASE_KW = {
    'cancer', 'leukemia', 'lymphoma', 'melanoma', 'glioblastoma', 'glioma',
    'myeloma', 'sarcoma', 'mesothelioma', 'neuroblastoma', 'retinoblastoma',
    'hepatoblastoma', 'medulloblastoma', 'astrocytoma', 'ependymoma',
    'diabetes', 'asthma', 'copd', 'fibrosis', 'cirrhosis', 'hepatitis',
    'arthritis', 'lupus', 'psoriasis', 'eczema', 'dermatitis',
    'alzheimer', 'parkinson', 'huntington', 'sclerosis', 'epilepsy',
    'stroke', 'infarction', 'atherosclerosis', 'cardiomyopathy',
    'hypertension', 'anemia', 'thrombosis', 'sepsis', 'pneumonia',
    'tuberculosis', 'malaria', 'hiv', 'aids', 'covid', 'influenza',
    'obesity', 'syndrome', 'disease', 'disorder', 'infection',
    'inflammation', 'crohn', 'colitis', 'pancreatitis', 'nephritis',
    'encephalitis', 'meningitis', 'osteosarcoma', 'chondrosarcoma',
    'rhabdomyosarcoma', 'leiomyosarcoma', 'liposarcoma',
    'adenoma', 'papilloma', 'polyp', 'dysplasia', 'metastasis',
    'metastatic', 'invasive', 'malignant', 'benign', 'tumor',
    'ipf', 'ibd', 'nash', 'nafld', 'ards',
    'dementia', 'carcinoma', 'adenocarcinoma', 'schizophrenia',
    'depression', 'bipolar', 'autism',
}
_DRUG_KW = {
    'cisplatin', 'doxorubicin', 'paclitaxel', 'docetaxel', 'gemcitabine',
    'methotrexate', 'cyclophosphamide', 'vincristine', 'etoposide',
    'irinotecan', 'carboplatin', 'oxaliplatin', 'fluorouracil', '5-fu',
    'tamoxifen', 'letrozole', 'anastrozole', 'trastuzumab', 'rituximab',
    'bevacizumab', 'cetuximab', 'erlotinib', 'gefitinib', 'sorafenib',
    'sunitinib', 'imatinib', 'nilotinib', 'dasatinib', 'vemurafenib',
    'olaparib', 'palbociclib', 'ribociclib', 'nivolumab', 'pembrolizumab',
    'ipilimumab', 'atezolizumab', 'durvalumab', 'avelumab',
    'dexamethasone', 'prednisolone', 'prednisone', 'hydrocortisone',
    'metformin', 'insulin', 'rapamycin', 'sirolimus', 'everolimus',
    'temozolomide', 'cytarabine', 'azacitidine', 'decitabine',
    'bortezomib', 'lenalidomide', 'thalidomide', 'venetoclax',
    'ibrutinib', 'acalabrutinib', 'ruxolitinib', 'tofacitinib',
    'retinoic acid', 'atra', 'tretinoin', 'isotretinoin',
    'aspirin', 'ibuprofen', 'celecoxib', 'statin', 'atorvastatin',
    'nutlin', 'jq1', 'bet inhibitor', 'hdac inhibitor', 'dnmt inhibitor',
    'pi3k inhibitor', 'mtor inhibitor', 'mek inhibitor', 'erk inhibitor',
    'jak inhibitor', 'cdk inhibitor', 'parp inhibitor', 'braf inhibitor',
    'alk inhibitor', 'egfr inhibitor', 'vegf inhibitor', 'bcl2 inhibitor',
    'proteasome inhibitor', 'topoisomerase inhibitor', 'kinase inhibitor',
    'tyrosine kinase inhibitor', 'tki', 'checkpoint inhibitor',
    'antibody', 'monoclonal', 'car-t',
}
_STIMULUS_KW = {
    'lps', 'lipopolysaccharide', 'phorbol', 'pma', 'tpa',
    'ifn', 'interferon', 'tnf', 'interleukin', 'il-',
    'tgf', 'egf', 'fgf', 'vegf', 'pdgf', 'ngf', 'bdnf',
    'wnt', 'notch', 'hedgehog', 'bmp',
    'heat shock', 'cold shock', 'hypoxia', 'hyperoxia',
    'uv', 'irradiation', 'radiation', 'gamma ray',
    'serum starvation', 'glucose deprivation', 'nutrient deprivation',
    'oxidative stress', 'ros', 'hydrogen peroxide', 'h2o2',
    'differentiation', 'reprogramming', 'transfection',
    'knockdown', 'knockout', 'overexpression', 'sirna', 'shrna',
    'crispr', 'cas9', 'gene editing',
    'stimulation', 'activation', 'inhibition',
    'agonist', 'antagonist', 'ligand',
    'vaccine', 'antigen', 'adjuvant', 'immunization',
    'co-culture', 'coculture', 'conditioned medium',
}
_CONTROL_KW = {'control', 'ctrl', 'baseline', 'reference', 'untreated', 'vehicle'}

_DRUG_NAMES_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(d) for d in sorted(_DRUG_KW, key=len, reverse=True)) + r')\b',
    re.IGNORECASE)
_DRUG_SUFFIX_RE = re.compile(
    r'\b\w*(?:mab|nib|tinib|ciclib|lisib|rafenib|zomib|'
    r'parib|olimus|vastatin|sartan|prazole|idine|etine|aline)\b',
    re.IGNORECASE)

# ── Disease identity groups (prevent merging distinct diseases) ──
_DISEASE_IDENTITY = {
    'alzheimer': 'alzheimer_disease', 'dementia': 'dementia_general',
    'parkinson': 'parkinson_disease', 'huntington': 'huntington_disease',
    'epilepsy': 'epilepsy', 'multiple sclerosis': 'multiple_sclerosis',
    'amyotrophic lateral sclerosis': 'als', 'als': 'als',
    'breast cancer': 'breast_cancer', 'lung cancer': 'lung_cancer',
    'non-small cell lung cancer': 'nsclc', 'small cell lung cancer': 'sclc',
    'prostate cancer': 'prostate_cancer',
    'colorectal cancer': 'colorectal_cancer', 'colon cancer': 'colorectal_cancer',
    'pancreatic cancer': 'pancreatic_cancer', 'liver cancer': 'liver_cancer',
    'kidney cancer': 'kidney_cancer', 'ovarian cancer': 'ovarian_cancer',
    'bladder cancer': 'bladder_cancer', 'stomach cancer': 'stomach_cancer',
    'gastric cancer': 'stomach_cancer', 'thyroid cancer': 'thyroid_cancer',
    'skin cancer': 'skin_cancer', 'brain cancer': 'brain_cancer',
    'cervical cancer': 'cervical_cancer', 'endometrial cancer': 'endometrial_cancer',
    'esophageal cancer': 'esophageal_cancer', 'head and neck cancer': 'head_neck_cancer',
    'glioblastoma': 'glioblastoma', 'glioma': 'glioma',
    'melanoma': 'melanoma', 'mesothelioma': 'mesothelioma',
    'neuroblastoma': 'neuroblastoma', 'medulloblastoma': 'medulloblastoma',
    'osteosarcoma': 'osteosarcoma',
    'acute myeloid leukemia': 'aml', 'acute lymphoblastic leukemia': 'all_leuk',
    'chronic myeloid leukemia': 'cml', 'chronic lymphocytic leukemia': 'cll',
    'diffuse large b-cell lymphoma': 'dlbcl',
    'hodgkin lymphoma': 'hodgkin', 'multiple myeloma': 'multiple_myeloma',
    'diabetes': 'diabetes', 'type 1 diabetes': 'diabetes_t1',
    'type 2 diabetes': 'diabetes_t2', 'type1diabetes': 'diabetes_t1',
    'type2diabetes': 'diabetes_t2', 'asthma': 'asthma', 'copd': 'copd',
    'fibrosis': 'fibrosis_general', 'pulmonary fibrosis': 'pulmonary_fibrosis',
    'idiopathic pulmonary fibrosis': 'ipf', 'cystic fibrosis': 'cystic_fibrosis',
    'liver fibrosis': 'liver_fibrosis',
    'cirrhosis': 'cirrhosis', 'hepatitis': 'hepatitis_general',
    'hepatitis b': 'hepatitis_b', 'hepatitis c': 'hepatitis_c',
    'rheumatoid arthritis': 'rheumatoid_arthritis',
    'osteoarthritis': 'osteoarthritis', 'psoriasis': 'psoriasis',
    'lupus': 'lupus', 'crohn': 'crohn', 'colitis': 'colitis',
    'ulcerative colitis': 'ulcerative_colitis',
    'covid': 'covid', 'influenza': 'influenza',
    'hiv': 'hiv', 'tuberculosis': 'tuberculosis', 'malaria': 'malaria',
    'sepsis': 'sepsis',
    'obesity': 'obesity', 'hypertension': 'hypertension',
    'atherosclerosis': 'atherosclerosis', 'cardiomyopathy': 'cardiomyopathy',
    'heart failure': 'heart_failure', 'stroke': 'stroke', 'anemia': 'anemia',
    'schizophrenia': 'schizophrenia', 'bipolar disorder': 'bipolar',
    'depression': 'depression', 'autism': 'autism',
}

def _get_disease_identity(text):
    """Return disease identity group (longer matches first). None if unknown."""
    tl = text.lower()
    for key in sorted(_DISEASE_IDENTITY.keys(), key=len, reverse=True):
        if key in tl:
            return _DISEASE_IDENTITY[key]
    return None


# ── Junk token patterns for cluster name cleaning ──
_JUNK_PATS = [
    re.compile(r'^[a-zA-Z]\d{1,2}$'),           # d4, a4, p3
    re.compile(r'^\d{1,3}$'),                     # pure short numbers
    re.compile(r'^[a-zA-Z]$'),                    # single letters
    re.compile(r'^[A-Z]{1,2}\d{2,4}$'),           # H524, A549, U87
    re.compile(r'^rep\d+$', re.I),                # rep1
    re.compile(r'^lane\d+$', re.I),               # lane1
    re.compile(r'^batch\d+$', re.I),              # batch1
    re.compile(r'^donor\d+$', re.I),              # donor1
    re.compile(r'^sample\d+$', re.I),             # sample1
    re.compile(r'^passage\d+$', re.I),            # passage5
    re.compile(r'^(si|sh|ko|wt|het|hom)\d*$', re.I),
]
_KEEP_SHORT = {
    'il', 'tnf', 'tgf', 'egf', 'fgf', 'vegf', 'igf', 'csf',
    'hiv', 'hcv', 'hbv', 'ebv', 'aml', 'cll', 'cml', 'all', 'dlbcl',
    'copd', 'ibd', 'sle', 'als', 'pd', 'ad', 't2d', 't1d',
    'akt', 'erk', 'jak', 'mek', 'p53', 'myc', 'bcl', 'brca',
    'lps', 'pma', 'dex', 'wt', 'ko', 'oe',
}


def _clean_cluster_name(name):
    """Strip meaningless junk tokens from a cluster name."""
    if not name or name.lower() == 'control':
        return name
    tokens = name.replace('_', ' ').split()
    cleaned = []
    for tok in tokens:
        ts = tok.strip('.,;:()-/')
        if not ts:
            continue
        if ts.lower() in _KEEP_SHORT:
            cleaned.append(tok)
            continue
        if any(p.match(ts) for p in _JUNK_PATS):
            continue
        cleaned.append(tok)
    result = ' '.join(cleaned).strip(' .,;:-_/')
    return result if result and len(result) >= 2 else name


def _clean_condition_label(raw_label):
    """Full condition label cleaning: negation → noise strip → synonyms → hierarchy.
    Returns a lowercase cleaned concept string."""
    text = str(raw_label)
    text = _IGNORE_PFX_RE.sub('', text)

    # Negation check BEFORE any synonym replacement
    if _detect_negation(text):
        return "control"

    # Strip noise
    text = _CONC_RE.sub('', text)
    text = _TIME_RE.sub('', text)
    text = _DOSE_RE.sub('', text)
    text = _DOSE_CYCLE_RE.sub('', text)
    text = _TEMP_RE.sub('', text)
    text = _SAMPLE_ID_RE.sub('', text)
    text = _VEHICLE_RE.sub('', text)
    text = _CELLLINE_RE.sub('', text)

    for pattern, replacement in _SYN_PATTERNS:
        text = pattern.sub(replacement, text)

    text = _BARE_NUMBER_RE.sub('', text)
    text = re.sub(r'\s*[+]\s*', ' ', text)
    text = _TRAIL_PUNCT_RE.sub('', text)
    text = _EXTRA_WS_RE.sub(' ', text).strip().lower()

    # Second negation check after cleaning
    if _detect_negation(text):
        return "control"

    # Hierarchy enforcement: Disease > Drug > Stimulus > Control
    has_disease = any(kw in text for kw in _DISEASE_KW)
    if has_disease:
        # Strip drug/stimulus words — keep disease
        text = _DRUG_NAMES_RE.sub('', text)
        text = _DRUG_SUFFIX_RE.sub('', text)
        for kw in _STIMULUS_KW:
            if len(kw) > 3:
                text = re.sub(r'\b' + re.escape(kw) + r'\b', '', text, flags=re.IGNORECASE)
        text = _TRAIL_PUNCT_RE.sub('', text)
        text = _EXTRA_WS_RE.sub(' ', text).strip()
    else:
        has_drug = bool(_DRUG_NAMES_RE.search(text)) or bool(_DRUG_SUFFIX_RE.search(text))
        if has_drug:
            for kw in _STIMULUS_KW:
                if len(kw) > 3:
                    text = re.sub(r'\b' + re.escape(kw) + r'\b', '', text, flags=re.IGNORECASE)
            text = _TRAIL_PUNCT_RE.sub('', text)
            text = _EXTRA_WS_RE.sub(' ', text).strip()

    if not text or len(text) < 2:
        text = "control"
    # Standalone 'normal', 'healthy', 'baseline' → control
    if text.lower().strip() in ('normal', 'healthy', 'baseline', 'healthy donor',
                                 'healthy control', 'negative control', 'reference'):
        text = "control"
    return text


def _titlecase_concept(concept):
    """Convert cleaned concept to Title Case, preserving known abbreviations."""
    _UPPER_ABBREVS = {
        'hiv', 'aids', 'covid', 'copd', 'aml', 'cml', 'cll', 'dlbcl',
        'nsclc', 'sclc', 'hcc', 'rcc', 'gbm', 'scc', 'tnbc', 'ibd',
        'sle', 'als', 'ards', 'nash', 'nafld', 'lps', 'pma', 'tnf',
        'il', 'ifn', 'tgf', 'egf', 'fgf', 'vegf', 'pdgf', 'pbmc',
        'dna', 'rna', 'mrna', 'pcr', 'er', 'her2', 'brca', 'crispr',
    }
    # Expand embedded number forms first
    _EXPAND_MAP = {
        'type2diabetes': 'Type 2 Diabetes',
        'type1diabetes': 'Type 1 Diabetes',
    }
    cl = concept.lower().strip()
    if cl in _EXPAND_MAP:
        return _EXPAND_MAP[cl]
    if cl == 'control':
        return 'Control'
    if cl in ('not specified', 'not_specified'):
        return 'Not Specified'
    words = concept.split()
    titled = []
    for w in words:
        if w.lower() in _UPPER_ABBREVS:
            titled.append(w.upper())
        elif '-' in w:
            titled.append('-'.join(part.capitalize() for part in w.split('-')))
        else:
            titled.append(w.capitalize())
    return ' '.join(titled)


def _classify_concept_fast(concept):
    """Rule-based fast concept category classification.
    Returns: 'disease', 'drug', 'stimulus', 'control', or 'unknown'."""
    text_lower = concept.lower()
    if text_lower in _CONTROL_KW or text_lower == 'control':
        return 'control'
    for kw in _CONTROL_KW:
        if kw in set(text_lower.split()):
            return 'control'
    for kw in _DISEASE_KW:
        if kw in text_lower:
            return 'disease'
    for kw in _DRUG_KW:
        if kw in text_lower:
            return 'drug'
    for kw in _STIMULUS_KW:
        if kw in text_lower:
            return 'stimulus'
    return 'unknown'


# ── Semantic clustering configuration ──
_STAGE1_DISTANCE_THRESHOLD = 0.25
_STAGE2_MERGE_THRESHOLD = 0.12
_DISEASE_DISTANCE_THRESHOLD = 0.18
_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
_CLASSIFIER_MODEL_NAME = 'facebook/bart-large-mnli'
_MIN_FREQ_CUTOFF = 5


def _try_load_embedding_model(log_func=None):
    """Try to load sentence-transformers model. Returns (model, device_str) or (None, None)."""
    _log = log_func or print
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        _log(f"[Harmonize] Loading embedding model ({_EMBEDDING_MODEL_NAME}) on {device_str}...")
        model = SentenceTransformer(_EMBEDDING_MODEL_NAME, device=device_str)
        return model, device_str
    except ImportError:
        _log("[Harmonize] sentence-transformers not installed — skipping semantic clustering.")
        _log("[Harmonize]   Install with: pip install sentence-transformers")
        return None, None
    except Exception as e:
        _log(f"[Harmonize] Could not load embedding model: {e}")
        return None, None


def _try_zeroshot_classify(unknowns, log_func=None):
    """Try zero-shot classification for unknown concepts. Returns dict or {}."""
    _log = log_func or print
    if not unknowns:
        return {}
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _log(f"[Harmonize] Zero-shot classifying {len(unknowns)} unknown concepts...")
        classifier = hf_pipeline("zero-shot-classification",
                                  model=_CLASSIFIER_MODEL_NAME, device=device)
        cat_labels = [
            "disease or medical condition",
            "drug or pharmaceutical treatment",
            "biological stimulus or perturbation",
            "healthy control or baseline"
        ]
        cat_map = {
            "disease or medical condition": "disease",
            "drug or pharmaceutical treatment": "drug",
            "biological stimulus or perturbation": "stimulus",
            "healthy control or baseline": "control",
        }
        results = {}
        batch_size = 32
        for i in range(0, len(unknowns), batch_size):
            batch = unknowns[i:i + batch_size]
            for concept in batch:
                try:
                    out = classifier(concept, cat_labels, multi_label=False)
                    best = out['labels'][0]
                    results[concept] = cat_map.get(best, 'unknown')
                except Exception:
                    results[concept] = 'unknown'
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log(f"[Harmonize] Zero-shot: classified {len(results)} concepts")
        return results
    except ImportError:
        _log("[Harmonize] transformers not installed — unknown concepts stay as-is.")
        return {c: 'unknown' for c in unknowns}
    except Exception as e:
        _log(f"[Harmonize] Zero-shot failed: {e}")
        return {c: 'unknown' for c in unknowns}


def _semantic_cluster_concepts(concepts, embedding_model, concept_categories,
                               device_str='cpu', log_func=None):
    """Cluster concepts with category + disease identity barriers.
    Returns {cluster_id: [concept_list]} or None on failure."""
    _log = log_func or print
    if len(concepts) <= 1:
        return {0: concepts} if concepts else {}
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        _log(f"[Harmonize] Generating embeddings for {len(concepts)} concepts...")
        embeddings = embedding_model.encode(concepts, show_progress_bar=False,
                                             device=device_str, batch_size=256)
        dist_matrix = cosine_distances(embeddings)

        # ── Category barriers (disease/drug/stimulus/control never mix) ──
        for i in range(len(concepts)):
            cat_i = concept_categories.get(concepts[i], 'unknown')
            for j in range(i + 1, len(concepts)):
                cat_j = concept_categories.get(concepts[j], 'unknown')
                if cat_i != cat_j and cat_i != 'unknown' and cat_j != 'unknown':
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
                if (cat_i == 'control') != (cat_j == 'control'):
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0

        # ── Disease identity barriers (Alzheimer ≠ Dementia, breast ≠ lung) ──
        barriers = 0
        for i in range(len(concepts)):
            if concept_categories.get(concepts[i]) != 'disease':
                continue
            id_i = _get_disease_identity(concepts[i])
            for j in range(i + 1, len(concepts)):
                if concept_categories.get(concepts[j]) != 'disease':
                    continue
                id_j = _get_disease_identity(concepts[j])
                if id_i is not None and id_j is not None and id_i != id_j:
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
                    barriers += 1
                elif (id_i is not None) != (id_j is not None):
                    if dist_matrix[i, j] > _DISEASE_DISTANCE_THRESHOLD:
                        dist_matrix[i, j] = 1.0
                        dist_matrix[j, i] = 1.0

        _log(f"[Harmonize] Category + disease barriers applied ({barriers} disease pairs blocked)")

        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=_STAGE1_DISTANCE_THRESHOLD,
            metric='precomputed', linkage='average'
        )
        labels = clustering.fit_predict(dist_matrix)

        clusters = {}
        for concept, label in zip(concepts, labels):
            clusters.setdefault(int(label), []).append(concept)

        _log(f"[Harmonize] Semantic clustering: {len(concepts)} concepts → {len(clusters)} clusters")
        return clusters

    except ImportError:
        _log("[Harmonize] scikit-learn not available for clustering")
        return None
    except Exception as e:
        _log(f"[Harmonize] Semantic clustering failed: {e}")
        return None


def _pick_canonical_name(concept_members, concept_categories, concept_map, value_counts):
    """Hierarchy-based canonical naming: Disease > Drug > Stimulus > Control."""
    from collections import Counter as _C
    cat_counts = _C()
    for concept in concept_members:
        cat = concept_categories.get(concept, 'unknown')
        weight = sum(value_counts.get(label, 1) for label in concept_map.get(concept, [concept]))
        cat_counts[cat] += weight

    dominant_cat = 'unknown'
    for cat in ['disease', 'drug', 'stimulus', 'control', 'unknown']:
        if cat_counts.get(cat, 0) > 0:
            dominant_cat = cat
            break

    if dominant_cat == 'control':
        return "Control"

    candidates = []
    for concept in concept_members:
        cat = concept_categories.get(concept, 'unknown')
        if cat == dominant_cat or dominant_cat == 'unknown':
            weight = sum(value_counts.get(l, 1) for l in concept_map.get(concept, [concept]))
            candidates.append((concept, weight))

    if not candidates:
        candidates = [(c, sum(value_counts.get(l, 1)
                              for l in concept_map.get(c, [c])))
                      for c in concept_members]

    candidates.sort(key=lambda x: (-x[1], len(x[0])))
    name = candidates[0][0].strip()
    return _clean_cluster_name(_titlecase_concept(name))


def _merge_similar_names(final_mapping, embedding_model, device_str='cpu', log_func=None):
    """Stage-2: merge near-duplicate cluster names via embeddings.
    Respects disease identity barriers."""
    _log = log_func or print
    unique_names = sorted(set(final_mapping.values()))
    if len(unique_names) <= 1:
        return final_mapping
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        embeddings = embedding_model.encode(unique_names, show_progress_bar=False,
                                             device=device_str)
        dist_matrix = cosine_distances(embeddings)

        # Block control from merging with non-control
        control_idx = {i for i, n in enumerate(unique_names) if n.lower() == 'control'}
        for i in control_idx:
            for j in range(len(unique_names)):
                if i != j:
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0

        # Block different disease identities
        for i in range(len(unique_names)):
            id_i = _get_disease_identity(unique_names[i])
            for j in range(i + 1, len(unique_names)):
                id_j = _get_disease_identity(unique_names[j])
                if id_i is not None and id_j is not None and id_i != id_j:
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0

        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=_STAGE2_MERGE_THRESHOLD,
            metric='precomputed', linkage='average'
        )
        labels = clustering.fit_predict(dist_matrix)

        name_groups = {}
        for name, label in zip(unique_names, labels):
            name_groups.setdefault(int(label), []).append(name)

        merge_map = {}
        for names in name_groups.values():
            if len(names) > 1:
                name_freq = {n: sum(1 for v in final_mapping.values() if v == n) for n in names}
                canonical = max(names, key=lambda n: (name_freq[n], -len(n)))
                for name in names:
                    if name != canonical:
                        merge_map[name] = canonical

        if merge_map:
            _log(f"[Harmonize] Stage-2 merged {len(merge_map)} near-duplicate names:")
            for old, new in sorted(merge_map.items()):
                _log(f"  '{old}' → '{new}'")
            for label in final_mapping:
                if final_mapping[label] in merge_map:
                    final_mapping[label] = merge_map[final_mapping[label]]
        else:
            _log("[Harmonize] Stage-2: no near-duplicates found")

        return final_mapping
    except Exception as e:
        _log(f"[Harmonize] Stage-2 merge failed: {e}")
        return final_mapping


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM LABEL CURATOR — Cross-experiment label collapsing via LLM judgment
#
#  No dictionaries, no fuzzy matching. The LLM reviews the label inventory
#  and decides which labels are the same biomedical concept.
#  Manual trigger only — user reviews proposed merges before applying.
# ═══════════════════════════════════════════════════════════════════════════════

class LLMCurator:
    """LLM-based label curator for cross-experiment harmonization.

    Workflow:
      1. Scan unique labels per field across the platform
      2. Pre-filter candidate pairs (likely same concept)
      3. Ask LLM: "Are these two labels the same biomedical concept?"
      4. Build merge map (short→long, rare→common)
      5. Present to user for review before applying

    Uses _ollama_post for LLM calls. No dictionaries, no fuzzy matching.
    """

    _CURATE_FIELDS = {'Condition', 'Tissue', 'Treatment'}

    def __init__(self, log_func=None):
        self.log = log_func or print
        self.merge_proposals = {}  # {field: [(from_label, to_label, llm_reason), ...]}

    def _find_candidates(self, labels_with_counts):
        """Pre-filter candidate pairs that MIGHT be the same concept.
        Only pairs worth asking the LLM about — saves LLM calls.

        Candidates:
          - One label's words are a subset of another's words
          - Short label (≤5 chars) could be abbreviation of longer
          - Same first word + similar length
        Returns list of (label_a, label_b, count_a, count_b) tuples.
        """
        items = sorted(labels_with_counts.items(), key=lambda x: -x[1])
        candidates = []
        seen = set()

        for i, (la, ca) in enumerate(items):
            la_words = set(la.lower().replace('-', ' ').replace('_', ' ').split())
            la_clean = la.lower().replace(' ', '').replace('-', '').replace('_', '')
            la_nums = re.findall(r'\d+', la)

            for j, (lb, cb) in enumerate(items):
                if i >= j:
                    continue
                pair_key = tuple(sorted([la, lb]))
                if pair_key in seen:
                    continue

                lb_words = set(lb.lower().replace('-', ' ').replace('_', ' ').split())
                lb_clean = lb.lower().replace(' ', '').replace('-', '').replace('_', '')
                lb_nums = re.findall(r'\d+', lb)

                # Numbers must match
                if la_nums != lb_nums:
                    continue

                # Skip if both are very short (likely different abbreviations)
                if len(la) <= 3 and len(lb) <= 3:
                    continue

                # Exact match after normalization — already handled by Phase 1.5
                if la_clean == lb_clean:
                    continue

                is_candidate = False

                # Candidate 1: word subset (e.g., "Leukemia" is subset of
                # "Acute Myeloid Leukemia")
                if la_words and lb_words:
                    if la_words.issubset(lb_words) or lb_words.issubset(la_words):
                        is_candidate = True

                # Candidate 2: short label could be abbreviation
                # ONLY if initials of longer label plausibly match
                shorter, longer = (la, lb) if len(la) <= len(lb) else (lb, la)
                if (len(shorter) <= 6 and shorter.replace('-','').replace(' ','').isupper()
                        and len(longer) > 8):
                    # Pre-check: initials or first-letter match
                    words = re.split(r'[\s\-_/]+', longer.strip())
                    initials = ''.join(w[0].upper() for w in words if w)
                    short_clean = shorter.upper().replace('-','').replace(' ','').replace('(','').replace(')','')
                    # Exact initials (AML=Acute Myeloid Leukemia)
                    if short_clean == initials:
                        is_candidate = True
                    # First letter matches + similar length to initials
                    # Catches compound-word abbreviations: GBM=GlioBlastoma Multiforme,
                    # HCC=HepatoCellular Carcinoma, NSCLC=Non-Small Cell Lung Cancer
                    elif (len(short_clean) >= 2 and short_clean[0] == initials[0:1]
                          and abs(len(short_clean) - len(initials)) <= 2):
                        is_candidate = True

                # Candidate 3: share ≥2 significant words
                common_words = la_words & lb_words
                stopwords = {'the', 'of', 'and', 'in', 'for', 'with', 'type', 'cell', 'cells'}
                sig_common = common_words - stopwords
                if len(sig_common) >= 2:
                    is_candidate = True

                if is_candidate:
                    seen.add(pair_key)
                    candidates.append((la, lb, ca, cb))

        return candidates

    def _ask_llm(self, label_a, label_b, field, context_labels=None):
        """Ask LLM if two labels represent the same biomedical concept.
        Returns (same: bool, canonical: str, reason: str).
        """
        context_str = ""
        if context_labels:
            top_labels = sorted(context_labels.items(), key=lambda x: -x[1])[:15]
            context_str = ("\nOther labels in this field on this platform:\n"
                           + ", ".join(f'"{l}" ({c})' for l, c in top_labels))

        prompt = f"""You are a biomedical terminology expert. I have two labels from a gene expression metadata field "{field}".

Label A: "{label_a}"
Label B: "{label_b}"
{context_str}

Question: Are these two labels referring to the SAME biomedical concept?

RULES:
- "AML" and "Acute Myeloid Leukemia" = SAME (AML is the abbreviation)
- "Breast Cancer" and "Breast Carcinoma" = SAME (synonyms)
- "HSV" and "HIV" = DIFFERENT (completely different viruses)
- "Control" and "Normal" = SAME in this context
- "Lung" and "Lung Cancer" = DIFFERENT (tissue vs disease)
- "MCF-7" and "MCF7" = SAME (formatting variant)
- "Cell Line: HeLa" and "HeLa" = SAME

Respond with EXACTLY one line in this format:
SAME|canonical_label|brief_reason
or
DIFFERENT|brief_reason

The canonical_label should be the most complete, informative form.
"""
        raw = _ollama_post(prompt, timeout=60)
        if not raw:
            return False, "", "LLM no response"

        line = raw.strip().split('\n')[0].strip()
        if line.upper().startswith('SAME'):
            parts = line.split('|')
            if len(parts) >= 3:
                return True, parts[1].strip(), parts[2].strip()
            elif len(parts) >= 2:
                return True, parts[1].strip(), "LLM confirmed same"
            else:
                return True, label_a if len(label_a) > len(label_b) else label_b, "LLM said same"
        return False, "", line

    def scan_and_propose(self, df, fields=None, progress_fn=None):
        """Scan labels, find candidates, ask LLM, return proposals.

        Args:
            df: DataFrame with label columns
            fields: list of columns to curate (default: Condition, Tissue, Treatment)
            progress_fn: callback(done, total, msg)

        Returns:
            dict: {field: [(from_label, to_label, reason, from_count, to_count), ...]}
        """
        fields = fields or [f for f in self._CURATE_FIELDS if f in df.columns]
        self.merge_proposals = {}
        total_calls = 0
        done_calls = 0

        # Count candidates per field
        all_candidates = {}
        for field in fields:
            vals = df[field].fillna('Not Specified').astype(str).str.strip()
            real = [v for v in vals if v.lower() not in
                    ('not specified', 'n/a', 'unknown', 'nan', '')]
            if not real:
                continue
            counter = Counter(real)
            candidates = self._find_candidates(counter)
            if candidates:
                all_candidates[field] = (counter, candidates)
                total_calls += len(candidates)

        if total_calls == 0:
            self.log("[Curator] No candidate pairs found — labels are already clean.")
            return {}

        self.log(f"[Curator] Found {total_calls} candidate pairs across "
                 f"{len(all_candidates)} fields. Asking LLM...")

        for field, (counter, candidates) in all_candidates.items():
            proposals = []
            self.log(f"[Curator] {field}: {len(candidates)} pairs to check")

            for la, lb, ca, cb in candidates:
                if progress_fn:
                    progress_fn(done_calls, total_calls,
                                f"{field}: checking '{la[:25]}' vs '{lb[:25]}'")

                same, canonical, reason = self._ask_llm(la, lb, field, counter)
                done_calls += 1

                if same:
                    # Merge less common → more common (or shorter → canonical)
                    if canonical and canonical not in (la, lb):
                        # LLM proposed a different canonical — use it if it matches one
                        can_norm = canonical.lower().strip()
                        if can_norm == la.lower().strip():
                            canonical = la
                        elif can_norm == lb.lower().strip():
                            canonical = lb
                        else:
                            canonical = la if ca >= cb else lb

                    if not canonical:
                        canonical = la if ca >= cb else lb

                    from_label = lb if canonical == la else la
                    from_count = cb if canonical == la else ca
                    to_count = ca if canonical == la else cb

                    proposals.append((from_label, canonical, reason, from_count, to_count))
                    self.log(f"  MERGE: '{from_label}' ({from_count}) → "
                             f"'{canonical}' ({to_count}) — {reason}")
                else:
                    self.log(f"  KEEP:  '{la}' ≠ '{lb}' — {reason}")

            if proposals:
                self.merge_proposals[field] = proposals

        total_merges = sum(len(v) for v in self.merge_proposals.values())
        self.log(f"[Curator] Done. {total_merges} merges proposed across "
                 f"{len(self.merge_proposals)} fields.")
        return self.merge_proposals

    @staticmethod
    def apply_merges(df, merge_proposals, log_func=None):
        """Apply proposed merges to a DataFrame.

        Args:
            df: DataFrame to modify
            merge_proposals: dict from scan_and_propose()
            log_func: logging function

        Returns: modified DataFrame
        """
        _log = log_func or print
        result = df.copy()
        total_changed = 0

        for field, proposals in merge_proposals.items():
            if field not in result.columns:
                continue
            for from_label, to_label, reason, from_count, to_count in proposals:
                mask = result[field].astype(str).str.strip() == from_label
                n = mask.sum()
                if n > 0:
                    result.loc[mask, field] = to_label
                    total_changed += n
                    _log(f"[Curator] {field}: '{from_label}' → '{to_label}' ({n} samples)")

        _log(f"[Curator] Applied {total_changed} label changes.")
        return result


def harmonize_labels(df, log_func=None):
    """Full label harmonization pipeline for LLM-extracted labels.

    For Condition columns (full pipeline inspired by merging_labels_faster):
      Phase 1: Regex pre-clean (negation → noise strip → synonyms → hierarchy)
      Phase 2: Frequency cutoff (drop rare labels < MIN_FREQ_CUTOFF)
      Phase 3: Category classification (rule-based + optional zero-shot)
      Phase 4: Semantic clustering with barriers (if sentence-transformers available)
               Falls back to string-overlap merging if ML unavailable.
      Phase 5: Canonical naming (hierarchy-based: Disease > Drug > Stimulus > Control)
      Phase 6: Stage-2 name merge (near-duplicate cluster names)

    For Tissue/Treatment: simple synonym dictionary lookup.

    Returns DataFrame with harmonized labels (_raw columns preserve originals).
    """
    _log = log_func or print
    result = df.copy()

    for col in list(result.columns):
        if col in ('GSM', 'gsm', 'series_id', 'gpl', '_platform'):
            continue
        if result[col].dtype.kind not in ('O', 'U', 'S'):
            continue

        col_lower = col.lower()

        # ── CONDITION columns: full clustering pipeline ──
        if col_lower in ('condition', 'disease', 'diagnosis', 'group', 'phenotype'):
            raw_col = col + "_raw"
            result[raw_col] = result[col].copy()

            unique_vals = [str(v) for v in result[col].dropna().unique()
                           if str(v).lower() not in ('not specified', 'parse error', 'nan')]
            if not unique_vals:
                result.drop(columns=[raw_col], inplace=True)
                continue

            val_counts = result[col].value_counts().to_dict()

            # ── Phase 1: Regex pre-clean → concept map ──
            _log(f"[Harmonize] {col}: Phase 1 — Pre-cleaning {len(unique_vals)} unique labels...")
            val_to_concept = {}
            concept_map = {}  # concept → [original_labels]
            for v in unique_vals:
                concept = _clean_condition_label(v)
                val_to_concept[v] = concept
                concept_map.setdefault(concept, []).append(v)

            concepts = list(concept_map.keys())
            neg_count = sum(1 for v in unique_vals if _detect_negation(str(v)))
            _log(f"[Harmonize]   Pre-clean: {len(unique_vals)} labels → {len(concepts)} concepts")
            if neg_count > 0:
                _log(f"[Harmonize]   Negation-aware: {neg_count} labels → Control")

            # ── Phase 2: Frequency cutoff (adaptive) ──
            # For small datasets (<100 samples), use cutoff=1 (no filtering)
            # For medium datasets (100-1000), use cutoff=2
            # For large datasets (>1000), use _MIN_FREQ_CUTOFF (default 5)
            n_samples = len(result)
            if n_samples < 100:
                freq_cutoff = 1
            elif n_samples < 1000:
                freq_cutoff = 2
            else:
                freq_cutoff = _MIN_FREQ_CUTOFF
            excluded_labels = set()
            for v in unique_vals:
                if val_counts.get(v, 0) < freq_cutoff:
                    excluded_labels.add(v)
            active_vals = [v for v in unique_vals if v not in excluded_labels]
            if excluded_labels:
                _log(f"[Harmonize]   Frequency cutoff (< {freq_cutoff}): "
                     f"{len(unique_vals)} → {len(active_vals)} labels "
                     f"({len(excluded_labels)} rare excluded)")
            # Rebuild concept map for active labels only
            active_concept_map = {}
            for v in active_vals:
                c = val_to_concept[v]
                active_concept_map.setdefault(c, []).append(v)
            active_concepts = list(active_concept_map.keys())

            # ── Phase 3: Category classification ──
            _log(f"[Harmonize] {col}: Phase 3 — Category classification ({len(active_concepts)} concepts)...")
            concept_categories = {}
            unknown_concepts = []
            for concept in active_concepts:
                cat = _classify_concept_fast(concept)
                if cat != 'unknown':
                    concept_categories[concept] = cat
                else:
                    unknown_concepts.append(concept)

            from collections import Counter as _Counter
            cat_dist = dict(_Counter(concept_categories.values()))
            _log(f"[Harmonize]   Rule-based: {len(concept_categories)} classified, "
                 f"{len(unknown_concepts)} unknown")
            _log(f"[Harmonize]   Categories: {cat_dist}")

            # Optional zero-shot for unknowns
            if unknown_concepts:
                zs_results = _try_zeroshot_classify(unknown_concepts, log_func=_log)
                for concept, cat in zs_results.items():
                    concept_categories[concept] = cat
                cat_dist = dict(_Counter(concept_categories.values()))
                _log(f"[Harmonize]   After zero-shot: {cat_dist}")

            # ── Phase 4: Semantic clustering ──
            _log(f"[Harmonize] {col}: Phase 4 — Semantic clustering...")
            embedding_model, device_str = _try_load_embedding_model(log_func=_log)

            final_map = {}
            used_semantic = False

            if embedding_model is not None:
                # Separate control from non-control
                non_control = [c for c in active_concepts
                               if concept_categories.get(c) != 'control']
                control_concepts = [c for c in active_concepts
                                     if concept_categories.get(c) == 'control']

                _log(f"[Harmonize]   Control: {len(control_concepts)} | "
                     f"Non-control to cluster: {len(non_control)}")

                # Control → "Control"
                for concept in control_concepts:
                    for label in active_concept_map.get(concept, []):
                        final_map[label] = "Control"

                # Cluster non-control concepts
                if non_control:
                    raw_clusters = _semantic_cluster_concepts(
                        non_control, embedding_model, concept_categories,
                        device_str=device_str, log_func=_log)

                    if raw_clusters is not None:
                        used_semantic = True
                        for cluster_id, concept_members in raw_clusters.items():
                            canonical = _pick_canonical_name(
                                concept_members, concept_categories,
                                active_concept_map, val_counts)
                            for concept in concept_members:
                                for label in active_concept_map.get(concept, []):
                                    final_map[label] = canonical

                # Catch any unmapped active labels
                for label in active_vals:
                    if label not in final_map:
                        concept = val_to_concept.get(label, "control")
                        final_map[label] = ("Control" if concept == "control"
                                             else _clean_cluster_name(_titlecase_concept(concept)))

                # ── Phase 6: Stage-2 name merge (near-duplicate cluster names) ──
                if used_semantic:
                    n_before = len(set(final_map.values()))
                    _log(f"[Harmonize] {col}: Phase 6 — Stage-2 name merge ({n_before} cluster names)...")
                    final_map = _merge_similar_names(final_map, embedding_model,
                                                      device_str=device_str, log_func=_log)
                    n_after = len(set(final_map.values()))
                    _log(f"[Harmonize]   Clusters: {n_before} → {n_after}")

                # Clean up embedding model memory
                del embedding_model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # ── Fallback: string-overlap merge (if no ML available) ──
            if not used_semantic:
                _log(f"[Harmonize]   Fallback: rule-based concept grouping + string-overlap merge")
                for concept, members in active_concept_map.items():
                    if concept == 'control':
                        canonical = 'Control'
                    else:
                        canonical = _clean_cluster_name(_titlecase_concept(concept))
                    for label in members:
                        final_map[label] = canonical

            # Map excluded (rare) labels: try to match to a clustered label
            if excluded_labels:
                clustered_names = set(final_map.values())
                for v in excluded_labels:
                    concept = val_to_concept.get(v, str(v).lower())
                    if concept == 'control':
                        final_map[v] = 'Control'
                    else:
                        cleaned = _clean_cluster_name(_titlecase_concept(concept))
                        if cleaned in clustered_names:
                            final_map[v] = cleaned
                        else:
                            # Try substring match against clustered names
                            matched = False
                            v_id = _get_disease_identity(str(v))
                            for cn in clustered_names:
                                cn_id = _get_disease_identity(cn)
                                if v_id and cn_id and v_id != cn_id:
                                    continue
                                if (cleaned.lower() in cn.lower() or
                                        cn.lower() in cleaned.lower()):
                                    final_map[v] = cn
                                    matched = True
                                    break
                            if not matched:
                                final_map[v] = cleaned

            # ── Apply mapping ──
            result[col] = result[col].apply(
                lambda x: final_map.get(str(x), str(x)) if pd.notna(x) else x)

            # ── Merge rare variants via string overlap (post-clustering cleanup) ──
            if not used_semantic:
                counts = result[col].value_counts()
                rare_thr = max(3, len(result) * 0.005)
                rare_vals = counts[counts < rare_thr].index.tolist()
                common_vals = counts[counts >= rare_thr].index.tolist()
                if rare_vals and common_vals:
                    merge_map = {}
                    for rv in rare_vals:
                        rv_id = _get_disease_identity(str(rv))
                        best, best_sc = None, 0
                        for cv in common_vals:
                            cv_id = _get_disease_identity(str(cv))
                            if rv_id and cv_id and rv_id != cv_id:
                                continue
                            rv_l, cv_l = str(rv).lower(), str(cv).lower()
                            if rv_l in cv_l or cv_l in rv_l:
                                sc = len(set(rv_l.split()) & set(cv_l.split()))
                                if sc > best_sc:
                                    best_sc, best = sc, cv
                        if best and best_sc > 0:
                            merge_map[rv] = best
                    if merge_map:
                        result[col] = result[col].replace(merge_map)

            # ── Log changes ──
            total_changes = (result[col] != result[raw_col]).sum()
            unique_before = result[raw_col].nunique()
            unique_after = result[col].nunique()

            mode_str = "semantic clustering" if used_semantic else "rule-based"
            _log(f"[Harmonize] {col} DONE ({mode_str}): {total_changes:,} values mapped, "
                 f"unique: {unique_before} → {unique_after}")
            if total_changes > 0:
                changed = result.loc[result[col] != result[raw_col], [raw_col, col]].drop_duplicates()
                for _, row in changed.head(15).iterrows():
                    _log(f"  '{row[raw_col]}' → '{row[col]}'")
            if total_changes == 0:
                result.drop(columns=[raw_col], inplace=True)

        # ── TISSUE columns: synonym dictionary ──
        elif col_lower in ('tissue', 'cell_type', 'organ'):
            _apply_synonym_dict(result, col, _TISSUE_SYNONYMS, _log)

        # ── TREATMENT columns: synonym dictionary ──
        elif col_lower in ('treatment', 'drug', 'stimulus', 'perturbation'):
            _apply_synonym_dict(result, col, _TREATMENT_SYNONYMS, _log)

    return result



def _apply_synonym_dict(df, col, syn_dict, log_func):
    """Simple synonym dictionary lookup for a column."""
    raw_col = col + "_raw"
    df[raw_col] = df[col].copy()
    sorted_keys = sorted(syn_dict.keys(), key=len, reverse=True)

    def _map_val(val):
        if pd.isna(val):
            return val
        vl = str(val).strip().lower()
        if vl in syn_dict:
            return syn_dict[vl]
        for key in sorted_keys:
            if key in vl:
                return syn_dict[key]
        return str(val).strip()

    df[col] = df[col].apply(_map_val)
    total = (df[col] != df[raw_col]).sum()
    if total > 0:
        ub = df[raw_col].nunique()
        ua = df[col].nunique()
        log_func(f"[Harmonize] {col}: {total:,} values mapped, unique: {ub} -> {ua}")
        changed = df.loc[df[col] != df[raw_col], [raw_col, col]].drop_duplicates()
        for _, row in changed.head(8).iterrows():
            log_func(f"  '{row[raw_col]}' → '{row[col]}'")
    else:
        df.drop(columns=[raw_col], inplace=True)


# ── Simple synonym dictionaries for Tissue / Treatment ──

_CONDITION_SYNONYMS = {
    'alzheimer': 'Alzheimer Disease', 'alzheimers': 'Alzheimer Disease',
    "alzheimer's": 'Alzheimer Disease', "alzheimer's disease": 'Alzheimer Disease',
    'alzheimer disease': 'Alzheimer Disease', 'ad': 'Alzheimer Disease',
    'parkinson': 'Parkinson Disease', "parkinson's": 'Parkinson Disease',
    'parkinson disease': 'Parkinson Disease', 'pd': 'Parkinson Disease',
    'control': 'Control', 'ctrl': 'Control', 'normal': 'Control',
    'healthy': 'Control', 'healthy control': 'Control',
    'healthy donor': 'Control', 'non-diseased': 'Control',
    'unaffected': 'Control', 'negative control': 'Control',
    'wild type': 'Control', 'wildtype': 'Control', 'wt': 'Control',
    # Specific cancers — NEVER collapse to generic "Cancer"
    'aml': 'Acute Myeloid Leukemia', 'acute myeloid leukemia': 'Acute Myeloid Leukemia',
    'cll': 'Chronic Lymphocytic Leukemia', 'all': 'Acute Lymphoblastic Leukemia',
    'cml': 'Chronic Myeloid Leukemia',
    'breast cancer': 'Breast Cancer', 'breast carcinoma': 'Breast Cancer',
    'lung cancer': 'Lung Cancer', 'nsclc': 'Non-Small Cell Lung Cancer',
    'sclc': 'Small Cell Lung Cancer', 'lung adenocarcinoma': 'Lung Adenocarcinoma',
    'liver cancer': 'Liver Cancer', 'hcc': 'Hepatocellular Carcinoma',
    'hepatocellular carcinoma': 'Hepatocellular Carcinoma',
    'colorectal cancer': 'Colorectal Cancer', 'colon cancer': 'Colorectal Cancer',
    'crc': 'Colorectal Cancer',
    'prostate cancer': 'Prostate Cancer', 'prostate carcinoma': 'Prostate Cancer',
    'pancreatic cancer': 'Pancreatic Cancer', 'pdac': 'Pancreatic Ductal Adenocarcinoma',
    'ovarian cancer': 'Ovarian Cancer', 'ovarian carcinoma': 'Ovarian Cancer',
    'melanoma': 'Melanoma', 'glioblastoma': 'Glioblastoma', 'gbm': 'Glioblastoma',
    'glioma': 'Glioma', 'neuroblastoma': 'Neuroblastoma',
    'multiple myeloma': 'Multiple Myeloma', 'myeloma': 'Multiple Myeloma',
    'lymphoma': 'Lymphoma', 'dlbcl': 'Diffuse Large B-Cell Lymphoma',
    'renal cell carcinoma': 'Renal Cell Carcinoma', 'rcc': 'Renal Cell Carcinoma',
    'bladder cancer': 'Bladder Cancer', 'gastric cancer': 'Gastric Cancer',
    'head and neck cancer': 'Head and Neck Cancer', 'hnscc': 'Head and Neck Squamous Cell Carcinoma',
    'thyroid cancer': 'Thyroid Cancer', 'endometrial cancer': 'Endometrial Cancer',
    # Other diseases
    'type 2 diabetes': 'Type 2 Diabetes', 't2d': 'Type 2 Diabetes',
    'type 1 diabetes': 'Type 1 Diabetes', 't1d': 'Type 1 Diabetes',
    'rheumatoid arthritis': 'Rheumatoid Arthritis', 'ra': 'Rheumatoid Arthritis',
    'systemic lupus erythematosus': 'Systemic Lupus Erythematosus', 'sle': 'Systemic Lupus Erythematosus',
    'multiple sclerosis': 'Multiple Sclerosis', 'ms': 'Multiple Sclerosis',
    'copd': 'COPD', 'asthma': 'Asthma', 'obesity': 'Obesity',
    'schizophrenia': 'Schizophrenia', 'depression': 'Major Depressive Disorder',
    'major depressive disorder': 'Major Depressive Disorder', 'mdd': 'Major Depressive Disorder',
    'bipolar disorder': 'Bipolar Disorder', 'autism': 'Autism Spectrum Disorder',
    'hiv': 'HIV Infection', 'tuberculosis': 'Tuberculosis', 'tb': 'Tuberculosis',
    'sepsis': 'Sepsis', 'psoriasis': 'Psoriasis', 'ibd': 'Inflammatory Bowel Disease',
    "crohn's disease": 'Crohn Disease', 'crohn disease': 'Crohn Disease',
    'ulcerative colitis': 'Ulcerative Colitis',
    'not specified': 'Not Specified', 'unknown': 'Not Specified',
    'na': 'Not Specified', 'n/a': 'Not Specified', 'none': 'Not Specified',
}

_TISSUE_SYNONYMS = {
    'blood': 'Whole Blood', 'whole blood': 'Whole Blood',
    'peripheral blood': 'Peripheral Blood', 'pb': 'Peripheral Blood',
    'peripheral blood mononuclear cells': 'PBMC', 'pbmc': 'PBMC', 'pbmcs': 'PBMC',
    'plasma': 'Plasma', 'serum': 'Serum', 'bone marrow': 'Bone Marrow',
    'brain': 'Brain', 'cerebral cortex': 'Cerebral Cortex', 'cortex': 'Cerebral Cortex',
    'hippocampus': 'Hippocampus', 'frontal cortex': 'Frontal Cortex',
    'prefrontal cortex': 'Prefrontal Cortex', 'cerebellum': 'Cerebellum',
    'liver': 'Liver', 'kidney': 'Kidney', 'heart': 'Heart',
    'lung': 'Lung', 'pancreas': 'Pancreas', 'spleen': 'Spleen',
    'colon': 'Colon', 'intestine': 'Intestine', 'skin': 'Skin',
    'muscle': 'Skeletal Muscle', 'skeletal muscle': 'Skeletal Muscle',
    'adipose': 'Adipose Tissue', 'adipose tissue': 'Adipose Tissue', 'fat': 'Adipose Tissue',
    'breast': 'Breast', 'prostate': 'Prostate', 'ovary': 'Ovary', 'thyroid': 'Thyroid',
    # Cell types → "Cell Type: X" format
    'fibroblast': 'Cell Type: Fibroblast', 'fibroblasts': 'Cell Type: Fibroblast',
    'macrophage': 'Cell Type: Macrophage', 'macrophages': 'Cell Type: Macrophage',
    't cell': 'Cell Type: T Cell', 't cells': 'Cell Type: T Cell',
    'b cell': 'Cell Type: B Cell', 'b cells': 'Cell Type: B Cell',
    'monocyte': 'Cell Type: Monocyte', 'monocytes': 'Cell Type: Monocyte',
    'neutrophil': 'Cell Type: Neutrophil', 'neutrophils': 'Cell Type: Neutrophil',
    'cardiomyocyte': 'Cell Type: Cardiomyocyte', 'cardiomyocytes': 'Cell Type: Cardiomyocyte',
    'hepatocyte': 'Cell Type: Hepatocyte', 'hepatocytes': 'Cell Type: Hepatocyte',
    'neuron': 'Cell Type: Neuron', 'neurons': 'Cell Type: Neuron',
    'astrocyte': 'Cell Type: Astrocyte', 'astrocytes': 'Cell Type: Astrocyte',
    'cd4+ t cell': 'Cell Type: CD4+ T Cell', 'cd8+ t cell': 'Cell Type: CD8+ T Cell',
    'dendritic cell': 'Cell Type: Dendritic Cell', 'dendritic cells': 'Cell Type: Dendritic Cell',
    'nk cell': 'Cell Type: NK Cell', 'nk cells': 'Cell Type: NK Cell',
    'endothelial cell': 'Cell Type: Endothelial Cell', 'endothelial cells': 'Cell Type: Endothelial Cell',
    'epithelial cell': 'Cell Type: Epithelial Cell', 'epithelial cells': 'Cell Type: Epithelial Cell',
    # Cell lines → "Cell Line: X" format
    'hela': 'Cell Line: HeLa', 'hek293': 'Cell Line: HEK293', 'hek-293': 'Cell Line: HEK293',
    'a549': 'Cell Line: A549', 'mcf7': 'Cell Line: MCF-7', 'mcf-7': 'Cell Line: MCF-7',
    'huvec': 'Cell Line: HUVEC', 'jurkat': 'Cell Line: Jurkat',
    'k562': 'Cell Line: K562', 'u937': 'Cell Line: U937',
    'thp1': 'Cell Line: THP-1', 'thp-1': 'Cell Line: THP-1',
    'sh-sy5y': 'Cell Line: SH-SY5Y', 'shsy5y': 'Cell Line: SH-SY5Y',
    'hepg2': 'Cell Line: HepG2', 'caco-2': 'Cell Line: Caco-2', 'caco2': 'Cell Line: Caco-2',
    'pc-3': 'Cell Line: PC-3', 'lncap': 'Cell Line: LNCaP',
    'mda-mb-231': 'Cell Line: MDA-MB-231', 'mcf10a': 'Cell Line: MCF10A',
    'hl60': 'Cell Line: HL-60', 'hl-60': 'Cell Line: HL-60',
    'raw264.7': 'Cell Line: RAW264.7', 'raw 264.7': 'Cell Line: RAW264.7',
    'hct116': 'Cell Line: HCT116', 'hct-116': 'Cell Line: HCT116',
    'sw480': 'Cell Line: SW480', 'ht29': 'Cell Line: HT-29', 'ht-29': 'Cell Line: HT-29',
    'u2os': 'Cell Line: U2OS', 'imr90': 'Cell Line: IMR-90', 'wi38': 'Cell Line: WI-38',
    'not specified': 'Not Specified', 'unknown': 'Not Specified',
    'na': 'Not Specified', 'n/a': 'Not Specified', 'none': 'Not Specified',
}

_TREATMENT_SYNONYMS = {
    'lps': 'LPS', 'lipopolysaccharide': 'LPS',
    'vehicle': 'Vehicle', 'dmso': 'DMSO', 'pbs': 'PBS',
    'untreated': 'Untreated', 'none': 'Untreated', 'no treatment': 'Untreated',
    'control': 'Untreated', 'mock': 'Mock',
    'dexamethasone': 'Dexamethasone', 'dex': 'Dexamethasone',
    'ifn-gamma': 'IFN-gamma', 'interferon gamma': 'IFN-gamma',
    'ifn-alpha': 'IFN-alpha', 'interferon alpha': 'IFN-alpha',
    'tnf': 'TNF-alpha', 'tnf-alpha': 'TNF-alpha', 'tnfa': 'TNF-alpha',
    'il-6': 'IL-6', 'il6': 'IL-6', 'il-1beta': 'IL-1beta',
    'tgf-beta': 'TGF-beta', 'tgfb': 'TGF-beta',
    'chemotherapy': 'Chemotherapy', 'radiation': 'Radiation', 'irradiation': 'Radiation',
    'not specified': 'Not Specified', 'unknown': 'Not Specified',
    'na': 'Not Specified', 'n/a': 'Not Specified',
}

_HARMONIZATION_DICTS = {
    'condition': _CONDITION_SYNONYMS,
    'tissue': _TISSUE_SYNONYMS,
    'treatment': _TREATMENT_SYNONYMS,
}


class InteractiveSubsetAnalyzerWindow(tk.Toplevel):
    """
    Provides:
    1. Data Table with keyword filtering
    2. PCA (Dimensionality Reduction) visualization
    3. Density Peak Clustering (DPC) analysis
    All scatter plots have click-to-inspect: click a sample to see its metadata.
    """
    def __init__(self, parent, app_ref, step2_dataframe: pd.DataFrame, source_description: str):
        super().__init__(parent)
        self.app_ref = app_ref
        self.df = step2_dataframe.copy()
        self.filtered_df = self.df.copy()
        self.title(f"Interactive Analyzer: {source_description}")
        self.geometry("1200x800")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1200x800+{(_sw-1200)//2}+{(_sh-800)//2}")
            self.minsize(600, 500)
        except Exception: pass

        self.numeric_cols = []
        self.current_grouping_cols = []
        self._detect_numeric_columns()

        # Auto-detect grouping column
        priority = ['Condition', 'Tissue', 'Treatment', 'Group', 'Cluster', 'series_id']
        for p in priority:
            matches = [c for c in self.df.columns if p.lower() in c.lower() and self.df[c].dtype == 'object']
            if matches:
                self.current_grouping_cols = [matches[0]]
                break
        if not self.current_grouping_cols:
            obj_cols = [c for c in self.df.columns if self.df[c].dtype == 'object' and c != 'GSM']
            if obj_cols:
                self.current_grouping_cols = [obj_cols[0]]

        self._setup_ui()
        self._populate_table_initial()

        if self.current_grouping_cols:
            self._run_pca_analysis()
            self._run_dpc_analysis()

    def _detect_numeric_columns(self):
        excluded = {'gsm', 'sample', 'id', 'index'}
        self.numeric_cols = [c for c in self.df.columns
                             if pd.api.types.is_numeric_dtype(self.df[c])
                             and c.lower() not in excluded]

    def _setup_ui(self):
        main_nb = ttk.Notebook(self)
        main_nb.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 0: Data Table
        tab_data = ttk.Frame(main_nb)
        main_nb.add(tab_data, text="Data Table")

        ctrl = ttk.Frame(tab_data)
        ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(ctrl, text="Keyword filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        e = ttk.Entry(ctrl, textvariable=self.filter_var, width=30)
        e.pack(side=tk.LEFT, padx=4)
        e.bind('<Return>', lambda ev: self._apply_keyword_filter())
        ttk.Button(ctrl, text="Filter", command=self._apply_keyword_filter).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Clear", command=self._clear_filter).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="   Group by:").pack(side=tk.LEFT, padx=(15, 2))
        self.grp_label = ttk.Label(ctrl,
            text=self.current_grouping_cols[0] if self.current_grouping_cols else "(click column header)",
            foreground="green" if self.current_grouping_cols else "red",
            font=('Segoe UI', 9, 'bold'))
        self.grp_label.pack(side=tk.LEFT)

        self.table_tree = ttk.Treeview(tab_data, show='headings', height=20)
        vsb = ttk.Scrollbar(tab_data, orient='vertical', command=self.table_tree.yview)
        hsb = ttk.Scrollbar(tab_data, orient='horizontal', command=self.table_tree.xview)
        self.table_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.table_tree.pack(fill=tk.BOTH, expand=True)
        self.table_tree.bind('<Button-1>', self._on_column_header_click)

        # Tab 1: PCA
        self.tab_pca = ttk.Frame(main_nb)
        main_nb.add(self.tab_pca, text="PCA (Dim. Reduction)")
        pca_ctrl = ttk.Frame(self.tab_pca)
        pca_ctrl.pack(fill=tk.X, pady=5)
        ttk.Button(pca_ctrl, text="Run PCA Analysis", command=self._run_pca_analysis).pack(side=tk.LEFT, padx=5)
        self.pca_canvas_frame = ScrollableCanvasFrame(self.tab_pca)
        self.pca_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Tab 2: DPC
        self.tab_dpc = ttk.Frame(main_nb)
        main_nb.add(self.tab_dpc, text="Density Peak Clustering")
        dpc_ctrl = ttk.Frame(self.tab_dpc)
        dpc_ctrl.pack(fill=tk.X, pady=5)
        ttk.Button(dpc_ctrl, text="Run DPC Analysis", command=self._run_dpc_analysis).pack(side=tk.LEFT, padx=5)
        self.dpc_canvas_frame = ScrollableCanvasFrame(self.tab_dpc)
        self.dpc_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _populate_table_initial(self):
        cols = list(self.df.columns)
        self.table_tree['columns'] = cols
        for c in cols:
            self.table_tree.heading(c, text=c)
            self.table_tree.column(c, width=100, anchor=tk.CENTER)
        self._refresh_table_data()

    def _set_active_grouping(self, cols):
        self.current_grouping_cols = cols
        self.grp_label.config(text=cols[0] if cols else "N/A",
                              foreground="green" if cols else "red")

    def _apply_keyword_filter(self):
        kw = self.filter_var.get().strip().lower()
        if not kw:
            self.filtered_df = self.df.copy()
        else:
            def check_row(row):
                for val in row:
                    if kw in str(val).lower():
                        return True
                return False
            mask = self.df.apply(check_row, axis=1)
            self.filtered_df = self.df[mask].copy()
        self._refresh_table_data()

    def _clear_filter(self):
        self.filter_var.set("")
        self.filtered_df = self.df.copy()
        self._refresh_table_data()

    def _refresh_table_data(self):
        self.table_tree.delete(*self.table_tree.get_children())
        for _, row in self.filtered_df.head(2000).iterrows():
            self.table_tree.insert('', tk.END, values=list(row))

    def _on_column_header_click(self, event):
        region = self.table_tree.identify_region(event.x, event.y)
        if region == 'heading':
            col = self.table_tree.identify_column(event.x)
            col_idx = int(col.replace('#', '')) - 1
            cols = list(self.df.columns)
            if 0 <= col_idx < len(cols):
                clicked_col = cols[col_idx]
                if self.df[clicked_col].dtype == 'object':
                    self._set_active_grouping([clicked_col])
                    self._run_pca_analysis()
                    self._run_dpc_analysis()

    # ─── PCA ───────────────────────────────────────────────────────
    def _run_pca_analysis(self):
        for w in self.pca_canvas_frame.scrollable_frame.winfo_children():
            w.destroy()

        if len(self.numeric_cols) < 2:
            ttk.Label(self.pca_canvas_frame.scrollable_frame,
                      text="Need at least 2 numeric (gene) columns for PCA.\n"
                           "Currently loaded data appears to be 1D or metadata only.",
                      foreground="red").pack(pady=20)
            return

        if not self.current_grouping_cols:
            ttk.Label(self.pca_canvas_frame.scrollable_frame,
                      text="Click a column header in Data Table to set grouping.",
                      foreground="orange").pack(pady=20)
            return

        X = self.df[self.numeric_cols].fillna(0)
        y_col = self.current_grouping_cols[0]
        y = self.df[y_col].astype(str)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X_scaled)

            self._plot_scatter(
                self.pca_canvas_frame.scrollable_frame,
                x=coords[:, 0], y=coords[:, 1], labels=y,
                title=f"PCA Plot (Colored by {y_col})",
                xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        except ImportError:
            ttk.Label(self.pca_canvas_frame.scrollable_frame,
                      text="scikit-learn is required for PCA.\npip install scikit-learn",
                      foreground="red").pack(pady=20)
        except Exception as e:
            ttk.Label(self.pca_canvas_frame.scrollable_frame,
                      text=f"PCA Error: {e}").pack()

    # ─── DPC ───────────────────────────────────────────────────────
    def _run_dpc_analysis(self):
        for w in self.dpc_canvas_frame.scrollable_frame.winfo_children():
            w.destroy()

        if not self.numeric_cols:
            ttk.Label(self.dpc_canvas_frame.scrollable_frame,
                      text="No numeric data for clustering.").pack()
            return

        if not self.current_grouping_cols:
            ttk.Label(self.dpc_canvas_frame.scrollable_frame,
                      text="Click a column header in Data Table to set grouping.",
                      foreground="orange").pack(pady=20)
            return

        X = self.df[self.numeric_cols].fillna(0).values
        y_col = self.current_grouping_cols[0]
        y = self.df[y_col].astype(str)

        try:
            from scipy.spatial.distance import pdist, squareform

            dists = squareform(pdist(X))
            dc = np.percentile(dists, 2)
            if dc == 0:
                dc = 1e-5
            rho = np.sum(np.exp(-(dists / dc) ** 2), axis=1) - 1
            delta = np.zeros(len(X))
            ord_rho = np.argsort(-rho)

            for i, idx in enumerate(ord_rho):
                if i == 0:
                    delta[idx] = dists[idx, :].max()
                else:
                    delta[idx] = dists[idx, ord_rho[:i]].min()

            self._plot_scatter(
                self.dpc_canvas_frame.scrollable_frame,
                x=rho, y=delta, labels=y,
                title=f"DPC Decision Graph (Colored by {y_col})",
                xlabel="Local Density (rho)",
                ylabel="Min Dist to Higher Density (delta)")
        except ImportError:
            ttk.Label(self.dpc_canvas_frame.scrollable_frame,
                      text="scipy is required for DPC.\npip install scipy",
                      foreground="red").pack(pady=20)
        except Exception as e:
            ttk.Label(self.dpc_canvas_frame.scrollable_frame,
                      text=f"DPC Error: {e}").pack()

    # ─── Unified Scatter with Click-to-Inspect ─────────────────────
    def _plot_scatter(self, parent_frame, x, y, labels, title, xlabel, ylabel):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
        import seaborn as sns

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        labels = np.asarray(labels, dtype=str)

        plot_frame = ttk.Frame(parent_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(bottom=0.14)

        unique_grps = sorted(set(labels))
        palette = sns.color_palette("husl", len(unique_grps))
        color_dict = {g: palette[i] for i, g in enumerate(unique_grps)}

        for grp in unique_grps:
            mask = labels == grp
            ax.scatter(x[mask], y[mask], c=[color_dict[grp]], label=grp,
                       s=50, alpha=0.7, edgecolors='k', lw=0.3,
                       picker=True, pickradius=5)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if len(unique_grps) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        fig.tight_layout()

        ax.text(0.5, -0.08,
                'Click points to inspect  •  Shift+click multi-select  •  Double-click clear',
                transform=ax.transAxes, fontsize=7.5, ha='center',
                color='#777', style='italic')

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Info table for clicked points ──
        info_frame = ttk.LabelFrame(parent_frame,
                                     text="Selected Samples (click points above)")
        info_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        # Show all metadata columns for clicked samples
        meta_cols = [c for c in self.df.columns if c not in self.numeric_cols or c == 'GSM']
        if len(meta_cols) > 12:
            meta_cols = meta_cols[:12]
        if not meta_cols:
            meta_cols = list(self.df.columns[:8])

        info_tree = ttk.Treeview(info_frame, columns=meta_cols,
                                  show='headings', height=5)
        for c in meta_cols:
            info_tree.heading(c, text=c)
            info_tree.column(c, width=100, anchor=tk.CENTER)
        isb = ttk.Scrollbar(info_frame, orient='vertical', command=info_tree.yview)
        info_tree.config(yscrollcommand=isb.set)
        isb.pack(side=tk.RIGHT, fill=tk.Y)
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
            d = ((x - mx) / xr) ** 2 + ((y - my) / yr) ** 2
            ci = int(np.argmin(d))
            lbl = labels[ci]
            key = f"{ci}"

            shift = bool(event.mouseevent.key == 'shift')

            if key in sel_anns:
                sel_anns[key].remove()
                del sel_anns[key]
                for item in info_tree.get_children():
                    info_tree.delete(item)
                # Re-add remaining
                for k2 in sel_anns:
                    i2 = int(k2)
                    row = self.df.iloc[i2]
                    vals = [str(row.get(c, '')) for c in meta_cols]
                    info_tree.insert('', tk.END, values=vals)
            else:
                if not shift:
                    for ann in sel_anns.values():
                        ann.remove()
                    sel_anns.clear()
                    for item in info_tree.get_children():
                        info_tree.delete(item)

                gsm = self.df.iloc[ci].get('GSM', f'#{ci}')
                ann = ax.annotate(
                    f"{gsm}\n{lbl}", (x[ci], y[ci]),
                    fontsize=7, fontweight='bold',
                    ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#FFEB3B', edgecolor='#333',
                              alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=0.8))
                sel_anns[key] = ann

                row = self.df.iloc[ci]
                vals = [str(row.get(c, '')) for c in meta_cols]
                info_tree.insert('', tk.END, values=vals)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', _on_pick)

class ScrollableCanvasFrame(ttk.Frame):
    """Scrollable frame using Canvas."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
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

class CompareDistributionsWindow(tk.Toplevel):
    def __init__(self, parent, app_ref, skip_autoload=False):
        super().__init__(parent)
        self.parent = parent
        self.app_ref = app_ref
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
        ttk.Label(header, text="Distribution Comparison", font=("Segoe UI", 16, "bold"), foreground="#333").pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="Ready", foreground="grey")
        self.status_label.pack(side=tk.RIGHT)

        # Splitter
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Controls
        control_pane = ttk.Frame(paned, width=350, padding=(0, 0, 5, 0))
        paned.add(control_pane, weight=0) 
        
        lf_data = ttk.LabelFrame(control_pane, text="1. Load Data")
        lf_data.pack(fill=tk.X, pady=5)
        ttk.Button(lf_data, text="Load Metadata (CSV)", command=self._load_labeled_file).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(lf_data, text="Clear Data", command=self._clear_user_data).pack(fill=tk.X, padx=5, pady=2)
        
        lf_group_info = ttk.Frame(control_pane)
        lf_group_info.pack(fill=tk.X, pady=2)
        ttk.Label(lf_group_info, text="Active Grouping:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_grouping = ttk.Label(lf_group_info, text="[None]", foreground="blue")
        self.lbl_grouping.pack(side=tk.LEFT, padx=5)
        
        lf_group = ttk.LabelFrame(control_pane, text="2. Select Groups")
        lf_group.pack(fill=tk.BOTH, expand=True, pady=5)
        self.loaded_files_listbox = tk.Listbox(lf_group, height=8, selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(lf_group, orient="vertical", command=self.loaded_files_listbox.yview)
        self.loaded_files_listbox.config(yscrollcommand=sb.set)
        self.loaded_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        lf_config = ttk.LabelFrame(control_pane, text="3. Analysis Parameters")
        lf_config.pack(fill=tk.X, pady=5)
        
        ttk.Label(lf_config, text="Target Gene(s):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=5)
        self.gene_entry = ttk.Entry(lf_config, font=("Segoe UI", 10)); self.gene_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(lf_config, text="Comparison Scope:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=5, pady=(5,0))
        self.comparison_mode = tk.StringVar(value="groups_only")
        scope_frame = ttk.Frame(lf_config)
        scope_frame.pack(fill=tk.X, padx=5, pady=3)
        self._cmp_scope_btns = {}
        for txt, val, clr in [("Groups Only", "groups_only", "#1565C0"),
                               ("vs Gene", "vs_gene", "#2E7D32"),
                               ("vs Platform", "vs_platform", "#E65100")]:
            btn = tk.Button(scope_frame, text=f" {txt} ",
                            font=('Segoe UI', 9, 'bold'), padx=8, pady=3, cursor='hand2',
                            relief=tk.SUNKEN if val == "groups_only" else tk.RAISED, bd=2,
                            bg=clr if val == "groups_only" else '#E0E0E0',
                            fg='white' if val == "groups_only" else '#333',
                            command=lambda v=val: self._set_cmp_scope(v))
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
            self._cmp_scope_btns[val] = (btn, clr)
        
        ttk.Label(lf_config, text="Platform (for BG):").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.platform_vars = {}
        plat_frame = ttk.Frame(lf_config)
        plat_frame.pack(fill=tk.X, padx=5)
        if hasattr(self.app_ref, 'gpl_datasets'):
            for p in sorted(self.app_ref.gpl_datasets.keys()):
                v = tk.BooleanVar()
                ttk.Checkbutton(plat_frame, text=p, variable=v).pack(anchor=tk.W)
                self.platform_vars[p] = v

        tk.Button(control_pane, text="  RUN ANALYSIS  ", command=self._run_analysis,
                  bg='#C62828', fg='white', font=('Segoe UI', 12, 'bold'),
                  padx=20, pady=8, cursor='hand2', relief=tk.RAISED, bd=2).pack(fill=tk.X, pady=10)

        tk.Button(control_pane, text="Multi-Label Query", command=self._add_query_group,
                  bg='#5C6BC0', fg='white', font=('Segoe UI', 10, 'bold'),
                  padx=15, pady=4, cursor='hand2', relief=tk.RAISED, bd=2).pack(fill=tk.X, pady=(0, 5))
        
        # Active Analysis View — scrollable
        nav_outer = ttk.LabelFrame(control_pane, text="Active Analysis View")
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
        
        # Tab 0: Data — GSE Experiment Browser
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
        tk.Label(dist_ctrl_frame, text="Plot:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        self._dist_mode_btns = {}
        self._dist_mode = tk.StringVar(value="both")
        for val, label, clr in [("density", " Density ", "#5C6BC0"),
                                  ("rug", " Rug ", "#7986CB"),
                                  ("both", " Both ", "#3F51B5")]:
            btn = tk.Button(dist_ctrl_frame, text=label,
                            font=('Segoe UI', 9, 'bold'), padx=10, pady=2, cursor='hand2',
                            relief=tk.SUNKEN if val == "both" else tk.RAISED, bd=2,
                            bg=clr if val == "both" else '#E0E0E0',
                            fg='white' if val == "both" else '#333',
                            command=lambda v=val: self._set_dist_mode(v))
            btn.pack(side=tk.LEFT, padx=2)
            self._dist_mode_btns[val] = (btn, clr)
        
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
        tk.Label(matrix_ctrl, text="Metric:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(2, 4))
        self.metric_combo = ttk.Combobox(matrix_ctrl, textvariable=self.metric_var, values=["Wasserstein", "Euclidean", "Jensen-Shannon"], state="readonly", font=('Segoe UI', 10))
        self.metric_combo.pack(side=tk.LEFT)
        
        tk.Label(matrix_ctrl, text="  Reference:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(10, 4))
        self.dist_ref_var = tk.StringVar(value="pairwise")
        self._ref_btns = {}
        for val, label, clr in [("pairwise", " Pairwise ", "#1565C0"),
                                  ("gene_mean", " Gene Mean ", "#2E7D32"),
                                  ("platform_mean", " Platform Mean ", "#E65100"),
                                  ("peaks", " Peaks (Mode) ", "#7B1FA2")]:
            btn = tk.Button(matrix_ctrl, text=label,
                            font=('Segoe UI', 9, 'bold'), padx=8, pady=2, cursor='hand2',
                            relief=tk.SUNKEN if val == "pairwise" else tk.RAISED, bd=2,
                            bg=clr if val == "pairwise" else '#E0E0E0',
                            fg='white' if val == "pairwise" else '#333',
                            command=lambda v=val: self._set_ref_mode(v))
            btn.pack(side=tk.LEFT, padx=2)
            self._ref_btns[val] = (btn, clr)

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
        self._dimred_btns = {}
        for val, clr in [("PCA", "#1565C0"), ("t-SNE", "#E65100")]:
            btn = tk.Button(dimred_ctrl, text=f" {val} ",
                            font=('Segoe UI', 9, 'bold'), padx=10, pady=2, cursor='hand2',
                            relief=tk.SUNKEN if val == "PCA" else tk.RAISED, bd=2,
                            bg=clr if val == "PCA" else '#E0E0E0',
                            fg='white' if val == "PCA" else '#333',
                            command=lambda v=val: self._set_dimred(v))
            btn.pack(side=tk.LEFT, padx=2)
            self._dimred_btns[val] = (btn, clr)
        ttk.Label(dimred_ctrl, text="Perplexity:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(15, 2))
        self.tsne_perplexity = tk.IntVar(value=30)
        ttk.Entry(dimred_ctrl, textvariable=self.tsne_perplexity, width=5).pack(side=tk.LEFT)
        tk.Button(dimred_ctrl, text="Run", command=self._run_dimred,
                  bg="#43A047", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.LEFT, padx=8)

        self.dimred_scroll_frame = ScrollableCanvasFrame(self.tab_dimred)
        self.dimred_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Tab 5: Clustering (DPC / K-means / DBSCAN)
        self.tab_cluster = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cluster, text="Clustering")

        cluster_ctrl = ttk.Frame(self.tab_cluster)
        cluster_ctrl.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(cluster_ctrl, text="Method:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=4)
        self.cluster_method = tk.StringVar(value="K-Means")
        self._cluster_btns = {}
        for val, clr in [("K-Means", "#C62828"), ("DBSCAN", "#1565C0"), ("DPC", "#7B1FA2")]:
            btn = tk.Button(cluster_ctrl, text=f" {val} ",
                            font=('Segoe UI', 9, 'bold'), padx=8, pady=2, cursor='hand2',
                            relief=tk.SUNKEN if val == "K-Means" else tk.RAISED, bd=2,
                            bg=clr if val == "K-Means" else '#E0E0E0',
                            fg='white' if val == "K-Means" else '#333',
                            command=lambda v=val: self._set_cluster_method(v))
            btn.pack(side=tk.LEFT, padx=2)
            self._cluster_btns[val] = (btn, clr)

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
        tk.Button(param_frame, text="Run", command=self._run_clustering,
                  bg="#43A047", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.RIGHT, padx=8)

        self.cluster_scroll_frame = ScrollableCanvasFrame(self.tab_cluster)
        self.cluster_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.btn_report = ttk.Button(main_frame, text="Generate Report (Folder)", command=self._generate_report, state=tk.DISABLED)
        self.btn_report.pack(side=tk.BOTTOM, anchor=tk.E, pady=5)

    def _set_ref_mode(self, val):
        """Toggle matrix reference mode with button appearance."""
        self.dist_ref_var.set(val)
        for v, (btn, clr) in self._ref_btns.items():
            if v == val:
                btn.config(relief=tk.SUNKEN, bg=clr, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
        self._calculate_matrix()

    def _set_dimred(self, val):
        """Toggle dimensionality reduction method."""
        self.dimred_method.set(val)
        for v, (btn, clr) in self._dimred_btns.items():
            if v == val:
                btn.config(relief=tk.SUNKEN, bg=clr, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')

    def _set_cluster_method(self, val):
        """Toggle clustering method."""
        self.cluster_method.set(val)
        for v, (btn, clr) in self._cluster_btns.items():
            if v == val:
                btn.config(relief=tk.SUNKEN, bg=clr, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')

    def _add_query_group(self):
        """Open Multi-Label Query Builder and add matching samples as a new group."""
        if self.full_dataset.empty:
            messagebox.showinfo("No Data", "Load data first (labels or metadata).", parent=self)
            return
        result = MultiLabelQueryDialog.open(self, self.full_dataset,
                                             title="Multi-Label Query — Create Custom Group")
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
        for v, (btn, clr) in self._cmp_scope_btns.items():
            if v == val:
                btn.config(relief=tk.SUNKEN, bg=clr, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')

    def _set_dist_mode(self, val):
        """Toggle distribution plot mode with button appearance."""
        self._dist_mode.set(val)
        for v, (btn, clr) in self._dist_mode_btns.items():
            if v == val:
                btn.config(relief=tk.SUNKEN, bg=clr, fg='white')
            else:
                btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='#333')
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
            # Remove rows that aren't valid GSM IDs
            valid_mask = df['GSM'].str.match(r'^GSM\d+$', na=False)
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                df = df[valid_mask].reset_index(drop=True)
                self.enqueue_log(f"[Load] Removed {n_invalid} rows with invalid GSM IDs")

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
            self.enqueue_log(f"[Load] Loaded {len(df):,} samples from {Path(fp).name}")

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
        self.enqueue_log(f"[Load] Resolving metadata for {total:,} GSMs from GEOmetadb...")

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
                self.enqueue_log(f"[Load] DB query error: {e}")

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

        self.enqueue_log(
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
                            self.enqueue_log(f"[Load] Auto-loading {gpl} expression data...")
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
                text=f"{len(df):,} samples loaded — no series_id column for experiment grouping")
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
            text=f"{len(df):,} samples across {n_gse:,} experiments — "
                 f"double-click to view samples")

    def _on_gse_tree_dblclick(self, event):
        """Handle double-click on GSE experiment — show all its GSMs."""
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
        top.title(f"Experiment {gse_id} — {len(subset)} samples")
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

        for _, row in subset.iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, '')) for c in cols])

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

        if self.full_dataset.empty: return
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
       
            fig, ax = plt.subplots(figsize=(calc_width, calc_height))
            
            # 2. Color Setup
            import seaborn as sns
            # Palette for active groups (Bright/Distinct)
            palette = sns.color_palette("husl", n_groups)
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
                        kde = gaussian_kde(data)
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
                        kde = gaussian_kde(data)
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
                        kde = gaussian_kde(data)
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
            ax.set_xlabel("Expression")
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
            plt.subplots_adjust(left=0.08, right=0.70, top=0.9, bottom=0.1)
            
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
                        kde = gaussian_kde(v)
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
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(df_mat, annot=True, fmt=".2f", cmap="viridis", ax=ax)
            
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

        fig, ax = plt.subplots(figsize=(max(9, n*0.5), 6))
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
                'Click points to inspect  •  Shift+click to multi-select  •  Double-click to clear',
                transform=ax.transAxes, fontsize=7.5, ha='center',
                color='#777777', style='italic')

        h = [mpatches.Patch(color=cdict[k], label=k) for k in keys]
        leg = ax.legend(handles=h, title="Groups", loc='upper left', bbox_to_anchor=(1.01, 1))
        self._make_legend_interactive(fig, leg, cdict, {}, keys)
        fig.tight_layout()

        # Rectangle selector for multi-sample selection
        X_2d = np.column_stack([all_x, all_y])
        self._add_rect_selector(fig, ax, X_2d, all_groups, all_gsms.tolist(), sf)

        self._embed_plot(fig, sf, "sep")

        # ── Info table for clicked points ──
        info_frame = ttk.LabelFrame(sf, text="Selected Points (click above)")
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
            # Stratified subsample — keep proportions per group
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
                title = f"PCA — {len(X):,} samples, {len(keys)} groups"
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
                title = f"t-SNE (perplexity={perp}) — {len(X):,} samples, {len(keys)} groups"

            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
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
            plt.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.08)

            # Rectangle selector for multi-sample selection
            self._add_rect_selector(fig, ax, X_2d, L, all_gsms, sf)

            self._embed_plot(fig, sf, "dimred")
            self.status_label.config(text=f"{method} complete — {len(X):,} samples plotted")

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

    def _add_rect_selector(self, fig, ax, X_2d, labels, gsms, scroll_frame):
        """Add a rectangle drag selector to scatter plots for multi-sample selection."""
        from matplotlib.widgets import RectangleSelector

        info_label = ttk.Label(scroll_frame, text="Drag a rectangle on the plot to select samples",
                               font=('Segoe UI', 9, 'italic'), foreground='#888')
        info_label.pack(pady=2)

        def on_select(eclick, erelease):
            x1, y1 = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
            x2, y2 = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)
            mask = (X_2d[:, 0] >= x1) & (X_2d[:, 0] <= x2) & (X_2d[:, 1] >= y1) & (X_2d[:, 1] <= y2)
            n_sel = mask.sum()
            if n_sel == 0:
                info_label.config(text="No samples in selection")
                return
            sel_labels = labels[mask]
            sel_gsms = [gsms[i] for i in np.where(mask)[0]]

            # Count per group
            from collections import Counter
            counts = Counter(sel_labels)
            summary = ", ".join(f"{k}: {v}" for k, v in counts.most_common(5))
            info_label.config(text=f"Selected {n_sel} samples — {summary}")

            # Show popup with details
            top = tk.Toplevel(self)
            top.title(f"Selected {n_sel} Samples")
            top.geometry("700x500")
            try:
                _sw, _sh = top.winfo_screenwidth(), top.winfo_screenheight()
                top.geometry(f"700x500+{(_sw-700)//2}+{(_sh-500)//2}")
            except: pass

            # Summary
            ttk.Label(top, text=f"Selected {n_sel} samples from rectangle region",
                      font=('Segoe UI', 11, 'bold')).pack(padx=10, pady=(10, 5))

            # Table
            tv_frame = ttk.Frame(top)
            tv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            cols = ("GSM", "Group")
            tree = ttk.Treeview(tv_frame, columns=cols, show="headings", height=15)
            vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            tree.heading("GSM", text="GSM")
            tree.heading("Group", text="Group")
            tree.column("GSM", width=200)
            tree.column("Group", width=300)
            for g, l in zip(sel_gsms, sel_labels):
                tree.insert("", tk.END, values=(g, l))

            # Save button
            btn_f = ttk.Frame(top, padding=5)
            btn_f.pack(fill=tk.X)
            def _save():
                path = filedialog.asksaveasfilename(defaultextension=".csv",
                    filetypes=[("CSV", "*.csv")], parent=top)
                if path:
                    pd.DataFrame({'GSM': sel_gsms, 'Group': sel_labels}).to_csv(path, index=False)
                    messagebox.showinfo("Saved", f"Saved {n_sel} samples", parent=top)
            ttk.Button(btn_f, text="Save to CSV", command=_save).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_f, text="Close", command=top.destroy).pack(side=tk.RIGHT, padx=5)

        rs = RectangleSelector(ax, on_select, useblit=True,
                                button=[1], interactive=True,
                                props=dict(facecolor='yellow', alpha=0.2, edgecolor='red', linewidth=2))
        # Keep reference to prevent GC
        if not hasattr(self, '_rect_selectors'):
            self._rect_selectors = []
        self._rect_selectors.append(rs)

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
        """DBSCAN clustering — finds arbitrary-shaped clusters, detects noise."""
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

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
            ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))

            # Right: contingency — true labels vs DBSCAN clusters
            uL = np.unique(L)
            non_noise = [c for c in unique_clusters if c != -1]
            all_c = non_noise + ([-1] if n_noise > 0 else [])
            contingency = np.zeros((len(uL), len(all_c)), dtype=int)
            for i, label in enumerate(uL):
                for j, c in enumerate(all_c):
                    contingency[i, j] = ((L == label) & (cluster_ids == c)).sum()

            im = ax2.imshow(contingency, aspect='auto', cmap='YlOrRd')
            c_labels = [f"C{c}" if c != -1 else "Noise" for c in all_c]
            ax2.set_xticks(range(len(all_c)))
            ax2.set_xticklabels(c_labels)
            ax2.set_yticks(range(len(uL)))
            ax2.set_yticklabels([str(l)[:25] for l in uL], fontsize=8)
            ax2.set_xlabel("DBSCAN Cluster")
            ax2.set_title("True Labels vs DBSCAN Clusters", fontsize=11, weight='bold')
            fig.colorbar(im, ax=ax2, label="Count")
            for i in range(len(uL)):
                for j in range(len(all_c)):
                    if contingency[i, j] > 0:
                        ax2.text(j, i, str(contingency[i, j]), ha='center', va='center',
                                fontsize=8, color='white' if contingency[i, j] > contingency.max()/2 else 'black')

            plt.subplots_adjust(left=0.06, right=0.85, top=0.90, bottom=0.08, wspace=0.35)
            X_2d_db = np.column_stack([all_scatter_x, all_scatter_y])
            self._add_rect_selector(fig, ax1, X_2d_db, np.array(all_scatter_labels), all_scatter_gsms, sf)
            self._embed_plot(fig, sf, "dbscan")

            # Stats
            stats_frame = ttk.LabelFrame(sf, text="DBSCAN Results", padding=8)
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

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
            ax1.set_title(f"K-Means (K={k_val}) — Silhouette={sil:.3f}, ARI={ari:.3f}",
                         fontsize=11, weight='bold')
            ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))

            # ── Plot 2: True labels vs clusters (contingency) ──
            uL = np.unique(L)
            contingency = np.zeros((len(uL), k_val), dtype=int)
            for i, label in enumerate(uL):
                for ci in range(k_val):
                    contingency[i, ci] = ((L == label) & (cluster_ids == ci)).sum()

            im = ax2.imshow(contingency, aspect='auto', cmap='YlOrRd')
            ax2.set_xticks(range(k_val))
            ax2.set_xticklabels([f"C{i}" for i in range(k_val)])
            ax2.set_yticks(range(len(uL)))
            ax2.set_yticklabels([str(l)[:25] for l in uL], fontsize=8)
            ax2.set_xlabel("Cluster")
            ax2.set_title("True Labels vs K-Means Clusters", fontsize=11, weight='bold')
            fig.colorbar(im, ax=ax2, label="Count")

            # Annotate cells
            for i in range(len(uL)):
                for j in range(k_val):
                    if contingency[i, j] > 0:
                        ax2.text(j, i, str(contingency[i, j]), ha='center', va='center',
                                fontsize=8, color='white' if contingency[i, j] > contingency.max()/2 else 'black')

            plt.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.08, wspace=0.35)
            X_2d_km = np.column_stack([all_scatter_x, all_scatter_y])
            self._add_rect_selector(fig, ax1, X_2d_km, np.array(all_scatter_labels), all_scatter_gsms, sf)
            self._embed_plot(fig, sf, "kmeans")

            # Stats summary
            stats_frame = ttk.LabelFrame(sf, text="K-Means Results", padding=8)
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

            fig, ax = plt.subplots(figsize=(max(10, len(self.current_data_map) * 0.4), 7))
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
            plt.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.08)
            if dpc_scatter_x:
                X_2d_dpc = np.column_stack([dpc_scatter_x, dpc_scatter_y])
                self._add_rect_selector(fig, ax, X_2d_dpc, np.array(dpc_scatter_labels), dpc_scatter_gsms, sf)
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

    def _embed_plot(self, fig, parent, key):
        if key in self.canvases: self.canvases[key].get_tk_widget().destroy(); self.toolbars[key].destroy(); plt.close(self.figs[key])
        c = self.FigureCanvasTkAgg(fig, master=parent); c.draw()
        c.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        t = self.NavigationToolbar2Tk(c, parent); t.update(); t.pack(side=tk.BOTTOM, fill=tk.X)
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
                    self.lbl_grouping.config(text=self.grouping_column, foreground="green")
                    self._update_group_list() 
                    self.loaded_files_listbox.select_set(0, tk.END)
                else:
                    self.lbl_grouping.config(text="[Click a Header to Group]", foreground="red")
    
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

        gsm_frame = ttk.LabelFrame(master, text="1. Select Sample ID Column (GSM)")
        gsm_frame.pack(padx=10, pady=5, fill=tk.X)
        self.gsm_var = tk.StringVar()
        self.gsm_combo = ttk.Combobox(gsm_frame, textvariable=self.gsm_var,
                                       values=self.columns, state="readonly", width=40)
        for col in self.columns:
            if 'gsm' in col.lower():
                self.gsm_var.set(col)
                break
        self.gsm_combo.pack(padx=5, pady=5)

        label_frame = ttk.LabelFrame(master, text="2. Select Grouping/Label Column(s)")
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
    ('human', 'mouse'): 'Human vs Mouse — ~16,000 one-to-one orthologs',
    ('human', 'rat'): 'Human vs Rat — ~15,000 one-to-one orthologs',
    ('mouse', 'rat'): 'Mouse vs Rat — ~17,000 one-to-one orthologs',
    ('human', 'zebrafish'): 'Human vs Zebrafish — ~10,000 orthologs',
    ('human', 'drosophila'): 'Human vs Drosophila — ~4,000 orthologs',
    ('human', 'c. elegans'): 'Human vs C. elegans — ~3,000 orthologs',
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


def _median_center_normalize(expr_dict):
    """Per-platform median centering. expr_dict = {plat: Series per gene}.
    Returns dict with same keys, values shifted so each platform median = global median."""
    all_vals = np.concatenate([s.values for s in expr_dict.values()])
    global_median = np.nanmedian(all_vals)
    result = {}
    for plat, series in expr_dict.items():
        plat_median = np.nanmedian(series.values)
        result[plat] = series - plat_median + global_median
    return result


def _quantile_normalize_cross(expr_dict):
    """Cross-platform quantile normalization for a single gene.
    Forces all platforms to share the same rank-distribution template.

    Algorithm:
        1. Sort each platform's values independently
        2. Resample sorted arrays to a common length (min across platforms)
        3. Compute rank-mean template (average of sorted values at each rank)
        4. Map each platform's values back using their rank positions

    This aligns the entire distribution shape, not just the median.
    More aggressive than median centering — use when distributions have
    different shapes (variance, skewness) across platforms.
    """
    if not expr_dict or len(expr_dict) < 2:
        return dict(expr_dict)

    # Clean NaN for rank computation
    clean = {p: s.dropna() for p, s in expr_dict.items()}
    if any(len(s) < 3 for s in clean.values()):
        # Too few values — fall back to median centering
        return _median_center_normalize(expr_dict)

    # Sort each platform
    sorted_arrs = {p: np.sort(s.values) for p, s in clean.items()}

    # Resample to common length (smallest platform)
    min_len = min(len(v) for v in sorted_arrs.values())
    resampled = {}
    for p, arr in sorted_arrs.items():
        indices = np.linspace(0, len(arr) - 1, min_len).astype(int)
        resampled[p] = arr[indices]

    # Compute rank-mean template
    stacked = np.vstack(list(resampled.values()))
    rank_means = stacked.mean(axis=0)

    # Map back: for each platform, assign rank-mean values based on original ranks
    result = {}
    for plat, series in expr_dict.items():
        vals = series.values.copy()
        non_nan_mask = ~np.isnan(vals)
        non_nan_vals = vals[non_nan_mask]

        if len(non_nan_vals) == 0:
            result[plat] = series.copy()
            continue

        # Compute ranks of non-NaN values
        ranks = np.argsort(np.argsort(non_nan_vals))
        # Scale ranks to [0, min_len-1]
        if len(non_nan_vals) > 1:
            scaled_ranks = (ranks / (len(non_nan_vals) - 1) * (min_len - 1)).astype(int)
        else:
            scaled_ranks = np.array([min_len // 2])
        scaled_ranks = np.clip(scaled_ranks, 0, min_len - 1)

        # Assign rank-mean values
        normalized = rank_means[scaled_ranks]
        vals[non_nan_mask] = normalized
        result[plat] = pd.Series(vals, index=series.index, name=series.name)

    return result


def _combat_correct(expr_df, batch_labels, bio_covariates=None):
    """ComBat batch correction with optional biological covariate protection.

    Args:
        expr_df:         genes x samples DataFrame
        batch_labels:    list of batch IDs (one per sample column)
        bio_covariates:  optional DataFrame (samples x covariates) with biological
                         variables to PROTECT from correction (e.g. Condition, Tissue).
                         When provided, ComBat removes only technical batch variance
                         while preserving biological group differences.

    Returns:
        (corrected_df, method_string)

    Falls back to median centering if pycombat unavailable or produces NaN.

    NOTE: ComBat requires multi-gene matrices (>= ~10 genes) for reliable empirical
    Bayes estimation. With fewer genes, it degenerates and may produce NaN.
    The caller (CrossPlatformAnalysisWindow) builds a multi-gene matrix for this reason.
    """
    # Validate input: need >= 2 batches for correction
    n_batches = len(set(batch_labels))
    if n_batches < 2:
        return expr_df.copy(), 'single_batch_noop'

    def _validate_combat_output(corrected, original):
        if corrected is None:
            return False
        nan_frac = np.isnan(corrected.values).sum() / corrected.size
        orig_nan_frac = np.isnan(original.values).sum() / original.size
        return nan_frac < orig_nan_frac + 0.1 and nan_frac < 0.5

    # Build covariate list(s) from biological labels for pycombat's mod parameter.
    # pycombat expects mod as a list (single covariate) or list of lists (multiple).
    # Each covariate list has one numeric/categorical entry per sample.
    mod = []
    if bio_covariates is not None and not bio_covariates.empty:
        try:
            if len(bio_covariates) == expr_df.shape[1]:
                for col_name in bio_covariates.columns:
                    # Convert categorical labels to numeric codes for pycombat
                    values = bio_covariates[col_name].astype(str).values
                    unique_vals = sorted(set(values))
                    val_to_code = {v: i for i, v in enumerate(unique_vals)}
                    coded = [val_to_code[v] for v in values]
                    mod.append(coded)
            else:
                mod = []
        except Exception:
            mod = []

    method_suffix = '_with_covariates' if mod else ''

    # Try pycombat
    for _try_import in range(2):
        try:
            if _try_import == 0:
                from pycombat import pycombat as _pycombat_fn
            else:
                from combat.pycombat import pycombat as _pycombat_fn
            if mod:
                corrected = _pycombat_fn(expr_df, batch_labels, mod=mod)
            else:
                corrected = _pycombat_fn(expr_df, batch_labels)
            if _validate_combat_output(corrected, expr_df):
                return corrected, f'combat{method_suffix}'
            else:
                break
        except ImportError:
            continue
        except Exception:
            # ComBat crashed — try without covariates as intermediate fallback
            if mod:
                try:
                    corrected = _pycombat_fn(expr_df, batch_labels)
                    if _validate_combat_output(corrected, expr_df):
                        return corrected, 'combat_no_covariates'
                except Exception:
                    pass
            break

    # Fallback: median centering per batch (robust, always works)
    corrected = expr_df.copy()
    global_median = np.nanmedian(expr_df.values)
    for batch in set(batch_labels):
        mask = [b == batch for b in batch_labels]
        batch_vals = corrected.loc[:, mask].values
        batch_median = np.nanmedian(batch_vals)
        if not np.isnan(batch_median) and not np.isnan(global_median):
            corrected.loc[:, mask] = batch_vals - batch_median + global_median
    return corrected, 'median_centering_fallback'


class CrossPlatformAnalysisWindow(tk.Toplevel):
    """Advanced multi-platform gene expression comparison with batch correction."""

    def __init__(self, parent, app_ref):
        super().__init__(parent)
        self.app = app_ref
        self.title("GeneVariate — Cross-Platform Analysis")
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
        # Header
        hdr = tk.Frame(self, bg="#1A237E", height=50)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Cross-Platform Gene Expression Analysis",
                 bg="#1A237E", fg="white", font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT, padx=15)
        tk.Label(hdr, text="Compare gene distributions, detect batch effects, find DE & conserved genes",
                 bg="#1A237E", fg="#90CAF9", font=('Segoe UI', 9, 'italic')).pack(side=tk.LEFT, padx=10)

        # ── Top config panel ──
        config_frame = ttk.LabelFrame(self, text="Analysis Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        # Platform selection
        plat_row = ttk.Frame(config_frame)
        plat_row.pack(fill=tk.X, pady=4)
        ttk.Label(plat_row, text="Platforms to compare:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Label(plat_row, text="(select 2 or more)",
                  font=('Segoe UI', 9, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=8)

        self._plat_checks_frame = ttk.Frame(config_frame)
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
                                         foreground='#1565C0')
        self._species_label.pack(fill=tk.X, pady=2)

        # Buttons
        btn_row = ttk.Frame(config_frame)
        btn_row.pack(fill=tk.X, pady=6)

        self._run_btn = tk.Button(btn_row, text="  Run Cross-Platform Analysis  ",
                                   command=self._start_analysis,
                                   bg="#1565C0", fg="white",
                                   font=('Segoe UI', 11, 'bold'),
                                   padx=20, pady=8, cursor="hand2",
                                   relief=tk.RAISED, bd=2)
        self._run_btn.pack(side=tk.LEFT, padx=5)

        self._export_btn = tk.Button(btn_row, text="  Export Report  ",
                                      command=self._export_full_report,
                                      bg="#2E7D32", fg="white",
                                      font=('Segoe UI', 10, 'bold'),
                                      padx=15, pady=6, cursor="hand2",
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
        for w in self._plat_checks_frame.winfo_children():
            w.destroy()
        self._plat_vars.clear()

        plats = sorted(self.app.gpl_datasets.keys())
        if not plats:
            ttk.Label(self._plat_checks_frame, text="No platforms loaded",
                      foreground='red').pack(side=tk.LEFT)
            return

        for plat in plats:
            var = tk.BooleanVar(value=True)
            self._plat_vars[plat] = var
            n_samples = len(self.app.gpl_datasets[plat])
            n_genes = len(self.app.gpl_gene_mappings.get(plat, {}))
            species = GPL_SPECIES.get(plat.split('_')[0], '?')
            cb = ttk.Checkbutton(self._plat_checks_frame,
                                  text=f"{plat} ({species}, {n_samples:,}s, {n_genes:,}g)",
                                  variable=var, command=self._on_plat_selection_change)
            cb.pack(side=tk.LEFT, padx=6)

        self._ref_combo['values'] = ['(auto)'] + plats
        self._ref_var.set('(auto)')
        self._on_plat_selection_change()

    def _on_plat_selection_change(self):
        """Update species comparison hint."""
        selected = [p for p, v in self._plat_vars.items() if v.get()]
        species_set = set()
        for p in selected:
            base = p.split('_')[0]
            sp = GPL_SPECIES.get(base, 'unknown')
            species_set.add(sp)

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
                text=f"Same-species comparison ({sp}) — direct gene symbol matching",
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

        thread = threading.Thread(target=self._run_analysis_thread,
                                   args=(selected,), daemon=True)
        thread.start()

    def _run_analysis_thread(self, selected_platforms):
        """Run full cross-platform analysis in background thread."""
        try:
            results = self._analyze_platforms(selected_platforms)
            self.after(0, lambda: self._display_results(results))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_analysis_error(str(e), tb))
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

    def _analyze_platforms(self, platforms):
        """Core analysis engine — returns comprehensive results dict."""
        from scipy.stats import ks_2samp, mannwhitneyu, spearmanr

        results = {
            'platforms': platforms,
            'reference': None,
            'batch_method': self._batch_var.get(),
            'pval_threshold': float(self._pval_var.get()),
            'delta_threshold': float(self._delta_var.get()),
        }

        # Determine reference
        ref = self._ref_var.get()
        if ref == '(auto)':
            ref = max(platforms,
                      key=lambda p: len(self.app.gpl_gene_mappings.get(p, {})))
        results['reference'] = ref

        self._update_progress(5, "Building gene inventories...")

        # ── 1. Gene Inventory ──────────────────────────────────────
        gene_sets = {}
        for plat in platforms:
            gmap = self.app.gpl_gene_mappings.get(plat, {})
            gene_sets[plat] = set(gmap.keys())

        all_genes = set()
        for gs in gene_sets.values():
            all_genes |= gs

        common_all = set.intersection(*gene_sets.values()) if gene_sets else set()
        pairwise_overlap = {}
        for i, a in enumerate(platforms):
            for b in platforms[i + 1:]:
                pairwise_overlap[(a, b)] = gene_sets[a] & gene_sets[b]

        unique_genes = {}
        for plat in platforms:
            others = set.union(*(gene_sets[p] for p in platforms if p != plat))
            unique_genes[plat] = gene_sets[plat] - others

        results['gene_sets'] = gene_sets
        results['all_genes'] = all_genes
        results['common_all'] = common_all
        results['pairwise_overlap'] = pairwise_overlap
        results['unique_genes'] = unique_genes

        self._update_progress(10, f"Analyzing {len(common_all):,} common genes across {len(platforms)} platforms...")
        self.app.enqueue_log(f"[XPlat] {len(common_all):,} common genes, "
                             f"{len(all_genes):,} total across {len(platforms)} platforms")

        # ── 2. Species detection ───────────────────────────────────
        species_map = {}
        for plat in platforms:
            base = plat.split('_')[0]
            species_map[plat] = GPL_SPECIES.get(base, 'unknown')
        results['species_map'] = species_map
        results['is_cross_species'] = len(set(species_map.values())) > 1

        # ── 3. Extract expression data for common genes ────────────
        self._update_progress(15, "Extracting expression data for common genes...")

        gene_expr = {}
        platform_medians = {}
        platform_means = {}
        platform_stds = {}

        for plat in platforms:
            df = self.app.gpl_datasets[plat]
            gmap = self.app.gpl_gene_mappings.get(plat, {})
            vals_all = []
            for gene in common_all:
                col = gmap.get(gene)
                if col and col in df.columns:
                    expr = pd.to_numeric(df[col], errors='coerce').dropna().values
                    if len(expr) > 0:
                        if gene not in gene_expr:
                            gene_expr[gene] = {}
                        gene_expr[gene][plat] = expr
                        vals_all.extend(expr.tolist())
            platform_medians[plat] = np.nanmedian(vals_all) if vals_all else 0
            platform_means[plat] = np.nanmean(vals_all) if vals_all else 0
            platform_stds[plat] = np.nanstd(vals_all) if vals_all else 0

        results['platform_medians'] = platform_medians
        results['platform_means'] = platform_means
        results['platform_stds'] = platform_stds

        # ── 4. Batch effect detection ──────────────────────────────
        self._update_progress(25, "Detecting batch effects...")

        batch_metrics = {}
        for plat in platforms:
            batch_metrics[plat] = {
                'median': platform_medians[plat],
                'mean': platform_means[plat],
                'std': platform_stds[plat],
                'median_shift_from_ref': platform_medians[plat] - platform_medians.get(ref, 0),
                'mean_shift_from_ref': platform_means[plat] - platform_means.get(ref, 0),
                'variance_ratio_to_ref': (platform_stds[plat] / max(platform_stds.get(ref, 1), 1e-10)) ** 2,
            }
        results['batch_metrics'] = batch_metrics

        median_vals = [platform_medians[p] for p in platforms]
        results['batch_effect_score'] = np.std(median_vals) / max(np.mean(list(platform_stds.values())), 1e-10)
        results['batch_effect_detected'] = results['batch_effect_score'] > 0.2

        # ── 5. Optional batch correction ───────────────────────────
        batch_method = self._batch_var.get()
        # Normalize the combo value for internal logic
        _batch_key = batch_method.replace(' (preserve biology)', '').strip()
        corrected_gene_expr = gene_expr

        if _batch_key != 'none' and len(gene_expr) > 0:
            self._update_progress(30, f"Applying batch correction ({batch_method})...")

            if _batch_key == 'median_centering':
                corrected_gene_expr = {}
                for gene, plat_data in gene_expr.items():
                    corrected = _median_center_normalize(
                        {p: pd.Series(v) for p, v in plat_data.items()})
                    corrected_gene_expr[gene] = {
                        p: s.values for p, s in corrected.items()}
                results['batch_correction_used'] = 'median_centering'

            elif _batch_key == 'quantile_normalization':
                self._update_progress(30, "Running cross-platform quantile normalization...")
                corrected_gene_expr = {}
                for gene, plat_data in gene_expr.items():
                    corrected = _quantile_normalize_cross(
                        {p: pd.Series(v) for p, v in plat_data.items()})
                    corrected_gene_expr[gene] = {
                        p: s.values for p, s in corrected.items()}
                results['batch_correction_used'] = 'quantile_normalization'

            elif _batch_key == 'combat':
                self._update_progress(30, "Running ComBat batch correction...")
                sample_genes = sorted(gene_expr.keys())[:5000]
                col_labels = []
                batch_labels = []
                for plat in platforms:
                    n = len(self.app.gpl_datasets[plat])
                    col_labels.extend([f"{plat}_{i}" for i in range(n)])
                    batch_labels.extend([plat] * n)

                # ── Build biological covariate matrix from ALL extracted labels ──
                # Auto-detect label columns: standard fields + user-created custom fields.
                # Columns NOT in this skip-set are assumed to be biological labels.
                _SKIP_COLS = {
                    'GSM', 'gsm', 'series_id', 'gpl', 'platform', '_platform',
                    'title', 'gsm_title', 'source_name', 'source_name_ch1',
                    'characteristics', 'characteristics_ch1', 'description',
                    'treatment_protocol', 'organism_ch1', 'geo_accession',
                    'Token_Match', 'Matched_Tokens',
                }
                bio_covariates = None
                _cov_rows = []
                _bio_cols_detected = set()
                _has_labels = False

                for plat in platforms:
                    df = self.app.gpl_datasets[plat]
                    n = len(df)
                    lbl_df = self.app.platform_labels.get(plat)

                    if lbl_df is not None and not lbl_df.empty:
                        # Auto-detect ALL label columns (standard + custom)
                        if not _bio_cols_detected:
                            _bio_cols_detected = {
                                c for c in lbl_df.columns
                                if c not in _SKIP_COLS
                                and not c.startswith('_')
                                and lbl_df[c].dtype == 'object'
                            }

                        # Find GSM column in both dataframes
                        gsm_col = next((c for c in df.columns if c.upper() == 'GSM'), None)
                        lbl_gsm = next((c for c in lbl_df.columns if c.upper() == 'GSM'), None)

                        if gsm_col and lbl_gsm:
                            lbl_indexed = lbl_df.drop_duplicates(lbl_gsm).set_index(lbl_gsm)
                            for _, row in df.iterrows():
                                gsm = str(row.get(gsm_col, '')).strip()
                                cov = {}
                                for bc in _bio_cols_detected:
                                    val = 'Unknown'
                                    if gsm in lbl_indexed.index and bc in lbl_indexed.columns:
                                        v = lbl_indexed.at[gsm, bc]
                                        if isinstance(v, pd.Series):
                                            v = v.iloc[0]
                                        v = str(v).strip()
                                        if v and v.lower() not in ('nan', 'none', 'not specified', ''):
                                            val = v
                                    cov[bc] = val
                                _cov_rows.append(cov)
                                _has_labels = True
                            continue

                    # No labels for this platform — fill with 'Unknown'
                    for _ in range(n):
                        _cov_rows.append({bc: 'Unknown' for bc in (_bio_cols_detected or {'_none'})})

                if _has_labels and _bio_cols_detected:
                    bio_covariates = pd.DataFrame(_cov_rows, index=col_labels)
                    # Drop '_none' placeholder if present
                    bio_covariates = bio_covariates.drop(columns=['_none'], errors='ignore')

                    # Remove columns with no variance (all same value)
                    for bc in list(bio_covariates.columns):
                        if bio_covariates[bc].nunique() <= 1:
                            bio_covariates = bio_covariates.drop(columns=[bc])
                    # Remove columns where >80% is 'Unknown' (too sparse to help)
                    for bc in list(bio_covariates.columns):
                        unknown_frac = (bio_covariates[bc] == 'Unknown').sum() / len(bio_covariates)
                        if unknown_frac > 0.8:
                            bio_covariates = bio_covariates.drop(columns=[bc])

                    if bio_covariates.empty:
                        bio_covariates = None
                        self._update_progress(30,
                            "ComBat: labels found but all uniform — "
                            "running without covariate protection")
                    else:
                        cov_names = list(bio_covariates.columns)
                        n_groups = sum(bio_covariates[c].nunique() for c in cov_names)
                        self._update_progress(30,
                            f"ComBat: protecting {len(cov_names)} label fields "
                            f"({n_groups} groups) from correction")
                        self.app.enqueue_log(
                            f"[XPlat] ComBat with biological covariates: "
                            f"{cov_names} ({n_groups} unique groups protected)")
                else:
                    self._update_progress(30,
                        "ComBat: no extracted labels found — "
                        "run LLM extraction first for biology-preserving correction")

                gene_rows = {}
                for gene in sample_genes:
                    row = []
                    for plat in platforms:
                        gmap = self.app.gpl_gene_mappings.get(plat, {})
                        col = gmap.get(gene)
                        df = self.app.gpl_datasets[plat]
                        if col and col in df.columns:
                            vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                        else:
                            vals = np.zeros(len(df))
                        row.extend(vals.tolist())
                    gene_rows[gene] = row

                expr_matrix = pd.DataFrame(gene_rows, index=col_labels).T
                self._update_progress(35,
                    f"ComBat: {expr_matrix.shape[0]} genes x {expr_matrix.shape[1]} samples"
                    f"{' (with covariates)' if bio_covariates is not None else ''}...")
                corrected_matrix, method_used = _combat_correct(
                    expr_matrix, batch_labels, bio_covariates=bio_covariates)

                # Validate ComBat output — check for NaN contamination
                nan_frac = np.isnan(corrected_matrix.values).sum() / max(1, corrected_matrix.size)
                if nan_frac > 0.5:
                    self._update_progress(35,
                        f"ComBat produced {nan_frac:.0%} NaN — falling back to median centering")
                    method_used = 'median_centering_fallback'
                    corrected_gene_expr = {}
                    for gene, plat_data in gene_expr.items():
                        corrected = _median_center_normalize(
                            {p: pd.Series(v) for p, v in plat_data.items()})
                        corrected_gene_expr[gene] = {
                            p: s.values for p, s in corrected.items()}
                else:
                    corrected_gene_expr = {}
                    for gene in sample_genes:
                        if gene in corrected_matrix.index:
                            corrected_gene_expr[gene] = {}
                            offset = 0
                            for plat in platforms:
                                n = len(self.app.gpl_datasets[plat])
                                vals = corrected_matrix.loc[gene].iloc[offset:offset + n].values
                                corrected_gene_expr[gene][plat] = vals.astype(float)
                                offset += n
                for gene in gene_expr:
                    if gene not in corrected_gene_expr:
                        corrected_gene_expr[gene] = gene_expr[gene]
                results['batch_correction_used'] = method_used
        else:
            results['batch_correction_used'] = 'none'

        results['corrected_gene_expr'] = corrected_gene_expr

        # ── 6. Per-gene cross-platform statistical comparison ──────
        self._update_progress(45, "Running per-gene statistical comparisons...")

        de_genes = []
        conserved_genes = []
        gene_stats = []

        total_genes = len(corrected_gene_expr)
        pval_thresh = results['pval_threshold']
        delta_thresh = results['delta_threshold']

        for gi, (gene, plat_data) in enumerate(corrected_gene_expr.items()):
            if gi % 500 == 0 and gi > 0:
                pct = 45 + int(45 * gi / max(total_genes, 1))
                self._update_progress(pct, f"Analyzing gene {gi:,}/{total_genes:,}...")

            if len(plat_data) < 2:
                continue

            active_plats = [p for p in platforms if p in plat_data and len(plat_data[p]) > 2]
            if len(active_plats) < 2:
                continue

            ref_vals = plat_data.get(ref)
            if ref_vals is None or len(ref_vals) < 3:
                ref_vals = plat_data[active_plats[0]]
                local_ref = active_plats[0]
            else:
                local_ref = ref

            ref_mean = float(np.nanmean(ref_vals))
            ref_std = float(np.nanstd(ref_vals))

            max_delta = 0.0
            platform_details = {}
            all_pvals = []
            is_de = False

            for other_plat in active_plats:
                if other_plat == local_ref:
                    platform_details[other_plat] = {
                        'mean': ref_mean, 'std': ref_std,
                        'n': len(ref_vals), 'pval': 1.0, 'delta_mean': 0.0,
                        'ks_stat': 0.0, 'effect_size': 0.0,
                    }
                    continue

                other_vals = plat_data[other_plat]
                other_mean = float(np.nanmean(other_vals))
                other_std = float(np.nanstd(other_vals))
                delta_mean = other_mean - ref_mean

                try:
                    _, pval_w = mannwhitneyu(ref_vals, other_vals, alternative='two-sided')
                except Exception:
                    pval_w = 1.0

                try:
                    ks_stat, ks_pval = ks_2samp(ref_vals, other_vals)
                except Exception:
                    ks_stat, ks_pval = 0.0, 1.0

                pooled_std = np.sqrt((ref_std ** 2 + other_std ** 2) / 2)
                effect_size = abs(delta_mean) / max(pooled_std, 1e-10)
                var_ratio = (other_std / max(ref_std, 1e-10)) ** 2

                platform_details[other_plat] = {
                    'mean': other_mean, 'std': other_std,
                    'n': len(other_vals), 'pval': pval_w,
                    'delta_mean': delta_mean,
                    'ks_stat': ks_stat, 'ks_pval': ks_pval,
                    'effect_size': effect_size,
                    'var_ratio': var_ratio,
                }

                all_pvals.append(pval_w)
                max_delta = max(max_delta, abs(delta_mean))

                if pval_w < pval_thresh and abs(delta_mean) > delta_thresh:
                    is_de = True

            entry = {
                'gene': gene, 'ref_platform': local_ref,
                'ref_mean': ref_mean, 'ref_std': ref_std,
                'n_platforms': len(active_plats),
                'max_abs_delta': max_delta,
                'min_pval': min(all_pvals) if all_pvals else 1.0,
                'max_pval': max(all_pvals) if all_pvals else 1.0,
                'platform_details': platform_details,
                'is_de': is_de,
            }
            gene_stats.append(entry)

        # BH correction on min_pvals
        if gene_stats:
            raw_pvals = np.array([g['min_pval'] for g in gene_stats])
            n_tests = len(raw_pvals)
            sorted_idx = np.argsort(raw_pvals)
            adj = np.ones(n_tests)
            for rank_i, idx in enumerate(sorted_idx):
                adj[idx] = raw_pvals[idx] * n_tests / (rank_i + 1)
            # Enforce monotonicity
            for i in range(n_tests - 2, -1, -1):
                adj[sorted_idx[i]] = min(adj[sorted_idx[i]],
                                          adj[sorted_idx[i + 1]] if i + 1 < n_tests else 1.0)
            adj = np.clip(adj, 0, 1)
            for i, gs in enumerate(gene_stats):
                gs['adj_pval'] = float(adj[i])

            de_genes = [g for g in gene_stats
                        if g['adj_pval'] < pval_thresh and g['max_abs_delta'] > delta_thresh]
            conserved_genes = [g for g in gene_stats
                               if g['max_pval'] > 0.5 and g['max_abs_delta'] < delta_thresh * 0.5]

        results['gene_stats'] = gene_stats
        results['de_genes'] = sorted(de_genes, key=lambda x: x.get('adj_pval', 1))
        results['conserved_genes'] = sorted(conserved_genes, key=lambda x: -x['max_pval'])
        results['n_de'] = len(de_genes)
        results['n_conserved'] = len(conserved_genes)
        results['n_tested'] = len(gene_stats)

        # ── 7. Pairwise platform similarity (Spearman on gene means) ──
        self._update_progress(95, "Computing platform similarity...")
        sorted_common = sorted(common_all)
        plat_mean_vectors = {}
        for plat in platforms:
            gmap = self.app.gpl_gene_mappings.get(plat, {})
            df = self.app.gpl_datasets[plat]
            vec = []
            for gene in sorted_common:
                col = gmap.get(gene)
                if col and col in df.columns:
                    vec.append(pd.to_numeric(df[col], errors='coerce').mean())
                else:
                    vec.append(np.nan)
            plat_mean_vectors[plat] = np.array(vec)

        plat_correlations = {}
        for i, a in enumerate(platforms):
            for b in platforms[i + 1:]:
                va, vb = plat_mean_vectors[a], plat_mean_vectors[b]
                mask = ~(np.isnan(va) | np.isnan(vb))
                if mask.sum() > 10:
                    corr, cpval = spearmanr(va[mask], vb[mask])
                else:
                    corr, cpval = np.nan, np.nan
                plat_correlations[(a, b)] = {'spearman': corr, 'pval': cpval,
                                              'n_genes_compared': int(mask.sum())}
        results['plat_correlations'] = plat_correlations

        self._update_progress(100, "Analysis complete!")
        self.app.enqueue_log(
            f"[XPlat] Done: {results['n_tested']:,} genes tested, "
            f"{results['n_de']:,} DE, {results['n_conserved']:,} conserved")
        return results

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
            same_sp = "same-species" if sp_a == sp_b else f"CROSS-SPECIES ({sp_a}<->{sp_b})"
            corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
            L.append(f"  {a} vs {b}: rho={corr_str}  ({n_cmp:,} genes, {shared:,} shared)  [{same_sp}]")

        L.append("")
        L.append("-" * 70)
        L.append("EXPRESSION COMPARISON RESULTS")
        L.append("-" * 70)
        L.append(f"Genes tested:                  {R['n_tested']:,}")
        L.append(f"DE genes (cross-platform):     {R['n_de']:,}  (adj p < {R['pval_threshold']}, |delta| > {R['delta_threshold']})")
        L.append(f"Conserved genes:               {R['n_conserved']:,}  (p > 0.5, |delta| < {R['delta_threshold'] * 0.5})")

        L.append("")
        L.append("-" * 70)
        L.append("BATCH EFFECT ASSESSMENT")
        L.append("-" * 70)
        score = R['batch_effect_score']
        detected = R['batch_effect_detected']
        status = "!! DETECTED -- consider batch correction" if detected else "OK No significant batch effect"
        L.append(f"Batch effect score:  {score:.4f}  ({status})")
        for plat in plats:
            bm = R['batch_metrics'][plat]
            L.append(f"  {plat}: median_shift={bm['median_shift_from_ref']:+.3f}, "
                     f"variance_ratio={bm['variance_ratio_to_ref']:.3f}")

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
        ttk.Label(self._overlap_tab,
                  text="Gene Overlap Matrix (shared gene count between each pair)",
                  font=('Segoe UI', 11, 'bold')).pack(anchor=tk.W, padx=10, pady=5)

        cols = ['Platform'] + plats
        tree = ttk.Treeview(self._overlap_tab, columns=cols, show='headings',
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

        ttk.Separator(self._overlap_tab).pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(self._overlap_tab,
                  text=f"Genes common to ALL {len(plats)} platforms: {len(R['common_all']):,}",
                  font=('Segoe UI', 11, 'bold'), foreground='#2E7D32').pack(anchor=tk.W, padx=10)

        list_frame = ttk.Frame(self._overlap_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        lb = tk.Listbox(list_frame, font=('Consolas', 9), height=15)
        sb = ttk.Scrollbar(list_frame, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(fill=tk.BOTH, expand=True)
        for gene in sorted(R['common_all']):
            lb.insert(tk.END, gene)

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
                      text="No significant cross-platform DE genes found — platforms are well-aligned.",
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
                  text="These genes have statistically similar distributions across all platforms "
                       "(p > 0.5, small delta). Ideal candidates for cross-platform normalization anchors.",
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

        nb = ttk.Notebook(self._unique_tab)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for plat in plats:
            genes = sorted(unique.get(plat, set()))
            tab = ttk.Frame(nb)
            nb.add(tab, text=f" {plat} ({len(genes):,}) ")
            lb = tk.Listbox(tab, font=('Consolas', 9))
            sb = ttk.Scrollbar(tab, command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb.pack(fill=tk.BOTH, expand=True)
            for gene in genes[:5000]:
                lb.insert(tk.END, gene)
            if len(genes) > 5000:
                lb.insert(tk.END, f"... and {len(genes) - 5000:,} more")

    def _fill_batch_tab(self, R):
        for w in self._batch_tab.winfo_children():
            w.destroy()

        plats = R['platforms']
        ref = R['reference']
        score = R['batch_effect_score']
        detected = R['batch_effect_detected']

        if detected:
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
        rec_frame = ttk.LabelFrame(self._batch_tab, text="Recommendations", padding=10)
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
        if not detected:
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


class GeoWorkflowGUI(ctk.CTk if _HAS_CTK else tk.Tk):
    """Complete GeneVariate main application window - Modern dark UI."""

    MAX_WORKERS = CONFIG['threading']['max_workers']
    METADATA_EXCLUSIONS = METADATA_EXCLUSIONS

    def __init__(self):
        super().__init__()
        self.title("GeneVariate 2.0 - Gene Expression Analysis Platform")
        self.geometry("1200x1050")
        try:
            _sw, _sh = self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"1200x1050+{(_sw-1200)//2}+{(_sh-1050)//2}")
            self.minsize(600, 500)
        except Exception: pass
        self.after_id = None

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
        self._user_data_dirs = []      # user-added directories to scan for GPL files
        self.default_labels_df = None  # loaded labels DataFrame (legacy compat)
        self._pending_gsm_filter = None  # for subset loading after download
        self._llm_update_count_fn = None  # callback for LLM window count updates
        self._dialog_active = False       # prevents concurrent modal dialog crashes (grab conflict)
        
        # ── Persistent GSE Cache ──
        # Stored in data_dir/gse_cache/gse_cache.json
        # Also checks old location: data_dir/agent_memory/recall_memory.json
        self._gse_cache_dir = os.path.join(self.data_dir, "gse_cache")
        self._gse_cache_path = os.path.join(self._gse_cache_dir, "gse_cache.json")
        # Fallback: old directory name from before rename
        self._gse_cache_path_old = os.path.join(self.data_dir, "agent_memory", "recall_memory.json")
        self._gse_saved_cache = self._load_persistent_cache()
        
        # ── App-level extraction settings (shared across all extraction paths) ──
        self._extraction_fields = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']
        self._extraction_custom_fields = []  # list of {'name': str, 'prompt': str}
        self._extraction_recall = True  # Phase 2 Re-extraction Phase 2 enabled
        self._extraction_fast_mode = True  # Default: Fast mode (Phase 1+1.5 only, instant results)
        self._extraction_ns_extra = set()  # Extra fields for NS curation (default: Condition, Tissue, Treatment only)
        self._llm_workers = 0  # 0 = auto from VRAM; set from extraction window
        self.platform_labels = {}     # {platform_name: DataFrame} per-platform labels
        self.labels_col_vars = {}      # col_name -> BooleanVar
        self.is_closing = False
        self.token_cache = {}
        self.ai_pipeline = None
        self.gene_dist_popup_root = None
        self.subset_dist_popup_root = None
        self.compare_window = None
        self.tracked_figures = {}
        self.current_device = 'cpu'  # GPU disabled
        agent_tools = [classify_sample]
        self.ai_agent = SampleClassificationAgent(
            tools_list=agent_tools,
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
        """Configure TTK styles."""
        style = ttk.Style(self)
        
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        style.configure("Add.TButton", 
                       font=('Segoe UI', 11, 'bold'), 
                       background="#C8E6C9", 
                       foreground="#2E7D32",
                       padding=(12, 8))
        style.map("Add.TButton",
                  background=[('active', '#A5D6A7'), ('pressed', '#81C784')])
        style.configure("Action.TButton",
                       font=('Segoe UI', 10, 'bold'),
                       padding=(12, 8))
    
    def _load_persistent_cache(self):
        """Load GSE context cache from disk (survives restarts).
        Checks new path first, falls back to old agent_memory/ location."""
        for path in [self._gse_cache_path, self._gse_cache_path_old]:
            if os.path.exists(path):
                try:
                    import json as _json
                    with open(path, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                    n_gse = len(data.get('gse_descriptions', {}))
                    n_cons = len(data.get('gse_consensus', {}))
                    info = data.get('_info', {})
                    print(f"[Cache] Loaded GSE cache from {path}: "
                          f"{n_gse} GSE descriptions, {n_cons} consensus entries "
                          f"(saved: {info.get('created', '?')})")
                    return data
                except Exception as e:
                    print(f"[Cache] Failed to load {path}: {e}")
        return None

    def _save_persistent_cache(self, agent):
        """Save agent GSE context cache to disk for next session."""
        try:
            import json as _json
            os.makedirs(self._gse_cache_dir, exist_ok=True)
            data = {
                "_info": {
                    "created": datetime.now().isoformat(),
                    "n_experiments": len(agent.gse_descriptions),
                    "n_consensus": len(agent.gse_consensus),
                    "n_corrected": agent.n_corrected,
                    "n_confirmed_ns": agent.n_confirmed,
                    "n_failed": agent.n_failed,
                },
                "gse_descriptions": agent.gse_descriptions,
                "gse_consensus": {
                    gse: {col: dict(counter) for col, counter in cols.items()}
                    for gse, cols in agent.gse_consensus.items()
                },
            }
            with open(self._gse_cache_path, 'w', encoding='utf-8') as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
            self.enqueue_log(f"[Cache] Saved to: {self._gse_cache_path}")
            self._gse_saved_cache = data
        except Exception as e:
            self.enqueue_log(f"[Cache] Save error: {e}")

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
            self.enqueue_log("[GSM→GSE] WARNING: GEOmetadb not loaded — cannot look up GSE")
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
        
        info_frame = tk.Frame(self, bg="#E8F5E9", relief=tk.RAISED, borderwidth=2)
        info_frame.place(relx=0.5, rely=0.05, anchor=tk.N)
        
        tk.Label(info_frame, text=" Welcome to GeneVariate!", 
                font=('Segoe UI', 11, 'bold'),
                bg="#E8F5E9", fg="#2E7D32").pack(padx=10, pady=5)
        tk.Label(info_frame, text=tip_text, 
                justify=tk.LEFT, bg="#E8F5E9").pack(padx=10, pady=5)
        
        def close_tip():
            info_frame.destroy()
        
        tk.Button(info_frame, text="Got it!", command=close_tip,
                 bg="#4CAF50", fg="white", relief=tk.FLAT).pack(pady=5)
        
        self.after(15000, lambda: info_frame.destroy() if info_frame.winfo_exists() else None)
    
    def _show_help(self):
        """Shows comprehensive help dialog."""
        help_win = tk.Toplevel(self)
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
GeneVariate 1.0 - Quick Start Guide

═══════════════════════════════════════════════════════════

OVERVIEW:
GeneVariate enables comprehensive analysis of gene expression data from 
GEO (Gene Expression Omnibus). Load entire platform datasets, select 
expression ranges visually, and use AI to classify and compare samples.

═══════════════════════════════════════════════════════════

WORKFLOW:

1. LOAD PLATFORMS
   - Click platform buttons to load datasets or use 'Download GPL'
   - ALL samples are loaded - this is essential for distribution analysis
   - Custom platforms: Click '+' button to load your own data

2. GENE DISTRIBUTION EXPLORER (Ctrl+G)
   - Enter gene symbols (comma-separated)
   - Select platforms to compare
   - Click "Plot Distributions"
   - DRAG rectangles on histograms to select expression ranges
   - Multiple regions can be selected per plot
   - Click "Analyze Selected Range(s)" for LLM extraction

3. AI CLASSIFICATION
   - Requires Ollama service (ollama.com)
   - Automatically classifies samples by condition, tissue, treatment
   - Uses semantic clustering to unify similar labels
   - Results open in Interactive Analyzer window

4. INTERACTIVE ANALYZER
   - Click column headers to change grouping/coloring
   - PCA tab: Dimensionality reduction visualization
   - DPC tab: Density Peak Clustering analysis
   - Filter by keywords to focus on specific samples

5. DISTRIBUTION COMPARISON
   - Load classified CSV files
   - Select multiple groups to compare
   - Statistical tests: Wilcoxon, Wasserstein distance
   - Toggle between rugs and density visualizations

STEP 1 (OPTIONAL): GEO DATABASE SEARCH
   - Search GEO for experiments by keywords
   - Filter by platform (e.g., GPL570)
   - Review and select experiments
   - Proceed to Step 2 for extraction

═══════════════════════════════════════════════════════════

TIPS & TRICKS:

- Click legend items to change colors (works in all plots!)
- Right-click on plots to save individual figures
- Use Ctrl+Mouse Wheel to zoom in plots
- Select multiple regions across different genes/platforms
- Export publication-ready figures (300+ DPI)

═══════════════════════════════════════════════════════════

KEYBOARD SHORTCUTS:

Ctrl+O     Load external CSV
Ctrl+G     Gene Distribution Explorer
Ctrl+D     Distribution Comparison
Ctrl+I     Interactive Analyzer
F1         This help screen
Ctrl+Q     Quit application

═══════════════════════════════════════════════════════════

TROUBLESHOOTING:

- "SQLite threading error": Restart analysis, issue is being handled
- "LLM service unavailable": Install/start Ollama (ollama.com)
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
Ctrl+I     Open Interactive Analyzer
F1         Show help
Ctrl+Q     Quit application
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts, parent=self)
    
    def _show_about(self):
        """Shows about dialog."""
        about_text = """
GeneVariate 1.0
Gene Expression Variability Analysis Tool

A comprehensive platform for analyzing gene expression 
patterns across large-scale GEO datasets.

Features:
- Multi-platform support (GPL570, GPL96, GPL6947, etc.)
- AI-powered label extraction
- Interactive distribution analysis
- PCA & Density Peak Clustering
- Publication-ready visualizations

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
                        self.progressbar["value"] = progress_val
                        
                        if hasattr(self, 'status_label'):
                            if progress_val >= 100:
                                self.status_label.config(text="OK Complete", foreground="green")
                            else:
                                self.status_label.config(
                                    text=f"Processing... {progress_val:.0f}%", 
                                    foreground="blue"
                                )
                    except (ValueError, IndexError):
                        pass
                else:
                    self.log_text.insert(tk.END, msg + chr(10))
                    
                    if "[ERROR]" in msg or "ERROR" in msg:
                        start_idx = self.log_text.search("[ERROR]", "end-2l", "end")
                        if start_idx:
                            self.log_text.tag_add("error", start_idx, "end-1c")
                    elif "[WARNING]" in msg or "WARNING" in msg:
                        start_idx = self.log_text.search("[WARNING]", "end-2l", "end")
                        if start_idx:
                            self.log_text.tag_add("warning", start_idx, "end-1c")
                    elif "[INFO]" in msg or "OK" in msg:
                        start_idx = self.log_text.index("end-2l")
                        self.log_text.tag_add("info", start_idx, "end-1c")
                    
                    self.log_text.see(tk.END)
                    self._log_msg_count += 1
                    if hasattr(self, 'log_status_label'):
                        self.log_status_label.config(text=f"Log: {self._log_msg_count} messages", foreground="blue")
                    
        except q.Empty:
            pass
        
        if self.winfo_exists():
            self.after_id = self.after(100, self.process_log_queue)
    
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

        tmp_sql_path = None
        try:
            file_size_mb = os.path.getsize(gz_path) / (1024*1024)
            self.enqueue_log(f"[DB] Loading GEOmetadb from: {gz_path} ({file_size_mb:.0f} MB)")
            print(f"[DB] Loading GEOmetadb into memory from {gz_path}...")
            self.enqueue_log("[DB] This may take a minute on first run...")
            
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
                tmp_sql_path = tmp.name
                with gzip.open(gz_path, "rb") as gzfi:
                    shutil.copyfileobj(gzfi, tmp)

            disk_conn = sqlite3.connect(tmp_sql_path)
            disk_conn.text_factory = lambda b: b.decode('utf-8', 'replace')

            self.gds_conn = sqlite3.connect(":memory:")
            self.gds_conn.text_factory = disk_conn.text_factory
            
            self.enqueue_log("[DB] Copying database to memory...")
            disk_conn.backup(self.gds_conn)
            disk_conn.close()
            
            os.remove(tmp_sql_path)
            
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
            if tmp_sql_path and os.path.exists(tmp_sql_path):
                os.remove(tmp_sql_path)
    
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
        """Checks if Ollama service is available with user-friendly messaging."""
        self.enqueue_log("[LLM] Checking for Ollama LLM service...")
        try:
            models = ollama.list()
            avail = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
            self.ai_pipeline = True
            self.enqueue_log(f"[LLM] OK Ollama service detected ({len(avail)} models: {', '.join(avail[:5])})")
            detected = _detect_ollama_model()
            if detected:
                self.enqueue_log(f"[LLM] OK Using model: {detected}")
            else:
                self.enqueue_log("[LLM] [!] No compatible model found. Pull one: ollama pull gemma2:9b")
        except Exception as e:
            self.enqueue_log("[LLM] [!] Ollama service not detected")
            self.enqueue_log("[LLM] LLM extraction will be unavailable")
            self.enqueue_log("[LLM] To enable: Install Ollama from https://ollama.com")
            self.enqueue_log(f"[LLM] Then pull a model: ollama pull gemma2:9b")
            self.ai_pipeline = None
    
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
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="gray", font=('Segoe UI', 9))
        self.status_label.pack(side=tk.LEFT)

        # ── Scrollable main content area ──
        # Progress bar + log stay at bottom outside scroll
        self._bottom_frame = ttk.Frame(self)
        self._bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        main_canvas = tk.Canvas(self, highlightthickness=0)
        main_vsb = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        self._main_sf = ttk.Frame(main_canvas)
        self._main_sf.bind("<Configure>",
                           lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        self._main_cw = main_canvas.create_window((0, 0), window=self._main_sf, anchor="nw")
        main_canvas.configure(yscrollcommand=main_vsb.set)
        main_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        main_canvas.bind("<Configure>",
                         lambda e: main_canvas.itemconfig(self._main_cw, width=e.width))
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

        plat_frame = ttk.LabelFrame(self._main_sf, text=" Load Gene Expression Platforms", padding=10)
        plat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_label = ttk.Label(plat_frame, text="Load GPL platforms to analyze gene distributions. Use 'Download GPL' to fetch any platform from NCBI GEO, or 'Add Custom Platform' to load your own data files.", foreground="gray", font=('Segoe UI', 9, 'italic'), wraplength=1100)
        info_label.pack(fill=tk.X, pady=(0, 10))
        
        # Dynamic platform buttons — discovered from data directory
        self._plat_btn_frame = ttk.Frame(plat_frame)
        self._plat_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self._refresh_platform_buttons()
        
        custom_frame = ttk.Frame(plat_frame)
        custom_frame.pack(fill=tk.X, padx=5, pady=8)
        
        ttk.Button(custom_frame, text="+ Add Custom Platform", command=self._load_custom_gpl_data, style="Add.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(custom_frame, text="Download GPL", command=self._open_gpl_downloader_window, style="Action.TButton").pack(side=tk.LEFT, padx=8)

        tk.Button(custom_frame, text="Refresh", command=self._refresh_platform_buttons,
                  font=('Segoe UI', 9), padx=8, cursor="hand2").pack(side=tk.LEFT, padx=5)

        
        self.loaded_plat_frame = ttk.Frame(plat_frame)
        self.loaded_plat_frame.pack(fill=tk.X, padx=5, pady=5)
        self.loaded_plat_label = ttk.Label(self.loaded_plat_frame, text="No platforms loaded yet", foreground="orange", font=('Segoe UI', 9, 'bold'))
        self.loaded_plat_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(self.loaded_plat_frame, text="+ Add Data Directory...",
                  command=self._add_data_directory,
                  bg="#1976D2", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.RIGHT, padx=5)

        tools_frame = ttk.LabelFrame(self._main_sf, text=" Analysis Tools", padding=10)
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(tools_frame, text="Use these tools after loading platforms:", font=('Segoe UI', 9, 'italic'), foreground='gray').pack(anchor=tk.W, pady=(0, 5))
        
        tools_btn_frame = ttk.Frame(tools_frame)
        tools_btn_frame.pack(fill=tk.X, pady=4)
        tools_btn_frame.columnconfigure((0, 1, 2), weight=1, uniform="toolbtn")
        
        _tool_font = ('Segoe UI', 10, 'bold')
        
        self.gene_explorer_btn = tk.Button(tools_btn_frame, text="Gene Distribution Explorer", command=self.show_gene_distribution_popup, bg="#9C27B0", fg="white", font=_tool_font, relief=tk.RAISED, bd=2, height=2, cursor="hand2", state=tk.DISABLED)
        self.gene_explorer_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self._add_button_hover(self.gene_explorer_btn, "#7B1FA2", "#9C27B0")
        
        self.compare_btn = tk.Button(tools_btn_frame, text="Compare Distributions", command=self.open_compare_window, bg="#FF9800", fg="white", font=_tool_font, relief=tk.RAISED, bd=2, height=2, cursor="hand2", state=tk.DISABLED)
        self.compare_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self._add_button_hover(self.compare_btn, "#F57C00", "#FF9800")

        self.dist_class_btn = tk.Button(tools_btn_frame, text="Distribution Classification", command=self._open_dist_classification, bg="#00897B", fg="white", font=_tool_font, relief=tk.RAISED, bd=2, height=2, cursor="hand2", state=tk.DISABLED)
        self.dist_class_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self._add_button_hover(self.dist_class_btn, "#00695C", "#00897B")
        
        # ── Label Source (inside tools_frame - always visible) ──────
        ttk.Separator(tools_frame, orient='horizontal').pack(fill=tk.X, pady=(8, 4))

        lbl_hdr = ttk.Frame(tools_frame)
        lbl_hdr.pack(fill=tk.X)
        ttk.Label(lbl_hdr, text="Sample Labels:",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.label_source_var = tk.StringVar(value="ai")
        ttk.Radiobutton(lbl_hdr, text="LLM (Ollama)",
                        variable=self.label_source_var, value="ai",
                        command=self._toggle_main_label_source).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Radiobutton(lbl_hdr, text="Label Files (per-platform)",
                        variable=self.label_source_var, value="file",
                        command=self._toggle_main_label_source).pack(side=tk.LEFT, padx=4)

        # File controls row (hidden until "file" selected)
        self.labels_file_row = ttk.Frame(tools_frame)
        tk.Button(self.labels_file_row, text="+ Add Label File...",
                  command=self._add_label_file,
                  bg="#1976D2", fg="white", font=("Segoe UI", 9, "bold"),
                  padx=10, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(self.labels_file_row, text="+ Add Folder...",
                  command=self._browse_labels_folder,
                  bg="#0D47A1", fg="white", font=("Segoe UI", 9),
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(self.labels_file_row, text="Set Labels Directory",
                  command=self._set_labels_directory,
                  bg="#388E3C", fg="white", font=("Segoe UI", 9),
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(self.labels_file_row, text="Clear All",
                  command=self._clear_all_labels,
                  bg="#757575", fg="white", font=("Segoe UI", 9),
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(self.labels_file_row, text="Curate Labels (LLM)",
                  command=self._open_llm_curator,
                  bg="#E65100", fg="white", font=("Segoe UI", 9, "bold"),
                  padx=8, cursor="hand2").pack(side=tk.LEFT, padx=2)
        ttk.Label(self.labels_file_row,
                  text="  (GPL ID auto-detected from filename)",
                  font=("Segoe UI", 8, "italic"), foreground="gray").pack(side=tk.LEFT, padx=4)

        # Per-platform loaded labels list
        self.labels_plat_frame = ttk.Frame(tools_frame)

        # Column checkboxes row (populated after load)
        self.labels_col_frame = ttk.Frame(tools_frame)
        self.labels_col_vars = {}

        # Status line
        self.labels_status_lbl = ttk.Label(tools_frame,
                                            text="LLM mode: samples labeled by Ollama during analysis.",
                                            font=("Segoe UI", 8, "italic"), foreground="gray")
        self.labels_status_lbl.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.step1_frame = ttk.LabelFrame(self._main_sf, text=" Step 1: Discover Experiments (Optional)", padding=10)
        self.step1_frame.pack(fill=tk.X, padx=5, pady=5)
        
        collapse_frame = ttk.Frame(self.step1_frame)
        collapse_frame.pack(fill=tk.X)
        
        self.step1_collapsed = tk.BooleanVar(value=True)
        self.step1_toggle_btn = ttk.Button(collapse_frame, text=" Show", command=self._toggle_step1)
        self.step1_toggle_btn.pack(side=tk.LEFT)
        
        ttk.Label(collapse_frame, text="Search GEO database for experiments by keywords", font=('Segoe UI', 9, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=10)
        
        self.step1_content = ttk.Frame(self.step1_frame)
        
        ttk.Label(self.step1_content, text="Platform Filter (e.g., GPL570,GPL96):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.platform_entry = ttk.Entry(self.step1_content, width=60)
        self.platform_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        
        ttk.Label(self.step1_content, text="Keywords (comma-separated):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.filter_entry = ttk.Entry(self.step1_content, width=60)
        self.filter_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        self.step1_content.grid_columnconfigure(1, weight=1)
        
        s1_btn_frame = ttk.Frame(self.step1_content)
        s1_btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        
        tk.Button(s1_btn_frame, text="  Search GEO Database  ", command=self.start_extraction, bg="#2E7D32", fg="white", font=('Segoe UI', 11, 'bold'), padx=20, pady=8, cursor="hand2", relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=10)
        
        self.gse_frame = ttk.LabelFrame(self._main_sf, text=" Step 1.5: Selected Experiments for Analysis", padding=10)
        
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
        
        _gse_font = ('Segoe UI', 9, 'bold')
        
        tk.Button(gse_btn_frame, text="Save Selected for Step 2", command=self._save_selected_gses,
                  bg="#1565C0", fg="white", font=_gse_font, height=2,
                  relief=tk.RAISED, bd=2, cursor="hand2").grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        
        tk.Button(gse_btn_frame, text="Select All",
                  command=lambda: self.gse_listbox.select_set(0, tk.END),
                  bg="#546E7A", fg="white", font=_gse_font, height=2,
                  relief=tk.RAISED, bd=2, cursor="hand2").grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        
        tk.Button(gse_btn_frame, text="Clear Selection",
                  command=lambda: self.gse_listbox.selection_clear(0, tk.END),
                  bg="#78909C", fg="white", font=_gse_font, height=2,
                  relief=tk.RAISED, bd=2, cursor="hand2").grid(row=0, column=2, padx=4, pady=3, sticky="ew")
        
        tk.Button(gse_btn_frame, text="Review Details", command=self._review_gse_details,
                  bg="#00796B", fg="white", font=_gse_font, height=2,
                  relief=tk.RAISED, bd=2, cursor="hand2").grid(row=0, column=3, padx=4, pady=3, sticky="ew")

        # Row 2: Download Expression Data button (spans full width)
        gse_btn_frame.columnconfigure(4, weight=1, uniform="gsebtn")
        self._download_expr_btn = tk.Button(
            gse_btn_frame,
            text="Download Expression Data for Selected Experiments",
            command=self._download_selected_expression,
            bg="#E65100", fg="white", font=_gse_font, height=2,
            relief=tk.RAISED, bd=2, cursor="hand2")
        self._download_expr_btn.grid(row=1, column=0, columnspan=4, padx=4, pady=3, sticky="ew")

        # Download progress
        self._dl_progress_frame = ttk.Frame(self.gse_frame)
        self._dl_progress_bar = ttk.Progressbar(self._dl_progress_frame, mode='determinate', length=400)
        self._dl_progress_bar.pack(fill=tk.X, padx=5)
        self._dl_progress_label = ttk.Label(self._dl_progress_frame, text="", font=('Consolas', 8), foreground='gray')
        self._dl_progress_label.pack(anchor='w', padx=5)
        lab_frame = ttk.LabelFrame(self._main_sf, text=" Step 2: Sample Classification & Analysis", padding=10)
        lab_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.step2_status_label = ttk.Label(lab_frame, text="Ready to extract labels from Step 1 or external file", foreground="blue", font=('Segoe UI', 9))
        self.step2_status_label.pack(pady=5)
        
        btn_row = ttk.Frame(lab_frame)
        btn_row.pack(pady=8)
        # Use grid for perfect symmetry
        btn_row.columnconfigure((0, 1, 2), weight=1, uniform="step2btn")
        
        _s2_font = ('Segoe UI', 10, 'bold')
        _s2_w = 22
        _s2_h = 2
        
        self.load_csv_btn = tk.Button(
            btn_row, text="  Load External CSV  ", command=self.load_external_file_for_step2,
            bg="#00796B", fg="white", font=_s2_font, width=_s2_w, height=_s2_h,
            relief=tk.RAISED, bd=2, cursor="hand2")
        self.load_csv_btn.grid(row=0, column=0, padx=6, pady=4, sticky="ew")
        self._add_button_hover(self.load_csv_btn, "#00695C", "#00796B")
        
        self.ai_label_btn = tk.Button(
            btn_row, text="  LLM Extraction  ", command=self._open_llm_extraction_window,
            bg="#1565C0", fg="white", font=_s2_font, width=_s2_w, height=_s2_h,
            relief=tk.RAISED, bd=2, cursor="hand2")
        self.ai_label_btn.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        self._add_button_hover(self.ai_label_btn, "#0D47A1", "#1565C0")
        
        self.manual_label_btn = tk.Button(
            btn_row, text="  Manual Labeling  ", command=self.run_manual_labeling,
            bg="#2E7D32", fg="white", font=_s2_font, width=_s2_w, height=_s2_h,
            relief=tk.RAISED, bd=2, cursor="hand2")
        self.manual_label_btn.grid(row=0, column=2, padx=6, pady=4, sticky="ew")
        self._add_button_hover(self.manual_label_btn, "#1B5E20", "#2E7D32")

        

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
        
        self.log_text = tk.Text(log_container, wrap=tk.WORD, font=('Consolas', 9), bg='#F5F5F5')
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_log = ttk.Scrollbar(log_container, command=self.log_text.yview)
        sb_log.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb_log.set)
        
        self.log_text.tag_configure("error", foreground="red", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("warning", foreground="orange", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("info", foreground="green")
        self.log_text.tag_configure("header", foreground="blue", font=('Consolas', 9, 'bold'))
    
    def _add_button_hover(self, button, hover_color, normal_color):
        """Adds hover effect to buttons."""
        button.bind("<Enter>", lambda e: button.config(bg=hover_color))
        button.bind("<Leave>", lambda e: button.config(bg=normal_color))

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
        artist_groups: optional list of lists — each inner list is a group of
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
        """Toggles Step 1 visibility."""
        if self.step1_collapsed.get():
            self.step1_content.pack(fill=tk.X, pady=5)
            self.step1_toggle_btn.config(text=" Hide")
            self.step1_collapsed.set(False)
        else:
            self.step1_content.pack_forget()
            self.step1_toggle_btn.config(text=" Show")
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
        """Universal progress update — callable from any process/thread.
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
                if value is not None:
                    self.progressbar["value"] = value
                    pct = value / max(1, self.progressbar["maximum"]) * 100
                    self.progress_label.config(text=f"{pct:.0f}%")
                if text is not None:
                    self.progress_status.config(text=text)
                if value == 0 or (value is not None and value >= self.progressbar["maximum"]):
                    if value == 0:
                        self.progress_status.config(text="")
                        self.progress_label.config(text="0%")
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
                with open(filepath, 'w') as f:
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
            self.loaded_plat_label.config(text="No platforms loaded yet", foreground="orange")
            self.gene_explorer_btn.config(state=tk.DISABLED)
            self.compare_btn.config(state=tk.DISABLED)
            self.dist_class_btn.config(state=tk.DISABLED)
        elif not has_loaded and has_available:
            # Files on disk but nothing fully loaded — Gene Explorer can quick-load genes
            avail_list = ', '.join(sorted(available.keys())[:6])
            extra = f" +{len(available)-6} more" if len(available) > 6 else ""
            self.loaded_plat_label.config(
                text=f"No platforms fully loaded | {len(available)} available on disk: {avail_list}{extra}\n"
                     f"Use Gene Distribution Explorer for quick gene-only loading",
                foreground="#E65100")
            self.gene_explorer_btn.config(state=tk.NORMAL)
            self.compare_btn.config(state=tk.NORMAL)
            self.dist_class_btn.config(state=tk.NORMAL)
        else:
            plat_names = []
            total_samples = 0
            
            for plat_name, plat_df in self.gpl_datasets.items():
                samples = len(plat_df)
                total_samples += samples
                plat_names.append(f"{plat_name} ({samples:,} samples)")
            
            n_extra = len(available) - len(self.gpl_datasets)
            extra_text = f" | {n_extra} more available on disk" if n_extra > 0 else ""
            status_text = (f"OK Loaded {len(self.gpl_datasets)} platform(s): "
                          + ", ".join(plat_names)
                          + f"\nTotal: {total_samples:,} samples{extra_text}")
            
            self.loaded_plat_label.config(text=status_text, foreground="green")
            
            self.gene_explorer_btn.config(state=tk.NORMAL)
            self.compare_btn.config(state=tk.NORMAL)
            self.dist_class_btn.config(state=tk.NORMAL)

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
        self.status_label.config(text=f"Loading {gpl_id}...", foreground="blue")
        self.update_idletasks()

        # 1. Check already-downloaded files in data_dir/GPL_ID/ subdirectory
        if hasattr(self, 'data_dir') and self.data_dir:
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

        # 2. Check discovered files (from _discover_available_platforms)
        #    This finds files in: data_dir (flat), user-added dirs, subdirectories
        available = self._discover_available_platforms()
        discovered_path = available.get(gpl_id) or available.get(gpl_id.upper())
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
        self.status_label.config(text=f"Downloading {gpl_id}...", foreground="blue")

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

    def _refresh_platform_buttons(self):
        """Dynamically build platform load buttons from discovered data files, organized by species."""
        for w in self._plat_btn_frame.winfo_children():
            w.destroy()

        available = self._discover_available_platforms()
        loaded = set(self.gpl_datasets.keys())

        if not available and not loaded:
            ttk.Label(self._plat_btn_frame,
                      text="No platforms found. Use 'Download GPL' or 'Add Custom Platform' to get started.",
                      font=('Segoe UI', 10), foreground='#888').pack(pady=15)
            return

        # Organize by species
        from collections import OrderedDict
        species_groups = OrderedDict()
        all_plats = sorted(set(list(available.keys()) + list(loaded)))
        for plat in all_plats:
            sp = GPL_SPECIES.get(plat, 'Other').title()
            if sp not in species_groups:
                species_groups[sp] = []
            is_loaded = plat in loaded
            n = len(self.gpl_datasets[plat]) if is_loaded else 0
            species_groups[sp].append((plat, is_loaded, n))

        # Sort: Human first, Mouse second, Rat third, then alphabetical
        priority = {'Human': 0, 'Mouse': 1, 'Rat': 2}
        sorted_species = sorted(species_groups.keys(), key=lambda s: (priority.get(s, 99), s))

        for species in sorted_species:
            platforms = species_groups[species]
            sp_frame = ttk.LabelFrame(self._plat_btn_frame, text=f"{species} Platforms")
            sp_frame.pack(fill=tk.X, padx=5, pady=3)

            for plat, is_loaded, n in platforms:
                bf = ttk.Frame(sp_frame)
                bf.pack(side=tk.LEFT, padx=5, pady=4)

                if is_loaded:
                    text = f"✓ {plat}\n({n:,} samples)"
                    btn = tk.Button(bf, text=text, bg="#43A047", fg="white",
                                    font=('Segoe UI', 9, 'bold'), width=16, height=2,
                                    state=tk.DISABLED, relief=tk.SUNKEN)
                else:
                    text = f"{plat}\n(click to load)"
                    btn = tk.Button(bf, text=text,
                                    command=lambda p=plat: self._smart_load_gpl(p),
                                    bg="#1565C0", fg="white",
                                    font=('Segoe UI', 9, 'bold'), width=16, height=2,
                                    cursor="hand2", relief=tk.RAISED)
                btn.pack()

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
    
    def _discover_available_platforms(self):
        """Find available GPL platform files.
        Scans:
          1. data_dir (genevariate/data/ where GPL Downloader saves)
          2. Project root (genevariate/ — parent of gui/)
          3. Program directory (genevariate/gui/ where app.py lives)
          4. Current working directory
          5. User-added directories
          6. Common output directory names (AI_agent_gemma2_results*)
        Returns dict of {gpl_id: file_path}.
        """
        available = dict(self.gpl_available_files)  # start with cached

        dirs_to_scan = set()
        
        # Primary: data_dir (should be genevariate/data/ after _find_data_dir fix)
        if hasattr(self, 'data_dir') and self.data_dir:
            dirs_to_scan.add(Path(self.data_dir))
        
        # Project root (genevariate/) — parent of gui/ where app.py lives
        _prog_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        _proj_root = _prog_dir.parent  # genevariate/
        dirs_to_scan.add(_proj_root)
        
        # Project root/data/ explicitly
        _proj_data = _proj_root / 'data'
        if _proj_data.exists():
            dirs_to_scan.add(_proj_data)
        
        # Program directory itself (genevariate/gui/)
        dirs_to_scan.add(_prog_dir)
        
        # Current working directory
        _cwd = Path(os.getcwd())
        dirs_to_scan.add(_cwd)
        _cwd_data = _cwd / 'data'
        if _cwd_data.exists():
            dirs_to_scan.add(_cwd_data)
        
        # User-added directories
        for ud in getattr(self, '_user_data_dirs', []):
            dirs_to_scan.add(Path(ud))
        
        # Common output directory names from old versions
        for parent in [_proj_root, _prog_dir, _cwd]:
            for pattern in ['AI_agent*', 'results*', 'output*', 'gpl_data*']:
                import glob as _glob_mod
                for match in _glob_mod.glob(str(parent / pattern)):
                    if os.path.isdir(match):
                        dirs_to_scan.add(Path(match))

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
                                best = max(candidates, key=lambda f: f.stat().st_size)
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
                    f"[Discovery] Scanned {', '.join(str(p) for p in dirs_to_scan)} — "
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
        self.enqueue_log(f"[DataDir] Added: {d} — found {len(available)} platform(s) total")
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
            if isinstance(child, ttk.LabelFrame) and 'Select Platforms' in str(child.cget('text')):
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
        Much faster than loading the entire platform — reads header first,
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
            import gzip
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
                             f"{len(all_cols):,} columns — loading subset...")

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

    def _load_gpl_data(self, gpl_name, file_path, metadata_path=None, gsm_filter=None):
        """
        Load GPL platform data.
        
        Args:
            gpl_name: Platform identifier (e.g., "GPL570")
            file_path: Path to expression data CSV
            metadata_path: Optional path to metadata CSV (for GPL570)
            gsm_filter: Optional set/list of GSM IDs — if provided, only keep these samples
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
            self.status_label.config(text=f"Loading {gpl_name}...", foreground="blue")
            self.update_idletasks()
            
            self.enqueue_log(f"[{gpl_name}] Reading expression data (this may take a moment)...")
            _is_gz = str(full_file_path).endswith('.gz')
            data_df = pd.read_csv(full_file_path, compression="gzip" if _is_gz else "infer", low_memory=False)
            self.progressbar["value"] = 30
            self.update_idletasks()
            
            if gpl_name == "GPL570" and full_meta_path:
                if full_meta_path.exists():
                    self.enqueue_log(f"[{gpl_name}] Loading metadata...")
                    _meta_gz = str(full_meta_path).endswith('.gz')
                    meta_df = pd.read_csv(full_meta_path, compression="gzip" if _meta_gz else "infer", low_memory=False)
                    
                    if len(meta_df) == len(data_df):
                        data_df = pd.concat([meta_df.reset_index(drop=True), data_df.reset_index(drop=True)], axis=1)
                        self.enqueue_log(f"[{gpl_name}] OK Metadata merged ({len(meta_df.columns)} columns)")
                    else:
                        self.enqueue_log(f"[{gpl_name}] [!] Metadata row count mismatch ({len(meta_df)} vs {len(data_df)}), skipping merge")
                else:
                    self.enqueue_log(f"[{gpl_name}] [!] Metadata file not found at {full_meta_path}")
            
            self.progressbar["value"] = 50
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
                    lookup_result = self._fast_gsm_lookup(unique_gsms)
                    
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
            
            self.progressbar["value"] = 70
            self.update_idletasks()
            
            self.gpl_datasets[gpl_name] = data_df
            
            gene_map = {}
            excluded_upper = {col.upper() for col in self.METADATA_EXCLUSIONS}
            
            coerced_cols = 0
            for col in data_df.columns:
                col_upper = col.upper()
                if col_upper not in excluded_upper:
                    if pd.api.types.is_numeric_dtype(data_df[col]):
                        gene_map[col_upper] = col
                    else:
                        # Downloaded platforms may have expression values as strings
                        # Try to coerce - if >50% convert to numeric, treat as expression
                        try:
                            test = pd.to_numeric(data_df[col], errors='coerce')
                            pct_numeric = test.notna().sum() / max(1, len(test))
                            if pct_numeric > 0.5:
                                data_df[col] = test
                                gene_map[col_upper] = col
                                coerced_cols += 1
                        except:
                            pass
            
            if coerced_cols > 0:
                self.enqueue_log(f"[{gpl_name}] Converted {coerced_cols} columns from string -> numeric")
                # Update stored df with coerced types
                self.gpl_datasets[gpl_name] = data_df
            
            self.gpl_gene_mappings[gpl_name] = gene_map
            
            self.progressbar["value"] = 100
            self.update_idletasks()
            
            mem_usage_mb = data_df.memory_usage(deep=True).sum() / 1024**2
            
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
        """Starts GSE extraction thread - COMPLETE VERSION."""
        if self.current_extraction_thread and self.current_extraction_thread.is_alive():
            messagebox.showwarning("Busy", "GSE extraction is already running.\n\nPlease wait for it to complete.", parent=self)
            return
        
        gz_path = CONFIG['paths']['geo_db']
        if not gz_path or not os.path.exists(gz_path):
            messagebox.showerror(
                "Database Required", 
                "GEOmetadb.sqlite.gz is required for GSE extraction.\n\n"
                "Please download from:\n"
                "https://gbnci.cancer.gov/geo/GEOmetadb.sqlite.gz\n\n"
                f"Expected at: {gz_path}", 
                parent=self
            )
            return
        
        plat_filter = self.platform_entry.get().strip()
        tokens = self.filter_entry.get().strip()
        
        if not tokens:
            messagebox.showerror("Input Required", "Please enter at least one keyword to search for.\n\nExample: cancer, breast, treatment", parent=self)
            return
        
        self.step1_results_df = None
        self.step2_data_df = None
        self.step1_gse_keywords = {}
        self.step1_gse_descriptions = {}
        self.gse_to_keep_for_step2 = []
        self.gse_listbox.delete(0, tk.END)
        self.gse_frame.pack_forget()
        self.step2_status_label.config(text="Searching GEO database...", foreground="blue")
        
        self.enqueue_log("[Step 1] Starting GEO database search...")
        self.enqueue_log(f"[Step 1] Keywords: {tokens}")
        if plat_filter:
            self.enqueue_log(f"[Step 1] Platform filter: {plat_filter}")
        else:
            self.enqueue_log("[Step 1] No platform filter - searching all platforms")
        
        self.update_progress(value=0)
        self.status_label.config(text="Searching GEO database...", foreground="blue")
        
        self.current_extraction_thread = ExtractionThread(
            gz_path=CONFIG['paths']['geo_db'],
            plat_filter=plat_filter,
            search_tokens=tokens,
            log_func=self.enqueue_log,
            on_finish=self.on_extraction_finish,
            gui_ref=self
        )
        self.current_extraction_thread.start()
    

    
    
    def on_extraction_finish(self):
        """Callback when extraction finishes — opens review window."""
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
        
        self.progressbar["value"] = 100
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
    #  Step 1.5 — Download Expression Data for Selected Experiments
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
                    os.path.join(self.data_dir, gpl,
                                 f"{gpl.lower()}_all_samples_normalized_scaled_with_nans.csv.gz"),
                    os.path.join(self.data_dir, f"{gpl}_data.csv.gz"),
                ]
                for c in candidates:
                    if os.path.exists(c) and os.path.getsize(c) > 1_000_000:
                        # File exists and is >1MB — offer to load it instead of re-downloading
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
                from genevariate.core.ns_repair_pipeline import load_db_to_memory
                from genevariate.core.gpl_downloader import GPLDownloader
                from genevariate.config import CONFIG

                output_dir = str(CONFIG['paths']['data'])
                geo_db = str(CONFIG['paths']['geo_db'])

                # Load GEOmetadb
                self.after(0, lambda: self._dl_progress_label.config(
                    text="Loading GEOmetadb..."))
                conn = load_db_to_memory(geo_db, log_fn=self.enqueue_log)
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
                            # Platform exists but none of our GSEs match — download a sample
                            filtered_gses = gse_list[:min(n_gse, 50)]
                            self.enqueue_log(
                                f"[Download {gpl_id}] "
                                f"GSE IDs not in GEOmetadb gse_gpl — "
                                f"downloading {len(filtered_gses)} directly")

                        # Override the info's gse_list with our filtered list
                        info['gse_list'] = filtered_gses
                        info['total_series'] = len(filtered_gses)

                        # Check if full platform file exists — protect it
                        full_csv = os.path.join(
                            output_dir, gpl_id,
                            f"{gpl_id.lower()}_all_samples_normalized_scaled_with_nans.csv.gz")
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
                                    f"all test values NaN — skipping load")
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
                self.after(0, lambda: messagebox.showerror("Error",
                    f"Download failed: {e}", parent=self))
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

        # ── PIPELINE OVERVIEW ──
        info_frame = ttk.LabelFrame(main, text=" Extraction Pipeline Overview", padding=6)
        info_frame.pack(fill=tk.X, padx=PAD, pady=(PAD, 4))

        info_text = (
            "Phase 1 — Raw LLM Extraction (always runs)\n"
            "  Each sample classified by gemma2:9b → Condition, Tissue, Age, Treatment, Treatment_Time.\n"
            "  VRAM-aware parallel workers. RAW output — no .title(), no synonym dictionaries.\n\n"
            "Phase 1.5 — Per-GSE Label Collapsing (always runs)\n"
            "  Within each experiment (GSE) only. Two strict rules:\n"
            "    1. Exact match (case/space/hyphen)  2. Abbreviation initials (AD → Alzheimer Disease)\n"
            "  Numeric guard: Mut12 ≠ Mut10. NO fuzzy matching, NO substring, NO synonyms.\n\n"
            "Phase 2 — 'Not Specified' Recovery (optional, background)\n"
            "  Fetches experiment descriptions from NCBI GEO website. Uses GSE context + sibling\n"
            "  sample consensus to recover missing labels. Only: Condition, Tissue, Treatment.\n\n"
            "Phase 3 — LLM Curator (optional, manual trigger)\n"
            "  LLM reviews label inventory across experiments: 'Are AML and Acute Myeloid Leukemia\n"
            "  the same concept?' Pre-filters candidates → LLM judges → user reviews → apply.\n"
            "  Button: 'Curate Labels (LLM)' in the label management bar or Region Analysis."
        )
        info_lbl = tk.Label(info_frame, text=info_text, font=('Consolas', 8),
                            justify=tk.LEFT, anchor='nw', fg='#333', bg='#F5F5F5',
                            padx=8, pady=6, relief=tk.GROOVE)
        info_lbl.pack(fill=tk.X)

        # ── 1. DATA SOURCE ──
        src_frame = ttk.LabelFrame(main, text=" Data Source", padding=PAD)
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

        tk.Button(exp_r2, text="Search", command=_search_experiments,
                  bg="#4CAF50", fg="white", font=('Segoe UI', 8, 'bold'),
                  padx=6, cursor="hand2").pack(side=tk.LEFT, padx=(20, 6))
        exp_status = ttk.Label(exp_r2, text="", foreground="gray", font=('Segoe UI', 8))
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
        ext_count_lbl = ttk.Label(file_row, text="", foreground="blue")
        ext_count_lbl.pack(side=tk.LEFT, padx=4)
        win._ext_file_df = None

        # ── 2. LABELS TO EXTRACT ──
        lbl_frame = ttk.LabelFrame(main, text=" Labels to Extract", padding=PAD)
        lbl_frame.pack(fill=tk.X, padx=PAD, pady=4)

        field_vars = {}
        standard_fields = [
            ("Tissue", "Tissue type (e.g., Brain, Blood, Liver)"),
            ("Condition", "Disease condition (e.g., Alzheimer, Cancer, Control)"),
            ("Treatment", "Treatment applied (e.g., LPS, Vehicle, Chemotherapy)"),
            ("Treatment_Time", "Treatment duration (e.g., 24h, 7 days)"),
            ("Age", "Age of subject (e.g., 35 years, postnatal day 7)"),
        ]
        for i, (fname, fdesc) in enumerate(standard_fields):
            var = tk.BooleanVar(value=True)
            field_vars[fname] = var
            ttk.Checkbutton(lbl_frame, text=f"{fname}", variable=var,
                            command=_update_sample_count).grid(row=i, column=0, sticky=tk.W, padx=4)
            ttk.Label(lbl_frame, text=fdesc, foreground="gray",
                      font=('Segoe UI', 8, 'italic')).grid(row=i, column=1, sticky=tk.W, padx=8)

        # Custom fields
        ttk.Separator(lbl_frame, orient='horizontal').grid(row=len(standard_fields), column=0, columnspan=3, sticky=tk.EW, pady=6)
        ttk.Label(lbl_frame, text="Custom Labels (prompt engineering):",
                  font=('Segoe UI', 9, 'bold')).grid(row=len(standard_fields)+1, column=0, columnspan=2, sticky=tk.W)

        custom_frame = ttk.Frame(lbl_frame)
        custom_frame.grid(row=len(standard_fields)+2, column=0, columnspan=3, sticky=tk.EW, pady=4)
        win._custom_fields_list = []
        win._custom_widgets = []

        def _add_custom_field():
            idx = len(win._custom_fields_list)
            row_f = ttk.Frame(custom_frame)
            row_f.pack(fill=tk.X, pady=2)
            ttk.Label(row_f, text="Name:", font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=2)
            name_e = ttk.Entry(row_f, width=15)
            name_e.pack(side=tk.LEFT, padx=2)
            ttk.Label(row_f, text="Extraction prompt:", font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=4)
            prompt_e = ttk.Entry(row_f, width=40)
            prompt_e.pack(side=tk.LEFT, padx=2)
            prompt_e.insert(0, "string (describe what to extract and give examples)")
            def _remove():
                row_f.destroy()
                if (name_e, prompt_e) in win._custom_fields_list:
                    win._custom_fields_list.remove((name_e, prompt_e))
            tk.Button(row_f, text="×", command=_remove, fg="red", font=('Segoe UI', 9, 'bold'),
                      bd=0, padx=4, cursor="hand2").pack(side=tk.LEFT, padx=4)
            win._custom_fields_list.append((name_e, prompt_e))

        tk.Button(lbl_frame, text="+ Add Custom Label", command=_add_custom_field,
                  bg="#FF9800", fg="white", font=('Segoe UI', 8, 'bold'),
                  padx=8, cursor="hand2").grid(row=len(standard_fields)+3, column=0, sticky=tk.W, pady=4)
        ttk.Label(lbl_frame, text="Define your own extraction criteria using prompt engineering",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).grid(
                      row=len(standard_fields)+3, column=1, sticky=tk.W, padx=8)

        # ── 2b. OLLAMA SETTINGS ──
        ollama_frame = ttk.LabelFrame(main, text=" Ollama Settings", padding=PAD)
        ollama_frame.pack(fill=tk.X, padx=PAD, pady=4)

        # URL
        url_row = ttk.Frame(ollama_frame)
        url_row.pack(fill=tk.X, pady=2)
        ttk.Label(url_row, text="Ollama URL:").pack(side=tk.LEFT)
        ollama_url_var = tk.StringVar(value=_OLLAMA_URL)
        ttk.Entry(url_row, textvariable=ollama_url_var, width=30).pack(side=tk.LEFT, padx=4)
        def _test_ollama():
            global _OLLAMA_URL, _OLLAMA_MODEL
            _OLLAMA_URL = ollama_url_var.get().strip()
            try:
                r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
                if r.status_code == 200:
                    models = [m.get('name','?') for m in r.json().get('models',[])]
                    _OLLAMA_MODEL = models[0] if models else None
                    messagebox.showinfo("Ollama OK",
                        f"Connected to {_OLLAMA_URL}\n"
                        f"Models: {', '.join(models[:5])}\n"
                        f"Using: {_OLLAMA_MODEL}", parent=win)
                else:
                    messagebox.showerror("Ollama Error", f"HTTP {r.status_code}", parent=win)
            except Exception as e:
                messagebox.showerror("Connection Failed",
                    f"Cannot reach {_OLLAMA_URL}\n\n{e}\n\n"
                    f"Start Ollama: ollama serve", parent=win)
        ttk.Button(url_row, text="Test", command=_test_ollama).pack(side=tk.LEFT, padx=4)

        # ── Hardware Info ──
        hw_frame = ttk.LabelFrame(ollama_frame, text="Hardware", padding=4)
        hw_frame.pack(fill=tk.X, pady=(4, 6))

        hw_info_lbl = ttk.Label(hw_frame, text="Scanning hardware...",
                                foreground="gray", font=('Consolas', 8))
        hw_info_lbl.pack(anchor=tk.W)

        # GPU Workers
        gpu_worker_row = ttk.Frame(ollama_frame)
        gpu_worker_row.pack(fill=tk.X, pady=2)
        ttk.Label(gpu_worker_row, text="GPU workers:").pack(side=tk.LEFT)
        worker_var = tk.IntVar(value=0)  # 0 = auto from VRAM
        gpu_worker_spin = ttk.Spinbox(gpu_worker_row, from_=0, to=16,
                                       textvariable=worker_var, width=4)
        gpu_worker_spin.pack(side=tk.LEFT, padx=4)
        ttk.Label(gpu_worker_row, text="(0 = auto from VRAM)",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).pack(side=tk.LEFT)

        # CPU Workers
        cpu_worker_row = ttk.Frame(ollama_frame)
        cpu_worker_row.pack(fill=tk.X, pady=2)
        ttk.Label(cpu_worker_row, text="CPU workers:").pack(side=tk.LEFT)
        cpu_worker_var = tk.IntVar(value=0)  # 0 = auto
        cpu_worker_spin = ttk.Spinbox(cpu_worker_row, from_=0, to=64,
                                       textvariable=cpu_worker_var, width=4)
        cpu_worker_spin.pack(side=tk.LEFT, padx=4)
        ttk.Label(cpu_worker_row, text="(overflow when VRAM full)",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).pack(side=tk.LEFT)

        # Total display
        total_worker_lbl = ttk.Label(ollama_frame, text="Total workers: auto",
                                     foreground="#1565C0", font=('Segoe UI', 9, 'bold'))
        total_worker_lbl.pack(anchor=tk.W, pady=(2, 4))
        def _update_worker_total(*_):
            g = worker_var.get()
            c = cpu_worker_var.get()
            if g == 0 and c == 0:
                total_worker_lbl.config(text="Total workers: auto-detect")
            else:
                total_worker_lbl.config(text=f"Total workers: {g + c} ({g} GPU + {c} CPU)")
        worker_var.trace_add("write", _update_worker_total)
        cpu_worker_var.trace_add("write", _update_worker_total)

        # GPU / CPU status
        gpu_lbl = ttk.Label(ollama_frame, text="", foreground="gray",
                            font=('Segoe UI', 8, 'italic'))
        gpu_lbl.pack(anchor=tk.W)
        def _refresh_gpu():
            try:
                import os as _os, psutil as _ps
                gpus = detect_gpus()
                gpu_st, gpu_vr = check_ollama_gpu()
                auto_w = compute_ollama_parallel()
                if isinstance(auto_w, tuple):
                    auto_total, auto_gpu, auto_cpu = auto_w
                else:
                    auto_total, auto_gpu, auto_cpu = auto_w, auto_w, 0

                cpu_count = _os.cpu_count() or 1
                ram = _ps.virtual_memory()
                ram_free = ram.available / 1e9
                ram_total = ram.total / 1e9

                hw_lines = [f"CPU: {cpu_count} cores | RAM: {ram_free:.1f}/{ram_total:.1f} GB free"]
                if gpus:
                    for g in gpus:
                        hw_lines.append(
                            f"GPU: {g['name']} | VRAM: {g['free_vram_gb']:.1f}/{g['vram_gb']:.1f} GB free")
                    txt = (f"Ollama: {gpu_st} | "
                           f"Recommended: {auto_gpu} GPU + {auto_cpu} CPU = {auto_total} workers")
                else:
                    hw_lines.append("GPU: None detected (CPU-only mode)")
                    txt = f"Ollama: {gpu_st} | Recommended: {auto_total} CPU workers"

                hw_info_lbl.config(text="\n".join(hw_lines))
                gpu_lbl.config(text=txt)

                # Auto-set if both are 0
                if worker_var.get() == 0 and cpu_worker_var.get() == 0:
                    worker_var.set(auto_gpu)
                    cpu_worker_var.set(auto_cpu)
            except Exception as e:
                gpu_lbl.config(text=f"Detection error: {e}")
        win.after(500, _refresh_gpu)

        # ── 3. SAVE DIRECTORY ──
        save_frame = ttk.LabelFrame(main, text=" Save Directory", padding=PAD)
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
        est_frame = ttk.LabelFrame(main, text=" Time Estimation", padding=PAD)
        est_frame.pack(fill=tk.X, padx=PAD, pady=4)
        est_lbl = ttk.Label(est_frame, text="Select a data source to see estimation",
                            font=('Segoe UI', 10), foreground="blue")
        est_lbl.pack(anchor=tk.W)
        ttk.Label(est_frame, text="Based on ~0.52 samples/second (NVIDIA RTX 3060). Speed varies by hardware.",
                  foreground="gray", font=('Segoe UI', 8, 'italic')).pack(anchor=tk.W)
        _update_sample_count()

        # ── 5. PROGRESS ──
        prog_frame = ttk.LabelFrame(main, text=" Progress", padding=PAD)
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
            # ── Apply Ollama settings (stored at app level for Region Analysis too) ──
            global _OLLAMA_URL
            _OLLAMA_URL = ollama_url_var.get().strip() or "http://localhost:11434"
            gpu_w = worker_var.get()
            cpu_w = cpu_worker_var.get()
            total_w = gpu_w + cpu_w if (gpu_w + cpu_w) > 0 else 0
            self.ai_agent.MAX_WORKERS = total_w
            self._ollama_workers = total_w  # shared with Region Analysis
            self._ollama_gpu_workers = gpu_w
            self._ollama_cpu_workers = cpu_w
            self._ollama_url = _OLLAMA_URL
            self._llm_workers = total_w  # app-level for Region Analysis

            # Gather fields
            selected_fields = [f for f, v in field_vars.items() if v.get()]
            custom_list = []
            for name_e, prompt_e in win._custom_fields_list:
                n = name_e.get().strip()
                p = prompt_e.get().strip()
                if n:
                    custom_list.append({'name': n, 'prompt': p if p else f"string (extract {n})"})

            if not selected_fields and not custom_list:
                messagebox.showwarning("No Labels", "Select at least one label to extract.", parent=win)
                return

            # Sync to app-level settings (used by ALL extraction paths)
            self._extraction_fields = selected_fields
            self._extraction_custom_fields = custom_list
            self._extraction_recall = win._recall_var.get() and not win._fast_var.get()
            self._extraction_fast_mode = win._fast_var.get()

            # Set global phase flags from checkboxes
            global _ENABLE_PHASE15, _ENABLE_PHASE2, _GSE_WORKERS
            _ENABLE_PHASE15 = win._phase15_var.get()
            _ENABLE_PHASE2 = win._recall_var.get() and not win._fast_var.get()
            _GSE_WORKERS = {}  # clear worker cache for fresh run

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
                    # Use loaded platform - need GSM metadata from GEOmetadb
                    gsm_col = 'GSM' if 'GSM' in self.gpl_datasets[sel].columns else 'gsm'
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
                    # CRITICAL: Deduplicate by GSM — same sample can appear in multiple series
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
                f"Extract {len(selected_fields) + len(custom_list)} label(s) for {n:,} samples\n\n"
                f"Labels: {', '.join(selected_fields + [c['name'] for c in custom_list])}\n"
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

            # Log GPU status
            _gpus = detect_gpus()
            if _gpus:
                self.enqueue_log(f"[LLM] GPU: {_gpus[0]['name']} ({_gpus[0]['vram_gb']}GB, "
                                 f"{_gpus[0]['free_vram_gb']}GB free)")
            _gpu_st, _gpu_vr = check_ollama_gpu()
            if _gpu_st == "gpu":
                self.enqueue_log(f"[LLM] Ollama: GPU mode ({_gpu_vr}GB VRAM)")
            elif _gpu_st == "cpu":
                self.enqueue_log("[LLM] WARNING: Ollama running on CPU! Extraction will be slow.")
                self.enqueue_log("[LLM] Fix: CUDA_VISIBLE_DEVICES=0 OLLAMA_GPU_LAYERS=999 ollama serve")

            def _on_progress(done, total, speed, eta):
                try:
                    win.after(0, lambda: prog_bar.config(value=done))
                    win.after(0, lambda: prog_lbl.config(
                        text=f"{done:,}/{total:,} | {speed:.2f} smp/s | ETA: {self._format_eta(eta)}"))
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
                        # PHASE 1 + 1.5 COMPLETE — save & load immediately
                        # User can start using labels RIGHT NOW
                        # ═══════════════════════════════════════════════════
                        save_dir = save_dir_var.get()
                        os.makedirs(save_dir, exist_ok=True)

                        # Save Phase 1+1.5 results (raw extraction + per-GSE normalization)
                        phase1_df = result_df.copy()

                        # ── Merge series_id and metadata back from samples_df ──
                        # classify_sample only returns GSM + labels; Phase 2 needs series_id
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
                            self.enqueue_log("[LLM] series_id not in source data — looking up from GEOmetadb...")

                        # Ensure ALL samples have series_id (critical for Phase 2)
                        phase1_df = self._ensure_series_id(phase1_df)
                        fname_raw = f"{source_name}_labels_phase1.csv"
                        fpath_raw = os.path.join(save_dir, fname_raw)
                        phase1_df.to_csv(fpath_raw, index=False)
                        self.enqueue_log(f"[LLM] Phase 1+1.5 labels saved: {fpath_raw} ({len(phase1_df):,} samples)")

                        # Also save as clean _labels.csv (Phase 2 (NS Recovery) will overwrite if run later)
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
                            f"[LLM] Phase 1+1.5 COMPLETE: {len(phase1_df):,} samples, "
                            f"NS in Condition/Tissue/Treatment: {ns_count:,}. Labels loaded.")

                        # ── Ingest results into Memory Agent (vocabulary + clusters) ──
                        if _HAS_DETERMINISTIC and _MEMORY_AGENT is not None:
                            try:
                                _MEMORY_AGENT.ingest_extraction_results(
                                    phase1_df, platform=source_name,
                                    log_fn=self.enqueue_log)
                                stats = _MEMORY_AGENT.stats()
                                self.enqueue_log(
                                    f"[Memory] Vocabulary: {stats.get('vocabulary', {})} | "
                                    f"Clusters: {stats.get('clusters', {})}")
                            except Exception as e:
                                self.enqueue_log(f"[Memory] Ingest warning: {e}")

                        self.update_progress(
                            value=100,
                            text="Phase 1+1.5 complete — labels loaded")

                        win.after(0, lambda: prog_lbl.config(
                            text=f"Phase 1+1.5 done! {len(phase1_df):,} labels ready"))
                        win.after(0, lambda: prog_bar.config(value=prog_bar["maximum"]))
                        win.after(0, lambda: start_btn.config(state=tk.NORMAL))
                        win.after(0, lambda: stop_btn.config(state=tk.DISABLED))

                        fast_mode = self._extraction_fast_mode

                        # ═══════════════════════════════════════════════════
                        # ASK USER: Continue with Phase 2 + 3?
                        # Labels are already loaded and available
                        # ═══════════════════════════════════════════════════
                        if fast_mode or not win._recall_var.get():
                            # Fast mode or recall disabled — done
                            self._release_progress()  # extraction fully finished
                            win.after(0, lambda: messagebox.showinfo(
                                "Extraction Complete (Fast Mode)",
                                f"Extracted labels for {len(phase1_df):,} samples!\n\n"
                                f"Saved to: {fpath_raw}\n"
                                + (f"Labels: {', '.join(_summary_parts)}\n\n" if _summary_parts else "\n")
                                + f"'Not Specified' in Condition/Tissue/Treatment: {ns_count:,}\n\n"
                                f"Labels are LOADED — you can start analysis now.\n\n"
                                f"To improve labels later, click 'Run Phase 2 (NS Recovery)'\n"
                                f"in Region Analysis (recovers Not Specified entries\n"
                                f"and normalizes labels across experiments).",
                                parent=win))
                            return

                        # Ask user
                        def _ask_continue():
                            response = messagebox.askyesno(
                                "Phase 1+1.5 Complete — Labels Ready",
                                f"Extracted labels for {len(phase1_df):,} samples!\n"
                                + (f"Labels: {', '.join(_summary_parts)}\n" if _summary_parts else "")
                                + f"'Not Specified' in Condition/Tissue/Treatment: {ns_count:,}\n\n"
                                f"Labels are LOADED — you can start analysis now.\n\n"
                                f"{'─' * 50}\n"
                                f"Continue with Phase 2 (NS Recovery)?\n"
                                f"(runs in background, GUI stays responsive)\n\n"
                                f"PHASE 2 — 'Not Specified' Recovery:\n"
                                f"  • Fetches experiment descriptions from NCBI GEO\n"
                                f"  • Builds consensus from sibling samples in same GSE\n"
                                f"  • Re-extracts {ns_count:,} missing labels using\n"
                                f"    experiment context + sibling labels\n"
                                f"  • Only curates: Condition, Tissue, Treatment\n\n"
                                f"After Phase 2, you will be asked to run Phase 3\n"
                                f"(LLM Curator — cross-experiment label harmonization).\n\n"
                                f"{'─' * 50}\n"
                                f"Yes = run Phase 2 in background\n"
                                f"No  = keep Phase 1+1.5 labels as-is",
                                parent=win)

                            if response:
                                # Phase 2 inherits the progress bar ownership
                                self._run_phase2_background(
                                    win, phase1_df, source_name, save_dir,
                                    plat_id, prog_lbl, prog_bar)
                            else:
                                self._release_progress()  # extraction fully finished
                                self.enqueue_log("[LLM] User chose to skip Phase 2 — using Phase 1+1.5 labels")
                                # Still offer Phase 3 (LLM Curator) even without Phase 2
                                self.after(500, lambda: self._ask_phase3_curator(win))

                        win.after(100, _ask_continue)
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

            # ── Initialize Memory Agent (episodic memory + vocabulary) ──
            if _HAS_DETERMINISTIC:
                try:
                    mem_ok = init_memory_agent(self.data_dir, log_fn=self.enqueue_log)
                    if mem_ok:
                        self.enqueue_log("[Extraction] MemoryAgent ready (episodic log + vocabulary)")
                except Exception as e:
                    self.enqueue_log(f"[Extraction] MemoryAgent warning: {e}")

            # ── Build GSE contexts (sibling label consensus) ──
            if _HAS_DETERMINISTIC and _MEMORY_AGENT is not None:
                try:
                    init_gse_contexts(samples_df, gds_conn=self.gds_conn, log_fn=self.enqueue_log)
                except Exception as e:
                    self.enqueue_log(f"[Extraction] GSE context warning: {e}")

            # Log phase configuration
            self.enqueue_log(
                f"[Extraction] Phases: Phase 1 (Raw LLM) = ON | "
                f"Phase 1.5 (Collapse Agent) = {'ON' if _ENABLE_PHASE15 else 'OFF'} | "
                f"Phase 2 (GSE Context) = {'ON' if _ENABLE_PHASE2 else 'OFF'}")

            # ── Pre-fetch GSE experiment context for LLM ──
            if 'series_id' in samples_df.columns and self.gds_conn:
                try:
                    series_ids = samples_df['series_id'].dropna().unique().tolist()
                    prefetch_gse_context(self.gds_conn, series_ids, log_fn=self.enqueue_log)
                except Exception as e:
                    self.enqueue_log(f"[LLM] GSE prefetch warning: {e}")

            self._acquire_progress()  # register extraction as progress bar owner

            win._extraction_thread = LabelingThread(
                input_dataframe=samples_df,
                ai_agent=self.ai_agent,
                gui_log_func=self.enqueue_log,
                on_finish=_on_finish,
                fields=selected_fields,
                custom_fields=custom_list if custom_list else None,
                on_progress=_on_progress,
                gui_ref=win,
            )
            win._extraction_thread.start()

        def _stop_extraction():
            if win._extraction_thread:
                win._extraction_thread.stop()
                prog_lbl.config(text="Stopping... (waiting for current samples to finish)")
                stop_btn.config(state=tk.DISABLED)

        start_btn = tk.Button(btn_frame, text=" Start Extraction", command=_start_extraction,
                              bg="#1976D2", fg="white", font=('Segoe UI', 11, 'bold'),
                              padx=20, pady=8, cursor="hand2")
        start_btn.pack(side=tk.LEFT, padx=8)

        stop_btn = tk.Button(btn_frame, text=" Stop", command=_stop_extraction,
                             bg="#D32F2F", fg="white", font=('Segoe UI', 11, 'bold'),
                             padx=20, pady=8, cursor="hand2", state=tk.DISABLED)
        stop_btn.pack(side=tk.LEFT, padx=8)

        # Phase 2 Re-extraction Phase 2 checkbox
        recall_var = tk.BooleanVar(value=True)
        recall_cb = ttk.Checkbutton(btn_frame, text="Phase 2 (GSE Context)",
                                     variable=recall_var)
        recall_cb.pack(side=tk.LEFT, padx=10)

        # Phase 1.5 collapse agent checkbox
        phase15_var = tk.BooleanVar(value=True)
        phase15_cb = ttk.Checkbutton(btn_frame, text="Phase 1.5 (Collapse Agent)",
                                      variable=phase15_var)
        phase15_cb.pack(side=tk.LEFT, padx=10)

        # Fast mode checkbox
        fast_var = tk.BooleanVar(value=self._extraction_fast_mode)
        fast_cb = ttk.Checkbutton(btn_frame, text="Fast Mode",
                                   variable=fast_var)
        fast_cb.pack(side=tk.LEFT, padx=5)

        # Mode description
        mode_desc = ttk.Frame(main)
        mode_desc.pack(fill=tk.X, padx=10, pady=(0, 4))
        mode_info = tk.Label(mode_desc,
            text="Phase 1: Raw LLM extraction (Tissue, Condition, Treatment via gemma2:9b).\n"
                 "         Age/Treatment_Time: regex parsers (no LLM needed).\n"
                 "Phase 1.5: ReAct collapse agent — normalises labels against accumulated vocabulary.\n"
                 "Phase 2: GSE context rescue — uses sibling labels to resolve remaining NS fields.\n"
                 "Each phase can be disabled. Fast Mode = Phase 1(+1.5) only.",
            font=('Segoe UI', 8), fg='#555', justify=tk.LEFT, anchor='nw',
            wraplength=750)
        mode_info.pack(fill=tk.X)
        win._recall_var = recall_var
        win._phase15_var = phase15_var
        win._fast_var = fast_var
        win._recall_agent = None  # set after Phase 2 runs

        # ── Export buttons (Memory Agent) ──
        def _export_clusters():
            if not _HAS_DETERMINISTIC or _MEMORY_AGENT is None:
                messagebox.showinfo("No Data", "Run extraction first to build vocabulary.", parent=win)
                return
            d = filedialog.askdirectory(title="Select folder to save cluster files", parent=win)
            if not d: return
            exported = _MEMORY_AGENT.export_clusters_text(d, log_fn=self.enqueue_log)
            if exported:
                messagebox.showinfo("Exported",
                    f"Saved {len(exported)} cluster file(s) to:\n{d}\n\n" +
                    "\n".join(f"  • {f}" for f in exported), parent=win)
            else:
                messagebox.showinfo("No Clusters", "No string field clusters to export.", parent=win)

        def _save_memory_db():
            if not _HAS_DETERMINISTIC or _MEMORY_AGENT is None:
                messagebox.showinfo("No Data", "Run extraction first.", parent=win)
                return
            path = filedialog.asksaveasfilename(
                title="Save Memory Database",
                defaultextension=".db",
                filetypes=[("SQLite DB", "*.db"), ("All", "*.*")],
                initialfile="biomedical_memory.db",
                parent=win)
            if path:
                _MEMORY_AGENT.export_db(path, log_fn=self.enqueue_log)
                messagebox.showinfo("Saved", f"Memory DB saved to:\n{path}", parent=win)

        tk.Button(btn_frame, text="Export Clusters",
                  command=_export_clusters,
                  font=('Segoe UI', 9), padx=10, pady=4,
                  bg="#7B1FA2", fg="white", cursor="hand2").pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_frame, text="Save Memory DB",
                  command=_save_memory_db,
                  font=('Segoe UI', 9), padx=10, pady=4,
                  bg="#00695C", fg="white", cursor="hand2").pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_frame, text="View Cache",
                  command=lambda: self._show_memory_viewer(win),
                  font=('Segoe UI', 9), padx=10, pady=4).pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_frame, text="Close", command=win.destroy,
                  font=('Segoe UI', 10), padx=15, pady=8).pack(side=tk.RIGHT, padx=8)

        # ── Row 2: Load existing labels and process ──
        load_frame = ttk.Frame(main)
        load_frame.pack(fill=tk.X, pady=(5, 8), padx=10)
        ttk.Separator(load_frame, orient='horizontal').pack(fill=tk.X, pady=3)
        ttk.Label(load_frame, text="Or load existing labels:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        def _load_and_run_phase2_ns():
            """Load a raw labels CSV and run Phase 2 (NS recovery)."""
            fpath = filedialog.askopenfilename(
                title="Load Labels CSV for Phase 2 (NS Recovery)",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                parent=win)
            if not fpath:
                return
            try:
                df = pd.read_csv(fpath)
                if 'GSM' not in df.columns and 'gsm' not in df.columns:
                    for c in df.columns:
                        if c.lower() == 'gsm':
                            df = df.rename(columns={c: 'GSM'})
                            break
                if 'GSM' not in df.columns:
                    messagebox.showerror("Error", "CSV must have a GSM column.", parent=win)
                    return

                df['GSM'] = df['GSM'].astype(str).str.strip().str.upper()
                self.enqueue_log(f"[LLM] Loaded {len(df):,} labels from {os.path.basename(fpath)}")

                # Detect platform
                plat = "Unknown"
                if 'gpl' in df.columns:
                    plat = df['gpl'].dropna().iloc[0] if not df['gpl'].dropna().empty else plat
                elif src_var.get():
                    plat = src_var.get().split('_')[0]
                pid = plat.upper() if plat.upper().startswith('GPL') else plat
                sname = pid

                # Load into platform_labels immediately
                self.platform_labels[pid] = df.copy()
                self._rebuild_merged_labels()
                self.label_source_var.set("file")
                self._toggle_main_label_source()
                self._refresh_labels_display()

                sdir = save_dir_var.get()
                os.makedirs(sdir, exist_ok=True)

                # Ensure series_id exists (critical for Phase 2 GSE grouping)
                df = self._ensure_series_id(df)

                messagebox.showinfo("Labels Loaded",
                    f"Loaded {len(df):,} samples from:\n{os.path.basename(fpath)}\n\n"
                    f"Starting Phase 2 (NS recovery) in the background...", parent=win)

                self._run_phase2_background(
                    win, df, sname, sdir, pid, prog_lbl, prog_bar)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV:\n{e}", parent=win)

        tk.Button(load_frame, text="Load Labels → Run Phase 2 (NS Recovery)",
                  command=_load_and_run_phase2_ns,
                  bg='#E65100', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=10, pady=4, cursor='hand2').pack(side=tk.LEFT, padx=6)

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

        tk.Button(load_frame, text="Load Labels → LLM Curator (Phase 3)",
                  command=_load_and_run_curator,
                  bg='#7B1FA2', fg='white', font=('Segoe UI', 9, 'bold'),
                  padx=10, pady=4, cursor='hand2').pack(side=tk.LEFT, padx=6)

        load_desc = ttk.Frame(main)
        load_desc.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Label(load_desc,
            text="Phase 2 (NS Recovery): Upload labels CSV → recover 'Not Specified' entries "
                 "using NCBI GEO experiment context + sibling consensus.\n"
                 "Phase 3 (LLM Curator): Upload labels CSV → LLM reviews cross-experiment labels, "
                 "proposes merges (AML → Acute Myeloid Leukemia), user reviews before applying.",
            font=('Segoe UI', 8), fg='#555', justify=tk.LEFT, anchor='nw',
            wraplength=750).pack(fill=tk.X)

        # Store refs
        win._src_var = src_var
        win._plat_combo = plat_combo
        win._field_vars = field_vars

    def _run_phase2_background(self, win, phase1_df, source_name, save_dir, plat_id, prog_lbl, prog_bar):
        """Run Phase 2 (NS Recovery) in background thread.
        Labels from Phase 1+1.5 are already loaded and available for analysis.
        When Phase 2 completes, labels are silently upgraded in place.
        Phase 3 (LLM Curator) is a separate manual trigger.
        """
        self.enqueue_log("[LLM] Starting Phase 2 (NS Recovery) in background...")

        # ── Visual indicator: change main window title ──
        _original_title = self.title()
        self._phase2_ns_running = True

        def _set_title(text):
            try:
                self.after(0, lambda: self.title(text))
            except: pass

        _set_title(f"{_original_title}  ●  Phase 2 (NS Recovery) Running...")

        def _update_ui(text, pct=None, log=True):
            try:
                if prog_lbl is not None:
                    win.after(0, lambda: prog_lbl.config(text=text))
                if pct is not None and prog_bar is not None:
                    win.after(0, lambda: prog_bar.config(value=pct))
                self.update_progress(value=pct if pct else 1, text=f"⟳ {text}")
                if log:
                    self.enqueue_log(f"[Phase 2 (NS Recovery)] {text}")
            except Exception:
                pass

        def _bg_worker():
            result_df = phase1_df.copy()
            ns_after = 0
            ns_before = 0
            try:
                # ── Phase 2: Phase 2 Re-extraction ──
                _update_ui("Phase 2: Building memory context...", 10)
                recall_agent = ContextRecallExtractor(
                    log_func=self.enqueue_log,
                    saved_cache=self._gse_saved_cache)

                _NS_CURATE = {'Condition', 'Tissue', 'Treatment'}
                label_cols = [c for c in result_df.columns
                              if c in _NS_CURATE and result_df[c].dtype == 'object']
                ns_before = sum(
                    result_df[c].astype(str).str.strip().isin(_NOT_SPECIFIED_VALUES).sum()
                    for c in label_cols)
                self.enqueue_log(f"[Phase2] Phase 1 had {ns_before:,} 'Not Specified' entries")

                if ns_before > 0 and recall_agent.build_context(result_df):
                    def _recall_progress(done, total, msg):
                        pct = 10 + int(done / max(1, total) * 50)
                        _update_ui(f"Phase 2: {done}/{total} — {msg}", pct, log=False)
                        if done % 20 == 0 or done == total:
                            self.enqueue_log(
                                f"[Phase 2] {done}/{total} NS samples processed")

                    result_df = recall_agent.run_recall_pass(
                        result_df,
                        progress_fn=_recall_progress,
                        extra_fields=self._extraction_ns_extra)

                    ns_after = sum(
                        result_df[c].astype(str).str.strip().isin(_NOT_SPECIFIED_VALUES).sum()
                        for c in label_cols)
                    recovered = ns_before - ns_after
                    self.enqueue_log(
                        f"[Phase2] Recovered {recovered:,} labels "
                        f"({ns_before:,} → {ns_after:,} Not Specified)")
                    try:
                        win._recall_agent = recall_agent
                    except: pass
                    self._save_persistent_cache(recall_agent)

                # Phase 3 (harmonize_labels) removed from automatic pipeline.
                # User runs harmonization separately if needed.
                # Raw + context-collapsed labels are the output.

                # ── Save final version ──
                _update_ui("Saving labels...", 90)

                raw_cols = [c for c in result_df.columns if c.endswith('_raw')]
                if raw_cols:
                    try:
                        result_df.to_csv(
                            os.path.join(save_dir, f"{source_name}_labels_raw.csv"), index=False)
                    except: pass

                clean_df = result_df.drop(columns=raw_cols, errors='ignore')
                fpath = os.path.join(save_dir, f"{source_name}_labels.csv")
                clean_df.to_csv(fpath, index=False)
                self.enqueue_log(f"[LLM] Final harmonized labels saved: {fpath}")

                # ── Silently upgrade loaded labels (on MAIN thread) ──
                pid = plat_id.upper()
                _clean = clean_df.copy()
                def _upgrade_labels():
                    try:
                        if pid.startswith('GPL'):
                            self.platform_labels[pid] = _clean
                            self._rebuild_merged_labels()
                            self._refresh_labels_display()
                    except Exception:
                        pass
                try:
                    self.after(0, _upgrade_labels)
                except Exception:
                    pass

                _update_ui(f"Phase 2 (NS Recovery) complete! Labels upgraded ({len(clean_df):,} samples)", 100)
                self._phase2_ns_running = False
                _set_title(_original_title)
                # Release progress bar ownership after 3 seconds
                def _clear_prog():
                    self._release_progress()
                try:
                    self.after(3000, _clear_prog)
                except: pass

                # Build improvement summary
                _summary = []
                for _c in clean_df.columns:
                    if _c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform') \
                            and clean_df[_c].dtype.kind in ('O','U','S'):
                        _summary.append(f"{_c}: {clean_df[_c].nunique()} unique")

                def _show_done():
                    try:
                        messagebox.showinfo(
                            "Phase 2 (NS Recovery) Complete — Labels Upgraded",
                            f"Background processing finished!\n\n"
                            f"Labels upgraded for {len(clean_df):,} samples.\n"
                            f"Saved to: {fpath}\n\n"
                            + (f"Labels: {', '.join(_summary)}\n\n" if _summary else "")
                            + f"NS recovered: {ns_before - ns_after if ns_before > 0 else 0}\n\n"
                            f"Your analysis tools now use the improved labels.",
                            parent=win)
                    except: pass
                    # Ask Phase 3 (LLM Curator)
                    try:
                        self.after(500, lambda: self._ask_phase3_curator(win))
                    except: pass

                try:
                    win.after(0, _show_done)
                except: pass

                # Auto-refresh if called from Region Analysis window
                def _auto_refresh():
                    try:
                        if hasattr(win, '_refresh_labels_from_app'):
                            win._refresh_labels_from_app()
                    except: pass
                try:
                    win.after(500, _auto_refresh)
                except: pass

            except Exception as e:
                self.enqueue_log(f"[LLM] Phase 2 (NS Recovery) background error: {e}")
                _update_ui(f"Phase 2 (NS Recovery) error: {e}", 0)
                self._phase2_ns_running = False
                _set_title(_original_title)
                self._release_progress()  # release on error too

        t = threading.Thread(target=_bg_worker, daemon=True)
        t.start()
        self.enqueue_log("[LLM] Phase 2 (NS Recovery) running in background — you can continue working")

    def _show_memory_viewer(self, llm_win):
        """Open a window showing the agent's GSE context cache contents."""
        agent = getattr(llm_win, '_recall_agent', None)
        
        # Also check for saved cache on disk
        mem_path = self._gse_cache_path
        
        if agent is None and not os.path.exists(mem_path):
            messagebox.showinfo(
                "No Memory Yet",
                "The Phase 2 Re-extraction Agent hasn't run yet.\n\n"
                "Run an extraction with 'Phase 2 Re-extraction (Phase 2)' enabled,\n"
                "or check if gse_cache/gse_cache.json exists on disk.",
                parent=llm_win)
            return
        
        viewer = tk.Toplevel(llm_win)
        viewer.title("Agent GSE Context Viewer")
        viewer.geometry("900x700")
        
        # Top info bar
        info_frame = ttk.Frame(viewer, padding=5)
        info_frame.pack(fill=tk.X)
        
        if agent:
            ttk.Label(info_frame, text="Memory source: LIVE (current session)",
                      font=('Segoe UI', 10, 'bold'), foreground='green').pack(side=tk.LEFT, padx=5)
            ttk.Label(info_frame,
                      text=f"  |  {len(agent.gse_descriptions)} GSEs  |  "
                           f"{agent.n_corrected} corrected  |  "
                           f"{agent.n_confirmed} confirmed NS",
                      font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Label(info_frame, text=f"Memory source: {mem_path}",
                      font=('Segoe UI', 9, 'italic'), foreground='#666').pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = ttk.Frame(viewer, padding=3)
        btn_frame.pack(fill=tk.X)
        
        def _save_cache():
            if agent:
                self._save_persistent_cache(agent)
                messagebox.showinfo("Saved", f"Memory saved to:\n{self._gse_cache_path}", parent=viewer)
            else:
                messagebox.showinfo("No Live Memory", "No active agent — memory is on disk only.", parent=viewer)
        
        def _open_json():
            path = self._gse_cache_path
            if os.path.exists(path):
                import subprocess
                try:
                    subprocess.Popen(['xdg-open', path])
                except Exception:
                    messagebox.showinfo("Path", f"Open in text editor:\n{path}", parent=viewer)
            else:
                messagebox.showinfo("Not Found", f"No saved cache at:\n{path}", parent=viewer)
        
        tk.Button(btn_frame, text="Save Cache to Disk", command=_save_cache,
                  bg="#1976D2", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Open JSON File", command=_open_json,
                  font=('Segoe UI', 9), padx=10, cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        # Main content: text area with memory summary
        text_frame = ttk.Frame(viewer, padding=5)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        vsb = ttk.Scrollbar(text_frame, orient='vertical', command=text.yview)
        text.config(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        
        # Fill with memory content
        if agent:
            summary = agent.get_memory_summary()
            text.insert(tk.END, summary)
        elif os.path.exists(mem_path):
            try:
                import json as _json
                with open(mem_path, 'r') as f:
                    data = _json.load(f)
                
                info = data.get('_info', {})
                text.insert(tk.END, f"{'='*70}\n")
                text.insert(tk.END, f"  SAVED GSE CONTEXT CACHE\n")
                text.insert(tk.END, f"  Created: {info.get('created', '?')}\n")
                text.insert(tk.END, f"  Experiments: {info.get('n_experiments', '?')}\n")
                text.insert(tk.END, f"  Corrected: {info.get('n_corrected', '?')}\n")
                text.insert(tk.END, f"  Confirmed NS: {info.get('n_confirmed_ns', '?')}\n")
                text.insert(tk.END, f"{'='*70}\n\n")
                
                descs = data.get('gse_descriptions', {})
                text.insert(tk.END, f"  GSE DESCRIPTIONS ({len(descs)}):\n\n")
                for gse, desc in list(descs.items())[:30]:
                    text.insert(tk.END, f"  {gse}: {desc.get('title', '?')[:70]}\n")
                    summary_txt = desc.get('summary', '')[:120]
                    if summary_txt:
                        text.insert(tk.END, f"    {summary_txt}...\n")
                if len(descs) > 30:
                    text.insert(tk.END, f"\n  ... and {len(descs) - 30} more\n")
                
                consensus = data.get('gse_consensus', {})
                text.insert(tk.END, f"\n  GSE CONSENSUS LABELS ({len(consensus)}):\n\n")
                for gse, cols in list(consensus.items())[:20]:
                    title = descs.get(gse, {}).get('title', '?')[:50]
                    text.insert(tk.END, f"  {gse}: {title}\n")
                    for col, counts in cols.items():
                        if counts:
                            top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
                            vals = ", ".join(f'"{v}" ({c})' for v, c in top3)
                            text.insert(tk.END, f"    {col}: {vals}\n")
                
            except Exception as e:
                text.insert(tk.END, f"Error reading memory file: {e}")
        
        text.config(state=tk.DISABLED)

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
        
        inst_frame = ttk.Frame(label_win, relief=tk.RIDGE, borderwidth=2)
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
        
        cat_frame = ttk.LabelFrame(label_win, text="Select Category", padding=10)
        cat_frame.pack(fill=tk.X, padx=20, pady=10)
        
        category_var = tk.StringVar(value="Condition")
        
        ttk.Radiobutton(cat_frame, text="Condition (e.g., cancer, healthy, disease)", variable=category_var, value="Condition").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cat_frame, text="Tissue (e.g., liver, brain, blood)", variable=category_var, value="Tissue").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cat_frame, text="Treatment (e.g., drug A, control, untreated)", variable=category_var, value="Treatment").pack(anchor=tk.W, pady=2)
        
        input_frame = ttk.LabelFrame(label_win, text="Enter Label", padding=10)
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
                f"- Use Gene Explorer or Compare Distributions\n"
                f"- Use 'Interactive Analyzer'",
                parent=label_win
            )
            
            self.step2_status_label.config(
                text=f"OK {len(df_to_label):,} samples labeled manually - Ready for analysis", 
                foreground="green"
            )
            
            if messagebox.askyesno("Continue?", "Add another label category?", parent=label_win):
                label_entry.delete(0, tk.END)
                label_entry.focus()
            else:
                label_win.destroy()
        
        btn_frame = ttk.Frame(label_win)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Apply Label", command=apply_label, bg="#4CAF50", fg="white", font=('Segoe UI', 11, 'bold'), padx=30, pady=10, cursor="hand2").pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Cancel", command=label_win.destroy, bg="#757575", fg="white", font=('Segoe UI', 10), padx=20, pady=10).pack(side=tk.LEFT, padx=10)
    
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
            text=" Instructions: Select genes and platforms below, then click 'Plot'. "
                 "DRAG rectangles on histograms to select expression ranges. "
                 "Multiple regions can be selected. Click legend items to change colors!",
            wraplength=850,
            font=('Segoe UI', 9),
            foreground='#1976D2',
            background='#E3F2FD',
            padding=8,
            relief=tk.RAISED
        )
        inst_label.pack(fill=tk.X)
        
        plat_label_frame = ttk.LabelFrame(top_frame, text="Select Platforms", padding=5)
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
        tk.Button(dir_row, text="+ Add Data Directory...",
                  command=lambda: self._add_data_dir_and_refresh(popup),
                  bg="#1976D2", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.LEFT, padx=5)
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
        
        gene_label_frame = ttk.LabelFrame(top_frame, text="Enter Gene Symbols", padding=5)
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
        popup.label_status_indicator.pack(side=tk.LEFT, padx=6)
        # Update indicator based on main window state
        if self.label_source_var.get() == "file" and self.platform_labels:
            plats = ', '.join(sorted(self.platform_labels.keys()))
            total = sum(len(df) for df in self.platform_labels.values())
            cols = [c for c, v in self.labels_col_vars.items() if v.get()]
            popup.label_status_indicator.config(
                text=f"[Label] Per-platform labels: {plats} ({total:,} samples, {len(cols)} columns) -- LLM disabled",
                foreground="#1B5E20", background="#E8F5E9")
        elif self.label_source_var.get() == "file" and self.default_labels_df is not None:
            n = len(self.default_labels_df)
            cols = [c for c, v in self.labels_col_vars.items() if v.get()]
            popup.label_status_indicator.config(
                text=f"[Label] Labels loaded ({n:,} samples, {len(cols)} columns) -- LLM disabled",
                foreground="#1B5E20", background="#E8F5E9")
        else:
            popup.label_status_indicator.config(
                text="[Label] Labels: LLM Extraction (Ollama)  -  change in main window ^",
                foreground="#555", background="#F5F5F5")
        
        btn_frame = ttk.Frame(popup, padding=5)
        btn_frame.pack(fill=tk.X)
        
        plot_btn = tk.Button(
            btn_frame, 
            text=" Plot Distributions", 
            command=lambda: self._plot_histograms(popup),
            bg="#4CAF50", 
            fg="white", 
            font=('Segoe UI', 11, 'bold'), 
            padx=20, 
            pady=8, 
            cursor="hand2", 
            relief=tk.RAISED, 
            bd=2
        )
        plot_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(plot_btn, "#388E3C", "#4CAF50")
        
        popup.analyze_selection_btn = tk.Button(
            btn_frame, 
            text=" Analyze Selected Range(s)", 
            command=lambda: self._pre_analyze_dialog(popup),
            bg="#FF9800", 
            fg="white", 
            font=('Segoe UI', 10, 'bold'), 
            padx=15, 
            pady=8, 
            cursor="hand2", 
            state=tk.DISABLED, 
            relief=tk.RAISED, 
            bd=2
        )
        popup.analyze_selection_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(popup.analyze_selection_btn, "#F57C00", "#FF9800")
        
        popup.compare_btn = tk.Button(
            btn_frame, 
            text=" Compare Regions", 
            command=lambda: self._compare_regions_logic(popup),
            bg="#9C27B0", 
            fg="white", 
            font=('Segoe UI', 10, 'bold'), 
            padx=15, 
            pady=8, 
            cursor="hand2", 
            state=tk.DISABLED, 
            relief=tk.RAISED, 
            bd=2
        )
        popup.compare_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(popup.compare_btn, "#7B1FA2", "#9C27B0")
        
        # ── Highlight Region button (opens specification dialog) ──
        highlight_btn = tk.Button(
            btn_frame,
            text="Highlight Region...",
            command=lambda: self._open_highlight_region_dialog(popup),
            bg="#C62828",
            fg="white",
            font=('Segoe UI', 10, 'bold'),
            padx=15,
            pady=8,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        highlight_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(highlight_btn, "#B71C1C", "#C62828")

        clear_btn = tk.Button(
            btn_frame, 
            text=" Clear Selections", 
            command=lambda: self._clear_selections_logic(popup),
            bg="#757575", 
            fg="white", 
            font=('Segoe UI', 10), 
            padx=15, 
            pady=8, 
            cursor="hand2", 
            relief=tk.RAISED, 
            bd=2
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        self._add_button_hover(clear_btn, "#616161", "#757575")
        
        popup.selection_label = ttk.Label(
            btn_frame, 
            text="No regions selected", 
            font=('Segoe UI', 9, 'italic'), 
            foreground='gray'
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
                text="LLM mode: samples labeled by Ollama during analysis.",
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
                        # Single-value label column — still useful for identification
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

            # User-provided labels are kept AS-IS — no harmonization.
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

    def _ask_phase3_curator(self, parent_win=None):
        """Ask user if they want to run Phase 3 (LLM Curator) after Phase 1 or Phase 2."""
        if not self.platform_labels:
            return

        # Count unique labels to show in dialog
        merged = self.default_labels_df
        if merged is None or merged.empty:
            self._rebuild_merged_labels()
            merged = self.default_labels_df
        if merged is None or merged.empty:
            return

        _CURATE = {'Condition', 'Tissue', 'Treatment'}
        label_info = []
        for field in _CURATE:
            if field in merged.columns:
                n = merged[field].fillna('NS').astype(str).str.strip()
                real = [v for v in n if v.lower() not in
                        ('not specified', 'n/a', 'unknown', 'nan', '')]
                n_unique = len(set(real))
                label_info.append(f"  {field}: {n_unique} unique labels")

        parent = parent_win if parent_win else self
        response = messagebox.askyesno(
            "Phase 3 — LLM Curator (Cross-Experiment Harmonization)",
            f"Labels are ready. Run Phase 3 (LLM Curator)?\n\n"
            f"Current label inventory:\n"
            + "\n".join(label_info) + "\n\n"
            f"{'─' * 50}\n"
            f"PHASE 3 — LLM Curator:\n"
            f"  • Scans ALL unique labels across experiments\n"
            f"  • Finds candidate pairs (e.g., 'AML' vs 'Acute Myeloid Leukemia')\n"
            f"  • Asks LLM: 'Are these the same biomedical concept?'\n"
            f"  • You review proposed merges before applying\n"
            f"  • HSV ≠ HIV (LLM rejects) | AML = Acute Myeloid Leukemia (LLM confirms)\n\n"
            f"{'─' * 50}\n"
            f"Yes = open LLM Curator window\n"
            f"No  = keep labels as-is (run later via 'Curate Labels' button)",
            parent=parent)

        if response:
            self._open_llm_curator()

    def _open_llm_curator(self):
        """Open LLM Curator window for cross-experiment label harmonization.
        Reviews label inventory, asks LLM about similar labels, proposes merges.
        User reviews and confirms before applying.
        """
        if not self.platform_labels:
            messagebox.showinfo("No Labels", "Load label files first.", parent=self)
            return

        # Build merged df
        merged = self.default_labels_df
        if merged is None or merged.empty:
            self._rebuild_merged_labels()
            merged = self.default_labels_df
        if merged is None or merged.empty:
            messagebox.showinfo("No Labels", "No labels loaded.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("LLM Label Curator — Cross-Experiment Harmonization")
        win.geometry("900x700")
        try:
            win.transient(self)
        except: pass

        # ── Header ──
        hdr = ttk.Frame(win)
        hdr.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(hdr, text="LLM Label Curator",
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)

        # Label inventory summary
        from collections import Counter
        _CURATE = {'Condition', 'Tissue', 'Treatment'}
        summary_parts = []
        for field in _CURATE:
            if field in merged.columns:
                vals = merged[field].fillna('NS').astype(str).str.strip()
                real = [v for v in vals if v.lower() not in
                        ('not specified', 'n/a', 'unknown', 'nan', '')]
                n_unique = len(set(real))
                summary_parts.append(f"{field}: {n_unique} unique")

        tk.Label(hdr, text=f"  {', '.join(summary_parts)}  |  "
                 f"{len(merged):,} total samples",
                 font=("Segoe UI", 9), foreground="gray").pack(side=tk.LEFT, padx=10)

        # ── Ollama Settings ──
        settings_frame = ttk.LabelFrame(win, text=" Ollama", padding=4)
        settings_frame.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(settings_frame, text="URL:").pack(side=tk.LEFT)
        cur_url = tk.StringVar(value=_OLLAMA_URL)
        ttk.Entry(settings_frame, textvariable=cur_url, width=25).pack(side=tk.LEFT, padx=4)
        ttk.Label(settings_frame, text="Workers:").pack(side=tk.LEFT, padx=(8, 2))
        cur_workers = tk.IntVar(value=0)
        ttk.Spinbox(settings_frame, from_=0, to=8, textvariable=cur_workers, width=3).pack(side=tk.LEFT)
        ttk.Label(settings_frame, text="(0=auto)", foreground="gray",
                  font=('Segoe UI', 7, 'italic')).pack(side=tk.LEFT, padx=2)

        # ── Results area ──
        results_frame = ttk.LabelFrame(win, text=" Proposed Merges", padding=4)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Treeview for merge proposals
        cols = ('Field', 'From', 'To', 'Reason', 'From#', 'To#')
        tree = ttk.Treeview(results_frame, columns=cols, show='headings', height=18)
        for c in cols:
            tree.heading(c, text=c)
        tree.column('Field', width=80, stretch=False)
        tree.column('From', width=200)
        tree.column('To', width=200)
        tree.column('Reason', width=200)
        tree.column('From#', width=50, stretch=False)
        tree.column('To#', width=50, stretch=False)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=sb.set)

        # ── Progress ──
        prog_frame = ttk.Frame(win)
        prog_frame.pack(fill=tk.X, padx=10, pady=4)
        prog_bar = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate")
        prog_bar.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 8))
        prog_lbl = ttk.Label(prog_frame, text="Ready", font=('Segoe UI', 9))
        prog_lbl.pack(side=tk.LEFT)

        # ── Buttons ──
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        win._curator = None
        win._proposals = {}

        def _scan():
            """Run LLM scan in background thread."""
            global _OLLAMA_URL
            _OLLAMA_URL = cur_url.get().strip() or "http://localhost:11434"

            for item in tree.get_children():
                tree.delete(item)
            prog_lbl.config(text="Scanning labels...")
            prog_bar["value"] = 0
            scan_btn.config(state=tk.DISABLED)
            apply_btn.config(state=tk.DISABLED)

            def _progress(done, total, msg):
                try:
                    pct = int(done * 100 / max(1, total))
                    win.after(0, lambda: prog_bar.config(value=pct))
                    win.after(0, lambda: prog_lbl.config(text=f"{done}/{total}: {msg}"))
                except: pass

            def _bg():
                try:
                    curator = LLMCurator(log_func=self.enqueue_log)
                    proposals = curator.scan_and_propose(
                        merged, progress_fn=_progress)
                    win._curator = curator
                    win._proposals = proposals

                    def _update_ui():
                        for field, items in proposals.items():
                            for from_l, to_l, reason, fc, tc in items:
                                tree.insert('', tk.END, values=(
                                    field, from_l, to_l, reason, fc, tc))
                        total = sum(len(v) for v in proposals.values())
                        prog_lbl.config(text=f"Done: {total} merges proposed")
                        prog_bar["value"] = 100
                        scan_btn.config(state=tk.NORMAL)
                        if total > 0:
                            apply_btn.config(state=tk.NORMAL)
                    win.after(0, _update_ui)
                except Exception as e:
                    self.enqueue_log(f"[Curator] Error: {e}")
                    def _err():
                        prog_lbl.config(text=f"Error: {e}")
                        scan_btn.config(state=tk.NORMAL)
                    win.after(0, _err)

            threading.Thread(target=_bg, daemon=True).start()

        def _remove_selected():
            """Remove selected proposal from list."""
            sel = tree.selection()
            for item in sel:
                tree.delete(item)

        def _apply():
            """Apply accepted merges to loaded labels."""
            # Build proposals from treeview (user may have removed some)
            proposals = {}
            for item in tree.get_children():
                vals = tree.item(item, 'values')
                field, from_l, to_l, reason = vals[0], vals[1], vals[2], vals[3]
                fc, tc = int(vals[4]), int(vals[5])
                proposals.setdefault(field, []).append(
                    (from_l, to_l, reason, fc, tc))

            if not proposals:
                messagebox.showinfo("Nothing to Apply", "No merges in the list.", parent=win)
                return

            total = sum(len(v) for v in proposals.values())
            if not messagebox.askyesno("Apply Merges",
                    f"Apply {total} label merges?\n\n"
                    f"This will modify your loaded labels.\n"
                    f"Original labels are NOT overwritten on disk\n"
                    f"until you save.", parent=win):
                return

            # Apply to each platform's labels
            n_changed = 0
            for plat_id, plat_df in self.platform_labels.items():
                self.platform_labels[plat_id] = LLMCurator.apply_merges(
                    plat_df, proposals, log_func=self.enqueue_log)
                n_changed += 1

            self._rebuild_merged_labels()
            self._refresh_labels_display()

            # Save curated labels
            save_dir = os.path.join(self.data_dir, "labels")
            os.makedirs(save_dir, exist_ok=True)
            for plat_id, plat_df in self.platform_labels.items():
                fpath = os.path.join(save_dir, f"{plat_id}_labels.csv")
                plat_df.to_csv(fpath, index=False)
            self.enqueue_log(f"[Curator] Saved curated labels to {save_dir}")

            messagebox.showinfo("Applied",
                f"Applied {total} merges to {n_changed} platform(s).\n"
                f"Labels saved to {save_dir}", parent=win)
            prog_lbl.config(text=f"Applied {total} merges to {n_changed} platform(s)")

        scan_btn = tk.Button(btn_frame, text=" Scan Labels (LLM)", command=_scan,
                             bg="#1976D2", fg="white", font=('Segoe UI', 10, 'bold'),
                             padx=15, pady=5, cursor="hand2")
        scan_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="Remove Selected", command=_remove_selected,
                  bg="#757575", fg="white", font=('Segoe UI', 9),
                  padx=10, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)

        apply_btn = tk.Button(btn_frame, text=" Apply Merges", command=_apply,
                              bg="#388E3C", fg="white", font=('Segoe UI', 10, 'bold'),
                              padx=15, pady=5, cursor="hand2", state=tk.DISABLED)
        apply_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="Close", command=win.destroy,
                  font=('Segoe UI', 9), padx=10, pady=5).pack(side=tk.RIGHT, padx=4)

        ttk.Label(btn_frame,
                  text="  Scan → Review → Remove bad merges → Apply",
                  font=('Segoe UI', 8, 'italic'), foreground='gray').pack(side=tk.LEFT, padx=8)

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

    def _refresh_labels_display(self):
        """Update the per-platform label status display and column checkboxes."""
        # Clear old platform status
        for w in self.labels_plat_frame.winfo_children():
            w.destroy()

        if not self.platform_labels:
            self.labels_status_lbl.config(
                text="No label files loaded. Click '+ Add Label File...' to add per-platform labels.",
                foreground="gray")
            for w in self.labels_col_frame.winfo_children():
                w.destroy()
            self.labels_col_vars.clear()
            return

        # Show per-platform status with remove buttons
        for plat_id in sorted(self.platform_labels.keys()):
            row = ttk.Frame(self.labels_plat_frame)
            row.pack(fill=tk.X, pady=1)
            n = len(self.platform_labels[plat_id])
            loaded_as_expr = plat_id in (self.gpl_datasets or {})
            fg = "#1B5E20" if loaded_as_expr else "#555"
            marker = "[v]" if loaded_as_expr else "[ ]"
            cols = [c for c in self.platform_labels[plat_id].columns
                    if c not in ('GSM', '_platform')]
            ttk.Label(row, text=f"  {marker} {plat_id}: {n:,} samples, "
                      f"{len(cols)} columns",
                      font=("Segoe UI", 9), foreground=fg).pack(side=tk.LEFT)
            tk.Button(row, text="x", command=lambda p=plat_id: self._remove_platform_label(p),
                      bg="#EF5350", fg="white", font=("Segoe UI", 8, "bold"),
                      width=2, padx=0, pady=0, bd=1, cursor="hand2").pack(side=tk.LEFT, padx=4)

        # Collect all label columns across all platforms
        all_label_cols = set()
        for df in self.platform_labels.values():
            for c in df.columns:
                if c not in ('GSM', '_platform'):
                    if df[c].dtype == 'object' or df[c].nunique() < 200:
                        if df[c].nunique() > 1:
                            all_label_cols.add(c)

        # Populate column checkboxes (keep existing selections)
        old_selections = {c: v.get() for c, v in self.labels_col_vars.items()}
        for w in self.labels_col_frame.winfo_children():
            w.destroy()
        self.labels_col_vars.clear()

        if all_label_cols:
            ttk.Label(self.labels_col_frame, text="Use columns:",
                      font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=2)
            for c in sorted(all_label_cols):
                val = old_selections.get(c, True)
                var = tk.BooleanVar(value=val)
                self.labels_col_vars[c] = var
                ttk.Checkbutton(self.labels_col_frame, text=c,
                                variable=var).pack(side=tk.LEFT, padx=4)

        # Update status
        total = sum(len(df) for df in self.platform_labels.values())
        plats = ', '.join(sorted(self.platform_labels.keys()))
        self.labels_status_lbl.config(
            text=f"OK {len(self.platform_labels)} platform(s): {plats}  |  "
                 f"{total:,} total samples  |  LLM labeling disabled",
            foreground="#1B5E20")

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
                    f"Load Expression Data? — {gpl_id}",
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
                    f"Load Expression Data? — {len(missing)} Platform(s)",
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
                # Not fully loaded — try gene-only quick load
                self.enqueue_log(f"[Gene Explorer] {plat} not fully loaded — trying quick gene load...")
                self.status_label.config(text=f"Quick-loading {plat} genes...", foreground="blue")
                self.update_idletasks()
                ok = self._quick_load_genes(plat, genes)
                if not ok:
                    self.enqueue_log(f"[Gene Explorer] {plat}: quick load failed — genes may not be found")
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
                plot_win.title(f"{gene} - {plat} Distribution")
                plot_win.geometry("1100x800")
                try:
                    _sw, _sh = plot_win.winfo_screenwidth(), plot_win.winfo_screenheight()
                    plot_win.geometry(f"1100x800+{(_sw-1100)//2}+{(_sh-800)//2}")
                    plot_win.minsize(500, 400)
                except Exception: pass
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
                
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
                
                # Create canvas
                canvas = FigureCanvasTkAgg(fig, master=plot_win)
                canvas.draw()
                # DO NOT plt.close(fig) - canvas needs figure alive
                
                canvas_widget = canvas.get_tk_widget()
                
                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, plot_win)
                toolbar.update()
                toolbar.pack(side=tk.TOP, fill=tk.X)
                
                canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
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
        gene_colors = ['#1976D2', '#C62828', '#2E7D32', '#F57C00', '#7B1FA2',
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
            plot_win.title(f"{plat} - {len(gene_cols)} genes overlaid")
            plot_win.geometry("1100x800")
            try:
                _sw, _sh = plot_win.winfo_screenwidth(), plot_win.winfo_screenheight()
                plot_win.geometry(f"1100x800+{(_sw-1100)//2}+{(_sh-800)//2}")
                plot_win.minsize(500, 400)
            except Exception: pass

            fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

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

            ax.set_xlabel("Expression", fontsize=10)
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
            canvas_widget = canvas.get_tk_widget()
            toolbar = NavigationToolbar2Tk(canvas, plot_win)
            toolbar.update()
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

        chosen_color = '#FF6F00'
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
            foreground='#FF6F00'
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
            ax.set_title(f"{plat} - {col}\n[!] No data", color='orange', fontsize=10, weight='bold')
            ax.axis("off")
            return None
        
        dist_class = BioAI_Engine.analyze_gene_distribution(expr)
        
        num_bins = Plotter.get_optimal_bins(expr, method='auto')
        
        mean_val = expr.mean()
        std_val = expr.std()
        median_val = expr.median()
        
        title_text = f"{plat} - {col} | {dist_class} | n={len(expr):,} | u={mean_val:.2f} SD={std_val:.2f}"
        ax.set_title(title_text, fontsize=10, weight='bold', pad=15)
        
        plot_cfg = CONFIG['plotting']['histogram']
        counts, bins, patches = ax.hist(
            expr, 
            bins=num_bins,
            edgecolor=plot_cfg['edge_color'],
            alpha=plot_cfg['alpha'],
            color=plot_cfg['default_color'],
            linewidth=0.5
        )
        
        if len(counts) > 0 and max(counts) > 0:
            ax.set_ylim(0, max(counts) * 1.15)
        
        min_samples = CONFIG['plotting']['histogram']['min_samples_for_kde']
        min_variance = CONFIG['plotting']['histogram']['min_variance_for_kde']
        
        if len(expr) >= min_samples and np.var(expr) > min_variance:
            try:
                kde = gaussian_kde(expr)
                x_range = np.linspace(expr.min(), expr.max(), 200)
                kde_vals = kde(x_range)
                
                ax2 = ax.twinx()
                ax2.plot(x_range, kde_vals, 'r-', alpha=0.6, linewidth=2, label='KDE')
                ax2.set_ylabel('Density', fontsize=8, color='red')
                ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
                ax2.set_ylim(0, kde_vals.max() * 1.2)
                
                ax2.legend(loc='upper right', fontsize=7, framealpha=0.7)
                
            except Exception as e:
                pass
        
        ax.set_xlabel("Expression", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2, linestyle='--')
        
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
        dlg.title("Highlight Region — Specify Criteria")
        dlg.transient(popup)

        # Header
        tk.Label(dlg, text="Define Region to Highlight",
                 font=('Segoe UI', 13, 'bold'), bg='#1A237E', fg='white',
                 pady=8).pack(fill=tk.X)

        main = ttk.Frame(dlg, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Method selection ──
        method_frame = ttk.LabelFrame(main, text="Method", padding=8)
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
        dir_frame = ttk.LabelFrame(main, text="Direction", padding=8)
        dir_frame.pack(fill=tk.X, pady=5)

        dir_var = tk.StringVar(value="above")
        dir_row = ttk.Frame(dir_frame)
        dir_row.pack(fill=tk.X)
        for val, label in [("above", "Above threshold"), ("below", "Below threshold"),
                           ("between", "Between two values"), ("outside", "Outside (both tails)")]:
            ttk.Radiobutton(dir_row, text=label, variable=dir_var, value=val,
                            command=lambda: _update_preview()).pack(side=tk.LEFT, padx=8)

        # ── Value inputs ──
        val_frame = ttk.LabelFrame(main, text="Value(s)", padding=8)
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
        preview_frame = ttk.LabelFrame(main, text="Preview (computed thresholds per gene)", padding=6)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        preview_text = tk.Text(preview_frame, font=('Consolas', 9), height=6,
                                wrap=tk.WORD, state=tk.DISABLED, bg='#FAFAFA')
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
        tk.Button(btn_row, text="  Apply Highlight  ", command=_apply,
                  bg="#C62828", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="Cancel", command=dlg.destroy,
                  font=('Segoe UI', 10), padx=15, pady=4).pack(side=tk.RIGHT, padx=5)

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

        # If using file labels, skip dialog — no LLM extraction needed
        if self.label_source_var.get() == "file":
            self._analyze_selected_range(popup)
            return

        # ── LLM Extraction Settings Dialog ──
        dlg = tk.Toplevel(popup)
        dlg.title("LLM Extraction Settings")
        dlg.transient(popup)
        dlg.grab_set()

        ttk.Label(dlg, text="Select labels to extract for the selected region(s):",
                  font=('Segoe UI', 11, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="The LLM agent will classify samples in your selected range.",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        # Field checkboxes
        fields_frame = ttk.LabelFrame(dlg, text="Standard Fields", padding=10)
        fields_frame.pack(fill=tk.X, padx=15, pady=5)

        STANDARD_FIELDS = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']
        field_vars = {}
        for f in STANDARD_FIELDS:
            var = tk.BooleanVar(value=f in self._extraction_fields)
            field_vars[f] = var
            ttk.Checkbutton(fields_frame, text=f.replace('_', ' '), variable=var).pack(
                side=tk.LEFT, padx=8)

        # Custom fields
        custom_frame = ttk.LabelFrame(dlg, text="Custom Fields (optional)", padding=10)
        custom_frame.pack(fill=tk.X, padx=15, pady=5)

        custom_entries = []
        custom_rows_frame = ttk.Frame(custom_frame)
        custom_rows_frame.pack(fill=tk.X)

        def _add_custom_row(name_val="", prompt_val=""):
            row = ttk.Frame(custom_rows_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="Name:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)
            name_e = ttk.Entry(row, width=15)
            name_e.pack(side=tk.LEFT, padx=2)
            ttk.Label(row, text="Prompt:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)
            prompt_e = ttk.Entry(row, width=30)
            prompt_e.pack(side=tk.LEFT, padx=2)
            if name_val:
                name_e.insert(0, name_val)
            if prompt_val:
                prompt_e.insert(0, prompt_val)
            # Remove button
            def _remove():
                custom_entries.remove((name_e, prompt_e))
                row.destroy()
            tk.Button(row, text="✕", command=_remove, fg="red",
                      font=('Segoe UI', 8, 'bold'), padx=4, pady=0,
                      relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=4)
            custom_entries.append((name_e, prompt_e))

        # Pre-fill from saved custom fields
        for cf in self._extraction_custom_fields:
            _add_custom_row(cf.get('name', ''), cf.get('prompt', ''))

        tk.Button(custom_frame, text="+ Add Custom Field", command=_add_custom_row,
                  bg="#43A047", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, pady=2, cursor="hand2").pack(anchor=tk.W, pady=(5, 0))

        # Phase 2 Re-extraction checkbox
        recall_frame = ttk.Frame(dlg)
        recall_frame.pack(fill=tk.X, padx=15, pady=8)
        recall_var = tk.BooleanVar(value=self._extraction_recall)
        ttk.Checkbutton(recall_frame, text="Phase 2 Re-extraction (Phase 2)",
                        variable=recall_var).pack(side=tk.LEFT)

        # Memory status
        if self._gse_saved_cache:
            n_gse = len(self._gse_saved_cache.get('gse_descriptions', {}))
            ttk.Label(recall_frame, text=f"  ({n_gse} GSEs in memory)",
                      font=('Segoe UI', 8, 'italic'), foreground='green').pack(side=tk.LEFT)

        # ── Extraction Mode: Fast vs Full with descriptions ──
        mode_frame = ttk.LabelFrame(dlg, text=" Extraction Mode", padding=8)
        mode_frame.pack(fill=tk.X, padx=15, pady=5)
        fast_var = tk.BooleanVar(value=getattr(self, '_extraction_fast_mode', True))

        fast_rb = ttk.Radiobutton(mode_frame, text="Fast Mode (recommended for first pass)",
                                   variable=fast_var, value=True)
        fast_rb.pack(anchor=tk.W, padx=5)
        tk.Label(mode_frame,
            text="  Phase 1: Raw LLM extraction (VRAM-aware parallel workers)\n"
                 "  Phase 1.5: Per-GSE label collapsing (exact match + abbreviation initials)\n"
                 "  → Labels loaded instantly. Phase 2 / Curator can be run later.",
            font=('Segoe UI', 8), fg='#555', justify=tk.LEFT, anchor='nw').pack(anchor=tk.W, padx=20)

        full_rb = ttk.Radiobutton(mode_frame, text="Full Mode (higher quality, longer)",
                                   variable=fast_var, value=False)
        full_rb.pack(anchor=tk.W, padx=5, pady=(6, 0))
        tk.Label(mode_frame,
            text="  Phase 1: Raw LLM extraction (VRAM-aware parallel workers)\n"
                 "  Phase 1.5: Per-GSE label collapsing (exact + abbreviation)\n"
                 "  → Labels loaded. Then asks to continue:\n"
                 "  Phase 2: Recover 'Not Specified' labels using NCBI GEO experiment\n"
                 "    descriptions + sibling sample consensus (background)",
            font=('Segoe UI', 8), fg='#555', justify=tk.LEFT, anchor='nw').pack(anchor=tk.W, padx=20)

        # NS curation extra fields
        ns_frame = ttk.Frame(mode_frame)
        ns_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Label(ns_frame, text="'Not Specified' curation applies to: Condition, Tissue, Treatment",
                  font=('Segoe UI', 8), foreground='#555').pack(side=tk.LEFT)
        ns_extra = getattr(self, '_extraction_ns_extra', set())
        ns_age_var = tk.BooleanVar(value='Age' in ns_extra)
        ns_tt_var = tk.BooleanVar(value='Treatment_Time' in ns_extra)
        ttk.Checkbutton(ns_frame, text="+ Age", variable=ns_age_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(ns_frame, text="+ Treatment_Time", variable=ns_tt_var).pack(side=tk.LEFT, padx=5)

        # ── Ollama Settings ──
        ollama_s_frame = ttk.LabelFrame(dlg, text=" Ollama Settings", padding=6)
        ollama_s_frame.pack(fill=tk.X, padx=15, pady=5)
        # URL
        url_s_row = ttk.Frame(ollama_s_frame)
        url_s_row.pack(fill=tk.X, pady=2)
        ttk.Label(url_s_row, text="URL:").pack(side=tk.LEFT)
        ra_url_var = tk.StringVar(value=_OLLAMA_URL)
        ttk.Entry(url_s_row, textvariable=ra_url_var, width=28).pack(side=tk.LEFT, padx=4)

        # Hardware info
        ra_hw_lbl = ttk.Label(ollama_s_frame, text="Scanning...",
                              foreground="gray", font=('Consolas', 8))
        ra_hw_lbl.pack(anchor=tk.W, pady=(4, 2))

        # GPU Workers
        ra_gpu_row = ttk.Frame(ollama_s_frame)
        ra_gpu_row.pack(fill=tk.X, pady=2)
        ttk.Label(ra_gpu_row, text="GPU workers:").pack(side=tk.LEFT)
        ra_worker_var = tk.IntVar(value=0)
        ttk.Spinbox(ra_gpu_row, from_=0, to=16, textvariable=ra_worker_var,
                     width=3).pack(side=tk.LEFT, padx=4)
        ttk.Label(ra_gpu_row, text="(0=auto)", foreground="gray",
                  font=('Segoe UI', 7, 'italic')).pack(side=tk.LEFT)

        # CPU Workers
        ra_cpu_row = ttk.Frame(ollama_s_frame)
        ra_cpu_row.pack(fill=tk.X, pady=2)
        ttk.Label(ra_cpu_row, text="CPU workers:").pack(side=tk.LEFT)
        ra_cpu_var = tk.IntVar(value=0)
        ttk.Spinbox(ra_cpu_row, from_=0, to=64, textvariable=ra_cpu_var,
                     width=3).pack(side=tk.LEFT, padx=4)
        ttk.Label(ra_cpu_row, text="(overflow when VRAM full)", foreground="gray",
                  font=('Segoe UI', 7, 'italic')).pack(side=tk.LEFT)

        # Total + status
        ra_total_lbl = ttk.Label(ollama_s_frame, text="Total: auto",
                                 foreground="#1565C0", font=('Segoe UI', 9, 'bold'))
        ra_total_lbl.pack(anchor=tk.W, pady=2)

        def _ra_update_total(*_):
            g = ra_worker_var.get()
            c = ra_cpu_var.get()
            if g == 0 and c == 0:
                ra_total_lbl.config(text="Total workers: auto-detect")
            else:
                ra_total_lbl.config(text=f"Total: {g + c} ({g} GPU + {c} CPU)")
        ra_worker_var.trace_add("write", _ra_update_total)
        ra_cpu_var.trace_add("write", _ra_update_total)

        ra_gpu_lbl = ttk.Label(ollama_s_frame, text="", foreground="gray",
                               font=('Segoe UI', 8, 'italic'))
        ra_gpu_lbl.pack(anchor=tk.W)

        def _ra_gpu_info():
            try:
                import os as _os, psutil as _ps
                gpus = detect_gpus()
                auto_w = compute_ollama_parallel()
                if isinstance(auto_w, tuple):
                    auto_total, auto_gpu, auto_cpu = auto_w
                else:
                    auto_total, auto_gpu, auto_cpu = auto_w, auto_w, 0

                cpu_count = _os.cpu_count() or 1
                ram = _ps.virtual_memory()

                hw_parts = [f"CPU: {cpu_count} cores | RAM: {ram.available/1e9:.1f}/{ram.total/1e9:.1f} GB"]
                if gpus:
                    for g in gpus:
                        hw_parts.append(f"GPU: {g['name']} | VRAM: {g['free_vram_gb']:.1f}/{g['vram_gb']:.1f} GB")
                    ra_gpu_lbl.config(text=f"Recommended: {auto_gpu} GPU + {auto_cpu} CPU = {auto_total}")
                else:
                    hw_parts.append("GPU: None (CPU-only)")
                    ra_gpu_lbl.config(text=f"Recommended: {auto_total} CPU workers")

                ra_hw_lbl.config(text="\n".join(hw_parts))

                if ra_worker_var.get() == 0 and ra_cpu_var.get() == 0:
                    ra_worker_var.set(auto_gpu)
                    ra_cpu_var.set(auto_cpu)
            except Exception:
                pass
        dlg.after(300, _ra_gpu_info)

        # Buttons
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        def _ok():
            # Save settings
            selected = [f for f, v in field_vars.items() if v.get()]
            custom_list = []
            for name_e, prompt_e in custom_entries:
                n = name_e.get().strip()
                p = prompt_e.get().strip()
                if n:
                    custom_list.append({'name': n, 'prompt': p if p else f"string (extract {n})"})

            if not selected and not custom_list:
                messagebox.showwarning("No Labels", "Select at least one label to extract.", parent=dlg)
                return

            self._extraction_fields = selected
            self._extraction_custom_fields = custom_list
            self._extraction_recall = recall_var.get() and not fast_var.get()
            self._extraction_fast_mode = fast_var.get()
            ns_extra = set()
            if ns_age_var.get(): ns_extra.add('Age')
            if ns_tt_var.get(): ns_extra.add('Treatment_Time')
            self._extraction_ns_extra = ns_extra

            # Apply Ollama settings
            global _OLLAMA_URL
            _OLLAMA_URL = ra_url_var.get().strip() or "http://localhost:11434"
            self.ai_agent.MAX_WORKERS = ra_worker_var.get()  # 0 = auto from VRAM

            dlg.destroy()
            self._analyze_selected_range(popup)

        def _cancel():
            dlg.destroy()

        tk.Button(btn_frame, text="  Run Analysis  ", command=_ok,
                  bg="#FF9800", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=_cancel,
                  font=('Segoe UI', 10), padx=15, pady=6).pack(side=tk.RIGHT, padx=5)

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
                # Force numeric conversion - critical for downloaded platforms
                expr_col = pd.to_numeric(dfg[col], errors='coerce')
                subset = dfg[expr_col.between(low, high)]
                
                if subset.empty:
                    self.enqueue_log(f"[Analysis] Region {gene}[{low:.2f}–{high:.2f}]: empty after filter")
                    continue
                
                if 'GSM' not in subset.columns:
                    self.enqueue_log(f"[Analysis] [!] {plat} has no GSM column - attempting fallback")
                    # Try to find a GSM-like column
                    gsm_cands = [c for c in subset.columns if 'gsm' in c.lower()]
                    if gsm_cands:
                        subset = subset.rename(columns={gsm_cands[0]: 'GSM'})
                        self.enqueue_log(f"[Analysis] Found '{gsm_cands[0]}' -> GSM")
                    else:
                        # Create synthetic GSM IDs from index
                        subset = subset.copy()
                        subset['GSM'] = [f"SAMPLE_{j}" for j in range(len(subset))]
                        self.enqueue_log(f"[Analysis] No GSM column - created synthetic IDs for {len(subset)} samples")

                gsms = subset['GSM'].unique().tolist()
                expr_vals = pd.to_numeric(subset[col], errors='coerce').dropna().astype(float)

                # Store essential + all metadata columns for TOTAL platform scope
                # Keep GSM + expression + all string/metadata columns (series_id, titles, etc.)
                keep_cols = ['GSM', col]
                meta_cols = [c for c in dfg.columns
                             if c not in keep_cols
                             and (dfg[c].dtype == 'object'
                                  or c in ('series_id', 'title', 'source_name_ch1',
                                           'organism_ch1', 'characteristics_ch1'))]
                # Cap at reasonable memory: skip columns with >50% unique values AND >10k uniques
                for mc in meta_cols:
                    nuniq = dfg[mc].nunique()
                    if nuniq < 10000 or nuniq < len(dfg) * 0.5:
                        keep_cols.append(mc)
                platform_slim = dfg[keep_cols].copy()

                # Ensure GSM is uppercase for label matching
                if 'GSM' in platform_slim.columns:
                    platform_slim['GSM'] = platform_slim['GSM'].astype(str).str.strip().str.upper()

                # Labels are merged by RegionAnalysisWindow._precompute()
                # using platform_labels_df passed separately.

                region_specs.append({
                    'label': f"{gene}_R{len(region_specs)+1}_{plat}",
                    'gene': gene, 'platform': plat, 'column': col,
                    'range': (low, high),
                    'color': color if isinstance(color, str) else '#FF6F00',
                    'expression_values': expr_vals,
                    'gsm_list': gsms,
                    'platform_df': platform_slim,
                })

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
        progress_win.title("Region Analysis Pipeline")
        progress_win.geometry("500x220")
        progress_win.transient(popup)
        progress_win.grab_set()

        ttk.Label(progress_win, text="Running comprehensive region analysis...",
                  font=('Segoe UI', 11, 'bold')).pack(pady=(15, 5))

        prog_bar = ttk.Progressbar(progress_win, mode='determinate', length=400, maximum=100)
        prog_bar.pack(pady=8)

        status_lbl = ttk.Label(progress_win, text="Initializing...", foreground='gray', wraplength=450)
        status_lbl.pack(pady=5)

        detail_lbl = ttk.Label(progress_win, text="", foreground='#1976D2', font=('Segoe UI', 9))
        detail_lbl.pack()

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
            self.enqueue_log("[Labels] Using LLM extraction (Ollama)")

        # ── 4. Background processing thread ─────────────────────────────
        def process_regions():
            try:
                import tempfile, gzip

                n_regions = len(region_specs)

                # Step A: Load GEOmetadb once for all regions
                _update_status("Loading GEO database...", 5)
                gz_path = CONFIG['paths']['geo_db']

                with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
                    tmp_path = tmp.name
                    with gzip.open(gz_path, "rb") as gzfi:
                        tmp.write(gzfi.read())

                thread_conn = sqlite3.connect(tmp_path)
                thread_conn.text_factory = lambda b: b.decode('utf-8', 'replace')

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
                                fast_mode = getattr(self, '_extraction_fast_mode', False)
                                
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

                                # Apply stored Ollama settings (shared with LLM Extraction window)
                                global _OLLAMA_URL
                                _OLLAMA_URL = getattr(self, '_ollama_url', _OLLAMA_URL)
                                self.ai_agent.MAX_WORKERS = getattr(self, '_ollama_workers', 0)

                                # Initialize Memory Agent for deterministic extraction
                                if _HAS_DETERMINISTIC and _MEMORY_AGENT is None:
                                    try:
                                        init_memory_agent(self.data_dir, log_fn=self.enqueue_log)
                                    except Exception:
                                        pass
                                if _HAS_DETERMINISTIC and _MEMORY_AGENT is not None:
                                    try:
                                        init_gse_contexts(meta_df, gds_conn=self.gds_conn,
                                                          log_fn=self.enqueue_log)
                                    except Exception:
                                        pass

                                # Pre-fetch GSE context for LLM fallback
                                if 'series_id' in meta_df.columns and self.gds_conn:
                                    try:
                                        prefetch_gse_context(
                                            self.gds_conn,
                                            meta_df['series_id'].dropna().unique().tolist(),
                                            log_fn=self.enqueue_log)
                                    except Exception:
                                        pass

                                ai_labels = self.ai_agent.process_samples(
                                    meta_df,
                                    fields=self._extraction_fields,
                                    custom_fields=self._extraction_custom_fields or None,
                                    progress_fn=_extraction_progress)
                                
                                if not ai_labels.empty and not fast_mode:
                                    # Full mode: Phase 2 Re-extraction Phase 2
                                    if (self._extraction_recall
                                            and self.label_source_var.get() != "file"):
                                        _update_status(
                                            f"Region {r_idx+1}/{n_regions}: Phase 2 Re-extraction...",
                                            ai_pct + 25,
                                            "Phase 2: correcting 'Not Specified' labels"
                                        )
                                        try:
                                            recall_agent = ContextRecallExtractor(
                                                log_func=self.enqueue_log,
                                                saved_cache=self._gse_saved_cache)
                                            if recall_agent.build_context(ai_labels):
                                                ai_labels = recall_agent.run_recall_pass(
                                                    ai_labels,
                                                    extra_fields=getattr(self, '_extraction_ns_extra', None))
                                                self._save_persistent_cache(recall_agent)
                                        except Exception as re:
                                            self.enqueue_log(f"[Phase2] Region recall error: {re}")
                                    
                                    # Full mode: Harmonization + Clustering
                                    _update_status(
                                        f"Region {r_idx+1}/{n_regions}: Harmonizing...",
                                        ai_pct + 28,
                                        "Normalizing label variants"
                                    )
                                    # Phase 3 removed from pipeline — raw labels preserved
                                    ai_labels = self.apply_semantic_clustering(ai_labels)
                                elif not ai_labels.empty:
                                    # Fast mode: just clustering, no harmonization
                                    ai_labels = self.apply_semantic_clustering(ai_labels)
                                
                                self.update_progress(value=0)  # safe: won't reset if another extraction owns the bar
                            except Exception as e:
                                self.enqueue_log(f"[LLM Warning] Region {region['label']}: {e}")

                    region['ai_labels_df'] = ai_labels

                thread_conn.close()
                os.remove(tmp_path)

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

                def show_err():
                    try:
                        progress_win.destroy()
                    except tk.TclError:
                        pass
                    messagebox.showerror("Analysis Error", f"Error:\n\n{e}", parent=popup)

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
    
    def open_compare_window(self):
        """Opens Compare Distributions setup dialog — similar to Gene Explorer.
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
        dlg.title("Compare Distributions — Setup")
        dlg.transient(self)

        # Instructions
        ttk.Label(dlg, text="Compare gene distributions across platforms and conditions",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 5))

        top_frame = ttk.Frame(dlg, padding=10)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # ── Platform Selection ──
        plat_frame = ttk.LabelFrame(top_frame, text="Select Platforms", padding=5)
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
        tk.Button(dir_row, text="+ Add Data Directory...",
                  command=lambda: self._add_data_directory(),
                  bg="#1976D2", fg="white", font=('Segoe UI', 9, 'bold'),
                  padx=10, cursor="hand2").pack(side=tk.LEFT, padx=5)

        # ── Gene Input ──
        gene_frame = ttk.LabelFrame(top_frame, text="Enter Gene Symbols", padding=5)
        gene_frame.pack(fill=tk.X, pady=5)

        ttk.Label(gene_frame, text="Gene symbols (comma-separated, e.g., TP53, BRCA1, EGFR):",
                  font=('Segoe UI', 9, 'italic')).pack(fill=tk.X, padx=5)
        gene_entry = ttk.Entry(gene_frame, font=('Consolas', 11))
        gene_entry.pack(fill=tk.X, padx=5, pady=5)

        # ── Options ──
        opts_frame = ttk.LabelFrame(top_frame, text="Options", padding=5)
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
                      text=f"Labels loaded: {plats} — full comparison with PCA, enrichment, etc.",
                      font=('Segoe UI', 9, 'bold'), foreground='green').pack(anchor=tk.W)
            compare_mode = "labels"
        else:
            ttk.Label(label_status,
                      text="No labels loaded — will compare expression distributions only\n"
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

        tk.Button(btn_frame, text="  Run Comparison  ", command=_run,
                  bg="#FF9800", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                  font=('Segoe UI', 10), padx=15, pady=6).pack(side=tk.RIGHT, padx=5)

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
            # Each gene/platform becomes a group — this is what gets compared
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
                            win.lbl_grouping.config(text=col, foreground="green")
                            break
                    try:
                        win._refresh_data_table()
                    except: pass

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
            self.enqueue_log(f"[Compare] OK {len(compare_data)} distributions ready — click RUN ANALYSIS")
            win.lift()
            win.focus_force()

        except Exception as e:
            self.enqueue_log(f"[Compare] Error: {e}")
            import traceback
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    #  Distribution Classification — per-gene statistics
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

        # Setup dialog — pick platform
        dlg = tk.Toplevel(self)
        dlg.title("Distribution Classification — Setup")
        dlg.transient(self)

        ttk.Label(dlg, text="Classify gene distributions on a platform",
                  font=('Segoe UI', 12, 'bold')).pack(padx=15, pady=(15, 5))
        ttk.Label(dlg, text="Computes normality, skewness, kurtosis, modality,\n"
                            "and descriptive statistics for every gene.",
                  font=('Segoe UI', 9, 'italic'), foreground='#666').pack(padx=15, pady=(0, 10))

        plat_frame = ttk.LabelFrame(dlg, text="Select Platform", padding=5)
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

        tk.Button(btn_frame, text="  Run Classification  ", command=_run,
                  bg="#00897B", fg="white", font=('Segoe UI', 11, 'bold'),
                  padx=20, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy,
                  font=('Segoe UI', 10), padx=15, pady=6).pack(side=tk.RIGHT, padx=5)

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
        win.title(f"Distribution Classification — {plat_id} ({len(results)} genes)")
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
            from scipy.stats import gaussian_kde

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
                plot_frame = ttk.LabelFrame(win, text="Representative Gene Distributions", padding=5)
                plot_frame.pack(fill=tk.X, padx=10, pady=5)

                fig, axes = plt.subplots(1, min(5, len(example_genes)),
                                          figsize=(min(5, len(example_genes)) * 3.2, 4.5))
                if len(example_genes) == 1:
                    axes = [axes]

                colors = ['#1565C0', '#C62828', '#2E7D32', '#E65100', '#7B1FA2']
                for idx, (gene_row, ax_i) in enumerate(zip(example_genes, axes)):
                    gene_name = gene_row['Gene']
                    gene_col = gene_mapping.get(gene_name, gene_name)
                    if gene_col in df.columns:
                        vals = pd.to_numeric(df[gene_col], errors='coerce').dropna()
                        if len(vals) > 10:
                            try:
                                kde = gaussian_kde(vals)
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
                fig.suptitle(f"{plat_id} — Example Distributions by Classification Type",
                             fontsize=10, weight='bold', y=0.92)

                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, plot_frame)
                canvas.draw()
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

        for _, row in results_df.iterrows():
            vals = [str(row.get(c, '')) for c in cols]
            tree.insert("", tk.END, values=vals)

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

        tk.Button(btn_frame, text="  Plot Selected Genes  ", command=_plot_selected,
                  bg="#9C27B0", fg="white", font=('Segoe UI', 10, 'bold'),
                  padx=12, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=5)

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
        for _, row in sorted_df.iterrows():
            vals = [str(row.get(c, '')) for c in display_cols]
            tree.insert("", tk.END, values=vals)

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
                
                # User-provided labels are kept AS-IS — no harmonization.

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
        win.title("GeneVariate - GPL Auto-Downloader (Any Species)")
        win.geometry("1050x800")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"1050x800+{(_sw-1050)//2}+{(_sh-800)//2}")
            win.minsize(500, 400)
        except Exception: pass
        self._gpl_dl_window = win

        ttk.Label(win,
                  text="Download and preprocess any GPL platform from NCBI GEO. "
                       "Search by species or enter a GPL ID directly.",
                  foreground="gray", font=('Segoe UI', 9, 'italic'),
                  wraplength=900).pack(padx=15, pady=(10, 5))

        # ── Direct GPL download row ─────────────────────────────────
        direct_frame = ttk.LabelFrame(win, text="Direct GPL Download", padding=8)
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
        tk.Button(inp, text="Download & Process",
                  command=self._auto_download_gpl,
                  bg="#1B5E20", fg="white", font=("Segoe UI", 10, "bold"),
                  padx=15, pady=3).pack(side=tk.LEFT, padx=15)

        # ── Species Browser ─────────────────────────────────────────
        species_frame = ttk.LabelFrame(win, text="Browse Platforms by Species", padding=8)
        species_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Search row
        search_row = ttk.Frame(species_frame)
        search_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_row, text="Species / GPL ID:",
                  font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT)
        self._species_entry = ttk.Entry(search_row, width=30, font=('Segoe UI', 11))
        self._species_entry.pack(side=tk.LEFT, padx=8)
        self._species_entry.insert(0, "Mus musculus")
        tk.Button(search_row, text="Search",
                  command=self._search_species_gpls,
                  bg="#1565C0", fg="white", font=("Segoe UI", 10, "bold"),
                  padx=12, pady=2, cursor="hand2").pack(side=tk.LEFT, padx=4)
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

        # Results treeview
        self._species_status = ttk.Label(species_frame,
                                          text="Enter a species name and click Search, "
                                               "or use a quick button above.",
                                          font=('Segoe UI', 9, 'italic'),
                                          foreground='gray')
        self._species_status.pack(anchor=tk.W, pady=2)

        tree_frame = ttk.Frame(species_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("GPL", "Title", "Technology", "Samples", "Genes")
        self._species_tree = ttk.Treeview(tree_frame, columns=cols,
                                           show="headings", height=12)
        self._species_tree.heading("GPL", text="GPL ID")
        self._species_tree.heading("Title", text="Platform Title")
        self._species_tree.heading("Technology", text="Technology")
        self._species_tree.heading("Samples", text="# Samples")
        self._species_tree.heading("Genes", text="# Genes/Probes")

        self._species_tree.column("GPL", width=80, anchor=tk.CENTER)
        self._species_tree.column("Title", width=380)
        self._species_tree.column("Technology", width=160)
        self._species_tree.column("Samples", width=80, anchor=tk.CENTER)
        self._species_tree.column("Genes", width=100, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self._species_tree.yview)
        self._species_tree.configure(yscrollcommand=vsb.set)
        self._species_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click or select + button to use
        self._species_tree.bind('<Double-1>', self._on_species_gpl_select)

        sel_row = ttk.Frame(species_frame)
        sel_row.pack(fill=tk.X, pady=5)
        tk.Button(sel_row, text="Use Selected GPL",
                  command=self._use_selected_species_gpl,
                  bg="#E65100", fg="white", font=("Segoe UI", 10, "bold"),
                  padx=12, pady=3, cursor="hand2").pack(side=tk.LEFT, padx=5)
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

    def _quick_species_search(self, species):
        """Set species entry and search."""
        self._species_entry.delete(0, tk.END)
        self._species_entry.insert(0, species)
        self._search_species_gpls()

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
        self._species_status.config(text=f"Searching for '{species}'...", foreground="blue")
        self.update_progress(value=10, text=f"Searching GPLs: {species}")
        # Also update the GPL downloader's own progress bar
        try:
            self.auto_dl_progress_bar["maximum"] = 100
            self.auto_dl_progress_bar["value"] = 10
            self.auto_dl_status.config(text=f"Searching for '{species}'...", foreground="blue")
        except Exception:
            pass

        # Clear old results
        for item in self._species_tree.get_children():
            self._species_tree.delete(item)

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
                    # Fast query: get GPLs first WITHOUT sample count (avoids slow subquery)
                    query = """
                        SELECT gpl.gpl, gpl.title, gpl.technology, gpl.organism,
                               gpl.data_row_count
                        FROM gpl
                        WHERE LOWER(gpl.organism) LIKE ? OR LOWER(gpl.title) LIKE ?
                        ORDER BY gpl.data_row_count DESC LIMIT 200
                    """
                    pattern = f"%{species.lower()}%"
                    rows = self.gds_conn.execute(query, (pattern, pattern)).fetchall()
                    search_desc = f"'{species}'"

                def _update_status(text, pct=None):
                    try:
                        self.after(0, lambda: self._species_status.config(text=text, foreground="blue"))
                        if pct is not None:
                            self.after(0, lambda p=pct: self.auto_dl_progress_bar.config(value=p))
                            self.after(0, lambda t=text: self.auto_dl_status.config(text=t, foreground="blue"))
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
                        for gpl_id, title, tech, organism, n_probes, n_samples in rows_with_counts:
                            title = (title or "")[:80]
                            tech = (tech or "")[:30]
                            n_probes = n_probes or 0
                            self._species_tree.insert("", tk.END, values=(
                                gpl_id, title, tech,
                                f"{n_samples:,}" if n_samples else "?",
                                f"{n_probes:,}" if n_probes else "?"
                            ))
                        actual_org = rows[0][3] if rows and rows[0][3] else species
                        done_text = (f"Found {len(rows)} platform(s) for {search_desc} "
                                     f"({actual_org}). Double-click to download.")
                        self._species_status.config(text=done_text, foreground="green")
                        self.auto_dl_status.config(text=done_text, foreground="green")
                        self.auto_dl_progress_bar["value"] = 100
                        self.enqueue_log(f"[GPL Browser] Found {len(rows)} platforms for {search_desc}")
                        self.update_progress(value=0)
                        self.after(2000, lambda: self.auto_dl_progress_bar.config(value=0))
                    except Exception as e:
                        self._species_status.config(text=f"Error: {e}", foreground="red")
                        self.auto_dl_progress_bar["value"] = 0
                        self.update_progress(value=0)

                if not rows:
                    def _no_results():
                        no_text = (f"No platforms found for {search_desc}. "
                                   f"Try 'Homo sapiens' or 'GPL570'.")
                        self._species_status.config(text=no_text, foreground="orange")
                        self.auto_dl_status.config(text=no_text, foreground="orange")
                        self.auto_dl_progress_bar["value"] = 0
                        self.update_progress(value=0)
                    self.after(0, _no_results)
                else:
                    self.after(0, _populate)

            except Exception as e:
                self.enqueue_log(f"[GPL Browser] Search error: {e}")
                def _err():
                    self._species_status.config(text=f"Search error: {e}", foreground="red")
                    try:
                        self.auto_dl_status.config(text=f"Search error: {e}", foreground="red")
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
        Query platform info directly from self.gds_conn — same approach as
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

        return {
            'gpl_id':       gpl_id,
            'organism':     str(row[1] or 'Unknown'),
            'title':        str(row[2] or 'Unknown'),
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
        
        n_to_dl = min(info['total_series'], max_gse) if max_gse else info['total_series']
        if not messagebox.askyesno(f"Download {gpl_id}?",
            f"Platform: {info['title']}\nOrganism: {info['organism']}\n"
            f"Technology: {info['technology']}\nSeries: {n_to_dl} of {info['total_series']}\n\n"
            f"Output: GSM | series_id | genes (NaN preserved)\n\nProceed?", parent=self):
            return
        
        self.auto_dl_status.config(text=f"Starting {gpl_id}...", foreground="blue")
        if hasattr(self, 'auto_dl_progress_bar'):
            self.auto_dl_progress_bar["value"] = 0
        self.enqueue_log(f"[GPL-DL] Starting {gpl_id} ({info['organism']})...")
        
        def worker():
            try:
                result = downloader.run_with_info(info=info, max_gse=max_gse,
                    callback=lambda p, s, m: self.after(0, self._gpl_dl_progress, p, s, m))
                self.after(0, lambda: self._gpl_dl_done(result))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda _e=str(e), _tb=tb: self._gpl_dl_error(gpl_id, _e, _tb))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _gpl_dl_progress(self, pct, stage, msg):
        if pct is not None:
            self.progressbar["value"] = pct
            if hasattr(self, 'auto_dl_progress_bar'):
                self.auto_dl_progress_bar["value"] = pct
        self.auto_dl_status.config(text=f"[{stage}] {msg}", foreground="blue")
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
        win.title(f"{gpl_id} — Download Summary")
        win.geometry("800x600")
        try:
            _sw, _sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"800x600+{(_sw-800)//2}+{(_sh-600)//2}")
        except: pass

        # Header
        hdr = ttk.Frame(win)
        hdr.pack(fill=tk.X, padx=15, pady=(15, 5))
        species = GPL_SPECIES.get(gpl_id, result.get('organism', '?'))
        ttk.Label(hdr, text=f"{gpl_id} — {species.title()}",
                  font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT)

        # Stats frame
        stats = ttk.LabelFrame(win, text="Platform Statistics", padding=10)
        stats.pack(fill=tk.X, padx=15, pady=5)

        stats_data = [
            ("Total Samples (GSMs)", f"{result['n_samples']:,}"),
            ("Total Genes (probes)", f"{result['n_genes']:,}"),
            ("Experiments (GSEs)", f"{result.get('n_series', '?')}"),
            ("Species", species.title()),
            ("File", str(result.get('filepath', '?'))[-60:]),
        ]
        for i, (label, value) in enumerate(stats_data):
            ttk.Label(stats, text=f"{label}:", font=('Segoe UI', 10, 'bold')).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(stats, text=value, font=('Segoe UI', 10)).grid(
                row=i, column=1, sticky=tk.W, padx=10, pady=2)

        # Experiment list
        gse_frame = ttk.LabelFrame(win, text="Experiments (GSEs)", padding=5)
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
                      text=f"{len(gse_list)} experiments — double-click to open GEO page & view sample metadata",
                      font=('Segoe UI', 8, 'italic'), foreground='#888').pack(pady=2)
        else:
            ttk.Label(gse_frame, text="Experiment list not available (GEOmetadb not loaded)",
                      font=('Segoe UI', 10), foreground='#888').pack(pady=30)

        # Close button
        tk.Button(win, text="Close", command=win.destroy,
                  font=('Segoe UI', 10), padx=20, pady=5).pack(pady=10)

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
        top.title(f"{gse_id} — {len(meta_df)} samples (raw GEOmetadb metadata)")
        top.geometry("1300x700")
        try:
            _sw, _sh = top.winfo_screenwidth(), top.winfo_screenheight()
            top.geometry(f"1300x700+{(_sw-1300)//2}+{(_sh-700)//2}")
        except: pass

        # Summary
        ttk.Label(top, text=f"{gse_id}: {len(meta_df)} samples — all GEOmetadb fields shown",
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

        for _, row in meta_df.iterrows():
            tree.insert("", tk.END, values=[str(row.get(c, ''))[:100] for c in cols])

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
        _crash_log = open(os.path.join(os.path.expanduser('~'), 'genevariate_crash.log'), 'w')
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
