"""
NS Repair Pipeline - Main orchestrator for the GeneVariate NS repair process.

Coordinates the entire multi-phase repair pipeline:
    1. Load harmonized CSV platforms (repair mode) or raw GSM list (scratch mode)
    2. Build MemoryAgent from SQL DB (populate from .txt clusters if DB empty)
    3. Load GEOmetadb into RAM, fetch raw GSM metadata
    4. Scrape NCBI GEO for missing GSMs and GSE metadata (with file-based caching)
    5. Build GSEContext objects for each experiment
    6. Compute GPU/CPU hybrid parallel worker count
    7. Run Phase 1 extraction (scratch mode only)
    8. Dispatch parallel GSEWorkers with semaphore-based concurrency
    9. Live-flush results to CSV (never hold all rows in RAM)
   10. Save outputs: NS_repaired.csv, full_repaired.csv, collapse_report.csv, summary.txt

Queue protocol for GUI communication:
    {"type": "log",        "msg": str}
    {"type": "progress",   "pct": int, "label": str}
    {"type": "stats_live", ...}
    {"type": "done",       "success": bool}
    {"type": "watchdog",   "msg": str}
"""

import os
import re
import csv
import gzip
import json
import time
import shutil
import sqlite3
import hashlib
import logging
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from genevariate.core.extraction import (
    NS, LABEL_COLS, LABEL_COLS_SCRATCH, EXTRACTION_MODEL,
    EXTRACTION_PROMPT_TEMPLATE, parse_json_extraction, is_ns,
    format_raw_block, format_sample_for_extraction,
)
from genevariate.core.memory_agent import MemoryAgent, MEM_DB_NAME
from genevariate.core.gse_context import GSEContext
from genevariate.core.gse_worker import GSEWorker
from genevariate.core.ollama_manager import (
    DEFAULT_MODEL, DEFAULT_URL, MODEL_RAM_GB, DEFAULT_MODEL_GB,
    detect_gpus, compute_ollama_parallel, check_ollama_gpu,
    ollama_server_ok, start_ollama_server_blocking, start_ollama_cpu_server,
    kill_ollama, Watchdog, CPU_OLLAMA_URL,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

ALL_GPLS = ["GPL6947", "GPL96", "GPL570", "GPL10558"]

BATCH_SIZE = 200
CKPT_EVERY = 1000
NCBI_WORKERS = 5
NCBI_DELAY = 0.35

GSE_CACHE_FILE = "gse_meta_cache.json"
GSM_CACHE_FILE = "gsm_raw_cache.json"

NCBI_GSE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NCBI_GSM_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NCBI_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_NCBI_HEADERS = {
    "User-Agent": "GeneVariate/1.0 (NS Repair Pipeline)"
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Queue helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _qlog(q, msg: str):
    """Send a log message to the GUI queue."""
    if q is not None:
        try:
            q.put_nowait({"type": "log", "msg": msg})
        except Exception:
            pass
    logger.info(msg)


def _qprogress(q, pct: int, label: str = ""):
    """Send a progress update to the GUI queue."""
    if q is not None:
        try:
            q.put_nowait({"type": "progress", "pct": pct, "label": label})
        except Exception:
            pass


def _qstats(q, **kwargs):
    """Send live statistics to the GUI queue."""
    if q is not None:
        try:
            payload = {"type": "stats_live"}
            payload.update(kwargs)
            q.put_nowait(payload)
        except Exception:
            pass


def _qdone(q, success: bool, msg: str = ""):
    """Send completion signal to the GUI queue."""
    if q is not None:
        try:
            q.put_nowait({"type": "done", "success": success, "msg": msg})
        except Exception:
            pass


def _qwatchdog(q, msg: str):
    """Send a watchdog status to the GUI queue."""
    if q is not None:
        try:
            q.put_nowait({"type": "watchdog", "msg": msg})
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOmetadb helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _decompress_gz(db_path: str, log_fn=print) -> Optional[str]:
    """Decompress .sqlite.gz to a .sqlite file on disk. Returns path or None."""
    tmp_path = db_path.replace(".sqlite.gz", ".sqlite.tmp.sqlite")
    if os.path.exists(tmp_path):
        return tmp_path
    log_fn(f"[GEOmetadb] Decompressing {os.path.basename(db_path)}...")
    try:
        with gzip.open(db_path, "rb") as gz_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out, length=1024 * 1024 * 16)
        log_fn(f"[GEOmetadb] Decompressed to {os.path.basename(tmp_path)}")
        return tmp_path
    except Exception as exc:
        log_fn(f"[GEOmetadb] Decompress failed: {exc}")
        return None


def _ensure_indexes(conn: sqlite3.Connection, log_fn=print):
    """Create indexes on gsm/gse tables for fast lookups on disk-based DBs."""
    indexes = [
        ("idx_gsm_gsm", "gsm", "gsm"),
        ("idx_gsm_series", "gsm", "series_id"),
        ("idx_gsm_gpl", "gsm", "gpl"),
        ("idx_gse_gse", "gse", "gse"),
    ]
    for idx_name, table, col in indexes:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col})")
        except Exception:
            pass
    try:
        conn.commit()
    except Exception:
        pass
    log_fn("[GEOmetadb] Indexes verified")


def load_db_to_memory(db_path: str, log_fn=print,
                      force_disk: bool = False) -> Optional[sqlite3.Connection]:
    """
    Load GEOmetadb.sqlite with resource-aware strategy.

    On high-RAM devices (>= 10 GB free): loads entire DB into :memory: for speed.
    On low-RAM devices: opens directly from disk with WAL mode and indexes.

    Supports both plain .sqlite and gzipped .sqlite.gz files.
    Returns a connection (in-memory or disk-based) or None on failure.
    """
    if not db_path:
        log_fn("[GEOmetadb] No database path provided")
        return None

    db_path = str(db_path)

    # Resolve .gz to decompressed file on disk
    if db_path.endswith(".gz"):
        actual_path = _decompress_gz(db_path, log_fn)
        if actual_path is None:
            return None
    else:
        actual_path = db_path

    if not os.path.exists(actual_path):
        log_fn(f"[GEOmetadb] File not found: {actual_path}")
        return None

    file_mb = os.path.getsize(actual_path) / (1024 * 1024)

    # Decide: load into RAM or use disk-based access
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        # Need at least 2x the DB size as headroom + 2 GB for Ollama/OS
        need_gb = (file_mb / 1024) * 2 + 2
        use_memory = (not force_disk) and (free_gb > need_gb) and (free_gb > 6)
    except Exception:
        use_memory = False

    if use_memory:
        log_fn(f"[GEOmetadb] Loading {os.path.basename(actual_path)} "
               f"({file_mb:.0f} MB) into RAM (free: {free_gb:.1f} GB)...")
        try:
            t0 = time.time()
            disk_conn = sqlite3.connect(actual_path, timeout=60)
            mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            disk_conn.backup(mem_conn)
            disk_conn.close()
            elapsed = time.time() - t0
            log_fn(f"[GEOmetadb] Loaded into RAM in {elapsed:.1f}s")
            conn = mem_conn
        except (MemoryError, Exception) as exc:
            log_fn(f"[GEOmetadb] RAM load failed ({exc}), falling back to disk mode")
            use_memory = False

    if not use_memory:
        log_fn(f"[GEOmetadb] Opening {os.path.basename(actual_path)} "
               f"({file_mb:.0f} MB) from disk (low-RAM mode)...")
        try:
            t0 = time.time()
            conn = sqlite3.connect(actual_path, timeout=60,
                                   check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
            conn.execute("PRAGMA mmap_size=268435456")  # 256 MB memory-mapped I/O
            conn.execute("PRAGMA temp_store=MEMORY")
            _ensure_indexes(conn, log_fn)
            elapsed = time.time() - t0
            log_fn(f"[GEOmetadb] Disk mode ready in {elapsed:.1f}s "
                   f"(WAL + indexes + mmap)")
        except Exception as exc:
            log_fn(f"[GEOmetadb] Disk load failed: {exc}")
            return None

    # Verify key tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for required in ("gsm", "gse"):
        if required not in tables:
            log_fn(f"[GEOmetadb] WARNING: table '{required}' not found")

    return conn


def fetch_gsm_raw(conn: sqlite3.Connection,
                  gsm_ids: List[str]) -> Dict[str, dict]:
    """
    Fetch raw GSM metadata from in-memory GEOmetadb for a list of GSM IDs.

    Returns {gsm_id: {gsm_title, source_name, characteristics,
                       treatment_protocol, description, series_id, gpl}}.
    """
    if conn is None or not gsm_ids:
        return {}

    result = {}
    # Query in batches to avoid SQLite variable limit
    for i in range(0, len(gsm_ids), BATCH_SIZE):
        batch = gsm_ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        try:
            rows = conn.execute(f"""
                SELECT gsm, title, source_name_ch1, characteristics_ch1,
                       treatment_protocol_ch1, description, series_id, gpl
                FROM gsm
                WHERE gsm IN ({placeholders})
            """, batch).fetchall()
            for row in rows:
                gsm = row[0]
                result[gsm] = {
                    "gsm_title": row[1] or "",
                    "source_name": row[2] or "",
                    "characteristics": row[3] or "",
                    "treatment_protocol": row[4] or "",
                    "description": row[5] or "",
                    "series_id": row[6] or "",
                    "gpl": row[7] or "",
                }
        except Exception as exc:
            logger.warning(f"GEOmetadb query failed for batch {i}: {exc}")
            continue

    return result


def _fetch_gse_meta_from_db(conn: sqlite3.Connection,
                            gse_ids: List[str]) -> Dict[str, dict]:
    """Fetch GSE metadata (title, summary, overall_design) from GEOmetadb."""
    if conn is None or not gse_ids:
        return {}

    result = {}
    for i in range(0, len(gse_ids), BATCH_SIZE):
        batch = gse_ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        try:
            rows = conn.execute(f"""
                SELECT gse, title, summary, overall_design
                FROM gse
                WHERE gse IN ({placeholders})
            """, batch).fetchall()
            for row in rows:
                result[row[0]] = {
                    "title": row[1] or "",
                    "summary": row[2] or "",
                    "design": row[3] or "",
                }
        except Exception:
            continue

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  NCBI GEO scraping (with persistent JSON caches)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json_cache(cache_dir: str, filename: str) -> dict:
    """Load a JSON cache file from disk."""
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json_cache(cache_dir: str, filename: str, data: dict):
    """Save a JSON cache file to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Cache save failed for {filename}: {exc}")


def _scrape_one_gse(gse_id: str) -> Optional[dict]:
    """Scrape a single GSE page from NCBI GEO for title/summary/design."""
    try:
        params = {"acc": gse_id, "targ": "self", "form": "text", "view": "brief"}
        resp = requests.get(NCBI_GSE_URL, params=params, headers=_NCBI_HEADERS,
                            timeout=20)
        if resp.status_code != 200:
            return None

        text = resp.text
        title = ""
        summary = ""
        design = ""

        for line in text.splitlines():
            line_s = line.strip()
            if line_s.startswith("!Series_title"):
                title = line_s.split("=", 1)[-1].strip().strip('"')
            elif line_s.startswith("!Series_summary"):
                chunk = line_s.split("=", 1)[-1].strip().strip('"')
                summary = (summary + " " + chunk).strip() if summary else chunk
            elif line_s.startswith("!Series_overall_design"):
                chunk = line_s.split("=", 1)[-1].strip().strip('"')
                design = (design + " " + chunk).strip() if design else chunk

        if title or summary:
            return {"title": title, "summary": summary[:2000], "design": design[:1000]}
        return None

    except Exception:
        return None


def scrape_gse_meta(gse_ids: List[str],
                    log_fn=print,
                    progress_fn=None,
                    cache_dir: str = "") -> Dict[str, dict]:
    """
    Scrape NCBI GEO for GSE metadata (title, summary, design).

    Uses a persistent JSON cache file to avoid re-fetching.
    Returns {gse_id: {title, summary, design}}.
    """
    if not gse_ids:
        return {}

    # Load cache
    cache = _load_json_cache(cache_dir, GSE_CACHE_FILE) if cache_dir else {}
    cached_ids = set(cache.keys())
    to_scrape = [g for g in gse_ids if g not in cached_ids]

    if not to_scrape:
        log_fn(f"[NCBI] All {len(gse_ids)} GSE records found in cache")
        return {g: cache[g] for g in gse_ids if g in cache}

    log_fn(f"[NCBI] Scraping {len(to_scrape)} GSE records "
           f"({len(cached_ids)} cached)...")

    scraped = {}
    failed = 0

    def _scrape_worker(gse_id):
        time.sleep(NCBI_DELAY)
        return gse_id, _scrape_one_gse(gse_id)

    with ThreadPoolExecutor(max_workers=NCBI_WORKERS) as pool:
        futures = {pool.submit(_scrape_worker, g): g for g in to_scrape}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                gse_id, meta = future.result(timeout=30)
                if meta:
                    scraped[gse_id] = meta
                    cache[gse_id] = meta
                else:
                    failed += 1
            except Exception:
                failed += 1

            if progress_fn and done_count % 10 == 0:
                pct = int(100 * done_count / len(to_scrape))
                progress_fn(pct)

    # Save updated cache
    if cache_dir and scraped:
        _save_json_cache(cache_dir, GSE_CACHE_FILE, cache)

    log_fn(f"[NCBI] GSE scrape complete: {len(scraped)} fetched, {failed} failed")

    # Merge cached and freshly scraped
    result = {}
    for g in gse_ids:
        if g in cache:
            result[g] = cache[g]
    return result


def _scrape_one_gsm(gsm_id: str) -> Optional[dict]:
    """Scrape a single GSM record from NCBI GEO."""
    try:
        params = {"acc": gsm_id, "targ": "self", "form": "text", "view": "brief"}
        resp = requests.get(NCBI_GSM_URL, params=params, headers=_NCBI_HEADERS,
                            timeout=20)
        if resp.status_code != 200:
            return None

        text = resp.text
        fields = {
            "gsm_title": "",
            "source_name": "",
            "characteristics": "",
            "treatment_protocol": "",
            "description": "",
            "series_id": "",
            "gpl": "",
        }

        char_parts = []
        for line in text.splitlines():
            line_s = line.strip()
            if line_s.startswith("!Sample_title"):
                fields["gsm_title"] = line_s.split("=", 1)[-1].strip().strip('"')
            elif line_s.startswith("!Sample_source_name"):
                fields["source_name"] = line_s.split("=", 1)[-1].strip().strip('"')
            elif line_s.startswith("!Sample_characteristics"):
                val = line_s.split("=", 1)[-1].strip().strip('"')
                char_parts.append(val)
            elif line_s.startswith("!Sample_treatment_protocol"):
                val = line_s.split("=", 1)[-1].strip().strip('"')
                fields["treatment_protocol"] = (
                    (fields["treatment_protocol"] + "; " + val).strip("; ")
                    if fields["treatment_protocol"] else val
                )
            elif line_s.startswith("!Sample_description"):
                val = line_s.split("=", 1)[-1].strip().strip('"')
                fields["description"] = (
                    (fields["description"] + " " + val).strip()
                    if fields["description"] else val
                )
            elif line_s.startswith("!Sample_series_id"):
                fields["series_id"] = line_s.split("=", 1)[-1].strip().strip('"')
            elif line_s.startswith("!Sample_platform_id"):
                fields["gpl"] = line_s.split("=", 1)[-1].strip().strip('"')

        if char_parts:
            fields["characteristics"] = "\t".join(char_parts)

        if fields["gsm_title"] or fields["source_name"] or fields["characteristics"]:
            return fields
        return None

    except Exception:
        return None


def scrape_gsm_raw(missing_gsms: List[str],
                   log_fn=print,
                   progress_fn=None,
                   cache_dir: str = "") -> Dict[str, dict]:
    """
    NCBI fallback for GSMs missing from GEOmetadb.

    Uses a persistent JSON cache file.
    Returns {gsm_id: {gsm_title, source_name, characteristics, ...}}.
    """
    if not missing_gsms:
        return {}

    # Load cache
    cache = _load_json_cache(cache_dir, GSM_CACHE_FILE) if cache_dir else {}
    cached_ids = set(cache.keys())
    to_scrape = [g for g in missing_gsms if g not in cached_ids]

    if not to_scrape:
        log_fn(f"[NCBI] All {len(missing_gsms)} GSM records found in cache")
        return {g: cache[g] for g in missing_gsms if g in cache}

    log_fn(f"[NCBI] Scraping {len(to_scrape)} missing GSMs "
           f"({len(cached_ids)} cached)...")

    scraped = {}
    failed = 0

    def _scrape_worker(gsm_id):
        time.sleep(NCBI_DELAY)
        return gsm_id, _scrape_one_gsm(gsm_id)

    with ThreadPoolExecutor(max_workers=NCBI_WORKERS) as pool:
        futures = {pool.submit(_scrape_worker, g): g for g in to_scrape}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                gsm_id, meta = future.result(timeout=30)
                if meta:
                    scraped[gsm_id] = meta
                    cache[gsm_id] = meta
                else:
                    failed += 1
            except Exception:
                failed += 1

            if progress_fn and done_count % 20 == 0:
                pct = int(100 * done_count / len(to_scrape))
                progress_fn(pct)

    # Save updated cache
    if cache_dir and scraped:
        _save_json_cache(cache_dir, GSM_CACHE_FILE, cache)

    log_fn(f"[NCBI] GSM scrape complete: {len(scraped)} fetched, {failed} failed")

    result = {}
    for g in missing_gsms:
        if g in cache:
            result[g] = cache[g]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Platform / CSV loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_platform(gpl: str, base_dir: str) -> Optional[pd.DataFrame]:
    """
    Load a harmonized CSV platform file for a given GPL.

    Searches for files matching patterns:
        {base_dir}/{gpl}/{gpl}_harmonized.csv
        {base_dir}/{gpl}_harmonized.csv
        {base_dir}/{gpl}_data.csv.gz

    Returns a DataFrame with at least 'GSM' column, or None.
    """
    candidates = [
        os.path.join(base_dir, gpl, f"{gpl}_harmonized.csv"),
        os.path.join(base_dir, gpl, f"{gpl}_harmonized.csv.gz"),
        os.path.join(base_dir, f"{gpl}_harmonized.csv"),
        os.path.join(base_dir, f"{gpl}_harmonized.csv.gz"),
        os.path.join(base_dir, gpl, f"{gpl}_data.csv"),
        os.path.join(base_dir, gpl, f"{gpl}_data.csv.gz"),
        os.path.join(base_dir, f"{gpl}_data.csv.gz"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                # Normalise GSM column name
                for col in df.columns:
                    if col.upper() in ("GSM", "GEO_ACCESSION", "SAMPLE_ID"):
                        df.rename(columns={col: "GSM"}, inplace=True)
                        break
                if "GSM" not in df.columns and df.columns[0].startswith("GSM"):
                    df.rename(columns={df.columns[0]: "GSM"}, inplace=True)
                if "GSM" not in df.columns:
                    continue
                df["GSM"] = df["GSM"].astype(str).str.strip()
                df["platform"] = gpl
                return df
            except Exception as exc:
                logger.warning(f"Failed to load {path}: {exc}")
                continue

    return None


def load_all(base_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all GPL platform files from base_dir.

    Returns {gpl: DataFrame} for each successfully loaded platform.
    """
    result = {}
    for gpl in ALL_GPLS:
        df = load_platform(gpl, base_dir)
        if df is not None:
            result[gpl] = df
    return result


def load_gsm_list(gsm_file: str, platform_id: str = "") -> Optional[pd.DataFrame]:
    """
    Load a raw GSM list for scratch mode.

    Accepts .txt (one GSM per line) or .csv (must have GSM column).
    Returns a DataFrame with GSM column.
    """
    if not gsm_file or not os.path.exists(gsm_file):
        return None

    ext = os.path.splitext(gsm_file)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(gsm_file, low_memory=False)
        for col in df.columns:
            if col.upper() in ("GSM", "GEO_ACCESSION", "SAMPLE_ID"):
                df.rename(columns={col: "GSM"}, inplace=True)
                break
        if "GSM" not in df.columns:
            return None
    else:
        # Text file: one GSM per line
        with open(gsm_file, "r") as f:
            gsms = [line.strip() for line in f if line.strip().startswith("GSM")]
        if not gsms:
            return None
        df = pd.DataFrame({"GSM": gsms})

    df["GSM"] = df["GSM"].astype(str).str.strip()
    if platform_id:
        df["platform"] = platform_id
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  GSEContext builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gse_contexts(df: pd.DataFrame,
                        raw_meta: Dict[str, dict],
                        gse_meta: Dict[str, dict],
                        mem_agent: Optional[MemoryAgent],
                        label_cols: List[str],
                        log_fn=print) -> Dict[str, GSEContext]:
    """
    Build GSEContext objects from a platform DataFrame and raw metadata.

    Each GSM is assigned to its GSE experiment. Existing non-NS labels
    are loaded into the context so sibling-aware prompts work correctly.
    """
    contexts: Dict[str, GSEContext] = {}

    # Map GSM -> series_id from raw metadata
    gsm_to_gse = {}
    for gsm, raw in raw_meta.items():
        series = raw.get("series_id", "")
        if series:
            # series_id can be multi-valued (semicolon-separated)
            primary = series.split(";")[0].strip()
            if primary:
                gsm_to_gse[gsm] = primary

    # Also check if df has a series_id column
    if "series_id" in df.columns:
        for _, row in df.iterrows():
            gsm = str(row.get("GSM", ""))
            sid = str(row.get("series_id", ""))
            if gsm and sid and sid.lower() not in ("", "nan", "none"):
                gsm_to_gse.setdefault(gsm, sid.split(";")[0].strip())

    # Create contexts
    for gsm in df["GSM"].unique():
        gse_id = gsm_to_gse.get(gsm, "UNKNOWN")
        if gse_id not in contexts:
            contexts[gse_id] = GSEContext(gse_id)

        # Seed labels from existing DataFrame columns
        labels = {}
        row_data = df[df["GSM"] == gsm]
        if not row_data.empty:
            row = row_data.iloc[0]
            for col in label_cols:
                val = str(row.get(col, NS)) if col in row.index else NS
                if pd.isna(row.get(col)) or val.strip() == "":
                    val = NS
                labels[col] = val

        contexts[gse_id].add_sample(gsm, labels, mem_agent)

    # Set GSE metadata
    for gse_id, ctx in contexts.items():
        if gse_id in gse_meta:
            meta = gse_meta[gse_id]
            ctx.set_meta(
                meta.get("title", ""),
                meta.get("summary", ""),
                meta.get("design", ""),
            )

    log_fn(f"  Built {len(contexts)} GSE contexts "
           f"for {len(df)} samples")
    return contexts


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 extraction (scratch mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _phase1_scratch_extraction(df: pd.DataFrame,
                               raw_meta: Dict[str, dict],
                               model: str,
                               ollama_url: str,
                               label_cols: List[str],
                               watchdog: Optional[Watchdog],
                               log_fn=print,
                               progress_fn=None) -> pd.DataFrame:
    """
    Phase 1: Initial LLM extraction for scratch mode.

    Runs EXTRACTION_PROMPT_TEMPLATE against raw metadata to populate
    Tissue, Condition, Treatment columns from nothing.
    """
    try:
        import ollama as _ollama
    except ImportError:
        log_fn("[Phase 1] ollama library not installed -- skipping extraction")
        for col in label_cols:
            if col not in df.columns:
                df[col] = NS
        return df

    log_fn(f"[Phase 1] Extracting {len(label_cols)} fields for "
           f"{len(df)} samples using {model}...")

    for col in label_cols:
        if col not in df.columns:
            df[col] = NS

    extracted_count = 0
    total = len(df)

    for idx, row in df.iterrows():
        gsm = str(row["GSM"])
        raw = raw_meta.get(gsm, {})

        if not raw:
            continue

        # Skip if already has values
        all_filled = all(
            not is_ns(str(row.get(col, NS)))
            for col in label_cols
        )
        if all_filled:
            extracted_count += 1
            continue

        # Wait if watchdog paused
        if watchdog:
            watchdog.wait_if_paused()

        # Build prompt
        text = format_sample_for_extraction(raw)
        title_val = raw.get("gsm_title", "")[:80]
        source_val = raw.get("source_name", "")[:60]
        chars_val = raw.get("characteristics", "")[:250]

        prompt = (EXTRACTION_PROMPT_TEMPLATE
                  .replace("{TITLE}", title_val)
                  .replace("{SOURCE}", source_val)
                  .replace("{CHAR}", chars_val))

        # LLM call with retries
        llm_text = ""
        for attempt in range(3):
            try:
                resp = _ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "num_predict": 200},
                    keep_alive=-1,
                )
                if hasattr(resp, "message") and hasattr(resp.message, "content"):
                    llm_text = (resp.message.content or "").strip()
                elif isinstance(resp, dict):
                    llm_text = resp.get("message", {}).get("content", "").strip()
                if llm_text:
                    break
            except Exception as exc:
                err = str(exc).lower()
                if "out of memory" in err or "cudamalloc" in err:
                    time.sleep(8)
                else:
                    time.sleep(3 * (attempt + 1))

        if llm_text:
            parsed = parse_json_extraction(llm_text, label_cols)
            for col in label_cols:
                val = parsed.get(col, NS)
                if not is_ns(val) and is_ns(str(row.get(col, NS))):
                    df.at[idx, col] = val

        if watchdog:
            watchdog.record_call()

        extracted_count += 1
        if progress_fn and extracted_count % 50 == 0:
            pct = int(100 * extracted_count / total)
            progress_fn(pct)

    log_fn(f"[Phase 1] Extraction complete: {extracted_count}/{total} processed")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV flushing utilities
# ═══════════════════════════════════════════════════════════════════════════════

class CSVFlusher:
    """
    Incremental CSV writer that flushes rows to disk in batches.
    Never holds all rows in RAM simultaneously.
    """

    def __init__(self, path: str, columns: List[str]):
        self.path = path
        self.columns = columns
        self._buffer: List[dict] = []
        self._written = 0
        self._lock = threading.Lock()

        # Write header
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()

    def add(self, row: dict):
        """Add a row to the buffer."""
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= BATCH_SIZE:
                self._flush()

    def add_batch(self, rows: List[dict]):
        """Add multiple rows."""
        with self._lock:
            self._buffer.extend(rows)
            if len(self._buffer) >= BATCH_SIZE:
                self._flush()

    def _flush(self):
        """Flush buffer to disk (caller must hold lock)."""
        if not self._buffer:
            return
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns,
                                        extrasaction="ignore")
                writer.writerows(self._buffer)
            self._written += len(self._buffer)
            self._buffer.clear()
        except Exception as exc:
            logger.error(f"CSV flush error: {exc}")

    def finalize(self) -> int:
        """Flush remaining buffer and return total rows written."""
        with self._lock:
            self._flush()
        return self._written


class CollapseReporter:
    """Tracks collapse (normalization) actions for the final report."""

    def __init__(self):
        self._records: List[dict] = []
        self._lock = threading.Lock()

    def record(self, gsm: str, col: str, raw: str, final: str,
               rule: str, gse: str = "", platform: str = ""):
        with self._lock:
            self._records.append({
                "GSM": gsm,
                "Field": col,
                "Raw_Extracted": raw,
                "Final_Label": final,
                "Collapse_Rule": rule,
                "GSE": gse,
                "Platform": platform,
            })

    def to_dataframe(self) -> pd.DataFrame:
        with self._lock:
            if not self._records:
                return pd.DataFrame(columns=[
                    "GSM", "Field", "Raw_Extracted", "Final_Label",
                    "Collapse_Rule", "GSE", "Platform",
                ])
            return pd.DataFrame(self._records)


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_gse_workers(df: pd.DataFrame,
                          contexts: Dict[str, GSEContext],
                          raw_meta: Dict[str, dict],
                          mem_agent: Optional[MemoryAgent],
                          model: str,
                          ollama_url: str,
                          label_cols: List[str],
                          platform_id: str,
                          max_workers: int,
                          watchdog: Optional[Watchdog],
                          ns_flusher: CSVFlusher,
                          full_flusher: CSVFlusher,
                          collapse_reporter: CollapseReporter,
                          log_fn=print,
                          progress_fn=None,
                          q=None,
                          throttle_sem=None) -> Dict[str, int]:
    """
    Dispatch parallel GSE workers with semaphore-based concurrency.

    Each GSEWorker processes all NS samples within its GSE experiment.
    Results are live-flushed to CSV via the flusher objects.

    Returns statistics dict.
    """
    stats = {
        "total_samples": len(df),
        "ns_before": {col: 0 for col in label_cols},
        "ns_after": {col: 0 for col in label_cols},
        "repaired": {col: 0 for col in label_cols},
        "workers_launched": 0,
        "errors": 0,
    }

    # Count initial NS
    for col in label_cols:
        if col in df.columns:
            mask = df[col].apply(lambda v: is_ns(str(v)) if pd.notna(v) else True)
            stats["ns_before"][col] = int(mask.sum())

    # Group samples by GSE
    gsm_to_gse = {}
    for gse_id, ctx in contexts.items():
        for sample in ctx._samples:
            gsm_to_gse[sample["gsm"]] = gse_id

    gse_groups: Dict[str, List[str]] = defaultdict(list)
    for gsm in df["GSM"].unique():
        gse_id = gsm_to_gse.get(gsm, "UNKNOWN")
        gse_groups[gse_id].append(gsm)

    # Identify GSEs that have NS samples needing repair
    gse_queue = []
    for gse_id, gsms in gse_groups.items():
        has_ns = False
        for gsm in gsms:
            row = df[df["GSM"] == gsm]
            if row.empty:
                continue
            for col in label_cols:
                val = str(row.iloc[0].get(col, NS))
                if is_ns(val):
                    has_ns = True
                    break
            if has_ns:
                break
        if has_ns:
            gse_queue.append((gse_id, gsms))
        else:
            # No NS -- flush all rows as-is
            for gsm in gsms:
                row = df[df["GSM"] == gsm]
                if not row.empty:
                    row_dict = row.iloc[0].to_dict()
                    full_flusher.add(row_dict)

    log_fn(f"  {len(gse_queue)} GSEs have NS samples to repair "
           f"({len(gse_groups) - len(gse_queue)} already complete)")

    if not gse_queue:
        return stats

    # Use the pipeline's fluid throttle semaphore if available, else static
    sem = throttle_sem if throttle_sem is not None else threading.Semaphore(max_workers)
    completed = [0]
    total_gse = len(gse_queue)
    results_lock = threading.Lock()

    def _process_one_gse(gse_id: str, gsms: List[str]):
        """Process all samples for one GSE experiment."""
        nonlocal completed

        sem.acquire()
        try:
            ctx = contexts.get(gse_id)
            if ctx is None:
                ctx = GSEContext(gse_id)

            worker = GSEWorker(
                gse_id=gse_id,
                ctx=ctx,
                mem_agent=mem_agent,
                model=model,
                platform=platform_id,
                ollama_url=ollama_url,
                watchdog=watchdog,
            )

            for gsm in gsms:
                row = df[df["GSM"] == gsm]
                if row.empty:
                    continue

                row_dict = row.iloc[0].to_dict()
                raw = raw_meta.get(gsm, {})

                # Build input for GSEWorker
                gsm_row = dict(raw)
                gsm_row["gsm"] = gsm
                gsm_row["GSM"] = gsm
                if "title" not in gsm_row:
                    gsm_row["title"] = gsm_row.get("gsm_title", "")
                if "source_name_ch1" not in gsm_row:
                    gsm_row["source_name_ch1"] = gsm_row.get("source_name", "")
                if "characteristics_ch1" not in gsm_row:
                    gsm_row["characteristics_ch1"] = gsm_row.get("characteristics", "")

                # Check which columns need repair
                ns_cols_here = [
                    col for col in label_cols
                    if is_ns(str(row_dict.get(col, NS)))
                ]

                if not ns_cols_here:
                    # Already complete
                    full_flusher.add(row_dict)
                    continue

                # Wait if watchdog paused
                if watchdog:
                    watchdog.wait_if_paused()

                # Run worker
                try:
                    result = worker.repair_one(gsm, gsm_row, ns_cols=ns_cols_here)
                except Exception as exc:
                    logger.warning(f"Worker error for {gsm} in {gse_id}: {exc}")
                    with results_lock:
                        stats["errors"] += 1
                    full_flusher.add(row_dict)
                    continue

                # Extract collapse rules from result metadata
                collapse_rules = result.pop("_collapse_rules", {})

                # Merge results back
                was_repaired = False
                for col in label_cols:
                    new_val = result.get(col, NS)
                    old_val = str(row_dict.get(col, NS))

                    if is_ns(old_val) and not is_ns(str(new_val)):
                        row_dict[col] = new_val
                        was_repaired = True
                        with results_lock:
                            stats["repaired"][col] = stats["repaired"].get(col, 0) + 1

                        # Record collapse
                        collapse_reporter.record(
                            gsm=gsm, col=col,
                            raw=old_val, final=str(new_val),
                            rule=collapse_rules.get(col, "llm"),
                            gse=gse_id, platform=platform_id,
                        )

                # Flush to appropriate files
                full_flusher.add(row_dict)
                if was_repaired:
                    ns_flusher.add(row_dict)

                if watchdog:
                    watchdog.record_call()

        except Exception as exc:
            logger.error(f"GSE worker error for {gse_id}: {exc}")
            with results_lock:
                stats["errors"] += 1
        finally:
            sem.release()
            with results_lock:
                completed[0] += 1
                stats["workers_launched"] = completed[0]

            if progress_fn:
                pct = int(100 * completed[0] / total_gse)
                progress_fn(pct)

            _qstats(q, completed=completed[0], total_gse=total_gse,
                    repaired=dict(stats["repaired"]))

    # Launch workers
    log_fn(f"  Launching {total_gse} GSE workers "
           f"(max {max_workers} concurrent)...")

    threads = []
    for gse_id, gsms in gse_queue:
        t = threading.Thread(
            target=_process_one_gse,
            args=(gse_id, gsms),
            name=f"GSE-{gse_id}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join(timeout=7200)  # 2 hour max per GSE

    # Count final NS
    # Re-read the full CSV to count (since we flushed incrementally)
    for col in label_cols:
        stats["ns_after"][col] = max(
            0, stats["ns_before"][col] - stats["repaired"].get(col, 0)
        )

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary writer
# ═══════════════════════════════════════════════════════════════════════════════

def _write_summary(out_dir: str, platform_id: str, stats: dict,
                   label_cols: List[str], elapsed: float,
                   model: str, mem_agent: Optional[MemoryAgent]):
    """Write a human-readable summary.txt file."""
    path = os.path.join(out_dir, "summary.txt")
    lines = [
        "=" * 70,
        "  GeneVariate NS Repair Pipeline - Run Summary",
        "=" * 70,
        f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Platform      : {platform_id}",
        f"  Model         : {model}",
        f"  Total samples : {stats.get('total_samples', 0):,}",
        f"  Elapsed       : {timedelta(seconds=int(elapsed))}",
        f"  Workers used  : {stats.get('workers_launched', 0)}",
        f"  Errors        : {stats.get('errors', 0)}",
        "",
        "  NS Repair Results:",
        "  " + "-" * 50,
    ]

    for col in label_cols:
        before = stats.get("ns_before", {}).get(col, 0)
        repaired = stats.get("repaired", {}).get(col, 0)
        after = stats.get("ns_after", {}).get(col, 0)
        rate = (100.0 * repaired / before) if before > 0 else 0.0
        lines.append(
            f"  {col:20s}: {before:>6,} NS -> {after:>6,} NS "
            f"({repaired:>6,} repaired, {rate:.1f}%)"
        )

    lines.append("")

    # Memory agent stats
    if mem_agent:
        ms = mem_agent.stats()
        new_clusters = mem_agent.get_new_cluster_log()
        lines.extend([
            "  Memory Agent Stats:",
            "  " + "-" * 50,
            f"  Semantic labels : {ms.get('semantic', {})}",
            f"  Episodic entries: {ms.get('episodic', {})}",
            f"  KG triples      : {ms.get('kg_triples', 0):,}",
            f"  New clusters    : {len(new_clusters)}",
        ])
        if new_clusters:
            lines.append("")
            lines.append("  Newly created clusters:")
            for cl in new_clusters[:20]:
                lines.append(
                    f"    [{cl['col']}] {cl['cluster_name']!r} "
                    f"<- {cl['raw_label']!r}"
                )
            if len(new_clusters) > 20:
                lines.append(f"    ... and {len(new_clusters) - 20} more")

    lines.extend(["", "=" * 70])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def pipeline(config: dict, q=None):
    """
    Main NS repair pipeline.

    Args:
        config: Configuration dict with keys:
            mode            : "repair" or "scratch"
            base_dir        : Path to data directory (contains GPL subdirs)
            output_dir      : Path for output files
            platforms       : List of GPL IDs to process (repair mode)
            gsm_file        : Path to GSM list file (scratch mode)
            platform_id     : Platform ID for scratch mode
            model           : Ollama model name (default: gemma2:9b)
            ollama_url      : Ollama server URL
            geo_db_path     : Path to GEOmetadb.sqlite(.gz)
            memory_dir      : Path to LLM memory dir (cluster .txt files)
            max_workers     : Override parallel worker count (0 = auto)
            enable_phase1   : Run Phase 1 extraction (scratch only)
            enable_watchdog : Run resource watchdog
            label_cols      : Override label columns to repair

        q: queue.Queue for GUI communication (optional).
    """
    t0 = time.time()
    mode = config.get("mode", "repair")
    base_dir = config.get("base_dir", "")
    output_dir = config.get("output_dir", "")
    platforms = config.get("platforms", ALL_GPLS)
    gsm_file = config.get("gsm_file", "")
    platform_id = config.get("platform_id", "CUSTOM")
    model = config.get("model", DEFAULT_MODEL)
    ollama_url = config.get("ollama_url", DEFAULT_URL)
    geo_db_path = config.get("geo_db_path", "")
    memory_dir = config.get("memory_dir", "")
    max_workers_override = config.get("max_workers", 0)
    enable_phase1 = config.get("enable_phase1", True)
    enable_watchdog = config.get("enable_watchdog", True)
    label_cols = config.get("label_cols", None)

    # Determine label columns based on mode
    if label_cols is None:
        label_cols = list(LABEL_COLS_SCRATCH) if mode == "scratch" else list(LABEL_COLS)

    log_fn = lambda msg: _qlog(q, msg)

    log_fn("=" * 60)
    log_fn(f"  GeneVariate NS Repair Pipeline")
    log_fn(f"  Mode: {mode} | Model: {model}")
    log_fn(f"  Fields: {', '.join(label_cols)}")
    log_fn("=" * 60)

    # ── Validate inputs ──────────────────────────────────────────────────

    if mode == "scratch":
        if not gsm_file or not os.path.exists(gsm_file):
            log_fn(f"[ERROR] GSM file not found: {gsm_file}")
            _qdone(q, False, "GSM file not found")
            return
    elif mode == "repair":
        if not base_dir or not os.path.isdir(base_dir):
            log_fn(f"[ERROR] Base directory not found: {base_dir}")
            _qdone(q, False, "Base directory not found")
            return

    # ── Create output directory ──────────────────────────────────────────

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "scratch":
            output_dir = os.path.join(
                base_dir or os.path.dirname(gsm_file),
                f"{platform_id}_NS_repaired_final_results"
            )
        else:
            output_dir = os.path.join(
                base_dir, f"NS_repaired_final_results_{timestamp}"
            )

    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    log_fn(f"  Output: {output_dir}")

    # ── Ensure Ollama server is running ──────────────────────────────────

    _qprogress(q, 2, "Checking Ollama server...")

    if not ollama_server_ok(ollama_url):
        log_fn("[Ollama] Server not responding, attempting to start...")
        kill_ollama(log_fn)
        time.sleep(2)

        # Compute initial parallelism for server config
        gpus = detect_gpus()
        init_parallel = max(1, len(gpus) * 2) if gpus else 2
        proc = start_ollama_server_blocking(log_fn, num_parallel=init_parallel)
        if proc is None:
            log_fn("[ERROR] Could not start Ollama server")
            _qdone(q, False, "Ollama server failed to start")
            return
    else:
        log_fn("[Ollama] Server already running")

    # Check GPU offload status
    offload_type, vram_used = check_ollama_gpu(ollama_url)
    log_fn(f"  Offload: {offload_type} ({vram_used} GB VRAM)")

    # ── Compute parallel worker count ────────────────────────────────────

    if max_workers_override > 0:
        total_workers = max_workers_override
        gpu_workers = 0
        cpu_workers = total_workers
    else:
        total_workers, gpu_workers, cpu_workers = compute_ollama_parallel(model)

    log_fn(f"  Workers: {total_workers} total "
           f"({gpu_workers} GPU + {cpu_workers} CPU)")
    _qprogress(q, 5, "Server ready")

    # ── Start CPU Ollama if beneficial ───────────────────────────────────

    cpu_proc = None
    if cpu_workers > 0 and gpu_workers > 0:
        log_fn("  Starting CPU Ollama for hybrid dispatch...")
        cpu_proc = start_ollama_cpu_server(log_fn, num_parallel=cpu_workers)
        if cpu_proc:
            log_fn("  CPU Ollama running on port 11435")

    # ── Start watchdog with fluid scaling ─────────────────────────────────

    watchdog = None
    throttle_sem = None
    _throttle_current = [total_workers]

    if enable_watchdog:
        watchdog = Watchdog(
            log_fn=log_fn,
            stat_fn=lambda msg: _qwatchdog(q, msg),
        )
        watchdog._model = model
        watchdog._max_workers = total_workers

        # Shared throttle semaphore -- fluid worker scaling across ALL phases
        throttle_sem = threading.Semaphore(total_workers)
        _throttle_lock = threading.Lock()

        def _throttle_adjust(new_n: int):
            """Scale the shared throttle semaphore up or down. Thread-safe."""
            with _throttle_lock:
                old_n = _throttle_current[0]
                if new_n == old_n:
                    return
                if new_n > old_n:
                    for _ in range(new_n - old_n):
                        throttle_sem.release()
                else:
                    drained = 0
                    for _ in range(old_n - new_n):
                        got = throttle_sem.acquire(blocking=False)
                        if got:
                            drained += 1
                        else:
                            break
                _throttle_current[0] = new_n

        watchdog._adjust_concurrency = _throttle_adjust
        watchdog._target_parallel = total_workers
        watchdog.start()
        log_fn(f"  Watchdog started | {total_workers} fluid workers "
               f"(scales {Watchdog.MIN_WORKERS}-{total_workers}) | "
               f"scale down at CPU>{Watchdog.CPU_HIGH_PCT:.0f}% / "
               f"RAM>{Watchdog.RAM_HIGH_PCT:.0f}%")

    # ── Build MemoryAgent ────────────────────────────────────────────────

    _qprogress(q, 8, "Building memory agent...")

    mem_db_path = os.path.join(output_dir, MEM_DB_NAME)
    mem_agent = MemoryAgent(mem_db_path, ollama_url=ollama_url)

    # Check if DB has clusters; if not, try loading from .txt files
    ms = mem_agent.stats()
    total_clusters = sum(ms.get("clusters", {}).values())

    if total_clusters == 0 and memory_dir:
        log_fn("  Memory DB empty -- importing from cluster files...")
        mem_agent.build_from_clusters(memory_dir, log_fn=log_fn)
    elif total_clusters > 0:
        log_fn(f"  Memory DB loaded: {total_clusters} clusters")
        mem_agent.load_cache_all(log_fn=log_fn)

    _qprogress(q, 12, "Memory agent ready")

    # ── Load platform data ───────────────────────────────────────────────

    all_dfs: Dict[str, pd.DataFrame] = {}

    if mode == "scratch":
        log_fn(f"  Loading GSM list: {gsm_file}")
        df = load_gsm_list(gsm_file, platform_id)
        if df is None or df.empty:
            log_fn("[ERROR] No GSMs loaded from file")
            _qdone(q, False, "No GSMs loaded")
            if watchdog:
                watchdog.stop()
            return
        all_dfs[platform_id] = df
        log_fn(f"  Loaded {len(df)} GSMs for platform {platform_id}")
    else:
        log_fn(f"  Loading platforms: {', '.join(platforms)}")
        for gpl in platforms:
            df = load_platform(gpl, base_dir)
            if df is not None:
                all_dfs[gpl] = df
                log_fn(f"  {gpl}: {len(df):,} samples loaded")
            else:
                log_fn(f"  {gpl}: not found, skipping")

        if not all_dfs:
            log_fn("[ERROR] No platform data loaded")
            _qdone(q, False, "No platform data found")
            if watchdog:
                watchdog.stop()
            return

    total_samples = sum(len(df) for df in all_dfs.values())
    log_fn(f"  Total samples: {total_samples:,}")
    _qprogress(q, 18, f"Loaded {total_samples:,} samples")

    # ── Load GEOmetadb ───────────────────────────────────────────────────

    _qprogress(q, 20, "Loading GEOmetadb...")

    geo_conn = None
    if geo_db_path:
        geo_conn = load_db_to_memory(geo_db_path, log_fn=log_fn)

    # Auto-detect if not specified
    if geo_conn is None and base_dir:
        for candidate in [
            os.path.join(base_dir, "GEOmetadb.sqlite.gz"),
            os.path.join(base_dir, "GEOmetadb.sqlite"),
            os.path.join(os.path.dirname(base_dir), "data", "GEOmetadb.sqlite.gz"),
        ]:
            if os.path.exists(candidate):
                geo_conn = load_db_to_memory(candidate, log_fn=log_fn)
                if geo_conn:
                    break

    _qprogress(q, 28, "GEOmetadb loaded")

    # ── Process each platform ────────────────────────────────────────────

    try:
        grand_stats = {
            "total_samples": total_samples,
            "ns_before": {col: 0 for col in label_cols},
            "ns_after": {col: 0 for col in label_cols},
            "repaired": {col: 0 for col in label_cols},
            "workers_launched": 0,
            "errors": 0,
        }

        platform_idx = 0
        total_platforms = len(all_dfs)

        for gpl, df in all_dfs.items():
            platform_idx += 1
            pbase = 30 + int(60 * (platform_idx - 1) / total_platforms)
            pend = 30 + int(60 * platform_idx / total_platforms)

            log_fn("")
            log_fn(f"{'='*50}")
            log_fn(f"  Processing {gpl} ({len(df):,} samples) "
                   f"[{platform_idx}/{total_platforms}]")
            log_fn(f"{'='*50}")

            # Platform output directory
            plat_out = os.path.join(output_dir, f"{gpl}_NS_repaired_final_results")
            os.makedirs(plat_out, exist_ok=True)

            # ── Fetch raw metadata from GEOmetadb ────────────────────

            _qprogress(q, pbase, f"{gpl}: fetching metadata...")

            gsm_ids = df["GSM"].unique().tolist()
            raw_meta = {}

            if geo_conn is not None:
                raw_meta = fetch_gsm_raw(geo_conn, gsm_ids)
                log_fn(f"  GEOmetadb: {len(raw_meta):,}/{len(gsm_ids):,} GSMs found")

            # ── Scrape missing GSMs from NCBI ────────────────────────

            missing_gsms = [g for g in gsm_ids if g not in raw_meta]
            if missing_gsms:
                log_fn(f"  {len(missing_gsms):,} GSMs missing from GEOmetadb")
                _qprogress(q, pbase + 3, f"{gpl}: scraping {len(missing_gsms)} GSMs...")

                ncbi_raw = scrape_gsm_raw(
                    missing_gsms, log_fn=log_fn,
                    progress_fn=lambda pct: _qprogress(
                        q, pbase + 3 + int(5 * pct / 100),
                        f"{gpl}: scraping GSMs ({pct}%)"),
                    cache_dir=cache_dir,
                )
                raw_meta.update(ncbi_raw)
                log_fn(f"  After NCBI: {len(raw_meta):,}/{len(gsm_ids):,} GSMs")

            # ── Fetch/scrape GSE metadata ────────────────────────────

            # Collect all GSE IDs
            all_gse_ids = set()
            for gsm, raw in raw_meta.items():
                series = raw.get("series_id", "")
                for sid in series.split(";"):
                    sid = sid.strip()
                    if sid.startswith("GSE"):
                        all_gse_ids.add(sid)

            # Also check df for series_id column
            if "series_id" in df.columns:
                for sid in df["series_id"].dropna().unique():
                    for part in str(sid).split(";"):
                        part = part.strip()
                        if part.startswith("GSE"):
                            all_gse_ids.add(part)

            all_gse_ids = sorted(all_gse_ids)
            log_fn(f"  Found {len(all_gse_ids)} unique GSE experiments")

            gse_meta = {}
            if geo_conn is not None:
                gse_meta = _fetch_gse_meta_from_db(geo_conn, all_gse_ids)
                log_fn(f"  GEOmetadb: {len(gse_meta)}/{len(all_gse_ids)} GSE records")

            missing_gses = [g for g in all_gse_ids if g not in gse_meta]
            if missing_gses:
                _qprogress(q, pbase + 8,
                           f"{gpl}: scraping {len(missing_gses)} GSE records...")
                ncbi_gse = scrape_gse_meta(
                    missing_gses, log_fn=log_fn,
                    progress_fn=lambda pct: _qprogress(
                        q, pbase + 8 + int(4 * pct / 100),
                        f"{gpl}: scraping GSEs ({pct}%)"),
                    cache_dir=cache_dir,
                )
                gse_meta.update(ncbi_gse)

            log_fn(f"  GSE metadata: {len(gse_meta)} records total")

            # ── Build GSE contexts ───────────────────────────────────

            _qprogress(q, pbase + 13, f"{gpl}: building GSE contexts...")

            contexts = _build_gse_contexts(
                df, raw_meta, gse_meta, mem_agent, label_cols, log_fn
            )

            # ── Phase 1 extraction (scratch mode) ────────────────────

            if mode == "scratch" and enable_phase1:
                _qprogress(q, pbase + 15, f"{gpl}: Phase 1 extraction...")
                df = _phase1_scratch_extraction(
                    df, raw_meta, model, ollama_url, label_cols,
                    watchdog, log_fn,
                    progress_fn=lambda pct: _qprogress(
                        q, pbase + 15 + int(10 * pct / 100),
                        f"{gpl}: Phase 1 ({pct}%)"),
                )
                # Rebuild contexts after Phase 1
                contexts = _build_gse_contexts(
                    df, raw_meta, gse_meta, mem_agent, label_cols, log_fn
                )

            # ── Initialize CSV flushers ──────────────────────────────

            # Determine output columns
            out_cols = ["GSM"]
            if "platform" in df.columns:
                out_cols.append("platform")
            out_cols.extend(label_cols)
            # Add any other metadata columns
            for c in df.columns:
                if c not in out_cols:
                    out_cols.append(c)

            ns_flusher = CSVFlusher(
                os.path.join(plat_out, "NS_repaired.csv"), out_cols
            )
            full_flusher = CSVFlusher(
                os.path.join(plat_out, "full_repaired.csv"), out_cols
            )
            collapse_reporter = CollapseReporter()

            # ── Dispatch GSE workers ─────────────────────────────────

            repair_pct_start = pbase + 25
            repair_pct_end = pend - 5

            _qprogress(q, repair_pct_start, f"{gpl}: repairing NS labels...")

            plat_stats = _dispatch_gse_workers(
                df=df,
                contexts=contexts,
                raw_meta=raw_meta,
                mem_agent=mem_agent,
                model=model,
                ollama_url=ollama_url,
                label_cols=label_cols,
                platform_id=gpl,
                max_workers=total_workers,
                watchdog=watchdog,
                ns_flusher=ns_flusher,
                full_flusher=full_flusher,
                collapse_reporter=collapse_reporter,
                log_fn=log_fn,
                progress_fn=lambda pct: _qprogress(
                    q, repair_pct_start + int(
                        (repair_pct_end - repair_pct_start) * pct / 100),
                    f"{gpl}: repairing ({pct}%)"),
                q=q,
                throttle_sem=throttle_sem,
            )

            # Finalize flushers
            ns_written = ns_flusher.finalize()
            full_written = full_flusher.finalize()
            log_fn(f"  NS_repaired.csv: {ns_written:,} rows")
            log_fn(f"  full_repaired.csv: {full_written:,} rows")

            # Save collapse report
            collapse_df = collapse_reporter.to_dataframe()
            if not collapse_df.empty:
                collapse_path = os.path.join(plat_out, "collapse_report.csv")
                collapse_df.to_csv(collapse_path, index=False)
                log_fn(f"  collapse_report.csv: {len(collapse_df):,} entries")

            # Accumulate grand stats
            for col in label_cols:
                grand_stats["ns_before"][col] += plat_stats.get(
                    "ns_before", {}).get(col, 0)
                grand_stats["ns_after"][col] += plat_stats.get(
                    "ns_after", {}).get(col, 0)
                grand_stats["repaired"][col] += plat_stats.get(
                    "repaired", {}).get(col, 0)
            grand_stats["workers_launched"] += plat_stats.get(
                "workers_launched", 0)
            grand_stats["errors"] += plat_stats.get("errors", 0)

            # Log platform results
            log_fn(f"\n  {gpl} Results:")
            for col in label_cols:
                before = plat_stats.get("ns_before", {}).get(col, 0)
                repaired = plat_stats.get("repaired", {}).get(col, 0)
                if before > 0:
                    rate = 100.0 * repaired / before
                    log_fn(f"    {col}: {before:,} NS -> "
                           f"{before - repaired:,} NS "
                           f"({repaired:,} repaired, {rate:.1f}%)")

            _qprogress(q, pend, f"{gpl}: complete")

        # ── Write summary ────────────────────────────────────────────────

        elapsed = time.time() - t0
        _write_summary(output_dir, ", ".join(all_dfs.keys()),
                       grand_stats, label_cols, elapsed, model, mem_agent)

        # ── Final log ────────────────────────────────────────────────────

        log_fn("")
        log_fn("=" * 60)
        log_fn("  PIPELINE COMPLETE")
        log_fn("=" * 60)
        log_fn(f"  Elapsed: {timedelta(seconds=int(elapsed))}")
        log_fn(f"  Output:  {output_dir}")
        for col in label_cols:
            before = grand_stats["ns_before"].get(col, 0)
            repaired = grand_stats["repaired"].get(col, 0)
            if before > 0:
                rate = 100.0 * repaired / before
                log_fn(f"  {col}: {before:,} NS -> "
                       f"{before - repaired:,} NS "
                       f"({repaired:,} repaired, {rate:.1f}%)")
        if grand_stats["errors"] > 0:
            log_fn(f"  Errors: {grand_stats['errors']}")
        log_fn("=" * 60)

        _qprogress(q, 100, "Complete")
        _qdone(q, True, f"Pipeline complete in {timedelta(seconds=int(elapsed))}")

    except Exception as exc:
        log_fn(f"\n[FATAL ERROR] {exc}")
        log_fn(traceback.format_exc())
        _qdone(q, False, str(exc))

    finally:
        # Cleanup
        if watchdog:
            watchdog.stop()
        if cpu_proc:
            try:
                cpu_proc.terminate()
            except Exception:
                pass
        if geo_conn:
            try:
                geo_conn.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience entry points
# ═══════════════════════════════════════════════════════════════════════════════

def run_repair(base_dir: str,
               platforms: Optional[List[str]] = None,
               model: str = DEFAULT_MODEL,
               geo_db_path: str = "",
               memory_dir: str = "",
               output_dir: str = "",
               q=None):
    """Convenience wrapper for repair mode."""
    config = {
        "mode": "repair",
        "base_dir": base_dir,
        "platforms": platforms or ALL_GPLS,
        "model": model,
        "geo_db_path": geo_db_path,
        "memory_dir": memory_dir,
        "output_dir": output_dir,
    }
    pipeline(config, q)


def run_scratch(gsm_file: str,
                platform_id: str = "CUSTOM",
                model: str = DEFAULT_MODEL,
                geo_db_path: str = "",
                memory_dir: str = "",
                output_dir: str = "",
                q=None):
    """Convenience wrapper for scratch mode."""
    config = {
        "mode": "scratch",
        "gsm_file": gsm_file,
        "platform_id": platform_id,
        "base_dir": os.path.dirname(gsm_file),
        "model": model,
        "geo_db_path": geo_db_path,
        "memory_dir": memory_dir,
        "output_dir": output_dir,
    }
    pipeline(config, q)
