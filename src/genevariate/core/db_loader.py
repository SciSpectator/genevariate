"""
GEOmetadb Database Loader — single shared helper for the entire application.

All GEOmetadb access goes through open_geometadb() which:
  1. Decompresses .gz ONCE to a persistent file (streamed in chunks, not read())
  2. Opens in disk mode by default (WAL + indexes + tuned PRAGMAs)
  3. Only loads to :memory: on high-RAM devices when it is genuinely safe
  4. Adapts cache/mmap sizes to the device tier

This replaces 6 separate copy-pasted loading functions that all had different
(and often broken) approaches — some calling gzfi.read() to load the entire
decompressed file into RAM, others using tmpfs-backed tempfiles.
"""

import os
import gzip
import shutil
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

# Chunk size for streaming decompression (16 MB)
_DECOMPRESS_CHUNK = 1024 * 1024 * 16

# Filename patterns accepted during auto-discovery
_GEOMETADB_PATTERNS = (
    "GEOmetadb.sqlite",
    "GEOmetadb.sqlite.gz",
    "GEOmetadb.sqlite.tmp.sqlite",
    "GEOmetadb*.sqlite",
    "GEOmetadb*.sqlite.gz",
    "geometadb*.sqlite",
    "geometadb*.sqlite.gz",
)


def find_geometadb(extra_dirs: Optional[List[str]] = None,
                   log_fn: Callable = print) -> Optional[str]:
    """Scan the local filesystem for a GEOmetadb SQLite file.

    Returns the best (largest, most recent) match or None if nothing is found.
    Preference order when multiple matches exist:
        1. uncompressed .sqlite (faster)
        2. larger size (more complete DB)
        3. more recently modified

    Locations searched (dedup'd, existence-checked):
        * CONFIG['paths']['geo_db'] (configured default)
        * CONFIG['paths']['data']
        * Project root, its parent, and ./data under each
        * Current working directory and ./data
        * ~/Desktop, ~/Downloads, ~ (shallow only — no full-home walk)
        * Any `extra_dirs` passed in
    """
    candidates: List[Path] = []
    seen_dirs = set()

    def _add_dir(d):
        if not d:
            return
        p = Path(d).expanduser()
        if p in seen_dirs:
            return
        seen_dirs.add(p)
        if not p.exists() or not p.is_dir():
            return
        for pattern in _GEOMETADB_PATTERNS:
            try:
                for match in p.glob(pattern):
                    if match.is_file() and match.stat().st_size > 1024:
                        candidates.append(match)
            except (OSError, PermissionError):
                continue

    # 1. Config defaults
    cfg_geo = None
    cfg_data = None
    project_base = None
    try:
        from genevariate.config import CONFIG, BASE_DIR
        cfg_geo = CONFIG.get('paths', {}).get('geo_db')
        cfg_data = CONFIG.get('paths', {}).get('data')
        project_base = BASE_DIR
    except Exception:
        pass

    # If the configured default file itself exists, it's the preferred answer
    if cfg_geo and Path(cfg_geo).exists():
        log_fn(f"[GEOmetadb] Using configured path: {cfg_geo}")
        return str(cfg_geo)

    # Otherwise crawl likely directories
    _add_dir(cfg_data)
    if project_base:
        _add_dir(project_base)
        _add_dir(project_base / 'data')
        _add_dir(project_base.parent)
        _add_dir(project_base.parent / 'data')

    _add_dir(os.getcwd())
    _add_dir(Path(os.getcwd()) / 'data')
    _add_dir(Path.home() / 'Desktop')
    _add_dir(Path.home() / 'Downloads')
    _add_dir(Path.home())

    if extra_dirs:
        for d in extra_dirs:
            _add_dir(d)

    if not candidates:
        log_fn("[GEOmetadb] Auto-search found no local GEOmetadb file")
        return None

    # Sort: uncompressed first, then larger, then newer
    def _score(p: Path):
        is_gz = p.suffix == ".gz"
        try:
            size = p.stat().st_size
            mtime = p.stat().st_mtime
        except OSError:
            size, mtime = 0, 0
        return (0 if not is_gz else 1, -size, -mtime)

    candidates = sorted(set(candidates), key=_score)
    best = candidates[0]
    log_fn(f"[GEOmetadb] Auto-discovered: {best} "
           f"({best.stat().st_size / (1024**2):.0f} MB"
           + (f"; {len(candidates)-1} other candidate(s) ignored" if len(candidates) > 1 else "")
           + ")")
    return str(best)


def _persistent_decompress(gz_path: str,
                           log_fn: Callable = print) -> Optional[str]:
    """Decompress .sqlite.gz to a persistent .sqlite file next to the source.

    Streams in 16 MB chunks — never holds the whole file in RAM.
    Reuses the decompressed file on subsequent calls.
    Returns the path to the decompressed file, or None on failure.
    """
    out_path = gz_path.replace(".sqlite.gz", ".sqlite")
    if os.path.exists(out_path):
        return out_path

    # Use a .part suffix while writing so a crash won't leave a corrupt file
    part_path = out_path + ".part"
    log_fn(f"[GEOmetadb] Decompressing {os.path.basename(gz_path)} "
           f"(streaming, 16 MB chunks)...")
    try:
        t0 = time.time()
        written = 0
        with gzip.open(gz_path, "rb") as gz_in:
            with open(part_path, "wb") as f_out:
                while True:
                    chunk = gz_in.read(_DECOMPRESS_CHUNK)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    written += len(chunk)
                    # Progress every ~256 MB
                    if written % (256 * 1024 * 1024) < _DECOMPRESS_CHUNK:
                        log_fn(f"[GEOmetadb]   ...{written / (1024**3):.1f} GB written")
        os.rename(part_path, out_path)
        elapsed = time.time() - t0
        size_gb = written / (1024 ** 3)
        log_fn(f"[GEOmetadb] Decompressed to {os.path.basename(out_path)} "
               f"({size_gb:.1f} GB) in {elapsed:.0f}s")
        return out_path
    except Exception as exc:
        log_fn(f"[GEOmetadb] Decompress failed: {exc}")
        # Clean up partial file
        try:
            os.remove(part_path)
        except OSError:
            pass
        return None


def _ensure_indexes(conn: sqlite3.Connection):
    """Create indexes for fast lookups (idempotent)."""
    for idx, tbl, col in [
        ("idx_gsm_gsm", "gsm", "gsm"),
        ("idx_gsm_series", "gsm", "series_id"),
        ("idx_gsm_gpl", "gsm", "gpl"),
        ("idx_gse_gse", "gse", "gse"),
    ]:
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl}({col})")
        except Exception:
            pass
    try:
        conn.commit()
    except Exception:
        pass


def _apply_pragmas(conn: sqlite3.Connection, tier: str):
    """Set SQLite PRAGMAs adapted to the device tier."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA temp_store=MEMORY")

    if tier == "low":
        conn.execute("PRAGMA cache_size=-16384")    # 16 MB page cache
        conn.execute("PRAGMA mmap_size=67108864")    # 64 MB mmap
    elif tier == "medium":
        conn.execute("PRAGMA cache_size=-32768")    # 32 MB page cache
        conn.execute("PRAGMA mmap_size=134217728")   # 128 MB mmap
    else:
        conn.execute("PRAGMA cache_size=-65536")    # 64 MB page cache
        conn.execute("PRAGMA mmap_size=268435456")   # 256 MB mmap


def open_geometadb(db_path: str = None,
                   log_fn: Callable = print,
                   force_disk: bool = False) -> Optional[sqlite3.Connection]:
    """Open GEOmetadb with resource-aware strategy.

    This is THE function all code should call. It:
      1. Decompresses .gz once to a persistent file (streamed, not read())
      2. Decides disk vs RAM based on actual free memory right now
      3. Falls back to disk mode if RAM load fails
      4. Tunes PRAGMAs and indexes for the device tier

    Args:
        db_path:    Path to .sqlite or .sqlite.gz.  If None, uses config default.
        log_fn:     Logging callback.
        force_disk: Always use disk mode (skip RAM detection).

    Returns:
        sqlite3.Connection or None on failure.
    """
    # Resolve path — try in order: explicit arg, config default, local auto-search
    if db_path is None:
        try:
            from genevariate.config import CONFIG
            db_path = str(CONFIG['paths']['geo_db'])
        except Exception:
            db_path = None

    if not db_path or not os.path.exists(db_path):
        if db_path:
            log_fn(f"[GEOmetadb] Configured path not found ({db_path}); "
                   f"auto-searching local filesystem...")
        else:
            log_fn("[GEOmetadb] No path configured; auto-searching local filesystem...")
        found = find_geometadb(log_fn=log_fn)
        if found is None:
            log_fn("[GEOmetadb] Could not locate a local GEOmetadb .sqlite/.sqlite.gz file")
            return None
        db_path = found

    db_path = str(db_path)
    if not os.path.exists(db_path):
        log_fn(f"[GEOmetadb] File not found: {db_path}")
        return None

    # Step 1: Decompress .gz to persistent file (streamed, cached)
    if db_path.endswith(".gz"):
        actual_path = _persistent_decompress(db_path, log_fn)
        if actual_path is None:
            return None
    else:
        actual_path = db_path

    if not os.path.exists(actual_path):
        log_fn(f"[GEOmetadb] Decompressed file not found: {actual_path}")
        return None

    file_mb = os.path.getsize(actual_path) / (1024 * 1024)

    # Step 2: Decide disk vs RAM
    try:
        from genevariate.config import RESOURCE_TIER
        tier = RESOURCE_TIER.get('tier', 'low')
    except Exception:
        tier = 'low'

    use_memory = False
    if not force_disk and tier == 'high':
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / (1024 ** 3)
            need_gb = (file_mb / 1024) * 2 + 3  # 2x DB + 3 GB headroom
            use_memory = free_gb > need_gb
        except Exception:
            pass

    # Step 3: Open connection
    conn = None
    if use_memory:
        log_fn(f"[GEOmetadb] Loading {file_mb:.0f} MB into RAM "
               f"(tier={tier}, free={free_gb:.1f} GB)...")
        try:
            t0 = time.time()
            disk_conn = sqlite3.connect(actual_path, timeout=60)
            mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            mem_conn.text_factory = lambda b: b.decode('utf-8', 'replace')
            disk_conn.backup(mem_conn)
            disk_conn.close()
            conn = mem_conn
            log_fn(f"[GEOmetadb] Loaded into RAM in {time.time() - t0:.1f}s")
        except (MemoryError, Exception) as exc:
            log_fn(f"[GEOmetadb] RAM load failed ({exc}), falling back to disk")
            use_memory = False

    if not use_memory:
        log_fn(f"[GEOmetadb] Opening {file_mb:.0f} MB from disk "
               f"(tier={tier})...")
        try:
            t0 = time.time()
            conn = sqlite3.connect(actual_path, timeout=60,
                                   check_same_thread=False)
            conn.text_factory = lambda b: b.decode('utf-8', 'replace')
            _apply_pragmas(conn, tier)
            _ensure_indexes(conn)
            log_fn(f"[GEOmetadb] Disk mode ready in {time.time() - t0:.1f}s "
                   f"(WAL + indexes)")
        except Exception as exc:
            log_fn(f"[GEOmetadb] Open failed: {exc}")
            return None

    # Step 4: Verify
    if conn is not None:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for required in ("gsm", "gse"):
            if required not in tables:
                log_fn(f"[GEOmetadb] WARNING: table '{required}' not found")

    return conn
