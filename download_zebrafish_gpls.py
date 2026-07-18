"""
Download every zebrafish (Danio rerio) GPL microarray platform from GEO using
genevariate's GPLDownloader, placing outputs into ./Zebrafish_gpls.

Runs each platform in sequence; continues on error.
Progress is appended to Zebrafish_gpls/_run_log.txt.
"""

import os
import sys
import json
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from genevariate.core.gpl_downloader import GPLDownloader, query_gpl_info  # noqa: E402

DB_PATH = PROJECT / "src" / "genevariate" / "data" / "GEOmetadb.sqlite.tmp.sqlite"
OUT_DIR = PROJECT / "Zebrafish_gpls"
LOG_FILE = OUT_DIR / "_run_log.txt"
STATUS_FILE = OUT_DIR / "_status.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_status():
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_status(status):
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def list_zebrafish_gpls(conn):
    """Danio rerio microarray platforms with >=1 associated GSE series."""
    q = """
        SELECT g.gpl, g.title, g.technology,
               (SELECT COUNT(DISTINCT gse) FROM gse_gpl WHERE gpl = g.gpl) AS n_gse
        FROM gpl g
        WHERE g.organism = 'Danio rerio'
          AND g.technology IN (
              'in situ oligonucleotide',
              'spotted oligonucleotide',
              'spotted DNA/cDNA',
              'oligonucleotide beads'
          )
    """
    rows = conn.execute(q).fetchall()
    rows = [r for r in rows if (r[3] or 0) > 0]
    # largest first so headline platforms land soonest
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


def cb(pct, stage, msg):
    if pct is None:
        log(f"  [{stage}] {msg}")
    else:
        log(f"  [{stage}] {pct}% | {msg}")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    gpls = list_zebrafish_gpls(conn)
    log(f"Found {len(gpls)} zebrafish microarray platforms with GSE series")

    downloader = GPLDownloader(
        gds_conn=conn,
        output_base_dir=str(OUT_DIR),
        max_workers=3,
        download_timeout=180,
    )
    downloader.check_dependencies()

    status = load_status()

    for idx, (gpl_id, title, tech, n_gse) in enumerate(gpls, 1):
        prev = status.get(gpl_id, {})
        if prev.get("state") == "done":
            log(f"[{idx}/{len(gpls)}] {gpl_id}: skip (already done)")
            continue

        log(f"[{idx}/{len(gpls)}] {gpl_id} — {title!r} | {tech} | {n_gse} GSE")
        try:
            info = query_gpl_info(conn, gpl_id)
            result = downloader.run_with_info(info, callback=cb)
            status[gpl_id] = {
                "state": "done",
                "filepath": result["filepath"],
                "n_samples": result["n_samples"],
                "n_genes": result["n_genes"],
                "n_series": result["n_series"],
                "n_failed": result["n_failed"],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            log(f"  OK: {result['n_samples']} samples x {result['n_genes']} genes "
                f"-> {result['filepath']}")
        except Exception as exc:
            tb = traceback.format_exc()
            status[gpl_id] = {
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            log(f"  FAIL {gpl_id}: {type(exc).__name__}: {exc}")
            # one-line traceback summary only
            for ln in tb.strip().splitlines()[-3:]:
                log(f"    {ln}")
        finally:
            save_status(status)

    done = sum(1 for v in status.values() if v.get("state") == "done")
    err = sum(1 for v in status.values() if v.get("state") == "error")
    log(f"FINISHED. ok={done}, error={err}, total={len(gpls)}")


if __name__ == "__main__":
    main()
