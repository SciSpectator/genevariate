"""
Retry the zebrafish GPL platforms that failed on first run.

For each failure we choose the best available identifier column from the GPL
annotation table (GB_ACC, miRNA_ID, ENSEMBL_ID, CLONE_ID, ...) as a
`gene_col_override`.  Platforms whose series matrices have no expression table
(supplementary-files-only) or whose probe IDs are fundamentally incompatible
are skipped and noted.
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

# gene_col_override per platform (None = let auto-detect run again)
RETRY_PLAN = {
    "GPL3721":  "GB_ACC",
    "GPL7338":  "GB_ACC",
    "GPL7343":  "GB_ACC",
    "GPL13908": "GB_ACC",
    "GPL530":   "CLONE_ID",
    "GPL531":   "CLONE_ID",
    "GPL13540": "miRNA_ID",
    "GPL17951": "miRNA_LIST",
    "GPL20900": "ENSEMBL_ID",
    "GPL13832": None,         # original fail was a network glitch on annotation
    "GPL5746":  "SEQ_ID",     # Nimblegen tiling — SEQ_ID is the closest ID
    "GPL5783":  "SEQ_ID",
}

# Known unrecoverable — noted, not retried
SKIP = {
    "GPL3766":  "series matrix has no expression table (supplementary-file only)",
    "GPL3767":  "series matrix has no expression table (supplementary-file only)",
    "GPL24619": "series matrix has no expression table (supplementary-file only)",
    "GPL4481":  "probe IDs in expression data don't match annotation format",
    "GPL15957": "expression matrix uses miRNA names, annotation uses probe IDs — incompatible",
    "GPL10835": "aCGH/tiling array — annotation has no gene/ID column, only chromosomal coords",
    "GPL13500": "aCGH array — annotation has no gene/ID column, only chromosomal coords",
}


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


def cb(pct, stage, msg):
    if pct is None:
        log(f"  [{stage}] {msg}")
    else:
        log(f"  [{stage}] {pct}% | {msg}")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    status = load_status()

    downloader = GPLDownloader(
        gds_conn=conn,
        output_base_dir=str(OUT_DIR),
        max_workers=3,
        download_timeout=180,
    )
    downloader.check_dependencies()

    log(f"=== RETRY failed platforms ({len(RETRY_PLAN)} to attempt, "
        f"{len(SKIP)} skipped as unrecoverable) ===")

    # Mark skipped ones clearly in status
    for gpl_id, reason in SKIP.items():
        status[gpl_id] = {
            "state": "skipped",
            "reason": reason,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    save_status(status)

    for idx, (gpl_id, override) in enumerate(RETRY_PLAN.items(), 1):
        log(f"[{idx}/{len(RETRY_PLAN)}] {gpl_id} — gene_col_override={override!r}")
        try:
            info = query_gpl_info(conn, gpl_id)
            result = downloader.run_with_info(
                info,
                gene_col_override=override,
                callback=cb,
                clear_cache=True,   # force clean slate
            )
            status[gpl_id] = {
                "state": "done",
                "filepath": result["filepath"],
                "n_samples": result["n_samples"],
                "n_genes": result["n_genes"],
                "n_series": result["n_series"],
                "n_failed": result["n_failed"],
                "gene_col_used": result["gene_col_used"],
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            log(f"  OK: {result['n_samples']} samples x {result['n_genes']} "
                f"genes -> {result['filepath']}")
        except Exception as exc:
            status[gpl_id] = {
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "gene_col_tried": override,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            log(f"  FAIL {gpl_id}: {type(exc).__name__}: {exc}")
            for ln in traceback.format_exc().strip().splitlines()[-3:]:
                log(f"    {ln}")
        finally:
            save_status(status)

    # Final cleanup of raw_matrices + annotation caches for any retried platform
    import shutil
    for gpl_id in list(RETRY_PLAN.keys()):
        for sub in ("raw_matrices", "annotation"):
            d = OUT_DIR / gpl_id / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

    # Remove empty GPL folders (for retries that still failed)
    for d in list(OUT_DIR.glob("GPL*")):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    done = sum(1 for v in status.values() if v.get("state") == "done")
    err = sum(1 for v in status.values() if v.get("state") == "error")
    skp = sum(1 for v in status.values() if v.get("state") == "skipped")
    log(f"RETRY FINISHED. total done={done}, error={err}, skipped={skp}")


if __name__ == "__main__":
    main()
