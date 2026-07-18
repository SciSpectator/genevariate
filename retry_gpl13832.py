"""
Custom retry for GPL13832, whose annotation GEOparse chokes on
(Content-Length mismatch from acc.cgi). We download the SOFT file with
requests/curl and hand a local filepath to GEOparse, then run the rest
of the GeneVariate pipeline as usual.
"""

import sys
import os
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import GEOparse  # noqa: E402
from genevariate.core import gpl_downloader as gd  # noqa: E402
from genevariate.core.gpl_downloader import (  # noqa: E402
    GPLDownloader, query_gpl_info, _detect_gene_col, _NULL_SYMBOLS,
)

GPL_ID = "GPL13832"
DB_PATH = PROJECT / "src" / "genevariate" / "data" / "GEOmetadb.sqlite.tmp.sqlite"
OUT_DIR = PROJECT / "Zebrafish_gpls"
STATUS_FILE = OUT_DIR / "_status.json"
LOG_FILE = OUT_DIR / "_run_log.txt"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def patched_download_annotation(gpl_id, dest_dir, gene_col_override=None):
    """Load a locally pre-downloaded SOFT file."""
    import pandas as pd
    os.makedirs(dest_dir, exist_ok=True)
    soft_path = os.path.join(dest_dir, f"{gpl_id}.txt")
    if not os.path.exists(soft_path):
        raise FileNotFoundError(f"Expected pre-downloaded SOFT at {soft_path}")
    gpl = GEOparse.get_GEO(filepath=soft_path, silent=True)
    tbl = gpl.table
    avail = tbl.columns.tolist()
    gcol = gene_col_override if gene_col_override in avail else _detect_gene_col(avail)
    if gcol is None:
        raise ValueError(f"No gene column detected for {gpl_id}. Columns: {avail}")
    id_col = "ID" if "ID" in tbl.columns else tbl.columns[0]
    m = tbl[[id_col, gcol]].copy()
    m.columns = ["probe_id", "gene_symbol"]
    m["gene_symbol"] = m["gene_symbol"].astype(str).str.strip()
    m = m[m["gene_symbol"].notna() & ~m["gene_symbol"].isin(_NULL_SYMBOLS)]
    m = m.drop_duplicates()
    return {
        "mapping": m,
        "gene_col": gcol,
        "available_columns": avail,
        "n_probes": len(m),
        "n_genes": m["gene_symbol"].nunique(),
    }


def main():
    log(f"=== Custom retry for {GPL_ID} (GEOparse-bypass) ===")
    conn = sqlite3.connect(str(DB_PATH))
    status = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}

    # Monkey-patch so GPLDownloader picks up the local SOFT file
    gd.download_annotation = patched_download_annotation

    dl = GPLDownloader(gds_conn=conn, output_base_dir=str(OUT_DIR),
                       max_workers=3, download_timeout=240)
    dl.check_dependencies()

    def cb(pct, stage, msg):
        log(f"  [{stage}]" + (f" {pct}%" if pct is not None else "") + f" | {msg}")

    try:
        info = query_gpl_info(conn, GPL_ID)
        # Candidate gene columns to try, in priority order
        candidates = ["GENE_SYMBOL", "Gene_Symbol", "miRNA_ID", "MIRNA_ID",
                      "Symbol", "miRNA", "MB ACC", "Reporter_Source"]

        # Peek the SOFT file to pick the best available column
        soft = OUT_DIR / GPL_ID / "annotation" / f"{GPL_ID}.txt"
        gpl = GEOparse.get_GEO(filepath=str(soft), silent=True)
        cols = gpl.table.columns.tolist()
        log(f"  annotation columns: {cols}")
        chosen = next((c for c in candidates if c in cols), None)
        if chosen is None:
            chosen = _detect_gene_col(cols)
        log(f"  chosen gene_col_override = {chosen!r}")

        result = dl.run_with_info(info, gene_col_override=chosen,
                                  callback=cb, clear_cache=True)
        status[GPL_ID] = {
            "state": "done",
            "filepath": result["filepath"],
            "n_samples": result["n_samples"],
            "n_genes": result["n_genes"],
            "n_series": result["n_series"],
            "gene_col_used": result["gene_col_used"],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "note": "GEOparse bypassed for annotation (Content-Length mismatch)",
        }
        log(f"  OK: {result['n_samples']} samples x {result['n_genes']} genes")
    except Exception as exc:
        import traceback
        status[GPL_ID] = {
            "state": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        log(f"  FAIL {GPL_ID}: {type(exc).__name__}: {exc}")
        for ln in traceback.format_exc().strip().splitlines()[-5:]:
            log(f"    {ln}")

    STATUS_FILE.write_text(json.dumps(status, indent=2))

    # Cleanup caches
    for sub in ("raw_matrices", "annotation"):
        d = OUT_DIR / GPL_ID / sub
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    gdir = OUT_DIR / GPL_ID
    if gdir.is_dir() and not any(gdir.iterdir()):
        gdir.rmdir()

    done = sum(1 for v in status.values() if v.get("state") == "done")
    err = sum(1 for v in status.values() if v.get("state") == "error")
    skp = sum(1 for v in status.values() if v.get("state") == "skipped")
    log(f"FINAL. done={done}, error={err}, skipped={skp}")


if __name__ == "__main__":
    main()
