"""Retry only the platforms still in 'error' state."""

import sys, json, sqlite3, shutil, traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from genevariate.core.gpl_downloader import GPLDownloader, query_gpl_info  # noqa

DB_PATH = PROJECT / "src" / "genevariate" / "data" / "GEOmetadb.sqlite.tmp.sqlite"
OUT_DIR = PROJECT / "Zebrafish_gpls"
LOG_FILE = OUT_DIR / "_run_log.txt"
STATUS_FILE = OUT_DIR / "_status.json"

MAX_ATTEMPTS = 3


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    status = json.loads(STATUS_FILE.read_text())
    failed = [k for k, v in status.items() if v.get("state") == "error"]
    log(f"=== RETRY REMAINING ({len(failed)} still-failed) ===")

    dl = GPLDownloader(gds_conn=conn, output_base_dir=str(OUT_DIR),
                       max_workers=3, download_timeout=240)
    dl.check_dependencies()

    def cb(pct, stage, msg):
        log(f"  [{stage}]" + (f" {pct}%" if pct is not None else "") + f" | {msg}")

    for gpl_id in failed:
        success = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            log(f"[{gpl_id}] attempt {attempt}/{MAX_ATTEMPTS}")
            try:
                info = query_gpl_info(conn, gpl_id)
                result = dl.run_with_info(info, callback=cb, clear_cache=(attempt > 1))
                status[gpl_id] = {
                    "state": "done",
                    "filepath": result["filepath"],
                    "n_samples": result["n_samples"],
                    "n_genes": result["n_genes"],
                    "n_series": result["n_series"],
                    "gene_col_used": result["gene_col_used"],
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "recovered_on_attempt": attempt,
                }
                log(f"  OK: {result['n_samples']} samples x {result['n_genes']} genes")
                success = True
                break
            except Exception as exc:
                log(f"  FAIL {gpl_id} attempt {attempt}: {type(exc).__name__}: {exc}")
                status[gpl_id] = {
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": attempt,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            STATUS_FILE.write_text(json.dumps(status, indent=2))

        # Clean caches regardless of success
        for sub in ("raw_matrices", "annotation"):
            d = OUT_DIR / gpl_id / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        # Drop empty dir if still failed
        gdir = OUT_DIR / gpl_id
        if gdir.is_dir() and not any(gdir.iterdir()):
            gdir.rmdir()

    STATUS_FILE.write_text(json.dumps(status, indent=2))
    done = sum(1 for v in status.values() if v.get("state") == "done")
    err = sum(1 for v in status.values() if v.get("state") == "error")
    skp = sum(1 for v in status.values() if v.get("state") == "skipped")
    log(f"DONE retry-remaining. total done={done}, error={err}, skipped={skp}")


if __name__ == "__main__":
    main()
