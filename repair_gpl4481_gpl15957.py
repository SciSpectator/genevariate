"""
Operational fix for two zebrafish GPLs that failed the main pipeline due
to bugs in gpl_downloader.py.  We DO NOT edit the source module — we
monkey-patch two functions at runtime just for this one job:

  GPL4481 — zero-padded probe IDs in the series matrix (e.g. '001001001001')
            get merged against annotation IDs that pandas/GEOparse read as
            int64 and stripped of leading zeros.  Fix: width-match the
            shorter key side by left-padding with '0' before the merge.

  GPL15957 — the platform's `ID` column contains probe/spot names
             (BKG0, Ctr01-2M05, ...) while the series matrix indexes rows
             by miRNA name (dre-miR-455).  The annotation's `miRNA_ID`
             column holds the real join key.  Fix: let an override choose
             the annotation probe-ID column, independently of the gene
             column.
"""

import os
import re
import sys
import json
import sqlite3
import shutil
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import pandas as pd                                         # noqa: E402
import GEOparse                                             # noqa: E402
from genevariate.core import gpl_downloader as gd           # noqa: E402
from genevariate.core.gpl_downloader import (               # noqa: E402
    GPLDownloader, query_gpl_info, _detect_gene_col, _NULL_SYMBOLS,
)

DB_PATH = PROJECT / "src" / "genevariate" / "data" / "GEOmetadb.sqlite.tmp.sqlite"
OUT_DIR = PROJECT / "Zebrafish_gpls"
STATUS_FILE = OUT_DIR / "_status.json"
LOG_FILE = OUT_DIR / "_run_log.txt"


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# -------------------------------------------------------------------
# PATCH 1: aggregate_probes — zero-pad fallback
# -------------------------------------------------------------------
_original_aggregate_probes = gd.aggregate_probes


def patched_aggregate_probes(expression_df, mapping_df):
    expr = expression_df.copy()
    expr.index = expr.index.astype(str).str.strip()
    expr.index.name = 'probe_id'
    expr = expr.reset_index()

    mapping_clean = mapping_df.copy()
    mapping_clean['probe_id'] = mapping_clean['probe_id'].astype(str).str.strip()

    merged = expr.merge(mapping_clean, on='probe_id', how='inner')

    def _has_values(m):
        if m.empty:
            return False
        data = m.drop(columns=['probe_id', 'gene_symbol'])
        return data.count().sum() > 0

    if not _has_values(merged):
        # Try stripping trailing .0
        expr['probe_id'] = expr['probe_id'].str.replace(r'\.0$', '', regex=True)
        mapping_clean['probe_id'] = mapping_clean['probe_id'].str.replace(r'\.0$', '', regex=True)
        merged = expr.merge(mapping_clean, on='probe_id', how='inner')

    # NEW FALLBACK: zero-pad the shorter side to match the longer side when
    # both sides look entirely numeric.
    if not _has_values(merged):
        def _all_digits(series):
            s = series.dropna().astype(str)
            if s.empty:
                return False
            return s.str.fullmatch(r'\d+').all()

        if _all_digits(expr['probe_id']) and _all_digits(mapping_clean['probe_id']):
            ewid = expr['probe_id'].dropna().astype(str).str.len().max()
            mwid = mapping_clean['probe_id'].dropna().astype(str).str.len().max()
            target = int(max(ewid, mwid))
            expr['probe_id'] = expr['probe_id'].astype(str).str.zfill(target)
            mapping_clean['probe_id'] = mapping_clean['probe_id'].astype(str).str.zfill(target)
            merged = expr.merge(mapping_clean, on='probe_id', how='inner')
            log(f"  [repair] zero-padded probe IDs to width {target} -> "
                f"{len(merged):,} rows merged")

    if merged.empty:
        raise ValueError(
            "aggregate_probes still empty after zero-pad repair; probe ID "
            "formats truly incompatible."
        )

    data_cols = [c for c in merged.columns if c not in ('probe_id', 'gene_symbol')]
    if merged[data_cols].count().sum() == 0:
        raise ValueError("All expression values NaN after merge.")

    merged = merged.drop(columns=['probe_id'])
    return merged.groupby('gene_symbol').mean()


# -------------------------------------------------------------------
# PATCH 2: download_annotation — optional probe_col_override
# -------------------------------------------------------------------
def patched_download_annotation(gpl_id, dest_dir, gene_col_override=None,
                                probe_col_override=None, probe_zfill=None):
    """Like the original, but lets caller pick the annotation probe-ID column
    and optionally zero-pad numeric probe IDs to a fixed width."""
    os.makedirs(dest_dir, exist_ok=True)

    # Prefer a locally pre-downloaded SOFT file if we have one (for GPLs
    # where GEOparse's download is unreliable).
    soft_path = os.path.join(dest_dir, f"{gpl_id}.txt")
    if os.path.exists(soft_path) and os.path.getsize(soft_path) > 1024:
        gpl = GEOparse.get_GEO(filepath=soft_path, silent=True)
    else:
        try:
            gpl = GEOparse.get_GEO(geo=gpl_id, destdir=dest_dir, silent=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to download/parse GPL annotation for {gpl_id}: {exc}"
            ) from exc

    tbl = getattr(gpl, 'table', None)
    if tbl is None or tbl.empty:
        raise ValueError(f"No annotation table found for {gpl_id}")

    avail = tbl.columns.tolist()

    gcol = gene_col_override if gene_col_override in avail else _detect_gene_col(avail)
    if gcol is None:
        raise ValueError(
            f"No gene symbol column found for {gpl_id}. Columns: {avail}"
        )

    if probe_col_override and probe_col_override in avail:
        id_col = probe_col_override
    else:
        id_col = 'ID' if 'ID' in tbl.columns else tbl.columns[0]

    m = tbl[[id_col, gcol]].copy()
    m.columns = ['probe_id', 'gene_symbol']
    m['gene_symbol'] = m['gene_symbol'].astype(str).str.strip()
    m['probe_id']    = m['probe_id'].astype(str).str.strip()

    # Zero-pad numeric probe IDs if requested (fixes integer-coerced leading-zero loss)
    if probe_zfill:
        digit_mask = m['probe_id'].str.fullmatch(r'\d+')
        m.loc[digit_mask, 'probe_id'] = (
            m.loc[digit_mask, 'probe_id'].str.zfill(int(probe_zfill))
        )

    # Standard cleanups copied from the original
    mask3 = m['gene_symbol'].str.contains('///', na=False)
    if mask3.any():
        m.loc[mask3, 'gene_symbol'] = (
            m.loc[mask3, 'gene_symbol']
             .str.split(r'\s*///\s*').str[0].str.strip()
        )
    if gcol.lower() == 'gene_assignment':
        mask2 = m['gene_symbol'].str.contains('//', na=False)
        if mask2.any():
            def _extract(v):
                parts = re.split(r'\s*//\s*', str(v))
                return parts[1].strip() if len(parts) > 1 else parts[0].strip()
            m.loc[mask2, 'gene_symbol'] = m.loc[mask2, 'gene_symbol'].apply(_extract)

    m = m[m['gene_symbol'].notna() & ~m['gene_symbol'].isin(_NULL_SYMBOLS)]
    m = m.drop_duplicates()

    return {
        'mapping':           m,
        'gene_col':          gcol,
        'probe_col':         id_col,
        'available_columns': avail,
        'n_probes':          len(m),
        'n_genes':           m['gene_symbol'].nunique(),
    }


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
PLAN = [
    # (gpl_id, gene_col_override, probe_col_override, probe_zfill, note)
    ("GPL4481",  "GB_ACC",    None,       12,   "zero-pad probe IDs to 12 digits (leading-zero bug)"),
    ("GPL15957", "miRNA_ID",  "miRNA_ID", None, "join on miRNA_ID (ID col holds probe names)"),
]


def run_one(downloader, conn, gpl_id, gene_col, probe_col, probe_zfill, note, status):
    log(f"[{gpl_id}] {note}")

    def cb(pct, stage, msg):
        log(f"  [{stage}]" + (f" {pct}%" if pct is not None else "") + f" | {msg}")

    # Patch module-level function pointers; they will be hit inside run_with_info
    gd.aggregate_probes = patched_aggregate_probes

    # Wrap the patched download so we can inject probe_col_override without
    # changing run_with_info's signature.
    def _wrapped_annot(gpl_id_, dest_dir, gene_col_override=None):
        return patched_download_annotation(
            gpl_id_, dest_dir,
            gene_col_override=gene_col_override,
            probe_col_override=probe_col,
            probe_zfill=probe_zfill,
        )
    gd.download_annotation = _wrapped_annot

    try:
        info = query_gpl_info(conn, gpl_id)
        result = downloader.run_with_info(info, gene_col_override=gene_col,
                                          callback=cb, clear_cache=True)
        status[gpl_id] = {
            "state": "done",
            "filepath": result["filepath"],
            "n_samples": result["n_samples"],
            "n_genes": result["n_genes"],
            "n_series": result["n_series"],
            "gene_col_used": result["gene_col_used"],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
        }
        log(f"  OK: {result['n_samples']} samples x {result['n_genes']} genes")
    except Exception as exc:
        status[gpl_id] = {
            "state": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        log(f"  FAIL {gpl_id}: {type(exc).__name__}: {exc}")
        for ln in traceback.format_exc().strip().splitlines()[-4:]:
            log(f"    {ln}")


def main():
    log("=== Repair GPL4481 + GPL15957 (monkey-patched fix) ===")
    conn = sqlite3.connect(str(DB_PATH))
    status = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}

    downloader = GPLDownloader(gds_conn=conn, output_base_dir=str(OUT_DIR),
                               max_workers=3, download_timeout=240)
    downloader.check_dependencies()

    for gpl_id, gene_col, probe_col, probe_zfill, note in PLAN:
        run_one(downloader, conn, gpl_id, gene_col, probe_col, probe_zfill, note, status)
        STATUS_FILE.write_text(json.dumps(status, indent=2))

    # Clean up intermediate caches for these two
    for gpl_id, *_ in PLAN:
        for sub in ("raw_matrices", "annotation"):
            d = OUT_DIR / gpl_id / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        gdir = OUT_DIR / gpl_id
        if gdir.is_dir() and not any(gdir.iterdir()):
            gdir.rmdir()

    done = sum(1 for v in status.values() if v.get("state") == "done")
    err = sum(1 for v in status.values() if v.get("state") == "error")
    skp = sum(1 for v in status.values() if v.get("state") == "skipped")
    log(f"FINAL after repair. done={done}, error={err}, skipped={skp}")


if __name__ == "__main__":
    main()
