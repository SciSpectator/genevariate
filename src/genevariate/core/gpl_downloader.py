"""
genevariate/core/gpl_downloader.py
====================================
Automated GPL Platform Downloader & Preprocessor for GeneVariate.

Downloads ANY GPL platform from NCBI GEO, preprocesses expression data
(probe->gene mapping, log2 transform, quantile normalization), and outputs
a CSV.GZ file in GeneVariate's standard format:

    Columns:  GSM  |  series_id  |  GENE1  |  GENE2  |  GENE3  |  ...
    Rows:     Each row is one sample (GSM) with its GSE and expression values

NaN values are PRESERVED throughout -- never dropped or filled.
Supports ALL species.

Import in app.py:
    from genevariate.core.gpl_downloader import GPLDownloader, SPECIES_EXAMPLES

Dependencies:
    Required:  GEOparse  (pip install GEOparse)
    Optional:  qnorm     (pip install qnorm) -- numpy fallback if absent
"""

import os
import re
import gzip
import time
import logging
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import GEOparse
    _HAS_GEOPARSE = True
except ImportError:
    _HAS_GEOPARSE = False

try:
    import qnorm
    _HAS_QNORM = True
except ImportError:
    _HAS_QNORM = False

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------

GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

_GENE_COL_PATTERNS = [
    r'^Gene[\s_.]?Symbol$',
    r'^Symbol$',
    r'^GENE_SYMBOL$',
    r'^ILMN_Gene$',
    r'^GeneName$',
    r'^gene_assignment$',
    r'^ORF$',
    r'^GENE_NAME$',
    r'^Gene\.Symbol$',
    r'^gene_symbol$',
    r'^GeneSymbol$',
    r'^Associated[\s_.]Gene[\s_.]Name$',
    r'^GENE$',
    r'^gene$',
    r'^Gene_Name$',
    r'^NAME$',
    r'^Reporter[\s_]Name$',
    r'^SystematicName$',
    r'^SPOT_ID$',
]

_GENE_COL_KEYWORDS = ['symbol', 'gene_name', 'genename', 'gene.symbol']

_NULL_SYMBOLS = frozenset({
    '', 'nan', 'NaN', 'NA', 'N/A', 'n/a', '--', '---',
    'null', 'NULL', 'none', 'None', '.', 'undefined',
})

SPECIES_EXAMPLES = [
    ("\U0001f400 Rat 230 2.0",       "GPL1355"),
    ("\U0001f415 Canine 2.0",        "GPL3738"),
    ("\U0001f412 Rhesus",            "GPL3535"),
    ("\U0001f331 Arabidopsis ATH1",  "GPL198"),
    ("\U0001f9a0 E. coli",           "GPL3154"),
    ("\U0001f41f Zebrafish",         "GPL1319"),
    ("\U0001f416 Porcine",           "GPL3533"),
    ("\U0001fab0 Drosophila 2.0",    "GPL1322"),
    ("\U0001f41b C. elegans",        "GPL200"),
    ("\U0001f37a Yeast S98",         "GPL90"),
]


# -----------------------------------------------------------------------
# HELPER: Data verification
# -----------------------------------------------------------------------

def _data_report(df, label="DataFrame"):
    """Generate a quick data verification string for logging."""
    if df is None:
        return f"{label}: None"
    if df.empty:
        return f"{label}: EMPTY ({df.shape})"
    total = df.size
    non_nan = df.count().sum()  # count of non-NaN values
    pct = 100 * non_nan / total if total > 0 else 0
    # Sample a few actual values
    sample_vals = []
    for c in df.columns[:3]:
        first_valid = df[c].first_valid_index()
        if first_valid is not None:
            sample_vals.append(f"{c}[{first_valid}]={df.at[first_valid, c]}")
    sample_str = ", ".join(sample_vals[:3]) if sample_vals else "ALL NaN"
    return (f"{label}: {df.shape[0]:,}x{df.shape[1]:,}, "
            f"non-NaN: {non_nan:,}/{total:,} ({pct:.1f}%), "
            f"dtypes: {dict(df.dtypes.value_counts())}, "
            f"sample: [{sample_str}]")


# -----------------------------------------------------------------------
# STEP 1 -- Query GEOmetadb for Platform Info & GSE List
# -----------------------------------------------------------------------

def query_gpl_info(gds_conn, gpl_id):
    """
    Query the in-memory GEOmetadb for platform metadata and all GSE series.
    Returns dict: gpl_id, organism, title, technology, gse_list, total_series
    """
    # Case-insensitive match: GEOmetadb may store 'GPL570' or 'gpl570'
    plat = pd.read_sql_query(
        "SELECT gpl, title, organism, technology FROM gpl WHERE gpl = ? COLLATE NOCASE",
        gds_conn, params=[gpl_id]
    )
    if plat.empty:
        plat = pd.read_sql_query(
            "SELECT gpl, title, organism, technology FROM gpl WHERE UPPER(gpl) = UPPER(?)",
            gds_conn, params=[gpl_id]
        )
    if plat.empty:
        raise ValueError(
            f"Platform {gpl_id} not found in GEOmetadb.\n"
            f"Check the GPL ID or update your GEOmetadb.sqlite.gz file."
        )

    row = plat.iloc[0]
    gpl_db_value = str(row['gpl'])
    gse_df = pd.read_sql_query(
        "SELECT DISTINCT gse FROM gse_gpl WHERE gpl = ? COLLATE NOCASE",
        gds_conn, params=[gpl_db_value]
    )

    return {
        'gpl_id':       gpl_id,
        'organism':     str(row.get('organism', 'Unknown')),
        'title':        str(row.get('title', 'Unknown')),
        'technology':   str(row.get('technology', 'Unknown')),
        'gse_list':     gse_df['gse'].tolist(),
        'total_series': len(gse_df),
    }


# -----------------------------------------------------------------------
# STEP 2 -- Download Series Matrix Files from GEO FTP
# -----------------------------------------------------------------------

def _ftp_url(gse_id):
    num = gse_id.replace("GSE", "")
    nnn = (num[:-3] + "nnn") if len(num) > 3 else "nnn"
    return f"{GEO_FTP_BASE}/GSE{nnn}/{gse_id}/matrix/"


def download_one_matrix(gse_id, gpl_id, dest_dir, timeout=180):
    """Download a single series_matrix.txt.gz. Returns filepath or None."""
    base = _ftp_url(gse_id)
    candidates = [
        f"{gse_id}-{gpl_id}_series_matrix.txt.gz",
        f"{gse_id}_series_matrix.txt.gz",
    ]

    for fname in candidates:
        dest = os.path.join(dest_dir, fname)
        # Use cached file if it exists and is reasonably sized
        if os.path.exists(dest) and os.path.getsize(dest) > 500:
            return dest
        # Remove tiny/corrupt cached files
        if os.path.exists(dest):
            os.remove(dest)

        url = base + fname
        # Try up to 2 times (retry once on failure)
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=timeout, stream=True)
                if r.status_code == 200:
                    with open(dest, 'wb') as f:
                        for chunk in r.iter_content(65536):
                            f.write(chunk)
                    if os.path.getsize(dest) > 500:
                        return dest
                    if os.path.exists(dest):
                        os.remove(dest)
                elif r.status_code == 404:
                    break  # File doesn't exist, try next candidate
                else:
                    logger.warning("HTTP %d for %s (attempt %d)",
                                   r.status_code, url, attempt + 1)
                    if attempt == 0:
                        time.sleep(1)  # Brief pause before retry
            except requests.Timeout:
                logger.warning("Timeout downloading %s (attempt %d)", url, attempt + 1)
                if attempt == 0:
                    time.sleep(2)
            except (requests.RequestException, IOError) as exc:
                logger.warning("Error downloading %s: %s", url, exc)
                if os.path.exists(dest):
                    os.remove(dest)
                break  # Don't retry on connection errors
    return None


def batch_download(gse_list, gpl_id, dest_dir,
                   max_workers=4, timeout=180, callback=None):
    """Download series matrices in parallel. Returns (downloaded, failed)."""
    os.makedirs(dest_dir, exist_ok=True)
    downloaded, failed = [], []
    fail_reasons = []
    total = len(gse_list)
    if total == 0:
        return downloaded, failed

    # Use fewer workers to avoid rate limiting
    actual_workers = min(max_workers, 3)

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futs = {
            pool.submit(download_one_matrix, g, gpl_id, dest_dir, timeout): g
            for g in gse_list
        }
        for i, fut in enumerate(as_completed(futs)):
            gse = futs[fut]
            try:
                path = fut.result()
                if path:
                    downloaded.append((gse, path))
                else:
                    failed.append(gse)
                    if len(fail_reasons) < 5:
                        fail_reasons.append(f"{gse}: no valid file found")
            except Exception as exc:
                logger.warning("Download %s failed: %s", gse, exc)
                failed.append(gse)
                if len(fail_reasons) < 5:
                    fail_reasons.append(f"{gse}: {exc}")
            if callback:
                msg = f"Downloaded {i+1}/{total}: {gse}"
                if len(downloaded) > 0 or i < 5:
                    msg += f" (OK: {len(downloaded)}, fail: {len(failed)})"
                callback(int((i + 1) / total * 100), msg)

            # If first 10 all failed, report early
            if i == 9 and len(downloaded) == 0 and callback:
                callback(None,
                         f"WARNING: First 10 GSEs all failed! "
                         f"Reasons: {fail_reasons[:3]}. "
                         f"Check network connection and NCBI accessibility.")

    if failed and callback:
        callback(None,
                 f"Download summary: {len(downloaded)} OK, {len(failed)} failed. "
                 f"Sample failures: {fail_reasons[:3]}")

    return downloaded, failed


# -----------------------------------------------------------------------
# STEP 3 -- Parse Series Matrices (NaNs PRESERVED, GSE->GSM tracked)
# -----------------------------------------------------------------------

def parse_matrix(filepath, gse_id):
    """
    Parse one series_matrix.txt.gz into probes x GSMs DataFrame.
    NaN values are NEVER dropped.
    Returns (DataFrame, list_of_gsm_ids) or (None, []).
    """
    opener = gzip.open if filepath.endswith('.gz') else open
    in_table = False
    header = None
    rows = []

    try:
        with opener(filepath, 'rt', errors='replace') as fh:
            for line in fh:
                line = line.rstrip('\n\r')
                if line.startswith('!series_matrix_table_begin'):
                    in_table = True
                    continue
                if line.startswith('!series_matrix_table_end'):
                    break
                if not in_table:
                    continue
                parts = [p.strip('"') for p in line.split('\t')]
                if header is None:
                    header = parts
                else:
                    rows.append(parts)
    except Exception as exc:
        logger.warning("Parse error %s: %s", filepath, exc)
        return None, []

    if not header or not rows:
        return None, []

    try:
        df = pd.DataFrame(rows, columns=header).set_index(header[0])

        # Convert to numeric column by column — track failures
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Verify we got actual expression values
        non_nan = df.count().sum()
        total = df.size
        if total > 0 and non_nan == 0:
            # ALL values coerced to NaN — check what the raw data looked like
            raw_sample = rows[0][1:4] if rows else []
            logger.warning(
                "Parsed %s but ALL %d values are NaN! "
                "Raw data sample: %s. File may be corrupt or non-standard.",
                filepath, total, raw_sample
            )

        gsm_ids = df.columns.tolist()
        return df, gsm_ids
    except Exception as exc:
        logger.warning("DataFrame error %s: %s", filepath, exc)
        return None, []


def combine_matrices(downloaded_list, callback=None):
    """
    Parse all matrices, merge into one probes x GSMs table,
    and build GSM -> GSE mapping.
    NaN values are PRESERVED -- no dropna anywhere.
    Returns (combined_df, gsm_to_gse_dict)
    """
    frames = []
    gsm_to_gse = {}
    total = len(downloaded_list)
    total_parsed = 0
    total_empty = 0

    for i, (gse_id, fpath) in enumerate(downloaded_list):
        try:
            df, gsm_ids = parse_matrix(fpath, gse_id)
            if df is not None and df.shape[1] > 0:
                non_nan_count = df.count().sum()
                total_cells = df.size
                pct = 100 * non_nan_count / total_cells if total_cells > 0 else 0

                if non_nan_count == 0:
                    total_empty += 1
                    if callback:
                        callback(None,
                                 f"WARNING {gse_id}: {df.shape} but ALL NaN - skipping")
                else:
                    frames.append(df)
                    total_parsed += 1
                    for gsm in gsm_ids:
                        gsm_upper = gsm.strip().upper()
                        if gsm_upper not in gsm_to_gse:
                            gsm_to_gse[gsm_upper] = gse_id

                    # Log first successful parse in detail
                    if total_parsed == 1 and callback:
                        callback(None,
                                 f"First good parse {gse_id}: "
                                 f"{df.shape[0]} probes x {df.shape[1]} samples, "
                                 f"{pct:.0f}% non-NaN, "
                                 f"index sample: {list(df.index[:3])}, "
                                 f"value sample: {df.iloc[0, 0]}")

                if callback:
                    callback(int((i+1)/total*100),
                             f"Parsed {gse_id}: {df.shape[1]} samples, "
                             f"{df.shape[0]} probes, {pct:.0f}% non-NaN")
            else:
                if callback:
                    callback(None, f"Skipped {gse_id}: empty matrix")
        except Exception as exc:
            if callback:
                callback(None, f"Failed {gse_id}: {exc}")

    if not frames:
        raise ValueError(
            f"No expression data could be parsed from any downloaded GSE.\n"
            f"Parsed {total} files: {total_parsed} had data, "
            f"{total_empty} had all-NaN values.\n"
            f"The platform may use supplementary files instead of series matrices."
        )

    # Outer join: preserves all probes, mismatches become NaN (kept)
    combined = pd.concat(frames, axis=1, join='outer')

    # Remove duplicate GSM columns (keep first)
    dup = combined.columns.duplicated(keep='first')
    if dup.any():
        n = dup.sum()
        combined = combined.loc[:, ~dup]
        if callback:
            callback(None, f"Removed {n} duplicate GSM columns")

    return combined, gsm_to_gse


# -----------------------------------------------------------------------
# STEP 4 -- GPL Annotation -> Probe-to-Gene Mapping
# -----------------------------------------------------------------------

def _detect_gene_col(columns):
    for pat in _GENE_COL_PATTERNS:
        for c in columns:
            if re.match(pat, c, re.IGNORECASE):
                return c
    for c in columns:
        cl = c.lower().replace(' ', '_')
        for kw in _GENE_COL_KEYWORDS:
            if kw in cl and 'spot' not in cl:
                return c
    return None


def download_annotation(gpl_id, dest_dir, gene_col_override=None):
    """
    Download GPL annotation and build probe->gene_symbol mapping.
    Only the annotation junk entries (---, NA) are removed -- NOT expression NaNs.
    """
    if not _HAS_GEOPARSE:
        raise ImportError(
            "GEOparse is required for annotation download.\n"
            "Install with:  pip install GEOparse"
        )

    os.makedirs(dest_dir, exist_ok=True)
    try:
        gpl = GEOparse.get_GEO(geo=gpl_id, destdir=dest_dir, silent=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to download/parse GPL annotation for {gpl_id}: {exc}\n"
            f"Check network connection and that {gpl_id} is a valid GPL ID."
        ) from exc

    if gpl is None:
        raise ValueError(f"GEOparse returned None for {gpl_id}")

    tbl = getattr(gpl, 'table', None)
    if tbl is None or tbl.empty:
        raise ValueError(f"No annotation table found for {gpl_id}")

    avail = tbl.columns.tolist()

    if gene_col_override and gene_col_override in avail:
        gcol = gene_col_override
    else:
        gcol = _detect_gene_col(avail)

    if gcol is None:
        raise ValueError(
            f"Could not auto-detect gene symbol column for {gpl_id}.\n"
            f"Available columns:\n  {avail}\n"
            f"Pass the correct column name via gene_col_override."
        )

    id_col = 'ID'
    if id_col not in tbl.columns:
        cands = [c for c in tbl.columns
                 if c.upper() in ('ID', 'PROBE_ID', 'PROBEID')]
        id_col = cands[0] if cands else tbl.columns[0]

    m = tbl[[id_col, gcol]].copy()
    m.columns = ['probe_id', 'gene_symbol']
    m['gene_symbol'] = m['gene_symbol'].astype(str).str.strip()

    # Clean Affymetrix "GENE1 /// GENE2"
    mask3 = m['gene_symbol'].str.contains('///', na=False)
    if mask3.any():
        m.loc[mask3, 'gene_symbol'] = (
            m.loc[mask3, 'gene_symbol']
            .str.split(r'\s*///\s*').str[0].str.strip()
        )

    # Clean Agilent gene_assignment "NM_xxx // SYMBOL // desc"
    if gcol.lower() == 'gene_assignment':
        mask2 = m['gene_symbol'].str.contains('//', na=False)
        if mask2.any():
            def _agilent_extract(val):
                parts = re.split(r'\s*//\s*', str(val))
                return parts[1].strip() if len(parts) > 1 else parts[0].strip()
            m.loc[mask2, 'gene_symbol'] = (
                m.loc[mask2, 'gene_symbol'].apply(_agilent_extract)
            )

    # Remove ONLY annotation junk -- NOT expression data NaNs
    m = m[m['gene_symbol'].notna() & ~m['gene_symbol'].isin(_NULL_SYMBOLS)]
    m = m.drop_duplicates()

    return {
        'mapping':           m,
        'gene_col':          gcol,
        'available_columns': avail,
        'n_probes':          len(m),
        'n_genes':           m['gene_symbol'].nunique(),
    }


# -----------------------------------------------------------------------
# STEP 5 -- Aggregate Probes -> Genes (NaN-aware Mean)
# -----------------------------------------------------------------------

def aggregate_probes(expression_df, mapping_df):
    """
    Average probes per gene. NaN-aware mean.
    Handles probe ID type mismatches (int vs string, whitespace, etc).
    """
    expr = expression_df.copy()

    # Force probe IDs to clean strings on BOTH sides
    expr.index = expr.index.astype(str).str.strip()
    expr.index.name = 'probe_id'
    expr = expr.reset_index()

    mapping_clean = mapping_df.copy()
    mapping_clean['probe_id'] = mapping_clean['probe_id'].astype(str).str.strip()

    # Try direct merge first
    merged = expr.merge(mapping_clean, on='probe_id', how='inner')

    # If direct merge failed, try normalizing both sides more aggressively
    if merged.empty or merged.drop(columns=['probe_id', 'gene_symbol']).count().sum() == 0:
        # Try: strip trailing .0 from numeric-looking IDs (pandas reads "12345" as 12345.0)
        expr['probe_id'] = expr['probe_id'].str.replace(r'\.0$', '', regex=True)
        mapping_clean['probe_id'] = mapping_clean['probe_id'].str.replace(r'\.0$', '', regex=True)
        merged = expr.merge(mapping_clean, on='probe_id', how='inner')

    if merged.empty:
        expr_samples = expr['probe_id'].head(5).tolist()
        map_samples = mapping_clean['probe_id'].head(5).tolist()
        raise ValueError(
            f"Zero probes matched between expression data and annotation.\n"
            f"Expression probe IDs (first 5): {expr_samples}\n"
            f"Annotation probe IDs (first 5): {map_samples}\n"
            f"Probe IDs may be incompatible."
        )

    # Verify the merged data has actual values (not just NaN)
    data_cols = [c for c in merged.columns if c not in ('probe_id', 'gene_symbol')]
    non_nan = merged[data_cols].count().sum()
    if non_nan == 0:
        raise ValueError(
            f"Probes matched ({len(merged):,} rows) but ALL expression values are NaN.\n"
            f"The series matrix files may not contain expression data."
        )

    merged = merged.drop(columns=['probe_id'])
    result = merged.groupby('gene_symbol').mean()
    return result


# -----------------------------------------------------------------------
# STEP 6 -- Log2 + Quantile Normalization (NaNs PRESERVED)
# -----------------------------------------------------------------------

def normalize_expression(df):
    """
    Normalize expression data matching the GeneVariate standard:
      1) Replace negative values with 0
      2) Log2 transform ONLY values > 50  (element-wise, not log2(x+1) on all)
      3) Quantile normalize the full matrix with qnorm
      4) Replace any remaining negatives with 0
    NaN positions are preserved throughout.
    """
    arr = df.values.copy().astype(np.float64)
    nan_mask = np.isnan(arr)

    # Step 1: Replace negatives with 0
    arr[~nan_mask & (arr < 0)] = 0.0

    # Step 2: Log2 transform ONLY values > 50
    applied_log2 = False
    high_mask = ~nan_mask & (arr > 50)
    if high_mask.any():
        arr[high_mask] = np.log2(arr[high_mask])
        applied_log2 = True

    # Step 3: Quantile normalization on the full matrix
    if _HAS_QNORM:
        # qnorm cannot handle NaN — fill temporarily, normalize, restore NaN
        if nan_mask.any():
            # Fill NaN with column medians
            for j in range(arr.shape[1]):
                col = arr[:, j]
                col_nan = np.isnan(col)
                if col_nan.any():
                    med = np.nanmedian(col)
                    if np.isnan(med):
                        med = 0.0
                    col[col_nan] = med
                    arr[:, j] = col

        # Normalize full matrix (column-wise, matching user's method)
        qn_result = qnorm.quantile_normalize(pd.DataFrame(arr))
        arr = qn_result.values if hasattr(qn_result, 'values') else np.array(qn_result)

        # Restore NaN positions
        arr[nan_mask] = np.nan
    else:
        arr = _qnorm_numpy_arr(arr, nan_mask)

    # Step 4: Replace any remaining negatives with 0
    arr[~np.isnan(arr) & (arr < 0)] = 0.0

    return pd.DataFrame(arr, index=df.index, columns=df.columns), applied_log2


def _qnorm_numpy_arr(X, nan_mask):
    """Pure numpy column-wise quantile normalization. NaN preserved."""
    arr = X.copy()
    n_genes, n_samp = arr.shape

    # Fill NaN with column medians temporarily
    for j in range(n_samp):
        col = arr[:, j]
        nan_col = np.isnan(col)
        if nan_col.any():
            med = np.nanmedian(col)
            if np.isnan(med):
                med = 0.0
            col[nan_col] = med
            arr[:, j] = col

    # Standard column-wise quantile normalization
    sort_idx = np.argsort(arr, axis=0)
    sorted_arr = np.take_along_axis(arr, sort_idx, axis=0)
    rank_means = sorted_arr.mean(axis=1)

    result = np.empty_like(arr)
    for j in range(n_samp):
        ranks = np.argsort(np.argsort(arr[:, j]))
        result[:, j] = rank_means[ranks]

    result[nan_mask] = np.nan
    return result


# -----------------------------------------------------------------------
# STEP 7 -- Build Final Table & Save
# -----------------------------------------------------------------------

def save_genevariate_csv(normalised_df, gpl_id, output_dir, gsm_to_gse):
    """
    Build and save the final GeneVariate CSV.GZ.
    NaN = empty cells in CSV (preserved, never removed).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Transpose: genes x samples -> samples x genes
    t = normalised_df.T.copy()

    # Build GSM column
    gsm_values = [str(idx).strip().upper() for idx in t.index]

    # Build series_id column from tracked GSE mapping
    series_values = [gsm_to_gse.get(gsm, np.nan) for gsm in gsm_values]

    # Insert metadata columns at front
    t.insert(0, 'series_id', series_values)
    t.insert(0, 'GSM', gsm_values)
    t.reset_index(drop=True, inplace=True)

    fname = f"{gpl_id.lower()}_all_samples_normalized_scaled_with_nans.csv.gz"
    fpath = os.path.join(output_dir, fname)
    t.to_csv(fpath, index=False, compression='gzip')

    return fpath, t.shape


# =======================================================================
# GPLDownloader -- MAIN PUBLIC CLASS
# =======================================================================

class GPLDownloader:
    """
    One-call automated GPL download & preprocessing pipeline.

    Output CSV.GZ columns:
        GSM | series_id | GENE1 | GENE2 | GENE3 | ...

    Each row = one sample. NaN values NEVER removed.
    """

    def __init__(self, gds_conn, output_base_dir,
                 max_workers=4, download_timeout=180):
        self.gds_conn         = gds_conn
        self.output_base_dir  = str(output_base_dir)
        self.max_workers      = max_workers
        self.download_timeout = download_timeout

    @staticmethod
    def check_dependencies():
        missing = []
        if not _HAS_GEOPARSE:
            missing.append("GEOparse")
        if missing:
            raise ImportError(
                f"Missing required package(s): {', '.join(missing)}\n"
                f"Install with:  pip install {' '.join(missing)}"
            )

    @staticmethod
    def has_qnorm():
        return _HAS_QNORM

    def get_platform_info(self, gpl_id):
        return query_gpl_info(self.gds_conn, gpl_id.strip().upper())

    def get_annotation_columns(self, gpl_id):
        self.check_dependencies()
        d = os.path.join(self.output_base_dir, gpl_id, "annotation")
        try:
            gpl = GEOparse.get_GEO(geo=gpl_id, destdir=d, silent=True)
            if gpl is None or getattr(gpl, 'table', None) is None:
                return []
            return gpl.table.columns.tolist()
        except Exception as exc:
            logger.warning("Failed to get annotation columns for %s: %s", gpl_id, exc)
            return []

    def run(self, gpl_id, max_gse=0, gene_col_override=None, callback=None):
        """Full pipeline: query -> download -> parse -> annotate -> normalize -> save."""
        self.check_dependencies()
        gpl_id = gpl_id.strip().upper()
        info = query_gpl_info(self.gds_conn, gpl_id)
        return self.run_with_info(info, max_gse=max_gse,
                                 gene_col_override=gene_col_override,
                                 callback=callback)

    def run_with_info(self, info, max_gse=0, gene_col_override=None,
                      callback=None, clear_cache=False):
        """
        Full pipeline using PRE-QUERIED platform info (no SQLite access).
        clear_cache=True deletes previously downloaded raw matrices (use if previous download was corrupt).
        """
        self.check_dependencies()

        def cb(pct, stage, msg):
            if callback:
                callback(pct, stage, msg)

        def _verify(df, step_name):
            """Abort if data has no actual expression values."""
            non_nan = df.count().sum()
            total = df.size
            pct = 100 * non_nan / total if total > 0 else 0
            cb(None, "verify",
               f"[{step_name}] {df.shape[0]:,}x{df.shape[1]:,}, "
               f"non-NaN: {non_nan:,}/{total:,} ({pct:.1f}%)")
            if non_nan == 0:
                raise ValueError(
                    f"PIPELINE ABORT at '{step_name}':\n"
                    f"DataFrame is {df.shape[0]:,} x {df.shape[1]:,} "
                    f"but ALL values are NaN.\n"
                    f"Expression data was lost at this step."
                )

        gpl_id = info['gpl_id']

        cb(5, "query",
           f"Found {info['total_series']} GSE series for {gpl_id} "
           f"({info['organism']}, {info['title']})")

        gse_list = info['gse_list']
        if max_gse and max_gse > 0:
            gse_list = gse_list[:max_gse]
        if not gse_list:
            raise ValueError(f"No GSE series found for {gpl_id}.")

        # 2. Download (clear stale cache from previous failed attempts)
        dl_dir = os.path.join(self.output_base_dir, gpl_id, "raw_matrices")
        if clear_cache and os.path.isdir(dl_dir):
            import shutil
            n_old = len(os.listdir(dl_dir))
            if n_old > 0:
                cb(5, "download",
                   f"Clearing {n_old} cached files from previous download...")
                shutil.rmtree(dl_dir)

        cb(5, "download", f"Downloading {len(gse_list)} series matrices...")
        downloaded, failed = batch_download(
            gse_list, gpl_id, dl_dir,
            max_workers=self.max_workers,
            timeout=self.download_timeout,
            callback=lambda p, m: cb(5 + int((p or 0) * 0.30), "download", m),
        )
        cb(35, "download",
           f"Downloaded {len(downloaded)}/{len(gse_list)} "
           f"({len(failed)} failed)")
        if not downloaded:
            raise ValueError(
                f"Could not download any matrix files for {gpl_id}.\n"
                f"All {len(failed)} GSEs failed."
            )

        # 3. Parse & combine (NaNs preserved, GSE->GSM tracked)
        cb(35, "parse", "Parsing expression matrices...")
        combined, gsm_to_gse = combine_matrices(
            downloaded,
            callback=lambda p, m: cb(35 + int((p or 0) * 0.15), "parse", m),
        )
        _verify(combined, "parse+combine")

        cb(50, "parse",
           f"Combined: {combined.shape[0]:,} probes x "
           f"{combined.shape[1]:,} samples "
           f"(GSE mapping for {len(gsm_to_gse):,} GSMs)")

        # 4. Annotation
        cb(50, "annotate", f"Downloading {gpl_id} annotation...")
        ann_dir = os.path.join(self.output_base_dir, gpl_id, "annotation")
        ann = download_annotation(gpl_id, ann_dir,
                                  gene_col_override=gene_col_override)
        cb(60, "annotate",
           f"{ann['n_probes']:,} probes -> {ann['n_genes']:,} genes "
           f"(col: '{ann['gene_col']}')")

        # Verify probe ID overlap before merge
        expr_probes = set(combined.index.astype(str).str.strip())
        ann_probes = set(ann['mapping']['probe_id'].astype(str).str.strip())
        overlap = expr_probes & ann_probes
        cb(60, "annotate",
           f"Probe ID overlap: {len(overlap):,} / "
           f"{len(expr_probes):,} expression, "
           f"{len(ann_probes):,} annotation")

        if len(overlap) == 0:
            # Show samples to help debug
            raise ValueError(
                f"ZERO probe IDs match between expression and annotation!\n"
                f"Expression probes (sample): {list(expr_probes)[:5]}\n"
                f"Annotation probes (sample): {list(ann_probes)[:5]}\n"
                f"This usually means the platform uses a non-standard format."
            )

        # 5. Aggregate probes -> genes (NaN-aware mean)
        cb(60, "aggregate", "Averaging probes per gene (NaN-aware)...")
        gene_expr = aggregate_probes(combined, ann['mapping'])
        _verify(gene_expr, "aggregate (probe->gene)")

        cb(70, "aggregate",
           f"{gene_expr.shape[0]:,} genes x {gene_expr.shape[1]:,} samples")

        # 6. Normalize: clip negatives -> log2 values>50 -> quantile normalize -> clip negatives
        method = "qnorm" if _HAS_QNORM else "numpy"
        n_samp = gene_expr.shape[1]
        n_genes = gene_expr.shape[0]
        if n_samp > 10000:
            cb(75, "normalize",
               f"Normalizing {n_genes:,} genes x {n_samp:,} samples ({method}) "
               f"- large dataset, this may take several minutes...")
        else:
            cb(75, "normalize",
               f"Normalizing {n_genes:,} genes x {n_samp:,} samples ({method})...")

        normed, applied_log2 = normalize_expression(gene_expr)
        if applied_log2:
            cb(80, "normalize", "Applied log2 to values > 50")
        _verify(normed, "quantile normalize")

        # Report which method actually worked
        non_nan_after = normed.count().sum()
        cb(90, "normalize",
           f"Normalization complete: {non_nan_after:,} non-NaN values preserved")

        # 7. Save: GSM | series_id | GENE1 | GENE2 | ...
        cb(90, "save", "Saving: GSM + series_id + genes...")
        out_dir = os.path.join(self.output_base_dir, gpl_id)
        fpath, shape = save_genevariate_csv(
            normed, gpl_id, out_dir, gsm_to_gse
        )

        n_genes = shape[1] - 2   # subtract GSM and series_id columns
        n_samples = shape[0]

        cb(100, "done",
           f"Done: {n_samples:,} samples x {n_genes:,} genes -> "
           f"{os.path.basename(fpath)}")

        return {
            'filepath':       fpath,
            'gpl_id':         gpl_id,
            'organism':       info['organism'],
            'platform_title': info['title'],
            'technology':     info['technology'],
            'n_samples':      n_samples,
            'n_genes':        n_genes,
            'n_series':       len(downloaded),
            'n_failed':       len(failed),
            'gene_col_used':  ann['gene_col'],
            'log2_applied':   applied_log2,
        }
