"""Automated GPL Platform Downloader and Preprocessor for GeneVariate."""

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
from scipy.stats import rankdata

try:
    import GEOparse
    _HAS_GEOPARSE = True
except ImportError:
    _HAS_GEOPARSE = False

logger = logging.getLogger(__name__)

# CONSTANTS

GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

_GENE_COL_PATTERNS = [
    r'^UCSC[\s_.]?RefGene[\s_.]?Name$',
    r'^Closest[\s_.]?TSS[\s_.]?gene[\s_.]?name$',
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


# TECHNOLOGY CLASSIFIER

TECH_CATEGORIES = (
    "microarray",
    "bulk-rna-seq",
    "single-cell",
    "methylation",
    "sequencing-other",
    "other",
)

CATEGORY_LABELS = {
    "microarray":       "Microarray",
    "bulk-rna-seq":     "Bulk RNA-seq",
    "single-cell":      "Single-cell",
    "methylation":      "Methylation",
    "sequencing-other": "Sequencing (other)",
    "other":            "Other",
}

# Colors used by the GUI Treeview tagging.
CATEGORY_COLORS = {
    "microarray":       "#E3F2FD",  # light blue
    "bulk-rna-seq":     "#E8F5E9",  # light green
    "single-cell":      "#FFF3E0",  # light orange
    "methylation":      "#F3E5F5",  # light purple
    "sequencing-other": "#FFFDE7",  # light yellow
    "other":            "#FAFAFA",  # light grey
}

_SINGLE_CELL_PATTERNS = [
    r"\bsingle[\s\-]?cell\b",
    r"\bsc[\-_]?rna[\-_]?seq\b",
    r"\bsn[\-_]?rna[\-_]?seq\b",
    r"\bsingle[\s\-]?nucle(us|i)\b",
    r"\b10x\s*genomics\b",
    r"\bchromium\b",
    r"\bdrop[\s\-]?seq\b",
    r"\bsmart[\s\-]?seq2?\b",
    r"\bcel[\s\-]?seq2?\b",
    r"\bmars[\s\-]?seq\b",
    r"\bindrop(s|seq)?\b",
    r"\bbd\s*rhapsody\b",
    r"\bfluidigm\s*c1\b",
    r"\bparse\s*biosciences?\b",
    r"\bsplit[\s\-]?seq\b",
]

_METHYLATION_PATTERNS = [
    r"methylation",
    r"\bbisulfite\b",
    r"\binfinium\b",
    r"\bepic\b",
    r"\b450k?\b",
    r"\b850k?\b",
    r"\bmeth[\s\-]?seq\b",
    r"\brrbs\b",
    r"\bwgbs\b",
]

_BULK_RNASEQ_PATTERNS = [
    r"\brna[\s\-]?seq\b",
    r"\btranscriptom",
    r"\bmrna\b",
    r"\btotal\s+rna\b",
    r"\billumina\s+(hi|nova|next)seq",
    r"\billumina\s+genome\s+analyzer",
    r"\bion\s+torrent\b",
    r"\bab\s+solid\b",
    r"\bbgiseq\b",
    r"\bpacbio\b",
    r"\boxford\s+nanopore\b",
]

_CHIP_ATAC_PATTERNS = [
    r"\bchip[\s\-]?seq\b",
    r"\batac[\s\-]?seq\b",
    r"\bdnase[\s\-]?seq\b",
    r"\bcut\s*&?\s*run\b",
    r"\bcut\s*&?\s*tag\b",
    r"\bhi[\s\-]?c\b",
    r"\bmnase\b",
    r"\bribo[\s\-]?seq\b",
]


def classify_technology(technology, title=""):
    """Map GEOmetadb's (technology, title) pair to one of TECH_CATEGORIES."""
    tech = (technology or "").strip().lower()
    name = (title or "").strip().lower()
    blob = f"{tech} {name}"

    if any(re.search(p, blob) for p in _SINGLE_CELL_PATTERNS):
        return "single-cell"
    if any(re.search(p, blob) for p in _METHYLATION_PATTERNS):
        return "methylation"

    if "sequencing" in tech or tech in ("sage", "mpss"):
        if any(re.search(p, blob) for p in _CHIP_ATAC_PATTERNS):
            return "sequencing-other"
        if any(re.search(p, blob) for p in _BULK_RNASEQ_PATTERNS):
            return "bulk-rna-seq"
        return "sequencing-other"

    if "oligonucleotide" in tech or "spotted" in tech or "cdna" in tech \
            or "array" in tech or "beads" in tech or "bead" in tech \
            or tech in ("in situ oligonucleotide",):
        return "microarray"

    if any(re.search(p, blob) for p in _BULK_RNASEQ_PATTERNS):
        return "bulk-rna-seq"
    if any(re.search(p, blob) for p in _CHIP_ATAC_PATTERNS):
        return "sequencing-other"

    return "other"


def category_label(category):
    return CATEGORY_LABELS.get(category, category)


MEASUREMENT_LABELS = {
    "bulk-rna-seq": "log2 CPM (TMM)",
    "single-cell":  "log2 CPM (pseudobulk)",
    "methylation":  "beta (fraction methylated)",
}


def measurement_label(category):
    """Axis label for the quantity a platform of *category* reports."""
    return MEASUREMENT_LABELS.get(category, "expression")


# How each technology category reaches a gene x sample matrix.
INGESTION_ROUTES = {
    "microarray":       "series-matrix",
    "methylation":      "unsupported",
    "bulk-rna-seq":     "ncbi-counts",
    "single-cell":      "pseudobulk",
    "sequencing-other": "none",
    "other":            "series-matrix",
}

NO_GENE_MATRIX = {"sequencing-other"}


def ingestion_route(category):
    """Which ingestion path a technology category takes."""
    return INGESTION_ROUTES.get(category, "series-matrix")


# STEP 1 -- Query GEOmetadb for Platform Info & GSE List

def query_gpl_info(gds_conn, gpl_id):
    """Returns dict: gpl_id, organism, title, technology, gse_list, total_series."""
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
        "SELECT DISTINCT gse FROM gse_gpl WHERE gpl = ? COLLATE NOCASE "
        "ORDER BY gse",
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


# STEP 2 -- Download Series Matrix Files from GEO FTP

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


# STEP 3 -- Parse Series Matrices (NaNs PRESERVED, GSE->GSM tracked)

def parse_matrix(filepath, gse_id):
    """Returns (probes x GSMs DataFrame, list_of_gsm_ids) or (None, [])."""
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

        # Convert to numeric column by column - track failures
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Verify we got actual expression values
        non_nan = df.count().sum()
        total = df.size
        if total > 0 and non_nan == 0:
            # ALL values coerced to NaN - check what the raw data looked like
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
    """Returns (combined probes x GSMs df, gsm_to_gse_dict)."""
    frames = []
    parsed = []          # (gse_id, [GSM, ...]) in frame order
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
                    parsed.append((gse_id,
                                   [str(g).strip().upper() for g in gsm_ids]))

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

    size = {}
    candidates = {}
    for gse_id, gsms in parsed:
        size[gse_id] = max(size.get(gse_id, 0), len(gsms))
        for g in gsms:
            candidates.setdefault(g, set()).add(gse_id)
    gsm_to_gse = {g: max(c, key=lambda s: (size[s], s))
                  for g, c in candidates.items()}

    n_dup = combined.shape[1] - len(gsm_to_gse)
    if n_dup > 0:
        seen = set()
        keep = []
        for gse_id, gsms in parsed:
            for g in gsms:
                take = gsm_to_gse.get(g) == gse_id and g not in seen
                if take:
                    seen.add(g)
                keep.append(take)
        combined = combined.iloc[:, [i for i, k in enumerate(keep) if k]]
        if callback:
            callback(None, f"Removed {n_dup} duplicate GSM columns")

    return combined, gsm_to_gse


# STEP 4 -- GPL Annotation -> Probe-to-Gene Mapping

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
    """Download GPL annotation and build probe->gene_symbol mapping."""
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
    _as_written = m['gene_symbol'].copy()

    # Clean Affymetrix "GENE1 /// GENE2"
    mask3 = m['gene_symbol'].str.contains('///', na=False)
    if mask3.any():
        m.loc[mask3, 'gene_symbol'] = (
            m.loc[mask3, 'gene_symbol']
            .str.split(r'\s*///\s*').str[0].str.strip()
        )

    mask_semi = m['gene_symbol'].str.contains(';', na=False)
    if mask_semi.any():
        m.loc[mask_semi, 'gene_symbol'] = (
            m.loc[mask_semi, 'gene_symbol']
            .str.split(';').str[0].str.strip()
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

    kept = set(m['gene_symbol'])
    later = set()
    for val in _as_written:
        parts = [p.strip() for p in re.split(r'\s*///\s*|;', str(val))]
        later.update(p for p in parts[1:] if p)
    second_only = later - kept - set(_NULL_SYMBOLS)

    return {
        'mapping':            m,
        'gene_col':           gcol,
        'available_columns':  avail,
        'n_probes':           len(m),
        'n_genes':            m['gene_symbol'].nunique(),
        'n_genes_second_only': len(second_only),
    }


# STEP 5 -- Aggregate Probes -> Genes (NaN-aware Mean)

def aggregate_probes(expression_df, mapping_df):
    """Average probes per gene, NaN-aware mean."""
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


# STEP 6 -- Log2 + Quantile Normalization (NaN-aware, NaNs PRESERVED)

def _needs_log2(finite):
    """Decide whether a matrix of finite values is linear intensity."""
    if finite.size == 0:
        return False
    q_min, q25, q99, q_max = np.percentile(finite, [0.0, 25.0, 99.0, 100.0])
    return bool(q99 > 100.0 or (q_max - q_min > 50.0 and q25 > 0.0))


def _looks_like_counts(arr):
    """True when a matrix is RNA-seq read counts rather than intensities."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    if finite.min() < 0.0 or finite.max() < 1000.0:
        return False
    return bool(np.all(finite == np.floor(finite)))


def tmm_norm_factors(arr, min_shared_genes=100):
    """edgeR's TMM scaling factor per sample of a genes x samples count matrix."""
    n_samples = arr.shape[1]
    if n_samples < 2:
        return np.ones(n_samples)

    shared = np.isfinite(arr).all(axis=1) & (np.nansum(arr, axis=1) > 0)
    if int(shared.sum()) < min_shared_genes:
        return np.ones(n_samples)

    try:
        from rnanorm import TMM
    except ImportError as exc:
        raise RuntimeError(
            "TMM scaling of RNA-seq counts needs the 'rnanorm' package: "
            "pip install rnanorm. It is a required dependency of genevariate; "
            "an install that lacks it was built from an older pyproject."
        ) from exc

    counts = pd.DataFrame(arr[shared].T)
    factors = np.asarray(TMM().fit(counts).get_norm_factors(counts),
                         dtype=np.float64)
    factors[~np.isfinite(factors) | (factors <= 0.0)] = 1.0
    return factors


def normalize_counts(df, min_total_count=10):
    """Returns (log2(CPM+1) DataFrame over TMM effective library size, dropped_gene_count)."""
    arr = df.to_numpy(dtype=np.float64, copy=True)

    totals = np.nansum(arr, axis=1)
    keep = totals >= float(min_total_count)
    n_dropped = int((~keep).sum())
    if not keep.any():
        raise ValueError(
            f"Every gene has fewer than {min_total_count} total counts; "
            f"the matrix carries no usable expression.")
    arr = arr[keep]
    index = df.index[keep]

    lib = np.nansum(arr, axis=0)
    lib[lib <= 0] = np.nan                       # an empty library, not a zero
    eff_lib = lib * tmm_norm_factors(arr)        # edgeR effective library size
    arr = arr / eff_lib * 1e6                    # counts per million
    np.log2(arr + 1.0, out=arr)

    return pd.DataFrame(arr.astype(np.float32),
                        index=index, columns=df.columns), n_dropped


def normalize_expression(df):
    """Returns (log2-if-needed then quantile-normalized DataFrame, applied_log2)."""
    arr = df.to_numpy(dtype=np.float32, copy=True)

    applied_log2 = False
    finite = arr[np.isfinite(arr)]
    if _needs_log2(finite):
        arr[arr <= 0] = np.nan          # below the detection floor
        np.log2(arr, out=arr)
        applied_log2 = True

    _quantile_normalize_inplace(arr)

    return pd.DataFrame(arr, index=df.index, columns=df.columns), applied_log2


def _quantile_normalize_inplace(arr):
    """NaN-aware quantile normalization of a genes x samples matrix, in place."""
    n_genes, n_samples = arr.shape
    if n_genes < 2 or n_samples == 0:
        return arr

    grid = np.linspace(0.0, 1.0, n_genes)

    # Pass 1 -- reference distribution.
    reference = np.zeros(n_genes, dtype=np.float64)
    n_used = 0
    for j in range(n_samples):
        col = arr[:, j]
        finite = np.isfinite(col)
        n_ok = int(finite.sum())
        if n_ok < 2:
            continue
        vals = np.sort(col[finite].astype(np.float64))
        reference += np.interp(grid, np.linspace(0.0, 1.0, n_ok), vals)
        n_used += 1

    if n_used == 0:
        return arr
    reference /= n_used

    # Pass 2 -- map each sample onto the reference through its rank positions.
    for j in range(n_samples):
        col = arr[:, j]
        finite = np.isfinite(col)
        n_ok = int(finite.sum())
        if n_ok < 2:
            continue
        pos = (rankdata(col[finite], method='average') - 1.0) / (n_ok - 1.0)
        col[finite] = np.interp(pos, grid, reference)

    return arr


# STEP 7 -- Build Final Table & Save

#: Filename suffixes.
RAW_SUFFIX = "_all_samples_raw_with_nans.csv.gz"
NORMALIZED_SUFFIX = "_all_samples_normalized_scaled_with_nans.csv.gz"


def raw_csv_path(gpl_id, output_dir):
    return os.path.join(output_dir, f"{gpl_id.lower()}{RAW_SUFFIX}")


def normalized_csv_path(gpl_id, output_dir):
    return os.path.join(output_dir, f"{gpl_id.lower()}{NORMALIZED_SUFFIX}")


def write_atomic(path, write):
    """Call ``write(tmp)`` on a neighbouring path and rename it into place."""
    tmp = f"{path}.part"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path


def save_genevariate_csv(expr_df, gpl_id, output_dir, gsm_to_gse,
                         normalized=True):
    """Build and save a GeneVariate CSV.GZ from a genes x samples matrix."""
    os.makedirs(output_dir, exist_ok=True)

    # Transpose: genes x samples -> samples x genes
    t = expr_df.T.copy()

    # Build GSM column
    gsm_values = [str(idx).strip().upper() for idx in t.index]

    # Build series_id column from tracked GSE mapping
    series_values = [gsm_to_gse.get(gsm, np.nan) for gsm in gsm_values]

    # Insert metadata columns at front
    t.insert(0, 'series_id', series_values)
    t.insert(0, 'GSM', gsm_values)
    t.reset_index(drop=True, inplace=True)

    fpath = (normalized_csv_path(gpl_id, output_dir) if normalized
             else raw_csv_path(gpl_id, output_dir))
    write_atomic(fpath, lambda p: t.to_csv(p, index=False, compression='gzip'))

    return fpath, t.shape


def normalize_platform(gpl_id, output_dir, callback=None, chunksize=200_000,
                       modality="auto"):
    """Returns {'raw_path', 'output_path', 'shape', 'applied_log2', 'counts', 'genes_dropped'}."""
    def cb(pct, stage, msg):
        if callback:
            callback(pct, stage, msg)

    src = raw_csv_path(gpl_id, output_dir)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"No raw matrix for {gpl_id} at {src}. Download the platform "
            f"first -- normalization is a separate step and needs the raw "
            f"values.")

    cb(5, "normalize", f"Reading raw matrix: {os.path.basename(src)}")
    samples = pd.read_csv(src, compression='gzip')

    meta_cols = [c for c in ("GSM", "series_id") if c in samples.columns]
    gsm_to_gse = {}
    if "GSM" in samples.columns and "series_id" in samples.columns:
        gsm_to_gse = dict(zip(samples["GSM"].astype(str),
                              samples["series_id"]))

    gene_cols = [c for c in samples.columns if c not in meta_cols]
    if not gene_cols:
        raise ValueError(f"{src} carries no gene columns to normalize.")

    # genes x samples, which is the orientation normalize_expression expects.
    gene_expr = samples[gene_cols].apply(pd.to_numeric, errors='coerce').T
    gene_expr.columns = (samples["GSM"].astype(str).values
                         if "GSM" in samples.columns else gene_expr.columns)

    cb(30, "normalize",
       f"Normalizing {gene_expr.shape[0]:,} genes x "
       f"{gene_expr.shape[1]:,} samples...")

    is_counts = (modality == "counts") if modality != "auto" else \
        _looks_like_counts(gene_expr.to_numpy(dtype=np.float64, copy=False))

    if is_counts:
        cb(40, "normalize",
           "Matrix is RNA-seq counts - TMM effective library size, "
           "then CPM, then log2(CPM+1)")
        normed, n_dropped = normalize_counts(gene_expr)
        applied_log2 = True
        if n_dropped:
            cb(70, "normalize",
               f"Dropped {n_dropped:,} genes below the total-count floor "
               f"({normed.shape[0]:,} genes kept)")
    else:
        normed, applied_log2 = normalize_expression(gene_expr)
        n_dropped = 0
        if applied_log2:
            cb(70, "normalize",
               "Matrix was linear intensity - log2 applied to the whole matrix")
        else:
            cb(70, "normalize", "Matrix was already log2 - no rescaling applied")

    cb(85, "save", "Writing normalized matrix...")
    fpath, shape = save_genevariate_csv(normed, gpl_id, output_dir,
                                        gsm_to_gse, normalized=True)
    cb(100, "done", f"Normalized matrix saved: {os.path.basename(fpath)}")

    return {"raw_path": src, "output_path": fpath, "shape": shape,
            "applied_log2": applied_log2, "counts": is_counts,
            "genes_dropped": n_dropped}


# STEP 8 -- Platform gene list (for the interactive gene picker)

def list_platform_genes(gpl_id, dest_dir, gene_col_override=None):
    """Returns a dict: {genes, n_genes, gene_col, available_columns}."""
    ann = download_annotation(gpl_id, dest_dir,
                              gene_col_override=gene_col_override)
    genes = sorted({str(g).strip() for g in ann['mapping']['gene_symbol']
                    if str(g).strip()})
    return {
        'genes':             genes,
        'n_genes':           len(genes),
        'gene_col':          ann['gene_col'],
        'available_columns': ann['available_columns'],
    }


def downloaded_series(gpl_id, output_base_dir):
    """Accessions already fetched for a platform, read from its download caches."""
    found = set()
    base = os.path.join(output_base_dir, gpl_id)
    for sub in ("raw_matrices", "ncbi_counts"):
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = re.match(r"(GSE\d+)", name)
            if m:
                found.add(m.group(1))
    return found


def list_platform_series(gds_conn, gpl_id):
    """Returns a dict: {series: [{gse, title, summary, n_samples}], n_series}."""
    plat = pd.read_sql_query(
        "SELECT gpl FROM gpl WHERE gpl = ? COLLATE NOCASE",
        gds_conn, params=[gpl_id])
    if plat.empty:
        raise ValueError(f"Platform {gpl_id} not found in GEOmetadb.")
    gpl_db_value = str(plat.iloc[0]['gpl'])
    df = pd.read_sql_query(
        """
        SELECT g.gse            AS gse,
               COALESCE(m.title, '')   AS title,
               COALESCE(m.summary, '') AS summary
        FROM   gse_gpl g
        LEFT JOIN gse m ON m.gse = g.gse COLLATE NOCASE
        WHERE  g.gpl = ? COLLATE NOCASE
        GROUP BY g.gse
        """,
        gds_conn, params=[gpl_db_value])
    # per-GSE sample counts on this platform
    try:
        cnt = pd.read_sql_query(
            """
            SELECT s.series_id AS gse, COUNT(*) AS n
            FROM   gsm s
            WHERE  s.gpl = ? COLLATE NOCASE
            GROUP BY s.series_id
            """,
            gds_conn, params=[gpl_db_value])
        # series_id may pack several GSEs as 'GSE1,GSE2' -- explode
        counts = {}
        for _gse, _n in zip(cnt['gse'].astype(str), cnt['n']):
            for part in _gse.replace(';', ',').split(','):
                part = part.strip()
                if part:
                    counts[part] = counts.get(part, 0) + int(_n)
    except Exception:
        counts = {}

    series = []
    for gse, title, summary in zip(df['gse'].astype(str),
                                   df['title'].astype(str),
                                   df['summary'].astype(str)):
        desc = title.strip()
        summ = summary.strip()
        if summ:
            desc = f"{desc} - {summ}" if desc else summ
        series.append({
            'gse':       gse,
            'title':     title.strip(),
            'summary':   summ,
            'desc':      desc or "(no description in GEOmetadb)",
            'n_samples': counts.get(gse, 0),
        })
    series.sort(key=lambda r: (-r['n_samples'], r['gse']))
    return {'series': series, 'n_series': len(series)}


# STEP 9 -- GSE / sample metadata harvest

# GEO series-matrix header key -> our short field name.
_SERIES_META_KEYS = {
    "!Series_title":          "title",
    "!Series_summary":        "summary",
    "!Series_overall_design": "design",
}
_SAMPLE_META_KEYS = {
    "!Sample_title":                  "title",
    "!Sample_source_name_ch1":        "source_name",
    "!Sample_characteristics_ch1":    "characteristics",
    "!Sample_treatment_protocol_ch1": "treatment_protocol",
    "!Sample_description":            "description",
}
# Stable display order for the metadata picker / sidecar columns.
SERIES_META_FIELDS = ["title", "summary", "design"]
SAMPLE_META_FIELDS = ["title", "source_name", "characteristics",
                      "treatment_protocol", "description"]


def parse_series_metadata(filepath, gse_id):
    """Returns {'gse', 'series': {..}, 'samples': {GSM: {..}}}."""
    opener = gzip.open if str(filepath).endswith('.gz') else open
    gsms = []
    series = {}
    sample_cols = {v: [] for v in _SAMPLE_META_KEYS.values()}
    try:
        with opener(filepath, 'rt', errors='replace') as fh:
            for line in fh:
                if line.startswith('!series_matrix_table_begin'):
                    break
                if not line.startswith('!'):
                    continue
                parts = [p.strip().strip('"')
                         for p in line.rstrip('\n\r').split('\t')]
                key, vals = parts[0], parts[1:]
                if key == '!Sample_geo_accession':
                    gsms = [v.strip().upper() for v in vals]
                elif key in _SERIES_META_KEYS:
                    field = _SERIES_META_KEYS[key]
                    txt = ' '.join(v for v in vals if v).strip()
                    if txt:
                        series[field] = (series[field] + ' | ' + txt
                                         if series.get(field) else txt)
                elif key in _SAMPLE_META_KEYS:
                    sample_cols[_SAMPLE_META_KEYS[key]].append(vals)
    except Exception as exc:
        logger.warning("Metadata parse error %s: %s", filepath, exc)
        return {'gse': gse_id, 'series': {}, 'samples': {}}

    samples = {}
    for i, gsm in enumerate(gsms):
        rec = {'series_id': gse_id}
        for field, col_lists in sample_cols.items():
            pieces = [col[i] for col in col_lists
                      if i < len(col) and col[i]]
            if pieces:
                rec[field] = ' | '.join(pieces)
        samples[gsm] = rec
    return {'gse': gse_id, 'series': series, 'samples': samples}


def harvest_metadata(downloaded_list, metadata_opts, callback=None):
    """Returns (gse_meta: {gse: {field: val}}, sample_df)."""
    want_series = set((metadata_opts or {}).get('series', []))
    want_sample = set((metadata_opts or {}).get('sample', []))
    gse_meta = {}
    rows = []
    total = len(downloaded_list)
    for i, (gse_id, fpath) in enumerate(downloaded_list):
        meta = parse_series_metadata(fpath, gse_id)
        if want_series and meta['series']:
            gse_meta[gse_id] = {k: v for k, v in meta['series'].items()
                                if k in want_series}
        if want_sample:
            for gsm, rec in meta['samples'].items():
                row = {'GSM': gsm, 'series_id': gse_id}
                for f in SAMPLE_META_FIELDS:
                    if f in want_sample:
                        row[f] = rec.get(f, '')
                rows.append(row)
        if callback and (i % 50 == 0 or i == total - 1):
            callback(None, f"Harvested metadata {i + 1}/{total}")
    sample_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return gse_meta, sample_df


def save_metadata_sidecars(gse_meta, sample_df, gpl_id, output_dir):
    """Write <gpl>_gse_meta.json and <gpl>_sample_meta.csv.gz. Returns paths."""
    import json
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    if gse_meta:
        p = os.path.join(output_dir, f"{gpl_id.lower()}_gse_meta.json")

        def _dump(target):
            with open(target, 'w', encoding='utf-8') as fh:
                json.dump(gse_meta, fh, indent=2, ensure_ascii=False)

        paths['gse_meta'] = write_atomic(p, _dump)
    if sample_df is not None and not sample_df.empty:
        p = os.path.join(output_dir, f"{gpl_id.lower()}_sample_meta.csv.gz")
        paths['sample_meta'] = write_atomic(
            p, lambda t: sample_df.to_csv(t, index=False, compression='gzip'))
    return paths


# GPLDownloader -- MAIN PUBLIC CLASS

class GPLDownloader:
    """Automated GPL download and probe->gene mapping pipeline, RAW matrix only."""

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

    def get_platform_info(self, gpl_id):
        return query_gpl_info(self.gds_conn, gpl_id.strip().upper())

    def run(self, gpl_id, max_gse=0, gene_col_override=None, callback=None):
        """Download pipeline: query -> download -> parse -> annotate -> save raw."""
        self.check_dependencies()
        gpl_id = gpl_id.strip().upper()
        info = query_gpl_info(self.gds_conn, gpl_id)
        return self.run_with_info(info, max_gse=max_gse,
                                 gene_col_override=gene_col_override,
                                 callback=callback)

    def run_with_info(self, info, max_gse=0, gene_col_override=None,
                      callback=None, clear_cache=False,
                      gene_whitelist=None, metadata_opts=None,
                      gse_whitelist=None):
        """Download pipeline using PRE-QUERIED platform info (no SQLite access)."""
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
        if gse_whitelist:
            # user picked specific experiments
            want = {str(g).strip().upper() for g in gse_whitelist}
            gse_list = [g for g in gse_list if str(g).strip().upper() in want]
            cb(5, "query",
               f"Selected {len(gse_list)} of {info['total_series']} experiments")
        elif max_gse and max_gse > 0:
            gse_list = gse_list[:max_gse]
        if not gse_list:
            raise ValueError(f"No GSE series found for {gpl_id}.")

        route = ingestion_route(
            classify_technology(info.get('technology', ''),
                                info.get('title', '')))
        if route == "unsupported":
            raise ValueError(
                f"{gpl_id} ({info.get('title', '')}) is a methylation "
                f"platform. GeneVariate analyses expression values; "
                f"methylation beta values are bounded proportions, so the "
                f"normalisation and statistics here do not apply to them.")
        if route == "ncbi-counts":
            return self._run_ncbi_counts(info, gse_list, cb,
                                         gene_whitelist=gene_whitelist,
                                         metadata_opts=metadata_opts)

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

        # 3b. Optional GSE/sample metadata harvest (phase-2 label extraction).
        gse_meta, sample_meta_df = {}, None
        want_meta = bool(metadata_opts and (metadata_opts.get("series")
                                            or metadata_opts.get("sample")))
        if want_meta:
            cb(50, "metadata", "Harvesting GSE / sample metadata from headers...")
            gse_meta, sample_meta_df = harvest_metadata(
                downloaded, metadata_opts,
                callback=lambda p, m: cb(50, "metadata", m))
            n_smp = 0 if sample_meta_df is None else sample_meta_df.shape[0]
            cb(50, "metadata",
               f"Metadata: {len(gse_meta):,} GSE records, {n_smp:,} sample rows")

        # 4. Annotation
        ann_dir = os.path.join(self.output_base_dir, gpl_id, "annotation")
        _have = os.path.exists(os.path.join(ann_dir, f"{gpl_id}.txt"))
        cb(50, "annotate",
           f"{'Reading cached' if _have else 'Downloading'} {gpl_id} "
           f"annotation...")
        ann = download_annotation(gpl_id, ann_dir,
                                  gene_col_override=gene_col_override)
        _msg = (f"{ann['n_probes']:,} probes -> {ann['n_genes']:,} genes "
                f"(col: '{ann['gene_col']}')")
        if ann['n_genes_second_only']:
            _msg += (f"; {ann['n_genes_second_only']:,} further symbols are "
                     f"named only alongside another on a shared probe and have "
                     f"no probe of their own, so this platform cannot report "
                     f"them")
        cb(60, "annotate", _msg)

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

        # 6. No normalization here; normalize_platform() does that on request.
        cb(75, "save",
           f"Raw matrix ready: {gene_expr.shape[0]:,} genes x "
           f"{gene_expr.shape[1]:,} samples (not normalized)")

        # 6b. Optional gene subset.
        n_genes_total = int(gene_expr.shape[0])
        if gene_whitelist:
            wl = {str(g).strip().upper() for g in gene_whitelist
                  if str(g).strip()}
            keep = [g for g in gene_expr.index
                    if str(g).strip().upper() in wl]
            gene_expr = gene_expr.loc[keep]
            cb(85, "save",
               f"Gene subset: kept {gene_expr.shape[0]:,} of "
               f"{n_genes_total:,} genes")
            if gene_expr.shape[0] == 0:
                raise ValueError(
                    f"Gene whitelist matched 0 of {n_genes_total:,} genes on "
                    f"{gpl_id}. Check the gene symbols against this platform.")

        # 7. Save raw: GSM | series_id | GENE1 | GENE2 | ...
        cb(90, "save", "Saving raw matrix: GSM + series_id + genes...")
        out_dir = os.path.join(self.output_base_dir, gpl_id)
        fpath, shape = save_genevariate_csv(
            gene_expr, gpl_id, out_dir, gsm_to_gse, normalized=False
        )

        # 7b. Write metadata sidecars next to the expression CSV.
        meta_paths = {}
        if want_meta:
            meta_paths = save_metadata_sidecars(
                gse_meta, sample_meta_df, gpl_id, out_dir)
            if meta_paths:
                cb(95, "save",
                   f"Wrote metadata: {', '.join(os.path.basename(p) for p in meta_paths.values())}")

        n_genes = shape[1] - 2   # subtract GSM and series_id columns
        n_samples = shape[0]

        cb(100, "done",
           f"Done: {n_samples:,} samples x {n_genes:,} genes -> "
           f"{os.path.basename(fpath)} (raw - run 'Normalize Platform' next)")

        return {
            'filepath':       fpath,
            'raw_path':       fpath,
            'gpl_id':         gpl_id,
            'organism':       info['organism'],
            'platform_title': info['title'],
            'technology':     info['technology'],
            'n_samples':      n_samples,
            'n_genes':        n_genes,
            'n_series':       len(downloaded),
            'n_failed':       len(failed),
            'gene_col_used':  ann['gene_col'],
            'normalized':     False,
            'n_genes_total':  n_genes_total,
            'gene_subset':    bool(gene_whitelist),
            'metadata_files': meta_paths,
            'n_gse_meta':     len(gse_meta),
            'n_sample_meta':  (0 if sample_meta_df is None
                               else int(sample_meta_df.shape[0])),
        }

    def _run_ncbi_counts(self, info, gse_list, cb,
                         gene_whitelist=None, metadata_opts=None):
        """Build an RNA-seq platform matrix from NCBI's reprocessed counts."""
        from genevariate.core import rnaseq_counts

        gpl_id = info['gpl_id']
        out_dir = os.path.join(self.output_base_dir, gpl_id)
        counts_dir = os.path.join(out_dir, "ncbi_counts")

        cb(10, "download",
           f"RNA-seq platform: fetching NCBI reprocessed counts for "
           f"{len(gse_list)} series...")
        built = rnaseq_counts.build_counts_matrix(
            gse_list, counts_dir,
            callback=lambda p, m: cb(10 + int((p or 0) * 0.60), "download", m),
            timeout=self.download_timeout,
        )
        gene_expr = built['counts']
        gsm_to_gse = built['gsm_to_gse']

        if built['missing']:
            cb(70, "download",
               f"{len(built['missing'])} series have no NCBI counts "
               f"(not human/mouse RNA-seq, or not yet reprocessed) - "
               f"load those via ARCHS4")
        if built.get('failed'):
            cb(70, "download",
               f"{len(built['failed'])} series could not be checked - NCBI "
               f"did not answer. Their coverage is unknown, not absent; "
               f"re-run to include them.")

        gse_meta, sample_meta_df, meta_paths = {}, None, {}
        want_meta = bool(metadata_opts and (metadata_opts.get("series")
                                            or metadata_opts.get("sample")))
        if want_meta:
            cb(72, "metadata",
               f"Downloading series matrices of {len(built['reprocessed'])} "
               f"series for sample metadata...")
            dl_dir = os.path.join(out_dir, "raw_matrices")
            downloaded, _failed = batch_download(
                built['reprocessed'], gpl_id, dl_dir,
                max_workers=self.max_workers,
                timeout=self.download_timeout,
                callback=lambda p, m: cb(72 + int((p or 0) * 0.10),
                                         "metadata", m),
            )
            gse_meta, sample_meta_df = harvest_metadata(
                downloaded, metadata_opts,
                callback=lambda p, m: cb(82, "metadata", m))
            n_smp = 0 if sample_meta_df is None else sample_meta_df.shape[0]
            _msg = (f"Metadata: {len(gse_meta):,} GSE records, "
                    f"{n_smp:,} sample rows")
            if sample_meta_df is not None and not sample_meta_df.empty:
                _have = set(sample_meta_df["GSM"].astype(str))
                _with = sum(1 for g in gene_expr.columns if str(g) in _have)
                _n = int(gene_expr.shape[1])
                if _with < _n:
                    _msg += (f"; {_with:,} of the {_n:,} samples with counts "
                             f"have metadata, the other {_n - _with:,} have "
                             f"none and cannot be labelled")
            cb(85, "metadata", _msg)

        n_genes_total = int(gene_expr.shape[0])
        if gene_whitelist:
            wl = {str(g).strip().upper() for g in gene_whitelist
                  if str(g).strip()}
            keep = [g for g in gene_expr.index
                    if str(g).strip().upper() in wl]
            gene_expr = gene_expr.loc[keep]
            cb(87, "save",
               f"Gene subset: kept {gene_expr.shape[0]:,} of "
               f"{n_genes_total:,} genes")
            if gene_expr.shape[0] == 0:
                raise ValueError(
                    f"Gene whitelist matched 0 of {n_genes_total:,} genes on "
                    f"{gpl_id}. Check the gene symbols against this platform.")

        cb(90, "save", "Saving raw counts: GSM + series_id + genes...")
        fpath, shape = save_genevariate_csv(
            gene_expr, gpl_id, out_dir, gsm_to_gse, normalized=False)

        if want_meta:
            meta_paths = save_metadata_sidecars(
                gse_meta, sample_meta_df, gpl_id, out_dir)

        n_genes = shape[1] - 2
        n_samples = shape[0]
        cb(100, "done",
           f"Done: {n_samples:,} samples x {n_genes:,} genes of RAW COUNTS -> "
           f"{os.path.basename(fpath)} (run 'Normalize Platform' next)")

        return {
            'filepath':       fpath,
            'raw_path':       fpath,
            'gpl_id':         gpl_id,
            'organism':       info['organism'],
            'platform_title': info['title'],
            'technology':     info['technology'],
            'source':         'ncbi-counts',
            'n_samples':      n_samples,
            'n_genes':        n_genes,
            'n_series':       len(built['reprocessed']),
            'n_failed':       len(built['missing']),
            'missing_gse':    built['missing'],
            'unanswered_gse': built.get('failed') or [],
            'gene_col_used':  'NCBI GeneID -> Symbol',
            'normalized':     False,
            'n_genes_total':  n_genes_total,
            'gene_subset':    bool(gene_whitelist),
            'metadata_files': meta_paths,
            'n_gse_meta':     len(gse_meta),
            'n_sample_meta':  (0 if sample_meta_df is None
                               else int(sample_meta_df.shape[0])),
        }
