"""
genevariate/core/rnaseq_counts.py
=================================
NCBI GEO reprocessed RNA-seq counts as a GeneVariate ingestion source.

GEO series matrices are an array-era format: for RNA-seq they carry the sample
metadata but almost never the expression values, which the submitters leave in
per-series supplementary files of no fixed shape. NCBI solves this by
re-aligning the raw reads of human and mouse RNA-seq series itself and
publishing one uniform count table per series, keyed by Entrez GeneID. Those
tables are directly comparable across series, which is exactly what the rest of
GeneVariate assumes about a platform matrix.

This module turns a list of GSEs into the same genes x samples frame the
microarray path produces, so everything downstream -- label enrichment,
overdispersion, the pooled comparison -- runs unchanged.

Counts are returned RAW. Library-size correction belongs to the separate
normalization step, the same way the array path keeps download and
normalization apart.
"""

import gzip
import logging
import os
import re
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NCBI_DOWNLOAD = "https://www.ncbi.nlm.nih.gov/geo/download/"

#: The reprocessed count table for one series. The genome build is part of the
#: name (GRCh38.p13 today, something else tomorrow), so the name is discovered
#: from GEO rather than assembled here.
_RAW_COUNTS_RE = re.compile(r"^GSE\d+_raw_counts_.+_NCBI\.tsv\.gz$", re.I)

#: The GeneID -> symbol table for the species. Shared by every series of that
#: species and served WITHOUT an acc= parameter, unlike the count tables.
_ANNOT_RE = re.compile(r"^[A-Za-z]+\..+\.annot\.tsv\.gz$", re.I)


def list_reprocessed_files(gse, timeout=60, session=None, attempts=3):
    """Filenames NCBI publishes for *gse* under its RNA-seq reprocessing.

    An empty list means the series was never reprocessed -- it is not human or
    mouse RNA-seq, or NCBI has not gotten to it. That is the signal to fall
    back to ARCHS4, and it costs one small HTML fetch to learn.

    A failed lookup is NOT an empty list. NCBI serves this from a CGI endpoint
    that drops connections under load, and returning [] for that would report
    a perfectly well reprocessed series as absent -- silently shrinking the
    matrix and overstating how little NCBI has covered. Transient failures are
    retried; a persistent one raises.
    """
    get = (session or requests).get
    last = None
    for i in range(max(1, attempts)):
        try:
            resp = get(NCBI_DOWNLOAD, params={"acc": str(gse).strip().upper()},
                       timeout=timeout)
            resp.raise_for_status()
            return sorted(set(re.findall(r"file=([A-Za-z0-9._%-]+)",
                                         resp.text)))
        except Exception as exc:
            last = exc
            logger.warning("Listing reprocessed files for %s failed "
                           "(attempt %d/%d): %s", gse, i + 1, attempts, exc)
            time.sleep(2 ** i)
    raise RuntimeError(f"Could not ask NCBI what it has for {gse}: {last}")


def counts_filename(files):
    """The raw-count table among *files*, or None."""
    for name in files:
        if _RAW_COUNTS_RE.match(name):
            return name
    return None


def annotation_filename(files):
    """The species GeneID->symbol table among *files*, or None."""
    for name in files:
        if _ANNOT_RE.match(name):
            return name
    return None


def _fetch(filename, gse=None, timeout=300, session=None):
    """Download one reprocessed file and return its decompressed text.

    The count tables are addressed by series; the shared species annotation is
    not, and passing acc= for it returns a 404 HTML page.
    """
    params = {"type": "rnaseq_counts", "format": "file", "file": filename}
    if gse:
        params["acc"] = str(gse).strip().upper()
    get = (session or requests).get
    resp = get(NCBI_DOWNLOAD, params=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.content
    # A missing file is answered with an HTML error page and HTTP 200-ish
    # framing, so trust the gzip magic rather than the status line.
    if body[:2] != b"\x1f\x8b":
        raise ValueError(f"{filename} did not come back as gzip "
                         f"(got {len(body):,} bytes of {resp.headers.get('Content-Type')})")
    return gzip.decompress(body).decode("utf-8", "replace")


def download_counts(gse, dest_dir, timeout=300, session=None):
    """Cache the raw-count table for *gse* on disk; return its path or None.

    None means "not reprocessed" and is an ordinary outcome, not an error.
    """
    gse = str(gse).strip().upper()
    os.makedirs(dest_dir, exist_ok=True)

    files = list_reprocessed_files(gse, session=session)
    name = counts_filename(files)
    if not name:
        return None

    path = os.path.join(dest_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    text = _fetch(name, gse=gse, timeout=timeout, session=session)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def download_gene_annotation(gse, dest_dir, timeout=300, session=None):
    """Cache the species GeneID->symbol table; return its path or None.

    Which species table applies is a property of the series, so it is looked up
    through *gse*, but the file itself is shared and cached once per species.
    """
    os.makedirs(dest_dir, exist_ok=True)
    files = list_reprocessed_files(gse, session=session)
    name = annotation_filename(files)
    if not name:
        return None

    path = os.path.join(dest_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    text = _fetch(name, gse=None, timeout=timeout, session=session)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load_counts(path):
    """Read a reprocessed count table as GeneID x GSM integers."""
    df = pd.read_csv(path, sep="\t", index_col=0, compression="infer")
    df.index = df.index.astype(str).str.strip()
    df.index.name = "GeneID"
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def load_gene_map(path):
    """Read the annotation table as a GeneID -> gene symbol Series."""
    ann = pd.read_csv(path, sep="\t", usecols=["GeneID", "Symbol"],
                      dtype=str, compression="infer")
    ann["GeneID"] = ann["GeneID"].astype(str).str.strip()
    ann["Symbol"] = ann["Symbol"].astype(str).str.strip()
    ann = ann[ann["Symbol"].notna() & (ann["Symbol"] != "")
              & (ann["Symbol"].str.lower() != "nan")]
    return ann.drop_duplicates("GeneID").set_index("GeneID")["Symbol"]


def genes_from_counts(counts, gene_map):
    """Collapse a GeneID x GSM count table onto gene symbols.

    Counts from two GeneIDs that share a symbol are SUMMED, not averaged: a
    count is a number of reads, and reads assigned to either locus are reads
    for that gene. Averaging would silently halve the gene's depth. This is why
    the array path's NaN-aware mean is not reused here.

    A sample with no observed value for any of a symbol's GeneIDs stays NaN.
    Series are merged on GeneID with an outer join, so a series whose table
    does not carry a GeneID leaves a hole there, and a hole is not a count of
    zero: zero means the gene was looked for and no read was assigned to it,
    NaN means it was never looked for. Summing a hole to zero would invent a
    measurement, put it in the plateau that TMM and the distribution
    classifier read as "gene off", and report samples as switched off that
    were only never sequenced for that gene.
    """
    symbols = counts.index.map(gene_map)
    known = symbols.notna()
    if not known.any():
        raise ValueError(
            "No GeneID in the count table matched the NCBI annotation "
            f"(table starts with {list(counts.index[:5])})")
    out = counts[known].groupby(symbols[known].values, sort=True).sum(min_count=1)
    out.index.name = "gene_symbol"
    return out


def build_counts_matrix(gse_list, dest_dir, callback=None, timeout=300):
    """Assemble one genes x samples RAW count matrix from many GEO series.

    Returns ``{'counts', 'gsm_to_gse', 'reprocessed', 'missing', 'failed'}``.
    ``missing`` lists the GSEs NCBI has not reprocessed, so the caller can send
    those to ARCHS4. ``failed`` lists the ones NCBI would not answer for, which
    is a different thing entirely: those may well have counts and are worth
    retrying, so they are kept apart rather than reported as uncovered.

    Series are fetched one at a time on purpose. NCBI serves these from a CGI
    endpoint, and hammering it in parallel gets the whole run throttled, which
    costs far more than the serial fetches save.
    """
    def cb(pct, msg):
        if callback:
            callback(pct, msg)

    session = requests.Session()
    frames, gsm_to_gse, reprocessed, missing, failed = [], {}, [], [], []
    gene_map = None

    total = max(1, len(gse_list))
    for i, gse in enumerate(gse_list):
        gse = str(gse).strip().upper()
        pct = int(100 * i / total)
        try:
            path = download_counts(gse, dest_dir, timeout=timeout,
                                   session=session)
        except Exception as exc:
            logger.warning("Counts download failed for %s: %s", gse, exc)
            failed.append(gse)
            cb(pct, f"{gse}: NCBI would not answer ({exc}) - retry later, "
                    f"this is not a statement about coverage")
            continue

        if path is None:
            missing.append(gse)
            cb(pct, f"{gse}: not reprocessed by NCBI")
            continue

        if gene_map is None:
            try:
                ann_path = download_gene_annotation(gse, dest_dir,
                                                    timeout=timeout,
                                                    session=session)
                if ann_path:
                    gene_map = load_gene_map(ann_path)
                    cb(pct, f"Gene annotation: {len(gene_map):,} GeneIDs "
                            f"({os.path.basename(ann_path)})")
            except Exception as exc:
                logger.warning("Gene annotation failed via %s: %s", gse, exc)

        df = load_counts(path)
        frames.append(df)
        reprocessed.append(gse)
        cb(pct, f"{gse}: {df.shape[1]:,} samples x {df.shape[0]:,} GeneIDs")

    if not frames:
        raise ValueError(
            f"None of the {len(gse_list)} series have NCBI-reprocessed counts. "
            f"NCBI only reprocesses human and mouse RNA-seq; try ARCHS4.")
    if gene_map is None:
        raise ValueError(
            "Downloaded counts but no GeneID->symbol annotation, so the "
            "matrix cannot be keyed by gene.")

    cb(90, f"Merging {len(frames)} series on GeneID...")
    combined = pd.concat(frames, axis=1, join="outer")

    # One sample can appear in several series, most often because GEO
    # republishes a subseries inside a superseries. Charging it to whichever
    # series happened to be iterated first made series_id depend on the order
    # of gse_list, and series_id is the cluster variable every design-effect,
    # cluster bootstrap and grouped cross-validation is grouped on. Worse, the
    # study was taken last-writer-wins while the column was kept first-wins, so
    # the counts and the study recorded against them could come from different
    # series. This is the resolution the microarray path already uses: the
    # series carrying more samples wins, ties to the higher accession, and the
    # column kept is the winner's.
    parsed = [(g, list(f.columns)) for g, f in zip(reprocessed, frames)]
    size: dict = {}
    candidates: dict = {}
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
        cb(92, f"Removed {n_dup} duplicate GSM columns")

    genes = genes_from_counts(combined, gene_map)
    cb(100, f"Counts matrix: {genes.shape[0]:,} genes x "
            f"{genes.shape[1]:,} samples "
            f"({len(reprocessed)} series, {len(missing)} not reprocessed, "
            f"{len(failed)} unanswered)")

    return {
        "counts":      genes,
        "gsm_to_gse":  gsm_to_gse,
        "reprocessed": reprocessed,
        "missing":     missing,
        "failed":      failed,
    }
