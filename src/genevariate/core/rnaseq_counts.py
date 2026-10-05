"""NCBI GEO reprocessed RNA-seq counts, returned RAW, as an ingestion source."""

import gzip
import logging
import os
import re
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NCBI_DOWNLOAD = "https://www.ncbi.nlm.nih.gov/geo/download/"

#: The reprocessed count table for one series.
_RAW_COUNTS_RE = re.compile(r"^GSE\d+_raw_counts_.+_NCBI\.tsv\.gz$", re.I)

#: The GeneID -> symbol table for the species.
_ANNOT_RE = re.compile(r"^[A-Za-z]+\..+\.annot\.tsv\.gz$", re.I)


def list_reprocessed_files(gse, timeout=60, session=None, attempts=3):
    """Filenames NCBI publishes for *gse*; empty means not reprocessed, failure raises."""
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
    """Download one reprocessed file and return its decompressed text."""
    params = {"type": "rnaseq_counts", "format": "file", "file": filename}
    if gse:
        params["acc"] = str(gse).strip().upper()
    get = (session or requests).get
    resp = get(NCBI_DOWNLOAD, params=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.content
    if body[:2] != b"\x1f\x8b":
        raise ValueError(f"{filename} did not come back as gzip "
                         f"(got {len(body):,} bytes of {resp.headers.get('Content-Type')})")
    return gzip.decompress(body).decode("utf-8", "replace")


def _cached(dest_dir, ends_with, starts_with=""):
    """A file already in *dest_dir* whose name brackets match, or None."""
    try:
        names = sorted(os.listdir(dest_dir))
    except OSError:
        return None
    for n in names:
        if n.startswith(starts_with) and n.endswith(ends_with):
            p = os.path.join(dest_dir, n)
            if os.path.getsize(p) > 0:
                return p
    return None


def download_counts(gse, dest_dir, timeout=300, session=None):
    """Cache the raw-count table for *gse* on disk; None means not reprocessed."""
    gse = str(gse).strip().upper()
    os.makedirs(dest_dir, exist_ok=True)

    try:
        files = list_reprocessed_files(gse, session=session)
    except Exception:
        hit = _cached(dest_dir, ".tsv.gz", f"{gse}_raw_counts")
        if hit:
            return hit
        raise
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
    """Cache the species GeneID->symbol table; return its path or None."""
    os.makedirs(dest_dir, exist_ok=True)
    try:
        files = list_reprocessed_files(gse, session=session)
    except Exception:
        hit = _cached(dest_dir, ".annot.tsv.gz")
        if hit:
            return hit
        raise
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
    """Collapse a GeneID x GSM count table onto gene symbols, summing shared symbols."""
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
    """Returns ``{'counts', 'gsm_to_gse', 'reprocessed', 'missing', 'failed'}``."""
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
        if failed and not missing:
            raise ValueError(
                f"NCBI would not answer for any of the {len(gse_list)} series "
                f"({', '.join(failed)}), so whether it has counts for them is "
                f"unknown. Retry when it responds.")
        if failed:
            raise ValueError(
                f"{len(missing)} of {len(gse_list)} series are not "
                f"reprocessed by NCBI; it would not answer for the other "
                f"{len(failed)} ({', '.join(failed)}), so those are unknown "
                f"rather than uncovered. Retry, or load the uncovered ones "
                f"from ARCHS4.")
        raise ValueError(
            f"None of the {len(gse_list)} series have NCBI-reprocessed counts. "
            f"NCBI only reprocesses human and mouse RNA-seq; try ARCHS4.")
    if gene_map is None:
        raise ValueError(
            "Downloaded counts but no GeneID->symbol annotation, so the "
            "matrix cannot be keyed by gene.")

    cb(90, f"Merging {len(frames)} series on GeneID...")
    combined = pd.concat(frames, axis=1, join="outer")

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
