"""The entity links an extracted label file carries, and what they mean."""

from __future__ import annotations

import re

import pandas as pd

#: Fields the normalization pass resolves. Sex and Age never enter it.
ENTITY_FIELDS = ("Tissue", "Condition", "Treatment")

CELL_LINE_FIELD = "Tissue"

SPAN_SEP = "; "

#: Stages that carry accessions, deepest first.
LINKED_STAGES = ("final_", "phase2_")

#: Stages that carry a value but no accession.
VERBATIM_STAGES = ("phase1b_", "phase1_")

MESH = "MeSH"
CELLOSAURUS = "Cellosaurus"
LOCAL = "Local (OOV)"
UNLINKED = "Unlinked"

KIND_TISSUE = "Tissue"
KIND_CELL_LINE = "Cell line"
KIND_MIXED = "Mixed"
KIND_UNRESOLVED = "Unresolved"

KIND_SUFFIX = "_kind"

_CVCL = re.compile(r"^CVCL_\w+$", re.I)
_MINTED = re.compile(r"^(?:ART|OOV)-[A-Z]-[0-9A-F]+$", re.I)
_MESH = re.compile(r"^[A-Z]\d+$")
_NOT_SPECIFIED = {"", "not specified", "nan", "none", "na", "n/a"}


def _text(value) -> str:
    """One cell as stripped text, with a missing value reading as blank."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):       # arrays and other non-scalars
        pass
    return str(value).strip()


def namespace(accession) -> str:
    """Which vocabulary *accession* belongs to, or ``""`` when it is blank."""
    a = _text(accession)
    if not a:
        return ""
    if _CVCL.match(a):
        return CELLOSAURUS
    if _MINTED.match(a):
        return LOCAL
    if _MESH.match(a):
        return MESH
    return UNLINKED


def spans(value) -> list:
    """The separate concepts inside one label cell, in extraction order."""
    v = _text(value)
    if not v or v.lower() in _NOT_SPECIFIED:
        return []
    return [p.strip() for p in v.split(";")]


def _aligned(value, ids) -> list:
    """Pair each span of *value* with the span of *ids* in the same position."""
    parts = spans(value)
    if not parts:
        return []
    acc = [p.strip() for p in _text(ids).split(";")]
    acc += [""] * (len(parts) - len(acc))
    return list(zip(parts, acc[:len(parts)]))


_SUFFIXES = (("id", "_id"), ("mesh_id", "_mesh_id"), ("cell_id", "_cell_id"),
             ("stage_col", "_stage"), ("curated", "_curated"))


def field_columns(columns, field: str) -> dict:
    """The column names holding *field*'s value and its accessions."""
    have = {str(c) for c in columns}
    value, stage = "", ""
    for prefix in LINKED_STAGES + VERBATIM_STAGES:
        if f"{prefix}{field}" in have:
            value, stage = f"{prefix}{field}", prefix.rstrip("_")
            break
    if not value and field in have:
        value = field
    if not value:
        return {}

    out = {"stage": stage, "value": value}
    if stage in ("phase1", "phase1b"):
        return out
    prefixes = [f"{stage}_" if stage else "", *LINKED_STAGES]
    for key, suffix in _SUFFIXES:
        for prefix in prefixes:
            name = f"{prefix}{field}{suffix}"
            if name in have:
                out[key] = name
                break
    return out


def has_entity_links(df) -> bool:
    """True when *df* carries at least one resolved accession column."""
    if df is None or getattr(df, "empty", True):
        return False
    return any("id" in field_columns(df.columns, f) for f in ENTITY_FIELDS)


def linked_stage(df) -> str:
    """Which pass the entity links in *df* came from, or ``""`` if none."""
    if df is None or getattr(df, "empty", True):
        return ""
    for f in ENTITY_FIELDS:
        cols = field_columns(df.columns, f)
        if "id" in cols:
            return cols["stage"]
    return ""


def entity_table(df) -> pd.DataFrame:
    """One row per distinct label value, with the entity it resolved to."""
    empty = pd.DataFrame(columns=["Field", "Value", "Accession", "Source",
                                  "Stage", "n"])
    if df is None or getattr(df, "empty", True):
        return empty

    rows = {}
    for field in ENTITY_FIELDS:
        cols = field_columns(df.columns, field)
        if "id" not in cols:
            continue          # verbatim, or no normalization pass ever ran
        values = df[cols["value"]]
        ids = df[cols["id"]]
        cells = (df[cols["cell_id"]] if "cell_id" in cols
                 else pd.Series("", index=df.index))
        stages = (df[cols["stage_col"]] if "stage_col" in cols
                  else pd.Series("", index=df.index))
        for value, acc, cell, stage in zip(values, ids, cells, stages):
            cell_by_span = [p.strip() for p in _text(cell).split(";")]
            stage_by_span = [p.strip() for p in _text(stage).split(";")]
            for i, (part, ident) in enumerate(_aligned(value, acc)):
                if not part:
                    continue
                own_cell = (cell_by_span[i] if i < len(cell_by_span) else "")
                src = namespace(ident)
                if own_cell:
                    src, ident = CELLOSAURUS, ident or own_cell
                key = (field, part, ident)
                rec = rows.get(key)
                if rec is None:
                    rows[key] = rec = {
                        "Field": field, "Value": part,
                        "Accession": ident, "Source": src or UNLINKED,
                        "Stage": (stage_by_span[i]
                                  if i < len(stage_by_span) else ""),
                        "n": 0}
                rec["n"] += 1

    if not rows:
        return empty
    out = pd.DataFrame(list(rows.values()))
    return out.sort_values(["Field", "n", "Value"],
                           ascending=[True, False, True]).reset_index(drop=True)


def kind_series(df, field: str = CELL_LINE_FIELD) -> pd.Series:
    """Per sample: is *field* a tissue, a catalogued cell line, or neither."""
    cols = field_columns(df.columns, field)
    if not cols or "id" not in cols:
        return pd.Series("", index=df.index, dtype=object)

    ids = df[cols["id"]]
    cells = (df[cols["cell_id"]] if "cell_id" in cols
             else pd.Series("", index=df.index))
    out = []
    for value, acc, cell in zip(df[cols["value"]], ids, cells):
        cell_by_span = ([] if pd.isna(cell)
                        else [p.strip() for p in str(cell).split(";")])
        n_cell = n_other = n_linked = 0
        pairs = _aligned(value, acc)
        for i, (part, ident) in enumerate(pairs):
            if not part:
                continue
            own_cell = cell_by_span[i] if i < len(cell_by_span) else ""
            src = CELLOSAURUS if own_cell else namespace(ident)
            if src == CELLOSAURUS:
                n_cell += 1
                n_linked += 1
            elif src == MESH:
                n_other += 1
                n_linked += 1
        if not pairs:
            out.append("")
        elif n_cell and n_other:
            out.append(KIND_MIXED)
        elif n_cell:
            out.append(KIND_CELL_LINE)
        elif n_linked:
            out.append(KIND_TISSUE)
        else:
            out.append(KIND_UNRESOLVED)
    return pd.Series(out, index=df.index, dtype=object)


def kind_column_name(field: str = CELL_LINE_FIELD) -> str:
    """Name of the derived column for *field*."""
    return f"{field}{KIND_SUFFIX}"


def add_kind_columns(df):
    """Add the derived kind column in place, when the links to derive it exist."""
    if df is None or getattr(df, "empty", True):
        return df
    name = kind_column_name()
    if name in df.columns:
        return df
    cols = field_columns(df.columns, CELL_LINE_FIELD)
    if "id" not in cols:
        return df
    kinds = kind_series(df)
    if kinds.replace("", pd.NA).dropna().nunique() < 1:
        return df
    df[name] = kinds
    return df


#: Stage prefixes the extractor writes in front of a field name.
LABEL_STAGE_PREFIXES = LINKED_STAGES + VERBATIM_STAGES + ("Classified_",)


def semantic_label_columns(columns) -> list:
    """The label-value columns among *columns*, one per extracted field."""
    from genevariate.core.geo_extract_driver import ALL_FIELDS

    wanted = {f.lower(): f for f in ALL_FIELDS}
    wanted.update({f"{f.lower()}{KIND_SUFFIX}": f for f in ENTITY_FIELDS})
    best = {}
    for col in columns:
        base = col
        rank = len(LABEL_STAGE_PREFIXES)
        for i, prefix in enumerate(LABEL_STAGE_PREFIXES):
            if col.startswith(prefix):
                base, rank = col[len(prefix):], i
                break
        key = base.lower()
        if key not in wanted:
            continue          # _id / _stage / gsm / gse / gpl / anything else
        if key not in best or rank < best[key][0]:
            best[key] = (rank, col)
    return [best[k][1] for k in wanted if k in best]


def cell_line_values(df, field: str = CELL_LINE_FIELD) -> set:
    """The values of *field* that are catalogued cell lines, for marking them."""
    tbl = entity_table(df)
    if tbl.empty:
        return set()
    hit = tbl[(tbl["Field"] == field) & (tbl["Source"] == CELLOSAURUS)]
    return set(hit["Value"].astype(str))


ID_LIKE_COLS = {'gsm', 'gse', 'gpl', 'series_id', 'sample', 'sample_id',
                'geo_accession', 'platform_id'}


def is_identifier_column(name) -> bool:
    """True for a column that names which sample, study or subject a row is."""
    s = str(name).strip().lower()
    return s in ID_LIKE_COLS or s.endswith('_id')


def label_value_columns(labels) -> list:
    """The testable label columns of a label file, extractor-made or not."""
    def _nunique(col) -> int:
        """Distinct values in a column, whatever the frame does with its name."""
        v = labels[col]
        if getattr(v, "ndim", 1) > 1:
            v = v.iloc[:, 0]
        return int(v.nunique(dropna=True))

    seen, cols = set(), []
    for c in semantic_label_columns(labels.columns):
        if c in seen or c not in labels.columns or _nunique(c) <= 1:
            continue
        seen.add(c)
        cols.append(c)
    if cols:
        return cols

    n = len(labels)
    out = []
    for c in labels.columns:
        if c in seen:
            continue
        seen.add(c)
        if str(c).strip().upper() == 'GSM' or is_identifier_column(c):
            continue
        k = _nunique(c)
        if k > 1 and (n < 20 or k < n * 0.5):
            out.append(c)
    return out


BATCH_FIELDS = ("_platform", "gpl", "series_id", "gse")

_BATCH_NAMES = {
    "platform", "gpl", "series", "series_id", "gse", "study", "study_id",
    "dataset", "dataset_id", "donor", "donor_id", "patient", "patient_id",
    "subject", "subject_id", "sample_id", "specimen_id",
    "assay", "chemistry", "library", "suspension_type", "suspension",
    "batch", "run", "lane", "plate", "flowcell",
}

_BATCH_PARTS = {"batch", "run", "lane", "plate", "flowcell", "flowcells"}


def _is_batch(col) -> bool:
    """True when *col* names how a sample was produced, not what it is."""
    raw = str(col).strip()
    if raw in BATCH_FIELDS:
        return True
    norm = re.sub(r"[^0-9a-z]+", "_", raw.lower()).strip("_")
    if not norm:
        return False
    if norm in _BATCH_NAMES:
        return True
    if norm.endswith("_id"):
        return True
    return any(p in _BATCH_PARTS for p in norm.split("_"))


