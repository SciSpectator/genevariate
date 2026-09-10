"""The entity links an extracted label file carries, and what they mean.

The label extractor publishes more than a value. For Tissue, Condition and
Treatment its normalization pass resolves every extracted span against a
controlled vocabulary and records the accession it landed on, so a label file
holds ``final_Tissue`` next to ``final_Tissue_id``, ``final_Tissue_mesh_id``,
``final_Tissue_cell_id`` and ``final_Tissue_stage``. Phase 1 and phase 1b are
the verbatim passes and carry no accession at all.

Three namespaces appear in that accession, and they do not mean the same
thing. ``D008099`` is a MeSH heading: the value named a concept the vocabulary
already knows. ``CVCL_0027`` is a Cellosaurus registration: the value did not
name a tissue, it named a catalogued cell line, and the extractor refuses that
identifier anywhere but Tissue. ``ART-T-00042`` was minted locally because no
vocabulary recognised the value, so it groups synonymous spellings but asserts
nothing about what the concept is. Reading all three as "the tissue" silently
mixes a liver biopsy, a hepatoma line grown in a flask and an unrecognised
phrase into one category, which is a difference no downstream analysis can
recover once the accessions are dropped.

A value is multi-span: phase 1 returns every span it found joined by ``"; "``,
and normalization keeps the accession lists positionally parallel to it,
blanks included, so span *i* of the value belongs to span *i* of every id
column.
"""

from __future__ import annotations

import re

import pandas as pd

#: Fields the normalization pass resolves. Sex and Age never enter it.
ENTITY_FIELDS = ("Tissue", "Condition", "Treatment")

#: A catalogued cell line answers Tissue only; the extractor refuses a CVCL
#: identifier in any other field, so this is where the distinction lives.
CELL_LINE_FIELD = "Tissue"

SPAN_SEP = "; "

#: Stages that carry accessions, deepest first. ``final_`` is the value that
#: ships; ``phase2_`` is the same resolution before curation.
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
#: No vocabulary recognised the value. A locally minted identifier belongs
#: here too: it groups spellings of the same phrase, but it does not attest
#: that the phrase names a tissue, so counting it as one would be an
#: assertion the extractor never made.
KIND_UNRESOLVED = "Unresolved"

#: Suffix of the column this module derives so a cell line can be told from a
#: tissue in any analysis, not only by eye.
KIND_SUFFIX = "_kind"

_CVCL = re.compile(r"^CVCL_\w+$", re.I)
#: Two out-of-vocabulary namespaces are minted, not one. ``ART-T-00042`` comes
#: from the vocabulary lookup and ``OOV-T-E7ED03862F`` from the normalization
#: pass. Both group spellings of one unrecognised phrase and neither attests
#: what the concept is, so both belong here. Matching only the first read the
#: second as a resolved MeSH heading - 203,300 values in the corpus analyzed
#: here, a fifth of every accession in it - which is precisely the confusion
#: this module's docstring says it exists to prevent.
_MINTED = re.compile(r"^(?:ART|OOV)-[A-Z]-[0-9A-F]+$", re.I)
#: A MeSH descriptor or supplementary concept. The digit run is not fixed
#: width: older headings carry six (``D008099``) and current ones nine
#: (``D000092302``), so anchoring the count would drop the newer half.
_MESH = re.compile(r"^[A-Z]\d+$")
_NOT_SPECIFIED = {"", "not specified", "nan", "none", "na", "n/a"}


def _text(value) -> str:
    """One cell as stripped text, with a missing value reading as blank.

    ``str(value or "")`` is wrong for anything that came out of a frame. A
    missing cell arrives as NaN, NaN is truthy, and ``str(nan)`` is the
    non-empty string ``"nan"`` - which then flows on as if it were content. It
    has already been read here as a Cellosaurus registration, as a MeSH
    accession and as a developmental stage, so the conversion happens in one
    place and every caller goes through it.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):       # arrays and other non-scalars
        pass
    return str(value).strip()


def namespace(accession) -> str:
    """Which vocabulary *accession* belongs to, or ``""`` when it is blank.

    An accession that matches none of the three shapes is reported as
    unlinked. It used to fall through to MeSH, which is the wrong default in
    the one direction that matters: it claims a controlled-vocabulary hit for
    a string that reached no vocabulary at all.
    """
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
    """Pair each span of *value* with the span of *ids* in the same position.

    Blanks are kept on both sides during the split, because an unresolved span
    leaves an empty entry in the id list and dropping it would slide every
    later accession onto the wrong label.
    """
    parts = spans(value)
    if not parts:
        return []
    acc = [p.strip() for p in _text(ids).split(";")]
    acc += [""] * (len(parts) - len(acc))
    return list(zip(parts, acc[:len(parts)]))


_SUFFIXES = (("id", "_id"), ("mesh_id", "_mesh_id"), ("cell_id", "_cell_id"),
             ("stage_col", "_stage"), ("curated", "_curated"))


def field_columns(columns, field: str) -> dict:
    """The column names holding *field*'s value and its accessions.

    Returns the deepest stage present, so a file exported straight from
    normalization and one that went through final assembly both read the same.
    The value column may be unprefixed while the accessions keep their stage
    prefix, which is what a frame assembled inside the program looks like, so
    the accessions are searched under the stage prefixes too. A verbatim stage
    is left bare on purpose: phase 1 resolved nothing, and lending it phase 2's
    accessions would claim a link it never made.
    """
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
    """One row per distinct label value, with the entity it resolved to.

    Counted over samples rather than over spans, so ``n`` reads as "this many
    samples carry this value" and matches what an enrichment reports.
    """
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
    """Per sample: is *field* a tissue, a catalogued cell line, or neither.

    A sample whose value names both -- "liver; HepG2" -- is neither, and is
    reported as mixed rather than silently counted as one of them.
    """
    cols = field_columns(df.columns, field)
    if not cols or "id" not in cols:
        return pd.Series("", index=df.index, dtype=object)

    ids = df[cols["id"]]
    cells = (df[cols["cell_id"]] if "cell_id" in cols
             else pd.Series("", index=df.index))
    out = []
    for value, acc, cell in zip(df[cols["value"]], ids, cells):
        # `cell or ""` is not safe here: a missing value arrives as NaN, NaN is
        # truthy, and `str(nan)` is the non-empty string "nan" - which then
        # reads as a Cellosaurus registration for every span of every row. A
        # frame that carries the cell-id column but has nothing in it came out
        # classified as a catalogued cell line from end to end: 683 of 734
        # samples in the case-study regions, against the ten per cent the
        # label files actually hold.
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
    """Add the derived kind column in place, when the links to derive it exist.

    Idempotent: a frame that already carries the column is left alone, so this
    can run on every refresh without rebuilding anything.
    """
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


#: Stage prefixes the extractor writes in front of a field name. ``final_`` is
#: the harmonised value; the rest are earlier passes, and ``Classified_`` is the
#: in-GUI labeller's own output.
LABEL_STAGE_PREFIXES = LINKED_STAGES + VERBATIM_STAGES + ("Classified_",)


def semantic_label_columns(columns) -> list:
    """The label-value columns among *columns*, one per extracted field.

    A label file carries far more than labels. For every field it also stores
    the ontology accession (``final_Tissue_id`` = MeSH ``D008168``) and the
    provenance of the mapping (``final_Tissue_stage`` = ``mesh``), plus the
    identifiers ``gsm``/``gse``/``gpl`` and the intermediate ``phase1b_*``
    pass. Those are metadata about a label, not labels, so testing them
    answers the same question several times over: "Lung" would enter an
    enrichment as ``final_Tissue``, ``phase1b_Tissue``, ``final_Tissue_id``
    and ``final_Tissue_stage`` with identical counts, inflating the apparent
    number of hits and -- because the tests are no longer distinct --
    corrupting the multiple-testing correction. ``gse`` is worse: it makes
    batch structure look like biology.

    So keep only the value column of each of the five fields the extractor
    actually produces, preferring the harmonised ``final_`` value when more
    than one stage is present.

    The one addition is the kind column this program derives from the
    accessions: whether a Tissue resolved to anatomy or to a catalogued cell
    line is a separate fact about the sample, not a restatement of the tissue
    value, so it is a test of its own rather than a duplicate one.
    """
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
    """The testable label columns of a label file, extractor-made or not.

    :func:`semantic_label_columns` only recognises the five fields the
    extractor writes. A label file the user curated themselves is just as
    valid a set of annotations and its columns can be named anything, so when
    no canonical field is present every column that is neither an identifier
    nor near-unique per sample is offered instead. A near-unique column is a
    per-sample note rather than a group and would put one sample in each cell.
    """
    def _nunique(col) -> int:
        """Distinct values in a column, whatever the frame does with its name.

        A label file the user wrote may carry the same column name twice, and
        then ``labels[name]`` is a frame rather than a series. Reading the
        first occurrence keeps that file usable instead of raising on it.
        """
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


#: Technical grouping columns that travel with the samples: they say how a
#: sample was produced, not what it is.
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
    """True when *col* names how a sample was produced, not what it is.

    Matching is on the normalised name rather than the literal one, because
    the same field reaches here spelled several ways: ``Classified_Platform``
    arrives as ``Platform`` once the alias prefix is stripped, a Census
    aggregate carries ``donor_id``, and a hand-made file may carry
    ``batch_run``. A trailing ``_id`` is decisive on its own - a column of
    identifiers groups samples by provenance whatever it is called - while a
    name like ``donor_sex`` keeps its biology, because only whole parts are
    matched and ``sex`` is not one of them.
    """
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


