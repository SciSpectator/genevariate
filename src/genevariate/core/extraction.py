"""
Extraction - LLM prompt templates, parsers, and deterministic collapse.

Contains:
    - Extraction prompt templates for raw GEO metadata
    - JSON response parsers
    - Phase 1.5 deterministic GSE-scoped label collapsing
    - Text cleaning and NS detection utilities
"""

import re
import json
from typing import Dict, List

NS = "Not Specified"
LABEL_COLS = ["Tissue", "Condition"]
LABEL_COLS_SCRATCH = ["Tissue", "Condition", "Treatment"]

EXTRACTION_MODEL = "gemma2:2b"

# ── Extraction prompt template ──
EXTRACTION_PROMPT_TEMPLATE = (
    "TASK: Read the metadata below and extract exactly what is written.\n"
    "Do NOT normalise, generalise, or map to any vocabulary -- copy the specific term.\n"
    "FIELDS:\n"
    "  Tissue    : anatomical tissue, organ, cell type, or cell line as written\n"
    "  Condition : disease, phenotype, or health status as written\n"
    "  Treatment : drug or stimulus as written. None/vehicle = Untreated.\n"
    "RULES:\n"
    "  - Copy the most specific term present (e.g. Alveolar Macrophages not Lung)\n"
    "  - If a cell type is named, use the cell type (e.g. NK cells not PBMC)\n"
    "  - Unknown or absent field = Not Specified\n"
    "  - Title Case. Output JSON only.\n"
    'METADATA: Title: {TITLE}\nSource: {SOURCE}\nCharacteristics: {CHAR}\n'
    'JSON SCHEMA: {{"Tissue":"", "Condition":"", "Treatment":""}}'
)


# ── Response parsers ──

def parse_json_extraction(text: str, cols: list) -> dict:
    """Parse JSON from LLM response using greedy regex."""
    result = {c: NS for c in cols}
    if not text:
        return result
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            for col in cols:
                for key in [col, col.replace(' ', '_'), col.lower(),
                            col.replace(' ', '_').lower()]:
                    if key in data:
                        val = str(data[key]).strip()
                        if val and val.lower() not in ('none', 'null', '', 'not specified'):
                            result[col] = val
                        break
            return result
    except Exception:
        pass
    return parse_combined(text, cols)


def parse_combined(text: str, ns_cols: List[str]) -> Dict[str, str]:
    """Parse combined extraction response (Tissue: X / Condition: Y format)."""
    result = {col: NS for col in ns_cols}
    for line in text.splitlines():
        line = line.strip()
        for col in ns_cols:
            prefix = f"{col}:"
            if line.lower().startswith(prefix.lower()):
                val = line[len(prefix):].strip().strip('"').strip("'")
                val = re.sub(r"^(tissue|condition|treatment)\s*:\s*", "", val,
                             flags=re.IGNORECASE).strip()
                if val:
                    result[col] = val
                break
    return result


# ── Text utilities ──

def sanitize(text, max_chars: int = 400) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', str(text or ""))
    return text.replace('\r', ' ').strip()[:max_chars]


def clean_output(text: str) -> str:
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`").strip('"').strip("'")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    out = lines[0]
    for prefix in ("tissue:", "condition:", "treatment:",
                    "tissue :", "condition :", "treatment :"):
        if out.lower().startswith(prefix):
            out = out[len(prefix):].strip()
            break
    return out.strip('"').strip("'").strip()


def is_ns(text: str) -> bool:
    """Check if text represents a Not Specified / unknown value."""
    return text.lower().strip() in {
        "not specified", "n/a", "none", "unknown", "na",
        "not available", "not applicable", "unclear", "unspecified",
        "missing", "undetermined", "insufficient", "insufficient information",
        "no information", "no data", ""
    }


# ── Prompt formatting ──

def format_raw_block(raw: dict) -> str:
    """Format raw GEO metadata fields into a text block for LLM."""
    fields = [
        ("Title", raw.get("gsm_title", "")),
        ("Source", raw.get("source_name", "")),
        ("Characteristics", raw.get("characteristics", "")),
        ("Treatment", raw.get("treatment_protocol", "")),
        ("Description", raw.get("description", "")),
    ]
    lines = []
    for label, val in fields:
        val = sanitize(val)
        if val:
            lines.append(f"{label}: {val}")
    return "\n".join(lines) or "(no metadata available)"


def format_sample_for_extraction(raw: dict) -> str:
    """Format raw GEO metadata as compact input for extraction model."""
    def _s(v):
        return str(v).strip().replace("\t", " ").replace("\n", " ") if v else ""
    title = _s(raw.get("gsm_title", ""))[:80]
    source = _s(raw.get("source_name", ""))[:60]
    char = _s(raw.get("characteristics", ""))[:250]
    treat = _s(raw.get("treatment_protocol", ""))[:80]
    parts = []
    if title:
        parts.append(f"title:{title}")
    if source:
        parts.append(f"source:{source}")
    if char:
        parts.append(f"char:{char}")
    if treat and treat.lower() not in ("none", "n/a", ""):
        parts.append(f"treatment:{treat}")
    return "{" + ", ".join(parts) + "}" if parts else "(no metadata)"


def task_prompt(col: str) -> str:
    """Shared task instruction for extraction prompts."""
    if col == "Tissue":
        return (
            "What tissue, organ, cell type, or cell line is this sample from?\n"
            "Answer with the tissue name only (e.g. brain, liver, PBMC, MCF-7, "
            "CD4+ T cells).\n"
            "If not mentioned or not clear: Not Specified"
        )
    else:
        return (
            "What disease, condition, or phenotype does this sample represent?\n"
            "Answer with the condition name only (e.g. Alzheimer Disease, "
            "breast cancer, Control, Normal, HSV-1 infection).\n"
            "Do NOT substitute one disease for another.\n"
            "If not mentioned or not clear: Not Specified"
        )


def prompt_extract_raw(gsm: str, col: str, raw: dict) -> str:
    """Step 1a: extract from raw metadata only (no GSE context)."""
    raw_block = format_raw_block(raw)
    return (
        f"Sample {gsm}:\n{raw_block}\n\n"
        f"{task_prompt(col)}\n\n"
        f"{col}:"
    )


def prompt_extract_with_gse(gsm: str, col: str, raw: dict,
                            ctx, mem_agent=None) -> str:
    """Step 1b: extraction with GSE context (title, summary, sibling labels)."""
    raw_block = format_raw_block(raw)
    gse_hint = ""
    if ctx.title:
        gse_hint += f"Experiment title  : {ctx.title}\n"
    if getattr(ctx, "summary", ""):
        gse_hint += f"Experiment summary: {ctx.summary[:400]}\n"
    if getattr(ctx, "design", ""):
        gse_hint += f"Overall design    : {ctx.design[:200]}\n"
    if gse_hint:
        gse_hint += "\n"

    gse_label_hint = ""
    if ctx.label_counts[col]:
        lines = []
        for lbl, cnt in ctx.label_counts[col].most_common():
            lines.append(f"  {lbl} ({cnt} sample{'s' if cnt > 1 else ''})")
        gse_label_hint = (
            f"Other samples in this experiment are labeled as:\n"
            + "\n".join(lines) + "\n\n"
        )

    return (
        f"{gse_hint}"
        f"{gse_label_hint}"
        f"Sample {gsm}:\n{raw_block}\n\n"
        f"{task_prompt(col)}\n\n"
        f"{col}:"
    )


def prompt_extract_combined(gsm: str, raw: dict, ctx,
                            ns_cols: List[str],
                            gse_block: str = "") -> str:
    """Combined prompt: extract Tissue AND Condition in one LLM call."""
    raw_block = format_raw_block(raw)
    gse_hint = gse_block
    sibling_block = ""
    for col in ns_cols:
        if ctx.label_counts[col]:
            lines = [f"  {lbl} ({cnt}x)"
                     for lbl, cnt in ctx.label_counts[col].most_common()]
            sibling_block += (
                f"{col} labels in this experiment:\n"
                + "\n".join(lines) + "\n\n"
            )
    answer_fmt = "\n".join(
        f"{col}: <value or Not Specified>" for col in ns_cols
    )
    return (
        f"{gse_hint}"
        f"{sibling_block}"
        f"Sample {gsm}:\n{raw_block}\n\n"
        f"Extract the following fields from this sample.\n"
        f"Use the sibling labels above as your vocabulary -- match their exact "
        f"phrasing when appropriate.\n"
        f"If a field cannot be determined: Not Specified\n\n"
        f"{answer_fmt}"
    )


def prompt_semantic_collapse(col: str, extracted: str,
                             candidates: List[str],
                             episodic_hits: List[dict] = None,
                             kg_hits: List[tuple] = None,
                             system_prompt: str = "") -> str:
    """Segmented context window prompt for memory-aware collapse agent."""
    sys_block = system_prompt if system_prompt else (
        f"You are a biomedical metadata normalization agent for GEO field: {col}. "
        f"Use the memory segments below in priority order."
    )
    if episodic_hits:
        ep_lines = []
        for h in episodic_hits[:3]:
            ep_lines.append(
                f"  canonical={h['canonical']}  "
                f"count={h['count']}  confidence={h['confidence']:.2f}")
        ep_block = ("[ENTITY MEMORY - Tier 3 Episodic]\n"
                    "Past resolutions:\n" + "\n".join(ep_lines))
    else:
        ep_block = "[ENTITY MEMORY]\nNo past resolutions found."

    if kg_hits:
        kg_lines = [f"  {extracted} --{r[1]}--> {r[0]}  (weight={r[2]:.2f})"
                    for r in kg_hits[:3]]
        kg_block = ("[WORKFLOW MEMORY - Tier 4 KG]\n" + "\n".join(kg_lines))
    else:
        kg_block = "[WORKFLOW MEMORY]\nNo KG triples found."

    cand_lines = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    kb_block = (f"[KNOWLEDGE BASE - Tier 2 Semantic]\n"
                f"Candidates for \"{extracted}\":\n{cand_lines}")

    user_block = (
        f"[USER PROMPT]\n"
        f"Normalize this extracted {col} label: {extracted!r}\n\n"
        "Reply with ONLY the exact candidate label string, or NO_MATCH.\n"
        "Answer:"
    )
    return "\n\n".join([sys_block, ep_block, kg_block, kb_block, user_block])


# ── Phase 1.5: Deterministic collapse ──

def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[^a-z0-9]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _compact(text: str) -> str:
    return _norm(text).replace(' ', '')

def _initials(text: str) -> str:
    return ''.join(w[0] for w in _norm(text).split() if w)

def _numbers(text: str) -> List[str]:
    return re.findall(r'\d+', text)

def _numeric_guard_ok(a: str, b: str) -> bool:
    na, nb = _numbers(a), _numbers(b)
    if not na or not nb:
        return True
    return sorted(na) == sorted(nb)


def phase15_collapse(extracted: str, ctx_labels: List[str]) -> tuple:
    """
    Phase 1.5 deterministic GSE-scoped label collapsing.

    Rules:
        1. Exact match after normalisation
        2. Abbreviation/initials match (with numeric guard)

    Returns (matched_label, rule_name) or (None, None).
    """
    if not extracted or not ctx_labels:
        return None, None

    compact_e = _compact(extracted)
    initials_e = _initials(extracted)

    for existing in ctx_labels:
        if not existing or is_ns(existing):
            continue
        if not _numeric_guard_ok(extracted, existing):
            continue

        compact_x = _compact(existing)
        initials_x = _initials(existing)

        # Rule 1: exact match (normalised)
        if compact_e == compact_x:
            return existing, "exact_match"

        # Rule 2a: extracted is abbreviation of existing
        if (len(compact_e) <= 6 and len(compact_e) >= 2
                and len(compact_x) > len(compact_e)
                and compact_e == initials_x
                and len(initials_x) >= 2):
            return existing, "abbreviation"

        # Rule 2b: existing is abbreviation of extracted
        if (len(compact_x) <= 6 and len(compact_x) >= 2
                and len(compact_e) > len(compact_x)
                and compact_x == initials_e
                and len(initials_e) >= 2):
            return existing, "abbreviation"

    return None, None


def rank_candidates_by_specificity(query: str, candidates: list) -> list:
    """Re-rank collapse candidates by specificity scoring."""
    ABBREV = {
        "nk": "natural killer cells", "pbmc": "peripheral blood mononuclear cell",
        "bm": "bone marrow", "sc": "spinal cord", "cns": "central nervous system",
        "dc": "dendritic cells", "treg": "regulatory t cell",
        "hspc": "hematopoietic stem progenitor cell",
        "msc": "mesenchymal stem cell", "ipsc": "induced pluripotent stem cell",
        "esc": "embryonic stem cell",
    }

    def _expand(text):
        words = text.lower().split()
        return " ".join(ABBREV.get(w.strip("+-()"), w) for w in words)

    CELL_TYPE_WORDS = {
        "cell", "cells", "macrophage", "monocyte", "lymphocyte", "neutrophil",
        "nk", "killer", "dendritic", "fibroblast", "neuron", "astrocyte",
        "microglia", "hepatocyte", "epithelial", "endothelial", "stem",
        "progenitor", "cd4", "cd8", "cd3", "cd19", "cd14", "treg",
    }
    ORGAN_WORDS = {
        "lung", "liver", "heart", "brain", "kidney", "spleen", "pancreas",
        "blood", "bone", "muscle", "skin", "thymus", "breast", "colon",
    }

    q_words = set(_expand(query).split())
    query_has_cell_type = bool(q_words & CELL_TYPE_WORDS)

    scored = []
    for cand_label, sim in candidates:
        c_words = set(_expand(cand_label).split())
        score = sim * 5
        overlap = q_words & c_words
        score += len(overlap) * 10
        extra = c_words - q_words
        score -= len(extra) * 3
        if len(c_words) <= len(q_words):
            score += 5
        cand_is_organ = bool(c_words & ORGAN_WORDS) and not bool(c_words & CELL_TYPE_WORDS)
        if query_has_cell_type and cand_is_organ:
            score -= 10
        scored.append((cand_label, sim, round(score, 3)))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
