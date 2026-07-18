"""Phase 2 (MeSH-only) collapse driver.

Replaces ``phase2.py`` and ``phase2_pubtator.py``. Resolves each Phase 1
raw label to a canonical MeSH descriptor name, or — if no MeSH
descriptor fits — mints an entry in the out-of-distribution (OOD) mesh
(ART-{T,C,X}-#####), backed by ``mesh.sqlite``'s
``ood_mesh_clusters`` table.

Output rule: returns canonical NAMES only, never MeSH IDs (per the user
spec). For composite Phase 1 labels (semicolon-joined), each component
is resolved independently and the canonical names are joined back with
``; `` in input order.

Tier order per component:

    1. Episodic recall
       Last decision logged for the same (raw_label, col) gets reused.

    2. Exact MeSH match
       ``mesh_terms.name`` or ``mesh_synonyms.synonym`` (NOCASE),
       category-gated by col (Tissue=A, Condition=C/F, Treatment=D/E).
       If exactly 1 hit, take it. If >1 hit, LLM picks among them.

    3. Existing OOD-mesh match
       ``ood_mesh_clusters.label`` or ``ood_mesh_synonyms.synonym``
       (NOCASE).

    4. Hybrid candidate gathering + LLM picker
       Pool MeSH candidates from two complementary sources:
         - PubTator3 normalize:   curated NCBI entity → MeSH (best on
           abbreviations like FTD/PBMC/MPTP and named drugs).
         - BioLORD top-K:         semantic similarity over MeSH names +
           scopes + synonyms (best on free-form anatomy/condition prose
           where PubTator3's NER is brittle, e.g. "whole blood",
           "human liver tissue").
       Both are col-gated to MeSH categories; results are deduped by
       MeSH ID and handed to gemma4:e2b which picks one or replies NONE.

    5. Mint OOD-mesh entry
       If the picker says NONE (or PubTator3 + BioLORD both produced
       nothing), mint a new ART-{T,C,X}-##### entry into the OOD mesh.

NS / empty inputs short-circuit to ``Not Specified``. PubTator3 use is
optional (network-dependent); set ``use_pubtator=False`` for fully
offline operation, in which case Tier 4 falls back to BioLORD-only.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    import requests as _requests          # type: ignore
except ImportError:                        # pragma: no cover
    _requests = None                       # only needed if LLM picker fires

from mesh_lookup import COL_CATS, MeshDB

try:
    from phase2_pubtator import PubTatorNormalizer    # type: ignore
except ImportError:                                    # pragma: no cover
    PubTatorNormalizer = None                          # type: ignore


NS = "Not Specified"
LABEL_COLS = ("Tissue", "Condition", "Treatment")

# Universal non-disease sample-state descriptors. Standard biomedical
# lab vocabulary for "control / normal / healthy / wild-type / untreated
# baseline" — universal across GEO, NOT eval-specific. A Phase-1/1b/1c
# Condition value matching one of these is the sample's documented
# sample-state — a FIRST-CLASS Condition value, NOT an absence of
# information. Phase 2 normalizes variants to a canonical surface form
# (so "control"/"controls"/"ctrl" all converge to "Control") and emits
# them through, instead of dropping to NS.
# Restricted to the Condition column: "vehicle"/"control"/"PBS" can
# legitimately be the Treatment in vehicle-control arms, so the
# Treatment column does NOT consult this set.
_CONDITION_CONTROL_CANONICAL: dict[str, str] = {
    "control":      "Control",
    "controls":     "Control",
    "ctrl":         "Control",
    "ctl":          "Control",
    "normal":       "Normal",
    "healthy":      "Healthy",
    "healthy control":  "Healthy Control",
    "healthy controls": "Healthy Control",
    "normal control":   "Normal Control",
    "normal controls":  "Normal Control",
    "healthy donor":    "Healthy Donor",
    "untreated":    "Untreated",
    "non-treated":  "Untreated",
    "nontreated":   "Untreated",
    "no treatment": "Untreated",
    "wt":           "Wild Type",
    "wild type":    "Wild Type",
    "wild-type":    "Wild Type",
    "wildtype":     "Wild Type",
    "baseline":     "Baseline",
    "mock":         "Mock",
    "sham":         "Sham",
    "naive":        "Naive",
    "naïve":        "Naive",
    "unstimulated": "Unstimulated",
    "unstim":       "Unstimulated",
    "non-tumor":    "Non-Tumor",
    "non-tumorous": "Non-Tumor",
    "non-malignant":"Non-Malignant",
    "non-disease":  "Non-Disease",
    "unaffected":   "Unaffected",
    "uninflamed":   "Uninflamed",
    "uninvolved":   "Uninvolved",
    "no condition": "Control",
    "no disease":   "Control",
    "none":         "Control",
}

# Generic Tissue placeholders that carry no organ signal. When the
# upstream pipeline emits one of these as the Tissue value, Phase 2
# would otherwise mint an OOD-mesh entry (or, worse, pick a
# semantically-near organ from BioLORD) — both of which are wrong
# because the source text genuinely doesn't specify an organ. Returning NS at Tier 0
# preserves the lack-of-signal honestly. Universal biomedical
# vocabulary, not eval-derived.
_TISSUE_GENERIC_PLACEHOLDERS = frozenset({
    "tumor", "tumour", "tumors", "tumours",
    "tissue", "tissues",
    "cell", "cells",
    "sample", "samples",
    "biopsy", "biopsies",
    "specimen", "specimens",
})

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_LLM_MODEL  = os.environ.get("PHASE2_MODEL", "gemma4-e2b-text:latest")
_LLM_NUM_CTX = int(os.environ.get("PHASE2_NUM_CTX", "4096"))
_TOP_K       = int(os.environ.get("PHASE2_TOPK",   "30"))

# Cache key for verifier verdicts. Bump whenever _VERIFIER_SYSTEM changes
# so cluster-scale fleets can invalidate stale verdicts without wiping
# the table. The picker prompt is included since a verdict only makes
# sense relative to the proposal that produced it.
_PROMPT_VERSION = os.environ.get(
    "PHASE2_PROMPT_VERSION", "v14-negation-tier0-2026-05-08"
)

# Cap study-context block at this many chars before injecting into picker /
# verifier prompts. Real GSE title+summary+overall_design typically fits in
# 600–1000 chars; truncating defends against pathological 50-kB blobs that
# would blow num_ctx and dilute the prompt's normalization rules.
_CONTEXT_CHARS = int(os.environ.get("PHASE2_CONTEXT_CHARS", "1200"))


# Universal anatomical adjective ↔ noun lexicon. Used to bridge the
# embedding gap between raws like "renal tissue" and MeSH descriptors
# like "Kidney" — BioLORD ranks the histology umbrella ("Parenchymal
# Tissue") above the organ when "tissue"/"parenchyma" dominate the
# embedding. Lexical re-ranking on raw-token overlap (after augmenting
# adjectives with their noun form) restores the organ to the top band
# before the LLM picker sees the candidate list.
#
# These pairs are stable English medical morphology — independent of any
# eval set, no entity leaks. Add new pairs only when they're general
# adjective↔noun morphology, never one-off case fixes.
_ANAT_ADJ: dict[str, str] = {
    "hepatic":     "liver",
    "renal":       "kidney",
    "pulmonary":   "lung",
    "cardiac":     "heart",
    "gastric":     "stomach",
    "cerebral":    "brain",
    "splenic":     "spleen",
    "intestinal":  "intestine",
    "colonic":     "colon",
    "esophageal":  "esophagus",
    "bronchial":   "bronchi",
    "thymic":      "thymus",
    "pancreatic":  "pancreas",
    "ovarian":     "ovary",
    "uterine":     "uterus",
    "prostatic":   "prostate",
    "mammary":     "breast",
    "muscular":    "muscle",
    "neuronal":    "neuron",
    "vascular":    "vessel",
    "osseous":     "bone",
    "skeletal":    "skeleton",
    "dermal":      "skin",
    "epidermal":   "epidermis",
    "lingual":     "tongue",
    "nasal":       "nose",
    "ocular":      "eye",
    "biliary":     "bile",
    "lymphatic":   "lymph",
    "salivary":    "saliva",
    "adrenal":     "adrenal",
}

# Generic-histology stop tokens. Stripped from BOTH raw and candidate
# names before computing overlap, so the score reflects the
# entity-bearing tokens (organ / cell-type) and not the sample-source
# phrasing.
_HISTOLOGY_STOP: set[str] = {
    "tissue", "tissues", "parenchyma", "parenchymal",
    "sample", "samples", "specimen", "specimens",
    "cell", "cells", "biopsy", "biopsies",
    "section", "sections", "block", "blocks",
    "of", "the", "a", "an",
}

_TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]+")

# Universal dose / duration / concentration / route stripper. Matches
# any number followed by a typical unit token (or a duration phrase like
# 'for 24 h') and replaces it with a space. Used by the dose-strip
# retry tier for Treatment raws like 'paclitaxel 10 nM 24 h' →
# 'paclitaxel'. No drug-name hardcoding — pure unit lexicon.
# Unit token — compound units (with "/") MUST come first in the
# alternation, otherwise `mg` will match before `mg/kg` and leave a
# stranded `/kg` fragment in the output.
_UNIT_TOKEN = (
    # compound (mass / volume / dose-rate)
    r"mg/kg|µg/kg|ug/kg|μg/kg|ng/kg"
    r"|ng/ml|ng/mL|mg/ml|mg/mL|µg/ml|ug/ml|μg/ml|µg/mL|ug/mL|μg/mL"
    r"|U/ml|U/mL|IU/ml|IU/mL"
    # molarity (mM / nM / µM / pM / fM) — order doesn't matter, no overlap
    r"|mM|nM|µM|uM|μM|pM|fM|M"
    r"|mmol|nmol|µmol|umol|mol"
    # mass — mg before m, ng before n, etc., bare 'g' last
    r"|mg|µg|ug|μg|ng|pg|kg|g"
    # volume
    r"|mL|ml|µL|uL|μL|nL|nl|L|l"
    # radiation
    r"|cGy|Gy"
    # activity
    r"|IU|U"
    # time — multi-letter first
    r"|hours|hour|hrs|hr"
    r"|mins|min"
    r"|secs|sec"
    r"|days|day"
    r"|weeks|week|wks|wk"
    r"|months|month|mo"
    r"|years|year|yrs|yr"
    r"|h|s|d|y"
)

_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:" + _UNIT_TOKEN + r")(?![A-Za-z])"
    r"|\b\d+(?:\.\d+)?\s*%"                   # percentages
    r"|\bfor\s+\d+\s*(?:" + _UNIT_TOKEN + r")(?![A-Za-z])"
    r"|\b(?:at|after|over)\s+\d+\s*(?:" + _UNIT_TOKEN + r")(?![A-Za-z])"
    r"|\b\d+\s*(?:fold|x)\b",                 # 10-fold, 5x
    re.IGNORECASE,
)


def _strip_dose(raw: str) -> str | None:
    """Strip numeric dose / duration / concentration / route tokens from
    a Treatment label. Returns the stripped form when it differs from the
    input (case-insensitive); else None.

    Examples:
      'paclitaxel 10 nM 24 h' → 'paclitaxel'
      'cisplatin 5 µg/mL'     → 'cisplatin'
      'metformin 100mg/kg'    → 'metformin'
      'IL-12 (10 ng) + butyrate (0.5 mM)'   → 'IL-12 () + butyrate ()'
        — caller still benefits from MeSH on each component.
      'cisplatin'             → None (no rewrite needed)
    """
    if not raw:
        return None
    s = _DOSE_RE.sub(" ", raw)
    # Drop now-empty parens and consecutive separators left by the strip.
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"[,\;\-]\s*[,\;\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;-")
    if not s or s.lower() == raw.lower():
        return None
    return s


def _augment_raw_tokens(raw: str) -> set[str]:
    """Tokenise the raw label, lowercase it, drop generic-histology
    stops, and add the noun form of any anatomical adjective. The result
    is a set used for token-overlap scoring against candidate names.
    """
    toks = {t.lower() for t in _TOK_RE.findall(raw or "")}
    aug: set[str] = set()
    for t in toks:
        aug.add(t)
        if t in _ANAT_ADJ:
            aug.add(_ANAT_ADJ[t])
    return aug - _HISTOLOGY_STOP


def _candidate_name_tokens(c: dict) -> set[str]:
    return {t.lower() for t in _TOK_RE.findall(c.get("name") or "")} \
        - _HISTOLOGY_STOP


def _augmented_query(raw: str) -> str | None:
    """Rewrite the raw into a normalised query by (a) replacing each
    anatomical adjective with its noun form and (b) dropping generic
    histology stop tokens. Returns the rewritten string when it differs
    from the original (case-insensitive); otherwise None — letting the
    caller skip the secondary retrieval round-trip when it would just
    re-issue the same query.

    Example: 'pulmonary tissue' → 'lung'; 'renal tissue' → 'kidney';
    'lung parenchyma' → 'lung'; 'lung' → None (no rewrite).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    out: list[str] = []
    for tok in _TOK_RE.findall(raw):
        low = tok.lower()
        if low in _HISTOLOGY_STOP:
            continue
        out.append(_ANAT_ADJ.get(low, low))
    if not out:
        return None
    rewritten = " ".join(out)
    return rewritten if rewritten.lower() != raw.lower() else None


def _format_context_block(context: str) -> str:
    """Render an optional study-context block for inclusion in the picker
    or verifier user message. Returns an empty string when no context is
    given, so callers can unconditionally splice it in.
    """
    ctx = (context or "").strip()
    if not ctx:
        return ""
    if len(ctx) > _CONTEXT_CHARS:
        ctx = ctx[:_CONTEXT_CHARS].rsplit(" ", 1)[0] + "…"
    # Single-line collapse so a long abstract doesn't visually dominate.
    ctx = " ".join(ctx.split())
    return f"study context: {ctx}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Polarity classifier prompt (Tier 0.5)
# ─────────────────────────────────────────────────────────────────────────────
# Universal grammatical pre-check applied BEFORE any cache or MeSH cascade.
# A raw whose meaning denies the entity (never X / no X / non-X / without X /
# absent / unaffected / un-X when negating, etc.) is, by universal rule, a
# missing observation = Not Specified — for ALL columns. We do NOT mint a
# separate "absence:<id>" lattice; "not treatment" means no treatment at
# all, "not disease" means disease state is unknown / not asserted, "not
# tissue" means no organ is asserted. Reproducibility comes from caching
# the verdict per (raw_lc, col, prompt_version); scalability from the
# universal collapse to NS (no auxiliary entity table to maintain).
#
# Temporal ASSERTIONS (former, ex-, previous, history-of, post-, current,
# recent, past tense in general) DO assert the entity — the subject DID
# undergo it. Those are ASSERT, not NEGATE.
_POLARITY_SYSTEM = (
    "You classify the polarity of a biomedical sample label as either\n"
    "ASSERT or NEGATE.\n"
    "\n"
    "Inputs:\n"
    "  - study context (optional) — free-form text from the experiment's\n"
    "    title / summary / overall_design or per-sample characteristics.\n"
    "    When the raw label contains a coded value (digit, single letter,\n"
    "    short token) that is decoded by an inline legend in the context\n"
    "    (e.g. 'category (1, 2, 3 = control, low-dose, high-dose): 1' →\n"
    "    1 means control, which encodes ABSENCE of the active condition),\n"
    "    use the legend to determine whether the resolved meaning is\n"
    "    asserted or negated. ALWAYS consult the context first when the\n"
    "    raw is digit-only or parens/colon-coded.\n"
    "  - raw label, column.\n"
    "\n"
    "Read the raw as a sentence describing one sample / patient. Ask:\n"
    "does this subject EVER possess / undergo / experience the entity\n"
    "named (or implied) by the raw?\n"
    "\n"
    "ASSERT — yes, at some point in time. Includes:\n"
    "  • plain present (e.g. an organ name, a disease name, a drug name)\n"
    "  • past tense / temporal qualifiers (history of <X>, previous <X>,\n"
    "    post-<X>, ex-<X>, former <X>)\n"
    "  • current / recent (currently on <X>, recently treated with <X>)\n"
    "  • affected / positive (subject is <X>-positive, has <X>, is\n"
    "    <X>-status)\n"
    "ASSERT covers any time the entity held for the subject, even if it\n"
    "is no longer holding now.\n"
    "\n"
    "NEGATE — no, the entity NEVER held for this subject. Any natural-\n"
    "language negation that scopes over the whole entity:\n"
    "  • negative adverbs / determiners (never, no, none, not)\n"
    "  • negative prepositions (without, free of, lacking)\n"
    "  • absence verbs (denies, lacks)\n"
    "  • negative status words (absent, unaffected, refused, declined,\n"
    "    'negative' as a status modifier)\n"
    "  • negative prefixes when they negate (non-, un-, dis- when it\n"
    "    means absence)\n"
    "  • bare 'never' / 'none' / 'no' as the whole label\n"
    "NEGATE encodes the negative arm of a contrast (control / no-\n"
    "exposure / no-treatment / disease-free).\n"
    "\n"
    "Edge cases (rules — NOT entity-specific examples):\n"
    "  • Empty / placeholder raws (n/a, not specified, unknown, ?, '-')\n"
    "    → answer ASSERT (they have their own NS short-circuit elsewhere\n"
    "    in the pipeline; you only see them if that short-circuit\n"
    "    missed; do not double-classify them as NEGATE here).\n"
    "  • Compound raws joined with ';' or '+' — classify the WHOLE\n"
    "    string. If ANY component is asserted, answer ASSERT. NEGATE\n"
    "    only when EVERY component is itself negated.\n"
    "  • Some compound forms beginning with 'non-' name positive\n"
    "    biomedical entities (canonical disease subtypes whose accepted\n"
    "    nomenclature contains 'non-' as part of the entity, not as a\n"
    "    negation operator). For these, a subject HAS the compound\n"
    "    entity → answer ASSERT. Test: if the compound has its own\n"
    "    accepted disease/entity name and a subject can be described\n"
    "    as 'having' it, then ASSERT. If, by contrast, the compound\n"
    "    decomposes cleanly into 'non' + (a known entity that the\n"
    "    subject does NOT have), then NEGATE.\n"
    "  • DO NOT be fooled by noun-phrase grammar. A negated agent-noun\n"
    "    (e.g. 'non-<doer>' or 'never-<doer>' where <doer> means 'one\n"
    "    who does X') describes a person — but the question is about\n"
    "    the ENTITY (the action / exposure / disease), not the social\n"
    "    role. A non-doer never did the action → entity = NEGATE.\n"
    "    Plurality / singularity of the role noun is irrelevant.\n"
    "  • DO NOT be fooled by adjective-of-absence forms. Compounds\n"
    "    ending in '-free', '-naive', or '-negative' applied to an\n"
    "    entity name encode ABSENCE of that entity (e.g. <X>-free =\n"
    "    no <X>; <X>-naive = no prior <X>; <X>-negative = lacks <X>)\n"
    "    → NEGATE.\n"
    "  • SUFFIX TEST for 'non-<X>' / 'no <X>' / '<X>-free' / 'never\n"
    "    <X>' / 'never had <X>' / 'absent <X>' / '<X>-negative' /\n"
    "    'lacking <X>': if the rest of the raw IS the bare entity (or\n"
    "    a clear morphological inflection of it — plural, gerund,\n"
    "    agent-noun derived from a verb) → NEGATE. ONLY treat the\n"
    "    prefixed form as ASSERT when the compound is itself a\n"
    "    recognized biomedical entity that the subject HAS (i.e. the\n"
    "    'non-<X>' compound has its own canonical name in disease /\n"
    "    drug nomenclature, not a literal negation of <X>).\n"
    "  • PARTICIPLE-NEGATION TEST. The prefixes 'un-' and 'non-' on a\n"
    "    PAST PARTICIPLE or on an AGENT NOUN derived from a verb\n"
    "    encode absence of that action — 'un-<verb-ed>' means the\n"
    "    subject did NOT undergo <verb>; 'non-<doer>' means the\n"
    "    subject is NOT the doer of <verb>. Both → NEGATE. (This is\n"
    "    the test that distinguishes 'non-<doer>' from 'non-<disease-\n"
    "    name>': the former is a negated verbal noun, the latter is a\n"
    "    canonical compound entity.)\n"
    "\n"
    "OUTPUT FORMAT (strict): reason in one short line, then on the\n"
    "VERY LAST line output exactly:\n"
    "  POLARITY: ASSERT\n"
    " or\n"
    "  POLARITY: NEGATE\n"
    "The parser requires the literal 'POLARITY:' prefix.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Picker prompt
# ─────────────────────────────────────────────────────────────────────────────
_PICKER_SYSTEM = (
    "You are a biomedical normalizer mapping a raw GEO sample label to\n"
    "the best MeSH descriptor. Inputs (in this order, when present):\n"
    "  - study context: free-form text from the experiment's title /\n"
    "    summary / overall_design. ALWAYS read this first; it usually\n"
    "    defines abbreviations, brand names, and study-specific terms\n"
    "    (e.g. an abstract that says 'AD = Alzheimer Disease' or\n"
    "    'patients received DEX (dexamethasone)' tells you exactly\n"
    "    how to expand short forms in the raw label).\n"
    "  - raw label: the surface form to normalize.\n"
    "  - column: Tissue / Condition / Treatment.\n"
    "  - candidates: numbered MeSH descriptors with [category] tag and\n"
    "    short scope. Output one PICK.\n"
    "\n"
    "DEFAULT BEHAVIOUR: pick the candidate that denotes the SAME\n"
    "biomedical entity as the raw — even when wording differs. Reply\n"
    "NONE only when no candidate denotes the same entity.\n"
    "\n"
    "Use the study context to disambiguate:\n"
    "  • If the raw is a short abbreviation (1–6 letters), look in the\n"
    "    context for the spelled-out form, then match the candidate\n"
    "    whose preferred name / scope corresponds to that expansion.\n"
    "  • If the raw is a brand or trade name and the context mentions\n"
    "    its generic active ingredient, pick the generic candidate.\n"
    "  • If multiple candidates fit the raw alone, prefer the one most\n"
    "    consistent with the experiment topic stated in the context.\n"
    "  • If no context is provided, fall back to general biomedical\n"
    "    knowledge to expand abbreviations / brand names.\n"
    "\n"
    "Wording differences that still mean SAME entity (always pick):\n"
    "  • Abbreviation ↔ its standard expansion. Use the study context\n"
    "    first; otherwise general biomedical knowledge.\n"
    "  • Brand / trade name ↔ the generic preferred name of the drug.\n"
    "  • Chemical / IUPAC / systematic name of a known drug ↔ that\n"
    "    drug's preferred name (NOT the parent chemical class).\n"
    "  • Dose / duration / concentration suffix on a compound — treat\n"
    "    the compound alone (strip the dose; the descriptor is the\n"
    "    compound).\n"
    "  • Cell-line / strain / species / age modifier on a tissue —\n"
    "    collapse to the underlying anatomy or cell-type descriptor.\n"
    "  • Adjectival form of an organ ↔ the organ's noun descriptor\n"
    "    (linguistic equivalent: an organ-adjective always names that\n"
    "    organ).\n"
    "  • '<organ-name> tissue' / '<organ-name> parenchyma' /\n"
    "    '<organ-name> sample' ↔ the organ's MeSH descriptor. The\n"
    "    organ-name modifier wins; the trailing generic-histology word\n"
    "    is just sample-source phrasing.\n"
    "  • Synonym / older nomenclature / minor spelling variant of the\n"
    "    same entity.\n"
    "  • Subtype IS explicitly named in the raw → pick that subtype.\n"
    "    Subtype is NOT named in the raw → pick the broader umbrella;\n"
    "    DO NOT invent a subtype the raw never mentions.\n"
    "\n"
    "Generic-histology trap: never pick a category-wide histology\n"
    "umbrella (descriptors whose meaning is just 'tissue', 'cells',\n"
    "'parenchyma', 'connective tissue', 'epithelium' etc.) when the raw\n"
    "names a SPECIFIC organ, drug, or disease. The organ-named\n"
    "candidate always beats the generic-histology candidate.\n"
    "\n"
    "When NOT to pick (reply NONE):\n"
    "  R1. The raw is a placeholder with NO biomedical content (none,\n"
    "      n/a, not specified). These cannot legitimately map to any\n"
    "      MeSH descriptor.\n"
    "      Note: 'control' / 'healthy' / 'normal' / 'untreated' /\n"
    "      'wild type' / 'baseline' for the Condition column are\n"
    "      handled BEFORE the picker (canonicalized to a fixed surface\n"
    "      form) — you will not see them here.\n"
    "  R2. Every candidate is a different specific entity that merely\n"
    "      shares a word with the raw — no candidate denotes the\n"
    "      raw's entity at all.\n"
    "  R3. Every candidate is in the wrong MeSH branch for the column.\n"
    "      Tissue requires category A (anatomy / cell / cell line);\n"
    "      Condition requires C or F (disease / mental disorder);\n"
    "      Treatment requires D or E (drug / biologic / procedure /\n"
    "      therapeutic technique). Cross-branch picks are NEVER valid.\n"
    "  R4. NEGATED RAW. The raw is grammatically negated — read it as\n"
    "      a sentence and ask: does the subject EVER possess / undergo\n"
    "      the entity, or NOT?\n"
    "        ASSERTION (always-true OR past-true): pick the bare entity.\n"
    "          The entity holds for this subject at some point in time.\n"
    "          A temporal qualifier (past tense, ex-, former-, previous,\n"
    "          history-of, post-) ASSERTS the entity in the past — the\n"
    "          subject DID undergo it. Pick the bare entity.\n"
    "        TRUE NEGATION (always-false): reply NONE. The raw denies\n"
    "          the entity ever held for this subject — natural-language\n"
    "          negation that scopes over the whole predicate (never had,\n"
    "          no, none, without, absent, denies, non-, un- when it\n"
    "          negates, etc.). This encodes the negative arm of a\n"
    "          contrast — semantically DISTINCT from the bare entity\n"
    "          AND from any treatment / agent / process that addresses\n"
    "          the entity. A negated raw is a separate class (no-\n"
    "          exposure / control), not the entity itself. Reply NONE\n"
    "          unless a candidate's preferred name itself denotes that\n"
    "          absence / no-exposure state.\n"
    "      The test is grammatical: ASSERTION → bare entity; TRUE\n"
    "      NEGATION → NONE. Applies universally across all columns.\n"
    "\n"
    "OUTPUT: reason briefly, then on the VERY LAST line output one of:\n"
    "  PICK: <integer>     (the chosen candidate index in [0..N-1])\n"
    "  PICK: NONE          (no candidate refers to the same entity)\n"
    "The parser requires the literal 'PICK:' prefix; without it the\n"
    "answer is treated as NONE.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Verifier prompt (Tier 4.5)
# ─────────────────────────────────────────────────────────────────────────────
_VERIFIER_SYSTEM = (
    "You are a biomedical normalization verifier. Inputs (in this order,\n"
    "when present):\n"
    "  - study context: free-form text from the experiment's title /\n"
    "    summary / overall_design. Use it to decode short abbreviations\n"
    "    or brand names in the raw label (the abstract often defines\n"
    "    them explicitly).\n"
    "  - raw label, column, picked MeSH descriptor (with category and\n"
    "    scope).\n"
    "Decide whether the pick names the SAME biomedical entity as the\n"
    "raw — even when wording differs.\n"
    "\n"
    "DEFAULT BEHAVIOUR: KEEP. Reject ONLY when one of R1..R5 below is\n"
    "clearly true; in all other cases approve. When the study context\n"
    "spells out an abbreviation or brand name and the pick matches that\n"
    "expansion, KEEP — the context is authoritative for that study.\n"
    "\n"
    "Wording differences that still mean SAME entity (always KEEP):\n"
    "  • Abbreviation ↔ its standard biomedical expansion.\n"
    "  • Brand / trade name ↔ the drug's generic preferred name.\n"
    "  • Chemical / IUPAC / systematic name ↔ a drug's preferred name.\n"
    "  • Dose / concentration / duration suffix stripped from a\n"
    "    compound — the compound is still the entity.\n"
    "  • Adjectival organ form ↔ that organ's noun descriptor.\n"
    "  • '<organ> tissue' / '<organ> parenchyma' / '<organ> sample'\n"
    "    rolled up to the organ descriptor. The organ-name modifier\n"
    "    is what names the entity; the trailing histology word is\n"
    "    sample-source phrasing.\n"
    "  • Generic disease wording rolled up to its parent class when\n"
    "    the raw does NOT specify a histological subtype.\n"
    "  • Synonym / older nomenclature / minor spelling variant.\n"
    "\n"
    "REJECT only when one of these is clearly true:\n"
    "\n"
    "R1. WRONG COLUMN BRANCH — the pick's MeSH category is wrong for\n"
    "    the column. Tissue requires A (anatomy / cell / cell line);\n"
    "    Condition requires C or F (disease / mental disorder);\n"
    "    Treatment requires D or E (drug / biologic / procedure /\n"
    "    therapeutic technique). Cross-branch picks are NEVER valid.\n"
    "\n"
    "R2. DIFFERENT SPECIFIC ENTITY — the pick is a clearly different\n"
    "    drug, disease, gene, or anatomy that merely shares a word\n"
    "    with the raw (the picker confused two unrelated entities\n"
    "    because of literal token overlap).\n"
    "\n"
    "R3. INFORMATION-FREE RAW — the raw is an empty placeholder with\n"
    "    NO biomedical content (none, n/a, not specified). Such raws\n"
    "    cannot legitimately map to anything; REJECT regardless of\n"
    "    pick. NOTE: standard biomedical abbreviations are NOT\n"
    "    information-free — they encode one specific entity and must\n"
    "    be KEPT against their canonical expansion. Healthy / control /\n"
    "    normal / untreated / wild-type sample-state words for the\n"
    "    Condition column are also NOT information-free — they are\n"
    "    canonicalized BEFORE the picker runs and never reach the\n"
    "    verifier under the canonical surface form.\n"
    "\n"
    "R4. WRONG SPECIFICITY DIRECTION — the picker chose a more\n"
    "    specific subtype than the raw mentions (raw is generic,\n"
    "    pick is a histological subtype the raw never names), or\n"
    "    chose a generic-histology umbrella (e.g. a 'tissue' or\n"
    "    'parenchyma' descriptor) when the raw names a SPECIFIC\n"
    "    organ that has its own descriptor available. Same-level\n"
    "    same-entity picks are fine.\n"
    "\n"
    "R5. TRUE-NEGATION RAW. The raw asserts the ABSENCE of an entity,\n"
    "    not the entity itself. Read the raw and decide whether the\n"
    "    subject EVER possesses / undergoes the entity:\n"
    "      ASSERTION — the entity holds at some point (now or past).\n"
    "        Temporal qualifiers (past tense, ex-, former-, previous,\n"
    "        history-of, post-, current, recent) ASSERT the entity. A\n"
    "        bare-entity pick is fine — KEEP it.\n"
    "      NEGATION — the raw denies the entity ever held for this\n"
    "        subject. Natural-language negation scoping over the whole\n"
    "        predicate (never had, no, none, without, absent, non-, un-\n"
    "        when it negates, denies, etc.). Absence is its own class —\n"
    "        DIFFERENT from the bare entity AND from any agent /\n"
    "        process that addresses it. REJECT a bare-entity pick OR a\n"
    "        related-intervention pick for a negated raw, UNLESS the\n"
    "        pick's preferred name itself denotes the absence /\n"
    "        no-exposure state.\n"
    "    Grammatical, column-universal.\n"
    "\n"
    "OUTPUT FORMAT (strict): produce ONE LINE PER ITEM in this exact\n"
    "order, each line filled with your judgement (no skipping, no\n"
    "merging), then the VERDICT. Before answering R5, you MUST answer\n"
    "the 'subject_undergoes_picked_entity' line. Read the raw,\n"
    "identify the SUBJECT (the sample / patient the raw describes),\n"
    "and ask literally: 'Does this subject ever undergo / possess /\n"
    "experience the picked entity?'. Picked entity is the descriptor\n"
    "named on the 'picked:' line above; ignore the descriptor's\n"
    "scope text for this step. Use temporal-inclusive reading: past,\n"
    "present, and history-of all count as YES. Only a true denial\n"
    "(never, none, no, without, non-, un- when negating, etc.)\n"
    "scoping over the entity yields NO.\n"
    "Then:\n"
    "  subject_undergoes_picked_entity: YES|NO\n"
    "  R1_wrong_branch:     YES|NO\n"
    "  R2_different_entity: YES|NO\n"
    "  R3_info_free_raw:    YES|NO\n"
    "  R4_wrong_specificity: YES|NO\n"
    "  R5_negated_raw:      YES|NO   (YES iff\n"
    "    subject_undergoes_picked_entity is NO AND the picked entity\n"
    "    is NOT itself an absence/cessation/withdrawal descriptor.)\n"
    "  VERDICT: KEEP   (when ALL five R-items are NO)\n"
    "         | REJECT (when ANY R-item is YES)\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
class Phase2Mesh:
    """Collapse Phase 1 raw labels to canonical MeSH names (or mint new
    out-of-distribution (OOD) mesh entries when no MeSH descriptor fits)."""

    def __init__(self, db: MeshDB | None = None, *, top_k: int = _TOP_K,
                 use_episodic: bool = True, use_picker: bool = True,
                 use_pubtator: bool = True, use_verifier: bool = True,
                 use_polarity: bool = True,
                 pubtator_sleep: float = 0.34,
                 cache: Any = None):
        """``cache`` is an optional ``GSEContextCache`` used for per-GSE
        canonical sibling-consistency. When provided AND ``gse_id`` is
        passed to ``collapse`` / ``collapse_record``, every sibling raw
        in the same experiment routes through ``gse_phase2_canon`` first
        (Tier 1.5 / 1.6) and falls back to the global cascade only on
        miss. Persisted decisions ensure two equivalent raws within one
        GSE collapse to the SAME canonical id.

        ``use_polarity`` toggles the Tier 0.5 negation short-circuit.
        Default on; turn off only for ablation studies.
        """
        self.db = db or MeshDB()
        self.top_k = top_k
        self.use_episodic = use_episodic
        self.use_picker = use_picker
        self.use_verifier = use_verifier
        self.use_polarity = use_polarity
        self.use_pubtator = use_pubtator and PubTatorNormalizer is not None
        self._pt = (PubTatorNormalizer(sleep=pubtator_sleep)
                    if self.use_pubtator else None)
        self._cache = cache

    # ── entry points ────────────────────────────────────────────────────
    def collapse(self, raw: str, col: str, context: str = "",
                 gse_id: str | None = None) -> dict:
        """Resolve one (Phase 1 raw, col) pair to canonical name(s).

        ``context`` is optional GSE-level free-form text (title + summary +
        overall_design from the source experiment). When provided, it is
        passed to the LLM picker / verifier so abbreviations and
        brand names defined in the experiment metadata can be expanded
        (e.g. an AD/PD/NSCLC abbreviation that the abstract spells out).

        ``gse_id`` opts into per-GSE sibling consistency: when set AND a
        ``cache`` was passed to ``__init__``, equivalent siblings within
        the same experiment collapse to the same canonical id without
        re-running the LLM cascade.
        """
        if col not in COL_CATS:
            raise ValueError(f"col must be one of {tuple(COL_CATS)}, got {col!r}")
        raw = (raw or "").strip()
        if not raw or raw.lower() == NS.lower():
            return {"label": NS, "components": [], "id": ""}

        # Composite split. Always split on ';'. For Treatment, also split
        # on ' + ' (with surrounding whitespace) so dosed combinations like
        # '10 ng IL-12 + 0.5 mM butyrate' resolve to two components instead
        # of dropping one. The space requirement keeps 'CD4+ T cells' and
        # 'CD8+/CD4+' (Tissue) intact since their '+' has no space before.
        sep = r";\s*|\s+\+\s+" if col == "Treatment" else r";\s*"
        parts = [p.strip() for p in re.split(sep, raw) if p.strip()]
        comps: list[dict] = []
        for p in parts:
            comps.append(self._resolve_one(p, col, context, gse_id))

        canonical_names = [c["name"] for c in comps]
        canonical_ids   = [c["id"]   for c in comps]
        return {
            "label":      "; ".join(canonical_names),
            "id":         "; ".join(canonical_ids),
            "components": comps,
        }

    def collapse_record(self, record: dict) -> dict:
        """Apply collapse() to all three label cols of a Phase 1 record.
        Output row contains the resolved canonical NAMES under the same
        keys; the MeSH / OOD-mesh IDs are kept under ``<col>_id`` for audit.

        ``record`` may contain a ``gse_context`` field (free-form text from
        title/summary/overall_design); when present it is forwarded to the
        picker/verifier so study-defined abbreviations can be expanded.
        """
        out = dict(record)
        ctx = (record.get("gse_context") or "").strip()
        gse_id = (record.get("gse") or record.get("series_id")
                  or record.get("gse_id") or None)
        for col in LABEL_COLS:
            res = self.collapse(record.get(col, NS), col, ctx, gse_id)
            out[col] = res["label"]
            out[f"{col}_id"] = res["id"]
            out[f"{col}_components"] = res["components"]
        return out

    # ── per-component resolution ────────────────────────────────────────
    def _resolve_one(self, label: str, col: str, context: str = "",
                     gse_id: str | None = None) -> dict:
        # ── Tier 0 — universal control-word canonicalization (Condition only) ──
        # Applied BEFORE any cache lookup so stale gse_phase2_canon rows
        # cannot bypass the surface-form unification. Universal biomedical
        # lab vocabulary; not eval-derived. Healthy / control / normal
        # sample-state IS a first-class Condition value — emit a canonical
        # surface form (so all variants converge in clustering) instead of
        # dropping to NS.
        if col == "Condition":
            canon = _CONDITION_CONTROL_CANONICAL.get(label.strip().lower())
            if canon is not None:
                return {"raw": label, "id": "", "name": canon,
                        "source": "condition_control_canonical"}

        # ── Tier 0 — generic Tissue placeholder stop-list ───────────────
        # Same role as the Condition control-word list, but for Tissue:
        # raws like 'tumor' / 'cells' / 'biopsy' carry no organ signal,
        # and any downstream MeSH / OOD-mesh resolution invents an organ that
        # isn't in the source. Return NS to preserve the lack-of-signal
        # honestly. Applied BEFORE the gse_canon lookup so a previously
        # poisoned cache row cannot bypass it.
        if col == "Tissue" and label.strip().lower() in _TISSUE_GENERIC_PLACEHOLDERS:
            return {"raw": label, "id": "", "name": NS,
                    "source": "tissue_generic_placeholder"}

        # ── Tier 0.5 — universal negation short-circuit ─────────────────
        # A grammatically negated raw (never X / no X / non-X / without X
        # / unaffected / un-X when it negates / etc.) means the entity
        # NEVER held for this subject — i.e. a missing observation. By
        # universal rule across all three columns, that is Not Specified.
        # We do NOT mint a separate "absence:<id>" lattice: "no
        # treatment" means no treatment at all (NaN), "no disease" means
        # disease state is unobserved, "no tissue" means no organ
        # asserted. The polarity verdict is cached per
        # (raw_lc, col, prompt_version) so the LLM call fires at most
        # once per unique surface and can be reused fleet-wide.
        if self.use_polarity:
            polarity = self._classify_polarity(label, col, context)
            if polarity == "NEGATE":
                return {"raw": label, "id": "", "name": NS,
                        "source": "negation_short_circuit"}

        # ── Tier 1.5 / 1.6 — per-GSE sibling consistency ─────────────────
        # When a cache + gse_id are available, siblings within the SAME
        # experiment must converge on ONE canonical id. We try three
        # progressively-fuzzier lookups before falling through to the
        # global cascade:
        #   1.5  exact-raw match in gse_phase2_canon
        #   1.6a morphology-rewritten raw ("renal tissue" → "kidney")
        #   1.6b token-overlap with any already-resolved sibling raw
        # Universal — relies only on adjective↔noun morphology + token
        # set comparison. No eval-entity hardcoding.
        if gse_id and self._cache is not None:
            raw_lc = label.strip().lower()
            hit = self._cache.get_gse_canon(gse_id, col, raw_lc)
            if hit:
                return {"raw": label, "id": hit["canon_id"],
                        "name": hit["canon_name"],
                        "source": f"gse_canon_exact(n={hit['n_uses']})"}

            aug_q = _augmented_query(label)
            if aug_q:
                hit = self._cache.get_gse_canon(gse_id, col, aug_q)
                if hit:
                    self._cache.set_gse_canon(
                        gse_id, col, raw_lc,
                        hit["canon_id"], hit["canon_name"])
                    return {"raw": label, "id": hit["canon_id"],
                            "name": hit["canon_name"],
                            "source": "gse_canon_morph"}

            raw_aug = _augment_raw_tokens(label)
            # Polarity guard: a negation-prefixed sibling and a bare-form
            # current label are NOT the same entity. "asthma" and "no
            # asthma diagnosis" pass naive token-set containment but
            # encode opposite truth values; without this guard, a single
            # negative-axis sibling minted earlier in the GSE would
            # capture all subsequent positive-axis siblings under its
            # OOD-mesh id (the GSE56553 asthma-study failure mode).
            _NEG_TOKENS = {"no", "not", "never", "absent", "negative",
                           "denied", "without", "unaffected", "neg",
                           "false", "non"}
            if raw_aug:
                raw_neg = bool(raw_aug & _NEG_TOKENS)
                for entry in self._cache.list_gse_canon(gse_id, col):
                    sib_aug = _augment_raw_tokens(entry["raw_lc"])
                    if not sib_aug:
                        continue
                    # Symmetric token-set containment after morphology
                    # augmentation. Either side fully contained in the
                    # other counts as "same entity, different surface" —
                    # but only when polarity matches.
                    if sib_aug <= raw_aug or raw_aug <= sib_aug:
                        sib_neg = bool(sib_aug & _NEG_TOKENS)
                        if raw_neg != sib_neg:
                            continue   # opposite-polarity sibling — skip
                        self._cache.set_gse_canon(
                            gse_id, col, raw_lc,
                            entry["canon_id"], entry["canon_name"])
                        return {"raw": label, "id": entry["canon_id"],
                                "name": entry["canon_name"],
                                "source": "gse_canon_overlap"}

        result = self._resolve_one_global(label, col, context)

        # Persist successful resolution back to the per-GSE canon so the
        # next sibling raw within this experiment gets the O(1) hit.
        if (gse_id and self._cache is not None
                and result.get("id") and result.get("name")):
            self._cache.set_gse_canon(
                gse_id, col, label,
                result["id"], result["name"])
        return result

    def _resolve_one_global(self, label: str, col: str,
                            context: str = "") -> dict:
        # Tier 1: episodic
        if self.use_episodic:
            history = self.db.get_resolution_history(label, col, k=1)
            if history:
                h = history[0]
                return {"raw": label, "id": h["output_id"],
                        "name": h["output_name"],
                        "source": f"episodic:{h['source']}"}

        # Tier 2: exact MeSH
        mesh_hits = self.db.lookup_mesh(label, col)
        if len(mesh_hits) == 1:
            return self._finalize_mesh(label, col, mesh_hits[0], "mesh-exact")
        if len(mesh_hits) > 1 and self.use_picker:
            chosen = self._pick(label, col, mesh_hits, context)
            if chosen is not None and self._verify_pick(label, col, chosen, context):
                return self._finalize_mesh(label, col, chosen, "mesh-exact-picked")

        # Tier 3: existing OOD-mesh entry
        ood = self.db.lookup_ood_mesh(label, col)
        if ood:
            self.db.record_resolution(label, col, ood["id"], ood["label"],
                                      "ood-mesh-existing")
            return {"raw": label, "id": ood["id"], "name": ood["label"],
                    "source": "ood-mesh-existing"}

        # Tier 4: hybrid candidate gathering + LLM picker
        if self.use_picker:
            cands: list[dict] = []
            seen_ids: set[str] = set()

            # 4a. PubTator3 (curated NCBI entity → MeSH).
            pt_cand = self._pt_candidate(label, col)
            if pt_cand and pt_cand["id"] not in seen_ids:
                cands.append(pt_cand)
                seen_ids.add(pt_cand["id"])

            # 4b. BioLORD top-K (semantic).
            for c in self.db.find_similar_mesh(label, col, k=self.top_k):
                if c["id"] in seen_ids:
                    continue
                cands.append(c)
                seen_ids.add(c["id"])

            # 4c. Anatomical-adjective augmentation. When the raw rewrites
            # to a different surface form ("renal tissue" → "kidney",
            # "pulmonary tissue" → "lung"), the original BioLORD query is
            # dominated by the histology stop tokens and may miss the
            # organ descriptor entirely. Issue an augmented exact-MeSH
            # lookup AND a secondary BioLORD query, merge any new
            # candidates in. Universal: pure adjective↔noun morphology.
            aug_q = _augmented_query(label)
            if aug_q:
                # Exact MeSH match on the rewritten form gets a high
                # confidence score so the lexical re-rank in _pick keeps
                # it at / near the top.
                for h in self.db.lookup_mesh(aug_q, col):
                    if h["id"] in seen_ids:
                        continue
                    h = {**h, "score": max(float(h.get("score") or 0.0), 0.95)}
                    cands.append(h)
                    seen_ids.add(h["id"])
                # Secondary BioLORD query on the rewritten form.
                for c in self.db.find_similar_mesh(aug_q, col, k=self.top_k):
                    if c["id"] in seen_ids:
                        continue
                    cands.append(c)
                    seen_ids.add(c["id"])

            if cands:
                chosen = self._pick(label, col, cands, context)
                if chosen is not None and self._verify_pick(label, col, chosen, context):
                    src = "mesh-pubtator" if chosen is pt_cand else "mesh-semantic"
                    return self._finalize_mesh(label, col, chosen, src)

        # Tier 4d: dose-strip retry (Treatment only).
        # When the picker returns NONE for a dose/duration-suffixed
        # compound (e.g. 'paclitaxel 10 nM 24 h'), the dose tokens
        # dominate the BioLORD embedding and pull MeSH candidates away
        # from the bare compound. Retry the cascade once with the
        # numeric dose / unit / duration tokens stripped. Universal —
        # purely numeric+unit lexicon, no drug-name hardcoding.
        if col == "Treatment" and self.use_picker:
            stripped = _strip_dose(label)
            if stripped and stripped.lower() != label.lower():
                # 4d-1: exact MeSH on bare compound.
                bare_hits = self.db.lookup_mesh(stripped, col)
                if len(bare_hits) == 1:
                    return self._finalize_mesh(
                        label, col, bare_hits[0], "mesh-dose-stripped-exact")
                if len(bare_hits) > 1:
                    chosen = self._pick(stripped, col, bare_hits, context)
                    if (chosen is not None
                            and self._verify_pick(stripped, col, chosen, context)):
                        return self._finalize_mesh(
                            label, col, chosen, "mesh-dose-stripped-picked")
                # 4d-2: PubTator + BioLORD on bare compound.
                bare_cands: list[dict] = []
                bare_seen: set[str] = set()
                pt2 = self._pt_candidate(stripped, col)
                if pt2:
                    bare_cands.append(pt2)
                    bare_seen.add(pt2["id"])
                for c in self.db.find_similar_mesh(stripped, col, k=self.top_k):
                    if c["id"] in bare_seen:
                        continue
                    bare_cands.append(c)
                    bare_seen.add(c["id"])
                if bare_cands:
                    chosen = self._pick(stripped, col, bare_cands, context)
                    if (chosen is not None
                            and self._verify_pick(stripped, col, chosen, context)):
                        src = ("mesh-dose-stripped-pubtator"
                               if chosen is pt2
                               else "mesh-dose-stripped-semantic")
                        return self._finalize_mesh(label, col, chosen, src)

        # Tier 5: mint new OOD-mesh entry
        minted = self.db.create_ood_mesh(label, col)
        self.db.record_resolution(label, col, minted["id"], minted["label"],
                                  "ood-mesh-minted")
        return {"raw": label, "id": minted["id"], "name": minted["label"],
                "source": "ood-mesh-minted"}

    def _pt_candidate(self, label: str, col: str) -> dict | None:
        """Run PubTator3 normalize and return one MeSH candidate (looked up
        in the local mesh.sqlite to enrich with name/scope/category) or None.

        We only emit the candidate when PT returns a MeSH ID that exists
        in our local MeSH DB AND whose category passes the col gate. PT
        misses (returns empty id) and out-of-category hits are dropped —
        the LLM picker only sees high-precision MeSH candidates.
        """
        if not self.use_pubtator or self._pt is None:
            return None
        try:
            r = self._pt.normalize(label, col)
        except Exception:                              # noqa: BLE001  network blip
            return None
        mid = (r or {}).get("id") or ""
        if not mid:
            return None
        row = self.db.con.execute(
            "SELECT id, name, category, scope FROM mesh_terms WHERE id = ?",
            (mid,),
        ).fetchone()
        if not row:
            return None
        cats = COL_CATS.get(col, ())
        if cats and row["category"] not in cats:
            return None
        return {"id": row["id"], "name": row["name"],
                "category": row["category"], "scope": row["scope"] or "",
                "score": 1.0}                           # PT is a curated hit

    def _finalize_mesh(self, label: str, col: str, hit: dict, source: str) -> dict:
        self.db.record_resolution(label, col, hit["id"], hit["name"], source)
        return {"raw": label, "id": hit["id"], "name": hit["name"],
                "source": source}

    # ── LLM picker ──────────────────────────────────────────────────────
    def _pick(self, label: str, col: str, candidates: list[dict],
              context: str = "") -> dict | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Lexical re-rank: pull candidates whose name tokens overlap
        # with the raw (after anatomical-adjective augmentation) into a
        # higher band. BioLORD score breaks ties within a band. This
        # fixes the generic-histology trap where "renal tissue" /
        # "pulmonary tissue" / "lung parenchyma" rank
        # 'Parenchymal Tissue' above 'Kidney' / 'Lung' on raw cosine
        # similarity. Universal — no eval-specific entity hardcoding.
        raw_aug = _augment_raw_tokens(label)

        def _rerank_key(c: dict) -> tuple:
            name_toks = _candidate_name_tokens(c)
            inter = raw_aug & name_toks
            # Coverage = how much of the candidate is explained by the
            # raw. Tighter banding so a fully-explained candidate
            # ('Lung' from raw 'pulmonary tissue', cov=1.0) outranks a
            # partially-explained sibling ('Pulmonary Alveoli',
            # cov=0.5) regardless of BioLORD score.
            coverage = len(inter) / max(1, len(name_toks))
            if not inter:
                band = 0
            elif coverage >= 1.0:
                band = 3
            elif coverage >= 0.5:
                band = 2
            else:
                band = 1
            return (-band, -float(c.get("score") or 0.0))

        candidates = sorted(candidates, key=_rerank_key)

        # Short-abbreviation substring guard. When the raw is ≤4 chars
        # (UC / IDC / OA / AD class), BioLORD frequently surfaces
        # MeSH descriptors that share a *substring* with the abbrev but
        # not a *token* (e.g. raw='UC' → candidate 'Urological
        # Manifestations'; raw='IDC' → 'Idiopathic Pulmonary Fibrosis').
        # If the top reranked candidate is band 0 (zero token overlap),
        # the LLM picker has no lexical anchor and tends to guess by
        # surface proximity. Fall through to mint instead — universal,
        # depends only on length + token-overlap, no entity hardcoding.
        if (candidates and len(label.strip()) <= 4):
            top = candidates[0]
            top_toks = _candidate_name_tokens(top)
            if not (raw_aug & top_toks):
                return None

        # Deterministic short-circuit: if the top candidate is a 1-token
        # MeSH descriptor that exactly matches the augmented raw token
        # set (e.g. raw='renal tissue' → aug={'renal','kidney'}, top
        # name='Kidney' → tokens={'kidney'} ⊆ aug), trust it without an
        # LLM call. This collapses adjective→organ + organ-noun cases
        # deterministically. Universal: depends only on adjective↔noun
        # morphology, not eval entities.
        if candidates and raw_aug:
            top = candidates[0]
            top_toks = _candidate_name_tokens(top)
            if (top_toks and len(top_toks) == 1
                    and top_toks.issubset(raw_aug)
                    and float(top.get("score") or 0.0) >= 0.45):
                return top

        # Cap candidates shown to the picker. With 30+ entries the
        # prompt becomes noisy and the LLM occasionally selects a
        # lower-banded distractor. Top-12 after re-rank always covers
        # the band-2 organ matches; band-0 generics never need to be
        # picked when a band-2 organ is available.
        candidates = candidates[:12]

        # Show category bracket + truncated scope so the LLM can apply
        # the col-branch rule (R4) and see the descriptor's actual meaning.
        lines = []
        for i, c in enumerate(candidates):
            scope = (c.get("scope") or "").strip().replace("\n", " ")
            if len(scope) > 220:
                scope = scope[:220].rsplit(" ", 1)[0] + "…"
            cat = f"[{c.get('category', '?')}]"
            lines.append(f"{i}. {c['name']} {cat}"
                         + (f" — {scope}" if scope else ""))
        ctx_block = _format_context_block(context)
        user = (
            f"{ctx_block}"
            f"raw label: {label}\n"
            f"column: {col}\n"
            f"candidates:\n  " + "\n  ".join(lines) + "\n\n"
            f"Pick:"
        )
        text = self._call_lm(_PICKER_SYSTEM, user).strip()
        # Prefer the explicit `PICK: <int|NONE>` terminal marker; fall back
        # to the LAST integer-or-NONE token if the marker is absent (older
        # gemma replies, or runaway reasoning that exceeds num_predict).
        m = re.search(r"PICK\s*:\s*(\d{1,2}|NONE)\b",
                      text, flags=re.IGNORECASE)
        tok: str | None = m.group(1).upper() if m else None
        if tok is None:
            ms = re.findall(r"\b(\d{1,2}|NONE)\b", text, flags=re.IGNORECASE)
            tok = ms[-1].upper() if ms else None
        if tok is None or tok == "NONE":
            return None
        try:
            i = int(tok)
        except ValueError:
            return None
        if 0 <= i < len(candidates):
            return candidates[i]
        return None

    # ── Tier 4.5 verifier ───────────────────────────────────────────────
    # ── Tier 0.5 polarity classifier ────────────────────────────────────
    def _classify_polarity(self, label: str, col: str,
                           context: str = "") -> str:
        """Return 'ASSERT' or 'NEGATE' for ``label`` in ``col`` context.

        Cached per (raw_lc, col, prompt_version) in
        ``polarity_decisions`` — first agent in the fleet to see a
        novel surface form pays the LLM call; every subsequent agent
        reads the cached verdict. On parse failure, defaults to
        ASSERT (conservative — keep the existing cascade).

        When ``context`` is provided AND the raw label contains a
        coded value (digit / paren-code / colon-code), the context is
        injected so an inline legend (e.g. '<field> (0,1,2 = level-A,
        level-B, level-C): X') can decode the code before
        classification. Coded raws are NOT cached — the same coded
        surface can encode different meanings in different studies;
        context-decoded verdicts are computed per call. Plain-prose
        raws are cached cluster-wide as before.
        """
        raw_lc = label.strip().lower()
        if not raw_lc:
            return "ASSERT"

        is_coded = bool(re.search(r"[\(:]\s*\d", raw_lc) or
                        re.fullmatch(r"\d+(\.\d+)?", raw_lc))

        if not is_coded:
            cached = self.db.get_polarity(raw_lc, col, _PROMPT_VERSION)
            if cached is not None:
                return cached

        ctx_block = _format_context_block(context) if (is_coded and context) else ""
        user = f"{ctx_block}raw label: {label}\ncolumn: {col}\n\nClassify polarity:"
        text = self._call_lm(_POLARITY_SYSTEM, user)
        m = re.search(r"POLARITY\s*:\s*(ASSERT|NEGATE)\b",
                      text, flags=re.IGNORECASE)
        polarity = m.group(1).upper() if m else "ASSERT"
        if not is_coded:
            self.db.cache_polarity(raw_lc, col, _PROMPT_VERSION, polarity)
        return polarity

    def _verify_pick(self, label: str, col: str, candidate: dict,
                     context: str = "") -> bool:
        """Self-check the picker's choice. Returns True to KEEP, False
        to REJECT (in which case the caller falls through to the next
        tier and ultimately mints an OOD-mesh entry). On parse failure, returns
        False (conservative).

        Cluster-scale path: consult the shared ``verifier_decisions``
        cache first — any agent in the fleet that already verified this
        ``(raw, col, picked_id)`` under the current prompt version saves
        every other agent the LLM call. Only on cache miss does the
        verifier LLM fire; the verdict is then written through for
        future fleet-wide reuse.
        """
        if not self.use_verifier:
            return True

        raw_lc    = label.strip().lower()
        picked_id = candidate.get("id", "")
        if picked_id:
            cached = self.db.get_verifier_verdict(
                raw_lc, col, picked_id, _PROMPT_VERSION,
            )
            if cached is not None:
                return cached == "KEEP"

        scope = (candidate.get("scope") or "").strip().replace("\n", " ")
        if len(scope) > 280:
            scope = scope[:280].rsplit(" ", 1)[0] + "…"
        cat = candidate.get("category", "?")
        ctx_block = _format_context_block(context)
        user = (
            f"{ctx_block}"
            f"raw label: {label}\n"
            f"column: {col}\n"
            f"picked: {candidate['name']} [{cat}]\n"
            f"scope: {scope or '(no scope note)'}\n\n"
            f"Run R1..R5 then output the VERDICT line."
        )
        text = self._call_lm(_VERIFIER_SYSTEM, user)
        m = re.search(r"VERDICT\s*:\s*(KEEP|REJECT)", text, flags=re.IGNORECASE)
        verdict = m.group(1).upper() if m else "REJECT"
        if picked_id:
            self.db.cache_verifier_verdict(
                raw_lc, col, picked_id, _PROMPT_VERSION, verdict,
            )
        return verdict == "KEEP"

    # ── cross-GSE promotion ─────────────────────────────────────────────
    def promote_global_canons(self, min_gses: int = 3) -> int:
        """Mirror unanimous per-GSE canonical decisions into the global
        episodic table. Call periodically (e.g. at end of a pipeline run)
        so future first-time studies see cluster-wide consensus on
        Tier 1 (episodic) and skip the LLM cascade entirely.

        Returns the number of (col, raw_lc) entries promoted.

        Safe — ``list_promote_candidates`` already drops any (col, raw_lc)
        with disagreement across GSEs. Idempotent: re-running just
        refreshes ``record_resolution`` rows in place.
        """
        if self._cache is None:
            return 0
        cands = self._cache.list_promote_candidates(min_gses=min_gses)
        n = 0
        for c in cands:
            try:
                self.db.record_resolution(
                    c["raw_lc"], c["col"],
                    c["canon_id"], c["canon_name"],
                    f"global-promote(n_gses={c['n_gses']})")
                n += 1
            except Exception as e:                          # pragma: no cover
                print(f"[phase2_mesh] promote skipped {c['raw_lc']!r}: {e!r}",
                      flush=True)
        return n

    @staticmethod
    def _call_lm(system: str, user: str) -> str:
        # vLLM opt-in (only when explicitly enabled). Falls through to
        # the original Ollama call below otherwise — Ollama path is
        # byte-identical to the pre-vLLM implementation.
        if os.environ.get("LLM_BACKEND", "ollama").lower() == "vllm":
            from llm_backend import chat as _vllm_chat
            return _vllm_chat(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                model=_LLM_MODEL, temperature=0.0, seed=0,
                num_predict=2048, num_ctx=_LLM_NUM_CTX,
                timeout=180, retries=3,
            )

        if _requests is None:
            raise RuntimeError("phase2_mesh: `requests` not installed")
        # gemma4 native reasoning is the pipeline-wide default per user
        # policy — pillar of extraction quality. Override only via the
        # THINK_MODE=false env var (the GUI exposes a checkbox for this).
        _think = os.environ.get("THINK_MODE", "true").lower() in ("1","true","yes")
        body = {
            "model":      _LLM_MODEL,
            "messages":   [{"role": "system", "content": system},
                           {"role": "user",   "content": user}],
            "stream":     False,
            "think":      _think,
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "seed":        0,
                "num_predict": 2048 if _think else 512,
                "num_ctx":     _LLM_NUM_CTX,
            },
        }
        r = _requests.post(_OLLAMA_URL.rstrip("/") + "/api/chat",
                           json=body, timeout=180)
        r.raise_for_status()
        data: Any = r.json()
        msg = data.get("message", {}) or {}
        # gemma4 with think:true splits output into `thinking` + `content`.
        # When num_predict is exhausted by reasoning, `content` can come back
        # empty — concatenate thinking so the caller's PICK/VERDICT regex
        # can still recover the answer if it appears in the chain-of-thought.
        return (msg.get("content") or "") + "\n" + (msg.get("thinking") or "")


__all__ = ["Phase2Mesh", "NS", "LABEL_COLS"]
