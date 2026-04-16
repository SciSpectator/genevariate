"""
Extraction - LLM prompt templates, parsers, and deterministic collapse.

Aligned with upstream LLM-Label-Extractor v2.2 (multi-value extraction
with semicolons, coded-value disambiguation for Condition/Treatment,
per-label agents, KV-cached system prompts for Phase 1c).

Contains:
    - Extraction prompt templates for raw GEO metadata
    - JSON response parsers (semicolon-aware multi-value)
    - Phase 1.5 deterministic GSE-scoped label collapsing
    - Text cleaning and NS detection utilities
"""

import re
import json
from typing import Dict, List

NS = "Not Specified"
LABEL_COLS = ["Tissue", "Condition"]
LABEL_COLS_SCRATCH = ["Tissue", "Condition", "Treatment"]

EXTRACTION_MODEL = "gemma4:e2b"   # gemma4 edge model — better reasoning, think=false for speed

# ── Per-label extraction prompts (gemma4:e2b, independent agents) ──
# Each label has its own focused prompt — no cross-label contamination.

_TISSUE_EXTRACT_PROMPT = (
    "TASK: Read ALL metadata below (Title, Source, Characteristics, Description, "
    "Experiment context) and extract the TISSUE / CELL TYPE / CELL LINE.\n"
    "Copy exactly what is written — do NOT normalise or generalise.\n"
    "\n"
    "SCAN EVERYTHING — tissue info can appear ANYWHERE:\n"
    "  - In Source field (most common location)\n"
    "  - In Characteristics (e.g. 'tissue: liver', 'cell type: T cells')\n"
    "  - In Description (e.g. 'RNA from brain cortex tissue')\n"
    "  - In Title (but NEVER extract sample IDs, patient codes, or batch numbers)\n"
    "\n"
    "WHAT TO EXTRACT (priority order):\n"
    "  1. Named CELL LINE (e.g. MCF-7, HeLa, Jurkat, A549, THP-1, HL-60)\n"
    "  2. CELL TYPE (e.g. CD4+ T cells, monocytes, fibroblasts, NK cells)\n"
    "  3. TISSUE / ORGAN (e.g. liver, brain, whole blood, skin biopsy)\n"
    "\n"
    "RULES:\n"
    "  - Read the ENTIRE metadata — do NOT skip any field\n"
    "  - Copy the MOST SPECIFIC term (e.g. Alveolar Macrophages not Lung)\n"
    "  - If a cell type is named, use the cell type (e.g. NK Cells not PBMC)\n"
    "  - Cancer/tumor tissue IS a valid tissue (e.g. 'breast tumor', 'colon carcinoma')\n"
    "  - 'whole blood', 'peripheral blood', 'liver', etc. ARE valid tissues\n"
    "  - If MULTIPLE tissues/cell types apply → list ALL separated by semicolons\n"
    "  - If unknown or genuinely absent from ALL fields = Not Specified\n"
    "  - Title Case.\n"
    "METADATA:\n  Title: {TITLE}\n  Source: {SOURCE}\n  Characteristics: {CHAR}\n"
    "ANSWER (tissue/cell type/cell line only, nothing else):"
)
_CONDITION_EXTRACT_PROMPT = (
    "TASK: Read ALL metadata below (Title, Source, Characteristics, Description, "
    "Experiment context) and extract the CONDITION / DISEASE STATE.\n"
    "Copy exactly what is written — do NOT normalise or generalise.\n"
    "\n"
    "SCAN EVERYTHING — disease info can appear ANYWHERE:\n"
    "  - In Title (e.g. 'Breast Cancer Patient 5')\n"
    "  - In Description (e.g. 'tumor tissue of pancreatic adenocarcinoma patient')\n"
    "  - In Characteristics key-value fields (e.g. 'disease state: X', 'diagnosis: X')\n"
    "  - In Experiment title/summary (e.g. 'Gene expression in Alzheimer's disease')\n"
    "\n"
    "WHAT TO EXTRACT (any of these count as condition):\n"
    "  - Any cancer/carcinoma/adenocarcinoma/sarcoma/lymphoma/leukemia/tumor\n"
    "  - Any disease name (e.g. Multiple Sclerosis, Asthma, HIV, Diabetes)\n"
    "  - Any infection (e.g. 'infected with Neisseria meningitidis')\n"
    "  - Phenotype (e.g. Obese, Morbidly Obese)\n"
    "  - Syndrome (e.g. Down Syndrome, CAIS = Complete Androgen Insensitivity Syndrome)\n"
    "  - Smoking status: 'smoking: 0' → Never Smoker, '1' → Former, '2' → Current\n"
    "  - Control / Healthy / Normal in a disease study = Control\n"
    "\n"
    "IMPORTANT — GEO metadata often uses CODED values (0/1, Y/N, Yes/No, True/False)\n"
    "to indicate presence or absence of a condition. The field NAME tells you WHAT\n"
    "the condition is, and the VALUE tells you whether this sample HAS it or not.\n"
    "  - Value indicates ABSENCE (0, N, No, None, negative, False, non-) → Control\n"
    "  - Value indicates PRESENCE (1, Y, Yes, positive, True) → extract the condition\n"
    "    name FROM THE FIELD NAME, not the coded value itself\n"
    "  - Numeric scales (0/1/2/3) often encode severity or categories — read the field\n"
    "    description to understand what each number means\n"
    "\n"
    "RULES:\n"
    "  - Read the ENTIRE metadata — do NOT skip any field\n"
    "  - Copy the MOST SPECIFIC condition name present\n"
    "  - If sample is healthy/control/normal in a disease study = Control\n"
    "  - If MULTIPLE conditions apply → list ALL separated by semicolons\n"
    "  - If unknown or genuinely absent from ALL fields = Not Specified\n"
    "  - Title Case.\n"
    "METADATA:\n  Title: {TITLE}\n  Source: {SOURCE}\n  Characteristics: {CHAR}\n"
    "ANSWER (condition/disease/status only, nothing else):"
)
_TREATMENT_EXTRACT_PROMPT = (
    "TASK: Read ALL metadata below (Title, Source, Characteristics, Description, "
    "Experiment context) and extract the TREATMENT / DRUG / INTERVENTION.\n"
    "Treatment = something DONE TO or GIVEN TO the patient/sample.\n"
    "\n"
    "SCAN EVERYTHING — treatment info can appear ANYWHERE:\n"
    "  - In Characteristics (e.g. 'treatment: Dexamethasone', 'compound: X')\n"
    "  - In Description (e.g. 'cells treated with 10nM estradiol')\n"
    "  - In Title (e.g. 'MCF7_Tamoxifen_24h')\n"
    "  - In Experiment summary (e.g. 'effect of drug X on gene expression')\n"
    "\n"
    "CORRECT EXAMPLES:\n"
    "  'treatment: Dexamethasone 10nM' → Dexamethasone 10nM\n"
    "  'compound: Carfilzomib' → Carfilzomib\n"
    "  'bariatric surgery: 1' → Bariatric Surgery\n"
    "  'infected with Neisseria meningitidis' → Neisseria Meningitidis Infection\n"
    "  'smoking: 1' or 'current/former smoker' → Smoking\n"
    "  'smoking: 0' or 'never smoker' → Not Specified (no exposure)\n"
    "  'treatment: vehicle' → Untreated\n"
    "\n"
    "NOT treatments (output Not Specified):\n"
    "  - Diseases/conditions (HIV+, Depression, Down syndrome, cancer)\n"
    "  - Tissues/organs (blood, liver, brain)\n"
    "  - Lab protocols (Illumina, TRIzol, RNA extraction, FFPE)\n"
    "  - Sample IDs, batch codes, patient identifiers\n"
    "\n"
    "IMPORTANT — coded values (0/1, Y/N, None) in treatment fields indicate\n"
    "presence or absence. If the value means NO treatment was applied → Not Specified.\n"
    "If the value means treatment WAS applied → extract the treatment name from context.\n"
    "\n"
    "RULES:\n"
    "  - Read the ENTIRE metadata — do NOT skip any field\n"
    "  - If no drug/compound/exposure was applied → Not Specified\n"
    "  - If MULTIPLE treatments apply → list ALL separated by semicolons\n"
    "  - Title Case.\n"
    "METADATA:\n  Title: {TITLE}\n  Source: {SOURCE}\n  Characteristics: {CHAR}\n"
    "ANSWER (treatment/drug/intervention only, nothing else):"
)
PER_LABEL_EXTRACT_PROMPTS = {
    "Tissue":    _TISSUE_EXTRACT_PROMPT,
    "Condition": _CONDITION_EXTRACT_PROMPT,
    "Treatment": _TREATMENT_EXTRACT_PROMPT,
}

# ── Per-label NS inference system prompts (KV-cached by Ollama) ──
_TISSUE_INFER_SYSTEM = (
    "You are a tissue/cell-type annotator. This sample is part of "
    "the following experiment:\n\n"
    "EXPERIMENT TITLE: {GSE_TITLE}\n"
    "EXPERIMENT SUMMARY: {GSE_SUMMARY}\n"
    "EXPERIMENT DESIGN: {GSE_DESIGN}\n\n"
    "Based on this experiment context, INFER the tissue or cell type.\n"
    "If the experiment uses a specific tissue, this sample is from that tissue.\n"
    "Be specific. Unknown = Not Specified.\n"
    "If MULTIPLE tissues/cell types → list ALL separated by semicolons."
)
_CONDITION_INFER_SYSTEM = (
    "You are a disease/condition annotator. This sample is part of "
    "the following experiment:\n\n"
    "EXPERIMENT TITLE: {GSE_TITLE}\n"
    "EXPERIMENT SUMMARY: {GSE_SUMMARY}\n"
    "EXPERIMENT DESIGN: {GSE_DESIGN}\n\n"
    "Based on this experiment context, INFER the disease or condition.\n"
    "If the experiment studies a disease, this sample has that condition.\n"
    "Control/healthy in disease study = Control.\n"
    "Be specific. Unknown = Not Specified.\n"
    "If MULTIPLE conditions → list ALL separated by semicolons."
)
_TREATMENT_INFER_SYSTEM = (
    "You are a treatment/drug annotator. This sample is part of "
    "the following experiment:\n\n"
    "EXPERIMENT TITLE: {GSE_TITLE}\n"
    "EXPERIMENT SUMMARY: {GSE_SUMMARY}\n"
    "EXPERIMENT DESIGN: {GSE_DESIGN}\n\n"
    "Was a drug, compound, or exposure APPLIED in this experiment?\n"
    "Treatment = drug, compound, surgical procedure, or active exposure (e.g. smoking).\n"
    "Disease names are NEVER treatments: HIV, Depression, Down Syndrome, MS,\n"
    "Cancer, Asthma, NAFLD, PSP, Obesity = these are CONDITIONS not treatments.\n"
    "'Control' or 'Normal' = NOT a treatment.\n"
    "Observational disease-vs-control studies have NO treatment → Not Specified.\n"
    "Vehicle/DMSO/PBS alone = Untreated. No drug applied = Not Specified.\n"
    "If MULTIPLE treatments → list ALL separated by semicolons."
)
PER_LABEL_INFER_SYSTEMS = {
    "Tissue":    _TISSUE_INFER_SYSTEM,
    "Condition": _CONDITION_INFER_SYSTEM,
    "Treatment": _TREATMENT_INFER_SYSTEM,
}

# ── Phase 1c: Per-label SYSTEM prompts (KV-cached, full-metadata re-extraction) ──
# For samples still NS after Phase 1a+1b. Uses system/user split:
#   - System prompt: domain expertise + rules (KV-cached by Ollama across calls)
#   - User prompt: full metadata with NO character limits (Description, Summary included)
# This catches labels missed by the truncated Phase 1a prompts.

_TISSUE_SYSTEM_PROMPT = (
    "You are a biomedical tissue extraction specialist. Your sole task is to "
    "identify the biological tissue, cell type, or cell line from GEO sample metadata.\n"
    "\n"
    "YOUR EXPERTISE:\n"
    "- You know all human anatomical tissues and organs\n"
    "- You know cell types: immune cells, epithelial, endothelial, stem cells, etc.\n"
    "- You know cell lines: HeLa (cervix), MCF-7 (breast), A549 (lung), HL-60 (blood),\n"
    "  THP-1 (monocyte), Jurkat (T-cell), K562 (CML), HEK-293 (kidney), etc.\n"
    "- You understand brain subregions: hippocampus, prefrontal cortex, cerebellum, etc.\n"
    "\n"
    "HOW TO READ THE METADATA:\n"
    "- Title: sample name, may contain tissue info or just an ID\n"
    "- Source: MOST RELIABLE field for tissue — check this first\n"
    "- Characteristics: key-value pairs, look for 'tissue:', 'cell type:', 'cell line:'\n"
    "- Description: free-text, may describe the biological material\n"
    "- Experiment context: the parent study's title and summary\n"
    "\n"
    "EXTRACTION RULES:\n"
    "1. Extract the MOST SPECIFIC tissue/cell type present\n"
    "2. Priority: Cell Line > Cell Type > Tissue/Organ\n"
    "3. Copy exactly what the text says — do NOT rename or generalize\n"
    "4. For tumor/cancer samples, extract the TISSUE not the disease\n"
    "   (e.g. 'breast tumor' → Breast, 'colon carcinoma tissue' → Colon)\n"
    "5. If NOTHING in any field indicates a tissue → Not Specified\n"
    "6. NEVER extract sample IDs, patient codes, or batch numbers from Title\n"
    "7. Title Case. One answer only. No explanation.\n"
    "\n"
    "EXAMPLES:\n"
    "  'source: whole blood' → Whole Blood\n"
    "  'cell type: CD4+ T cells' → CD4+ T Cells\n"
    "  'tissue: liver biopsy' → Liver Biopsy\n"
    "  'MCF-7 cell line treated with...' → MCF-7\n"
    "  'description: RNA from prefrontal cortex' → Prefrontal Cortex\n"
    "  'description: pancreatic adenocarcinoma patient' → Pancreas\n"
    "  Title is just 'Sample_001', no other info → Not Specified"
)

_CONDITION_SYSTEM_PROMPT = (
    "You are a biomedical condition extraction specialist. Your sole task is to "
    "identify the disease state, condition, or phenotype from GEO sample metadata.\n"
    "\n"
    "YOUR EXPERTISE:\n"
    "- You know all major human diseases, syndromes, and conditions\n"
    "- You recognize cancer types: carcinoma, adenocarcinoma, sarcoma, lymphoma,\n"
    "  leukemia, melanoma, glioma, neuroblastoma, myeloma\n"
    "- You recognize infections: bacterial (Neisseria, Staphylococcus), viral (HIV,\n"
    "  HCV, influenza), parasitic\n"
    "- You recognize genetic conditions: cystic fibrosis (CFTR mutations), Down\n"
    "  syndrome, Marfan syndrome, CAIS, Huntington's disease\n"
    "- You recognize phenotypes: obese, diabetic, hypertensive, smoker\n"
    "- 'Normal', 'Control', 'Healthy', 'Unaffected', 'Wild-type' ALL → Control\n"
    "\n"
    "HOW TO READ THE METADATA:\n"
    "- Title: may contain disease name, patient ID, or condition code\n"
    "- Source: rarely has condition info but check anyway\n"
    "- Characteristics: look for 'disease state:', 'diagnosis:', 'condition:',\n"
    "  'group:', 'subject status:', 'disease:', 'phenotype:', 'smoking:'\n"
    "- Description: free-text — diseases often described here\n"
    "- Experiment context: the parent study reveals the disease being studied\n"
    "\n"
    "CRITICAL: Scan ALL fields. Disease info can hide ANYWHERE:\n"
    "  'tumor tissue of pancreatic adenocarcinoma patient' → in Description\n"
    "  'CAIS' → abbreviation in Title\n"
    "  'infected with Neisseria meningitidis' → infection in Description\n"
    "\n"
    "EXTRACTION RULES:\n"
    "1. Extract the MOST SPECIFIC condition/disease present\n"
    "2. Copy exactly what the text states — do NOT rename or generalize\n"
    "3. Abbreviations are valid: 'CAIS', 'AML', 'HCC', 'T2D' — extract them\n"
    "4. Mutations that imply disease: 'CFTR D508' → CFTR D508\n"
    "5. Smoking: 'smoking: 0' → Never Smoker, '1' → Former, '2' → Current\n"
    "6. If sample is healthy/control/normal → Control\n"
    "7. If NO disease/condition info in ANY field → Not Specified\n"
    "8. Title Case. One answer only. No explanation.\n"
    "\n"
    "EXAMPLES:\n"
    "  'disease state: hepatocellular carcinoma' → Hepatocellular Carcinoma\n"
    "  'description: infected with Neisseria meningitidis' → Neisseria Meningitidis Infection\n"
    "  Title: 'ARD842, CAIS, F, GOF' → CAIS\n"
    "  'description: pancreatic ductal adenocarcinoma patient' → Pancreatic Ductal Adenocarcinoma\n"
    "  'group: control' → Control\n"
    "  No disease info anywhere → Not Specified"
)

_TREATMENT_SYSTEM_PROMPT = (
    "You are a biomedical treatment extraction specialist. Your sole task is to "
    "identify any drug, compound, intervention, or exposure applied to a GEO sample.\n"
    "Treatment = something DONE TO or GIVEN TO the patient/sample.\n"
    "\n"
    "YOUR EXPERTISE:\n"
    "- You know pharmaceutical drugs: Dexamethasone, Tamoxifen, Methotrexate, etc.\n"
    "- You know compounds: DMSO (vehicle), estradiol, retinoic acid, TGF-beta\n"
    "- You know interventions: surgery, radiation, gene knockdown (siRNA, shRNA, CRISPR)\n"
    "- You know exposures: smoking, UV irradiation, hypoxia, heat shock\n"
    "- Infections as experimental treatment: 'infected with virus X'\n"
    "- Vehicle controls: DMSO, PBS, saline, ethanol → Untreated\n"
    "\n"
    "HOW TO READ THE METADATA:\n"
    "- Characteristics: look for 'treatment:', 'compound:', 'drug:', 'agent:',\n"
    "  'stimulus:', 'exposure:', 'transfection:'\n"
    "- Description: free-text — often describes treatment in detail\n"
    "- Title: may encode treatment (e.g. 'MCF7_Tamoxifen_24h')\n"
    "- Experiment context: describes the treatment protocol\n"
    "\n"
    "NOT treatments (output Not Specified):\n"
    "- Diseases/conditions (HIV+, Depression, cancer)\n"
    "- Tissues/organs (blood, liver, brain)\n"
    "- Lab protocols (Illumina, TRIzol, RNA extraction, FFPE, RNAlater)\n"
    "- Sample IDs, batch codes, patient identifiers\n"
    "\n"
    "EXTRACTION RULES:\n"
    "1. Extract drug/compound/intervention with dose if present\n"
    "2. Vehicle controls (DMSO, PBS, saline) → Untreated\n"
    "3. 'smoking: 0' or 'never smoker' → Not Specified (no exposure)\n"
    "4. If NO treatment applied → Not Specified\n"
    "5. Title Case. One answer only. No explanation.\n"
    "\n"
    "EXAMPLES:\n"
    "  'treatment: Dexamethasone 10nM' → Dexamethasone 10nM\n"
    "  'compound: Carfilzomib' → Carfilzomib\n"
    "  'infected with influenza A' → Influenza A Infection\n"
    "  'treatment: vehicle (DMSO)' → Untreated\n"
    "  No treatment mentioned → Not Specified"
)

PER_LABEL_SYSTEM_PROMPTS = {
    "Tissue":    _TISSUE_SYSTEM_PROMPT,
    "Condition": _CONDITION_SYSTEM_PROMPT,
    "Treatment": _TREATMENT_SYSTEM_PROMPT,
}

# User prompt template for Phase 1c — metadata only, NO char limits
EXTRACT_USER_TEMPLATE = (
    "Title: {TITLE}\n"
    "Source: {SOURCE}\n"
    "Characteristics: {CHAR}"
)

_NS_TOKENS = ('none', 'null', '', 'not specified', 'n/a', 'unknown', 'na')


def _clean_value(v: str) -> str:
    """Strip wrapping punctuation and surrounding whitespace from a single value."""
    return str(v).strip().strip('"').strip("'").rstrip('.').strip()


def parse_single_label(text: str) -> str:
    """Parse label(s) from a per-label LLM agent response.

    Aligned with upstream LLM-Label-Extractor v2.2: supports MULTIPLE values
    separated by semicolons. Each value is cleaned individually and the
    deduplicated list is rejoined with '; '. NS sentinels are dropped.
    Falls back to JSON parsing if the response contains braces.
    """
    if not text:
        return NS
    text = text.strip()
    # If response contains JSON, extract the value(s)
    if '{' in text:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                vals = []
                for v in data.values():
                    v = _clean_value(v)
                    if v and v.lower() not in _NS_TOKENS:
                        # JSON values themselves may contain semicolons
                        for piece in v.split(';'):
                            piece = _clean_value(piece)
                            if piece and piece.lower() not in _NS_TOKENS \
                                    and piece not in vals:
                                vals.append(piece)
                if vals:
                    return '; '.join(vals)
            except Exception:
                pass
    # Plain text response — take the first non-empty answer line
    for line in text.splitlines():
        line = line.strip().rstrip('.')
        if line.lower().startswith(('answer:', 'tissue:', 'condition:',
                                     'treatment:', 'note:', 'rules:')):
            line = line.split(':', 1)[1].strip() if ':' in line else ''
        if not line or line.lower() in _NS_TOKENS:
            continue
        # Multi-value support: split on ';' and dedupe while preserving order
        pieces = [_clean_value(p) for p in line.split(';')]
        pieces = [p for p in pieces if p and p.lower() not in _NS_TOKENS]
        if pieces:
            seen = []
            for p in pieces:
                if p not in seen:
                    seen.append(p)
            return '; '.join(seen)
    return NS


# ── Per-label collapse prompts (single focused LLM call, replaces multi-turn ReAct) ──
_TISSUE_COLLAPSE_PROMPT = (
    "Map this tissue label to the best matching cluster name from the candidates.\n"
    "LABEL: {RAW_LABEL}\n"
    "CANDIDATES:\n{CANDIDATES}\n"
    "SIBLINGS:\n{SIBLING_LABELS}\n"
    "Pick the candidate that best matches the label. Cell type ≠ organ.\n"
    "If a candidate matches, output that exact candidate name.\n"
    "If NO candidate matches, output the label exactly as-is: {RAW_LABEL}\n"
    "One name only:"
)
_CONDITION_COLLAPSE_PROMPT = (
    "Map this condition label to the best matching cluster name from the candidates.\n"
    "LABEL: {RAW_LABEL}\n"
    "CANDIDATES:\n{CANDIDATES}\n"
    "SIBLINGS:\n{SIBLING_LABELS}\n"
    "Pick the candidate that best matches the label.\n"
    "If a candidate matches, output that exact candidate name.\n"
    "If NO candidate matches, output the label exactly as-is: {RAW_LABEL}\n"
    "One name only:"
)
_TREATMENT_COLLAPSE_PROMPT = (
    "Map this treatment label to the best matching cluster name from the candidates.\n"
    "LABEL: {RAW_LABEL}\n"
    "CANDIDATES:\n{CANDIDATES}\n"
    "SIBLINGS:\n{SIBLING_LABELS}\n"
    "Pick the candidate that best matches the label.\n"
    "Vehicle/DMSO/PBS = Untreated. Smoking/tobacco exposure = Smoking.\n"
    "Disease names (HIV, Cancer, MS, Depression) are NOT treatments → Not Specified.\n"
    "If a candidate matches, output that exact candidate name.\n"
    "If NO candidate matches, output the label exactly as-is: {RAW_LABEL}\n"
    "One name only:"
)
PER_LABEL_COLLAPSE_PROMPTS = {
    "Tissue":    _TISSUE_COLLAPSE_PROMPT,
    "Condition": _CONDITION_COLLAPSE_PROMPT,
    "Treatment": _TREATMENT_COLLAPSE_PROMPT,
}

# Legacy combined extraction prompt — kept for backward compatibility
EXTRACTION_PROMPT_TEMPLATE = (
    "TASK: Read the metadata below and extract exactly what is written.\n"
    "Do NOT normalise, generalise, or map to any vocabulary — copy the specific term.\n"
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

def sanitize(text, max_chars: int = -1) -> str:
    """Clean control chars; max_chars=-1 means no truncation."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', str(text or ""))
    text = text.replace('\r', ' ').strip()
    return text if max_chars < 0 else text[:max_chars]


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
    # NO truncation — full metadata goes to the LLM
    title = _s(raw.get("gsm_title", ""))
    source = _s(raw.get("source_name", ""))
    char = _s(raw.get("characteristics", ""))
    treat = _s(raw.get("treatment_protocol", ""))
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
        gse_hint += f"Experiment summary: {ctx.summary}\n"
    if getattr(ctx, "design", ""):
        gse_hint += f"Overall design    : {ctx.design}\n"
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
