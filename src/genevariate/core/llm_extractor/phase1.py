"""Phase 1 verbatim-extraction runtime.

Loads the per-label prompts previously optimised offline by
``compile_phase1.py`` from ``compiled/*.json`` and renders them into
chat messages in the ``[[ ## field ## ]]`` layout. The LM itself is
called with stdlib HTTP.

Public API:
    * ``NS``                     — sentinel "Not Specified" string
    * ``extract_tissue(raw)``    — per-sample tissue span
    * ``extract_condition(raw)`` — per-sample condition/disease span
    * ``extract_treatment(raw)`` — per-sample treatment span
    * ``Phase1Agent``            — ``.extract(raw)`` → {Tissue, Condition, Treatment}

Environment variables:
    * ``PHASE1_BACKEND``   — ``ollama`` (default) or ``sglang``
    * ``PHASE1_MODEL``     — default ``gemma4:e2b``
    * ``OLLAMA_URL``       — default ``http://localhost:11434``
    * ``SGLANG_URL``       — default ``http://localhost:30000/v1``

Reproducibility: temperature=0, seed=42, num_predict=-1.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

NS = "Not Specified"

# ─────────────────────────────────────────────────────────────────────────────
# Backend configuration — same env vars as the old DSPy module.
# ─────────────────────────────────────────────────────────────────────────────
_BACKEND = os.environ.get("PHASE1_BACKEND", "ollama").lower()
_MODEL   = os.environ.get("PHASE1_MODEL", "gemma4-e2b-text:latest")

if _BACKEND == "sglang":
    _SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000/v1")
    _CHAT_URL   = _SGLANG_URL.rstrip("/") + "/chat/completions"
    _IS_OPENAI  = True
else:
    _OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    _CHAT_URL   = _OLLAMA_URL.rstrip("/") + "/api/chat"
    _IS_OPENAI  = False


# ─────────────────────────────────────────────────────────────────────────────
# Schemas — the old dspy.Signature field lists, now as plain tuples.
# (name, type, description) mirrors dspy.InputField/OutputField exactly so the
# rendered system message is byte-equivalent to DSPy's ChatAdapter output.
# ─────────────────────────────────────────────────────────────────────────────
_Field = Tuple[str, str, str]

_SIG_TISSUE: Dict = {
    "artifact": "extract_tissue.json",
    "inputs": [
        ("title",              "str", "Sample title."),
        ("source",             "str", "Sample source_name field."),
        ("characteristics",    "str", "Sample characteristics field — usually key:value pairs separated by ';' or '|'."),
        ("treatment_protocol", "str", "Sample treatment/lab protocol text."),
        ("description",        "str", "Sample description."),
    ],
    "outputs": [
        ("tissue", "str", "Verbatim tissue/organ/cell span, or 'Not Specified'."),
    ],
}

_SIG_CONDITION: Dict = {
    "artifact": "extract_condition.json",
    "inputs": [
        ("title",              "str", "Sample title."),
        ("source",             "str", "Sample source_name field."),
        ("characteristics",    "str", "Sample characteristics field — usually key:value pairs."),
        ("treatment_protocol", "str", "Sample treatment/lab protocol text."),
        ("description",        "str", "Sample description."),
    ],
    "outputs": [
        ("condition", "str", "Verbatim condition / disease / phenotype, or 'Not Specified'."),
    ],
}

_SIG_TREATMENT: Dict = {
    "artifact": "extract_treatment.json",
    "inputs": [
        ("title",              "str", "Sample title."),
        ("source",             "str", "Sample source_name field."),
        ("characteristics",    "str", "Sample characteristics field — usually key:value pairs."),
        ("treatment_protocol", "str", "Sample treatment/lab protocol text — may contain lab handling (not a treatment) or the actual experimental treatment. Read carefully."),
        ("description",        "str", "Sample description."),
    ],
    "outputs": [
        ("treatment", "str", "Verbatim treatment / drug / intervention, or 'Not Specified'."),
    ],
}

_SIG_RESOLVE: Dict = {
    "artifact": "resolve_coded_value.json",  # may not exist → fallback used
    "inputs": [
        ("field",         "str", "Which label this is — one of 'Tissue', 'Condition', 'Treatment'. Helps disambiguate which legend slot applies."),
        ("raw_value",     "str", "The verbatim value just extracted by the Phase 1 per-label signature."),
        ("metadata_blob", "str", "All sample metadata concatenated (title, source, characteristics, treatment_protocol, description). Search HERE for a literal legend."),
    ],
    "outputs": [
        ("resolved", "str", "raw_value with coded tokens substituted by legend-text meanings, OR raw_value byte-identical when no legend applies, OR 'Not Specified' when resolution collapses to nothing."),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt fallbacks — used only when the compiled artifact is missing or
# corrupt. These mirror the Signature docstrings from the original DSPy
# module so behaviour is preserved even without the ``compiled/`` files.
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_TISSUE = """Extract the single best TISSUE / ORGAN / CELL-TYPE / CELL-LINE span for
THIS sample, copied VERBATIM from the metadata text.

Hierarchy (walk in order, stop at the first non-empty source):
  1. `cell line:` — the value IS the span.
  2. `cell type:` — only consulted when `cell line:` is absent.
  3. `tissue:` — only consulted when both cell_line and cell_type are absent.
  4. `organ:` / `anatomical site:` / `biopsy site:` — only when 1-3 absent.
  5. `source_name` — only when 1-4 absent. Extract verbatim.
  6. title / description — last resort.

Compound "<Organ> Cancer/Tumor/Carcinoma/Neoplasm": return the ORGAN noun alone
for spans with an EXPLICIT organ noun. Keep abbreviated cancers (OSCC, HCC,
TNBC, ...) and cell-line IDs whole.

If no tissue / organ / cell-type / cell-line is named, output: Not Specified.
"""

_FALLBACK_CONDITION = """Extract the CONDITION(S) of THIS SAMPLE — disease(s), pathological state,
OR explicit healthy / control / normal sample state — as it appears in the
metadata.

WHAT COUNTS (three equal-priority categories):
 (1) Disease names, phenotypes, stage / grade / severity markers.
 (2) Explicit healthy / control / normal sample state. When THIS sample's own
     metadata literally marks it as control / normal / healthy / non-disease /
     non-tumor / unaffected / uninflamed, EMIT that state verbatim. Do NOT
     emit "Not Specified" when a control / normal / healthy marker is
     documented for this sample.
 (3) Pathological phenotype markers without a named disease ("tumor", "lesion",
     "metastasis").

THIS-SAMPLE scoping: extract only the condition(s) or sample-state THIS sample
HAS. The GSE topic alone is not evidence — if the GSE studies a disease but
THIS sample's text says "control" / "healthy donor" / "non-tumor", THIS
sample's Condition is the control state, NOT the GSE disease.

Strip Tissue/cell-line descriptors around embedded disease names (e.g.
"glioblastoma stem-like cell line" → "glioblastoma").

Multiple conditions for one sample are joined with "; " in the order they
appear. Return "Not Specified" only when NO disease, phenotype, stage, OR
explicit healthy / control / normal marker appears anywhere in the metadata
for this sample.
"""

_FALLBACK_TREATMENT = """Extract the treatment(s) applicable to THIS sample — the drug(s),
compound(s), genetic perturbation(s), exposure(s), or clinical/experimental
procedure(s) applied to this particular sample.

Keep intervention names verbatim including dose/concentration tokens when
attached. Multi-treatment → join with "; ". Control / vehicle arms (DMSO,
shRNA_Ctrl, 0nM dasatinib) are valid spans. If no intervention applies, return
"Not Specified".

TREATMENT — UNIVERSAL DEFINITION (overrides any narrower phrasing above):
A Treatment is an in-vivo or in-vitro PERTURBATION applied to the
biological sample's source organism / tissue / cell line BEFORE the
molecular assay was run — drugs and their doses; genetic perturbations
(shRNA, siRNA, CRISPR, KO, OE, inducible expression, transgene);
irradiation; environmental exposures (hypoxia, heat shock, UV, smoke);
or clinical procedures. Text that describes the MOLECULAR ASSAY ITSELF
— chemical conversion of nucleic acids, hybridisation, library
construction, scan / sequencing protocol, kit names, reagent names,
instrument names — is NOT a Treatment, regardless of whether it
appears in a field literally named `treatment_protocol`. When the only
treatment-shaped text in the sample's metadata is assay-protocol text,
output exactly: Not Specified.
"""

_FALLBACK_RESOLVE = """Decode any coded token inside the extracted label value by consulting a
legend present in the same sample's metadata. If no legend applies, return
raw_value BYTE-IDENTICAL.

Legend shapes you must recognise (in order):

  1. EXPLICIT DEFINITION elsewhere in metadata: when a token in
     raw_value is defined in the sample text (e.g. a key/legend like
     "X = meaning"), substitute the definition.

  2. SELF-LEGENDING FIELD NAME — the field name itself names the
     condition / phenotype, and its value is a polarity axis. The
     field name acts AS the legend.
     Apply when raw_value has the form `<field>: <axis_token>`.
     The axis_token decides the polarity:
       - affirmative axis (yes, Y, 1, true, positive, pos, present,
         affected, case) → output the condition named in the field
         itself (drop trailing suffixes such as "diagnosis", ".status",
         "_yes_no", "_yn", "_status"; capitalise normally).
       - negative axis (no, N, 0, false, negative, neg, absent,
         unaffected, control) → output "Not Specified".
     For composite raw_value with multiple `<field>: <axis_token>`
     components separated by "; ", resolve each component independently
     under the same rule and join the surviving (non-NS) outputs with
     "; ".
     This is NOT world knowledge: the field name is literally present
     in the sample's metadata, and the rule is purely structural —
     "the field name names the condition when the value is on the
     affirmative axis."

  3. IN-FIELD PARENTHETICAL — code and name side by side, e.g.
       "RRMS (Relapsing Remitting Multiple Sclerosis)"
     keep the full clinical name.

If resolution collapses to nothing (e.g. all components on the negative
axis), return "Not Specified". Output BYTE-IDENTICAL only when none of
the three legend shapes apply.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Compiled-artifact loader — reads instructions + demos directly from JSON.
# ─────────────────────────────────────────────────────────────────────────────
_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "compiled")


def _load_artifact(spec: Dict, fallback: str) -> Dict:
    """Load ``{instructions, demos}`` from a MIPROv2 artifact; fall back to
    the Signature docstring if the file is missing or malformed."""
    path = os.path.join(_ARTIFACTS_DIR, spec["artifact"])
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ins = (data.get("signature") or {}).get("instructions") or fallback
            demos = data.get("demos") or []
            return {"instructions": ins, "demos": demos}
        except Exception as e:  # noqa: BLE001
            print(f"[phase1_runtime] could not load {spec['artifact']}: "
                  f"{e!r} — using fallback prompt")
    return {"instructions": fallback, "demos": []}


_PROMPT_TISSUE    = _load_artifact(_SIG_TISSUE,    _FALLBACK_TISSUE)
_PROMPT_CONDITION = _load_artifact(_SIG_CONDITION, _FALLBACK_CONDITION)
_PROMPT_TREATMENT = _load_artifact(_SIG_TREATMENT, _FALLBACK_TREATMENT)
_PROMPT_RESOLVE   = _load_artifact(_SIG_RESOLVE,   _FALLBACK_RESOLVE)


# ─────────────────────────────────────────────────────────────────────────────
# ChatAdapter-equivalent renderer. Produces the same system+user message
# layout DSPy uses (the ``[[ ## field ## ]]`` scheme).
# ─────────────────────────────────────────────────────────────────────────────
def _render_system(spec: Dict, instructions: str) -> str:
    lines: List[str] = ["Your input fields are:"]
    for i, (n, t, d) in enumerate(spec["inputs"], 1):
        lines.append(f"{i}. `{n}` ({t}): {d}")
    lines.append("Your output fields are:")
    for i, (n, t, d) in enumerate(spec["outputs"], 1):
        lines.append(f"{i}. `{n}` ({t}): {d}")
    lines.append("All interactions will be structured in the following way, "
                 "with the appropriate values filled in.")
    lines.append("")
    for n, _, _ in spec["inputs"]:
        lines.append(f"[[ ## {n} ## ]]")
        lines.append("{" + n + "}")
        lines.append("")
    for n, _, _ in spec["outputs"]:
        lines.append(f"[[ ## {n} ## ]]")
        lines.append("{" + n + "}")
        lines.append("")
    lines.append("[[ ## completed ## ]]")
    # DSPy indents each docstring line by 8 spaces when inlining the
    # instruction into the system message. Match that exactly.
    indented = "\n".join("        " + L for L in instructions.splitlines())
    lines.append(f"In adhering to this structure, your objective is: \n"
                 f"{indented}")
    return "\n".join(lines)


def _render_user(spec: Dict, values: Dict[str, str]) -> str:
    parts: List[str] = []
    for n, _, _ in spec["inputs"]:
        parts.append(f"[[ ## {n} ## ]]")
        parts.append(str(values.get(n, "") or ""))
        parts.append("")
    first_out = spec["outputs"][0][0]
    parts.append(
        f"Respond with the corresponding output fields, starting with the "
        f"field `[[ ## {first_out} ## ]]`, and then ending with the marker "
        f"for `[[ ## completed ## ]]`."
    )
    return "\n".join(parts)


def _render_demo_pair(spec: Dict, demo: Dict) -> List[Dict[str, str]]:
    """Render one few-shot demo as a user/assistant message pair in
    ChatAdapter format."""
    user = _render_user(spec, demo)
    asst_parts: List[str] = []
    for n, _, _ in spec["outputs"]:
        asst_parts.append(f"[[ ## {n} ## ]]")
        asst_parts.append(str(demo.get(n, "") or ""))
        asst_parts.append("")
    asst_parts.append("[[ ## completed ## ]]")
    return [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": "\n".join(asst_parts)},
    ]


def _build_messages(spec: Dict, prompt: Dict,
                    values: Dict[str, str]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": _render_system(spec, prompt["instructions"])},
    ]
    for demo in prompt.get("demos", []):
        msgs.extend(_render_demo_pair(spec, demo))
    msgs.append({"role": "user", "content": _render_user(spec, values)})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# HTTP — stdlib only, same reproducibility settings as the old DSPy LM
# (temperature=0, seed=42, num_predict=-1 on Ollama).
# ─────────────────────────────────────────────────────────────────────────────
def _http_post_json(url: str, body: Dict, timeout: int = 600) -> Dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _call_lm(messages: List[Dict[str, str]]) -> str:
    if _IS_OPENAI:
        body = {
            "model": _MODEL,
            "messages": messages,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": 16384,
        }
        resp = _http_post_json(_CHAT_URL, body)
        return resp["choices"][0]["message"]["content"]
    # Ollama
    # think=True is REQUIRED across the pipeline. The user has stated
    # multiple times that gemma4 reasoning is a pillar of this
    # extraction's quality; never disable it without explicit approval.
    # Override only via THINK_MODE=false env var (GUI exposes a toggle).
    _THINK = os.environ.get("THINK_MODE", "true").lower() in ("1", "true", "yes")
    body = {
        "model": _MODEL,
        "messages": messages,
        "stream": False,
        "think": _THINK,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            # think=True needs more headroom for the reasoning chain
            # plus the answer; bump num_predict accordingly.
            "num_predict": 1024 if _THINK else 256,
        },
    }
    resp = _http_post_json(_CHAT_URL, body)
    msg = resp.get("message", {})
    # gemma4 with think=True splits output into `thinking` + `content`.
    # When num_predict is exhausted by reasoning, content can be empty;
    # concatenate so caller's parser can still recover the answer.
    return (msg.get("content") or "") + ("\n" + msg.get("thinking", "") if _THINK else "")


# ─────────────────────────────────────────────────────────────────────────────
# Output parser — extract the first output field value out of the LM text.
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_RX = re.compile(r"\[\[ ## (?P<name>[^#]+?) ## \]\]")


def _parse_output(text: str, out_name: str) -> str:
    """Return the content between ``[[ ## out_name ## ]]`` and the next
    marker (another field or ``[[ ## completed ## ]]``). If no marker is
    present, treat the whole text as the output value."""
    if not text:
        return ""
    # Find all marker positions in order, to carve out each section.
    positions = [(m.start(), m.end(), m.group("name").strip())
                 for m in _FIELD_RX.finditer(text)]
    if not positions:
        return text.strip()
    for i, (_, end, name) in enumerate(positions):
        if name == out_name:
            next_start = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            return text[end:next_start].strip()
    # Target marker not found — return everything before the first marker as
    # a permissive fallback.
    return text[: positions[0][0]].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — NS normalisation + input marshalling. Copied from dspy_phase1 so
# callers see identical output.
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_ns(val) -> str:
    if not val:
        return NS
    v = str(val).strip().strip('"').strip("'")
    if not v or v.lower() in ("not specified", "none", "null", "n/a", "na"):
        return NS
    return v


def _args_from_raw(raw: Dict) -> Dict[str, str]:
    return {
        "title":              str(raw.get("gsm_title", "") or "").strip(),
        "source":             str(raw.get("source_name", "") or "").strip(),
        "characteristics":    str(raw.get("characteristics", "") or "").replace("\t", " ").strip(),
        "treatment_protocol": str(raw.get("treatment_protocol", "") or "").replace("\t", " ").strip(),
        "description":        str(raw.get("description", "") or "").replace("\t", " ").strip(),
    }


def _metadata_blob(args: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in args.items() if v)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — one entry point per label, plus Phase1Agent.
# ─────────────────────────────────────────────────────────────────────────────
def _predict_label(spec: Dict, prompt: Dict, out_name: str,
                   values: Dict[str, str]) -> str:
    messages = _build_messages(spec, prompt, values)
    try:
        raw = _call_lm(messages)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return ""
    return _parse_output(raw, out_name)


def _resolve_if_coded(field: str, raw_value: str,
                      args: Dict[str, str]) -> str:
    if _normalize_ns(raw_value) == NS:
        return NS
    values = {
        "field":         field,
        "raw_value":     raw_value,
        "metadata_blob": _metadata_blob(args),
    }
    try:
        out = _predict_label(_SIG_RESOLVE, _PROMPT_RESOLVE, "resolved", values)
    except Exception:
        return raw_value
    return _normalize_ns(out) if out else raw_value


def extract_tissue(raw: Dict) -> str:
    args = _args_from_raw(raw)
    out = _predict_label(_SIG_TISSUE, _PROMPT_TISSUE, "tissue", args)
    initial = _normalize_ns(out)
    return _resolve_if_coded("Tissue", initial, args)


# Field-level denial: `<word>: N|No|0|false|negative|absent|unaffected|none|neg`.
# When the LLM emits this verbatim (any disease-named field with a
# negative-axis value), the value is a denial of the field's disease —
# collapse to NS deterministically rather than relying on the legend
# resolver.
_DENIAL_RX = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.\- ]*\s*[:=]\s*"
    r"(N|No|0|false|negative|absent|unaffected|none|neg)\s*$",
    re.IGNORECASE,
)


def _is_denied_field_value(s: str) -> bool:
    if not s or NS.lower() in s.lower():
        return False
    return bool(_DENIAL_RX.match(s.strip()))


def extract_condition(raw: Dict) -> str:
    args = _args_from_raw(raw)
    out = _predict_label(_SIG_CONDITION, _PROMPT_CONDITION, "condition", args)
    initial = _normalize_ns(out)
    if _is_denied_field_value(initial):
        return NS
    return _resolve_if_coded("Condition", initial, args)


def extract_treatment(raw: Dict) -> str:
    args = _args_from_raw(raw)
    out = _predict_label(_SIG_TREATMENT, _PROMPT_TREATMENT, "treatment", args)
    initial = _normalize_ns(out)
    return _resolve_if_coded("Treatment", initial, args)


class Phase1Agent:
    """Phase 1 agent."""

    TOOLS = {
        "Tissue":    extract_tissue,
        "Condition": extract_condition,
        "Treatment": extract_treatment,
    }

    def extract(self, raw: Dict) -> Dict[str, str]:
        # 3 column extractors are independent LLM calls — fan them out
        # so a single sample fires Tissue+Condition+Treatment concurrently
        # into Ollama instead of paying 3× call latency serially.
        # LABEL_COL_WORKERS env var tunes the fan-out (1=serial, 3=full).
        import os
        from concurrent.futures import ThreadPoolExecutor
        n = max(1, min(int(os.environ.get("LABEL_COL_WORKERS", "3") or "3"),
                       len(self.TOOLS)))
        def _do(col):
            try:
                return col, self.TOOLS[col](raw)
            except Exception:
                return col, NS
        out: Dict[str, str] = {}
        if n <= 1:
            for col in self.TOOLS:
                out[col] = _do(col)[1]
            return out
        with ThreadPoolExecutor(max_workers=n) as ex:
            for col, val in ex.map(_do, list(self.TOOLS.keys())):
                out[col] = val
        return out

    def extract_field(self, raw: Dict, col: str) -> str:
        tool = self.TOOLS.get(col)
        if tool is None:
            return NS
        try:
            return tool(raw)
        except Exception:
            return NS


__all__ = [
    "NS",
    "extract_tissue", "extract_condition", "extract_treatment",
    "Phase1Agent",
]
