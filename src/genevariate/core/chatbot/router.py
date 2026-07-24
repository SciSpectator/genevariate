"""
Prompt -> tool router for the GeneVariate assistant.

``route(prompt, registry)`` returns an :class:`Action` naming the tool to run
and the params extracted from the prompt. It prefers a local LLM (ollama via
``llm_backend.chat``) constrained to emit a single JSON object; on any failure
— server down, model missing, malformed JSON, unknown tool — it falls back to a
deterministic keyword router. It never raises and never runs anything; the
confirmation card is the human gate.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .tools import Action, Tool


# -----------------------------------------------------------------
# LLM prompt construction
# -----------------------------------------------------------------
def _system_prompt(registry: Dict[str, Tool]) -> str:
    lines = [
        "You route a user's request to exactly ONE analysis tool.",
        "Reply with a SINGLE JSON object and nothing else:",
        '{"tool": <name|null>, "params": {<param>: <value>}, "confidence": <0..1>}',
        "Set tool to null if no tool fits. Only use these tools:",
    ]
    for t in registry.values():
        pnames = ", ".join(p.name for p in t.params) or "(none)"
        lines.append(f"- {t.name}: {t.description} params: {pnames}")
    lines.append("Examples of matching requests:")
    for t in registry.values():
        for ex in list(t.examples)[:2]:
            lines.append(f'  "{ex}" -> {{"tool": "{t.name}"}}')
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        break
        start = text.find("{", start + 1)
    return None


def _llm_route(prompt: str, registry: Dict[str, Tool]) -> Optional[Action]:
    try:
        from genevariate.core import ollama_manager as om
        from genevariate.core.llm_extractor import llm_backend
    except Exception:
        return None

    model = getattr(om, "DEFAULT_MODEL", "gemma4:e2b")
    try:
        if not om.ollama_server_ok():
            return None
        if not om.model_available(model):
            return None
    except Exception:
        return None

    messages = [
        {"role": "system", "content": _system_prompt(registry)},
        {"role": "user", "content": prompt},
    ]
    try:
        text = llm_backend.chat(messages, model=model, temperature=0.0,
                                num_predict=256, think=False, timeout=30)
    except Exception:
        return None
    obj = _extract_json(text)
    if not obj:
        return None
    tool = obj.get("tool")
    if tool is not None and tool not in registry:
        return None
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if tool is None:
        return Action(tool=None, source="llm", confidence=conf,
                      message="No matching tool.")
    coerced = registry[tool].coerce(params)
    return Action(tool=tool, params=coerced, confidence=conf, source="llm")


# -----------------------------------------------------------------
# Deterministic keyword fallback
# -----------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _keyword_route(prompt: str, registry: Dict[str, Tool]) -> Action:
    forced = _strong_intent(prompt, registry)
    if forced is not None:
        params = _extract_keyword_params(prompt, registry[forced])
        return Action(tool=forced, params=params, confidence=0.75,
                      source="keyword")
    ptok = _tokens(prompt)
    best_name, best_score = None, 0.0
    for t in registry.values():
        vocab = _tokens(t.name.replace("_", " ") + " " + t.description)
        for ex in t.examples:
            vocab |= _tokens(ex)
        overlap = len(ptok & vocab)
        # light boost when the tool name words appear verbatim
        name_hit = len(ptok & _tokens(t.name.replace("_", " ")))
        score = overlap + 1.5 * name_hit
        if score > best_score:
            best_name, best_score = t.name, score
    if best_name is None or best_score == 0:
        return Action(tool=None, source="keyword", confidence=0.0,
                      message="Could not match your request to a tool.")
    params = _extract_keyword_params(prompt, registry[best_name])
    conf = min(0.9, 0.3 + 0.1 * best_score)
    return Action(tool=best_name, params=params, confidence=conf,
                  source="keyword")


_VS = re.compile(r"([A-Za-z0-9_\- ]+?)\s+(?:vs\.?|versus|against)\s+([A-Za-z0-9_\- ]+)",
                 re.IGNORECASE)
_PLATFORM = re.compile(r"\bGPL\d+\b", re.IGNORECASE)
# an uppercase gene-symbol-like token (has a digit or is 2-8 all-caps letters)
_GENE = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")
# tokens that look like genes but never are, so we don't mis-extract them
_NOT_GENE = {"GPL", "GSM", "VS", "GO", "KEGG", "GSEA", "NGS", "DE", "RNA",
             "CSV", "TSV", "DESEQ2", "DESEQ", "QC", "CPM", "ID", "AI", "GEO",
             "MTX", "H5AD", "TP", "MSIGDB", "KS", "FDR", "NES"}


def _extract_gene(prompt: str) -> Optional[str]:
    for tok in _GENE.findall(prompt):
        up = tok.upper()
        if up in _NOT_GENE or _PLATFORM.fullmatch(up):
            continue
        return up
    return None


# High-precision deterministic overrides for a few request shapes the
# bag-of-words scorer routes wrong. Each rule fires only on strong, unambiguous
# signals so it never regresses the prompts the scorer already gets right.
_LOAD_VERBS = {"load", "download", "fetch", "get", "open", "import", "pull"}
_DIST_WORDS = {"distribution", "distributions", "bimodal", "multimodal",
               "unimodal", "histogram", "shape", "shapes", "skewed", "skew"}
_COMPARE_WORDS = {"compare", "side", "versus", "vs", "differ", "differs",
                  "across", "between"}
_MODALITY_HINT = ("modalit", "harmoni", "microarray", "rna-seq", "rnaseq",
                  "single-cell", "single cell", "scrna", "z-score", "zscore",
                  "batch")
_SC_HINT = ("single cell", "single-cell", "scrna", "sc-rna")


def _strong_intent(prompt: str, registry: Dict[str, Tool]) -> Optional[str]:
    low = prompt.lower()
    tok = _tokens(prompt)
    plats = {p.upper() for p in _PLATFORM.findall(prompt)}
    gene = _extract_gene(prompt)
    has_mod = any(h in low for h in _MODALITY_HINT)
    has_sc = any(h in low for h in _SC_HINT)

    # 1) a load/download verb + a named GPL id (and not single-cell) -> load it
    if plats and (tok & _LOAD_VERBS) and not has_sc \
            and "load_geo_platform" in registry:
        return "load_geo_platform"
    # 2) variability + enrichment mentioned together -> variability enrichment
    #    (the scorer otherwise leaks these to rank_genes)
    if "variab" in low and "enrich" in low \
            and "variability_enrichment" in registry:
        return "variability_enrichment"
    # 3) one gene + a distribution-shape word, no comparison/modality cue and at
    #    most one platform -> profile that single gene
    if gene and (tok & _DIST_WORDS) and not (tok & _COMPARE_WORDS) \
            and not has_mod and len(plats) < 2 \
            and "gene_distribution" in registry:
        return "gene_distribution"
    # 4) a distribution-shape word with NO specific gene named -> classify the
    #    whole platform's distributions rather than one gene
    if gene is None and (tok & _DIST_WORDS) and plats \
            and "classify_distributions" in registry:
        return "classify_distributions"
    # 3) one gene compared side-by-side across >=2 named GPL platforms with no
    #    modality/harmonisation cue -> plain compare (not compare_modalities)
    if gene and len(plats) >= 2 and (tok & _COMPARE_WORDS) and not has_mod \
            and "compare_gene" in registry:
        return "compare_gene"
    return None


def _extract_keyword_params(prompt: str, tool: Tool) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    names = {p.name for p in tool.params}
    m = _VS.search(prompt)
    if m and "case_label" in names:
        params["case_label"] = m.group(1).strip().split()[-1]
        params["control_label"] = m.group(2).strip().split()[0]
    # gene symbol (single, e.g. TP53) for the per-gene tools
    if "gene" in names:
        g = _extract_gene(prompt)
        if g:
            params["gene"] = g
    # platform id(s) — GPL#### mentioned in the prompt
    plats = [p.upper() for p in _PLATFORM.findall(prompt)]
    if plats:
        if "platforms" in names:
            params["platforms"] = list(dict.fromkeys(plats))
        elif "platform" in names:
            params["platform"] = plats[0]
    return tool.coerce(params)


# -----------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------
def route(prompt: str, registry: Dict[str, Tool]) -> Action:
    """Route ``prompt`` to a tool. LLM first, deterministic keyword fallback."""
    if not prompt or not prompt.strip():
        return Action(tool=None, source="none", message="Empty prompt.")
    action = _llm_route(prompt, registry)
    if action is not None and action.tool is not None:
        return action
    return _keyword_route(prompt, registry)
