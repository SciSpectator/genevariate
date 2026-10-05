"""Prompt -> tool router for the GeneVariate assistant."""
from __future__ import annotations

import json
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
        from genevariate.core import llm_client as llm_backend
    except Exception:
        return None

    model = llm_backend.default_model()
    try:
        # Ask the backend chat() will really use, not localhost Ollama.
        ready, _ = llm_backend.available(model)
    except Exception:
        return None
    if not ready:
        return None

    system = _system_prompt(registry)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    need = (len(system) + len(prompt)) // 3 + 512
    num_ctx = max(4096, 1 << (need - 1).bit_length())
    try:
        text = llm_backend.chat(messages, model=model, temperature=0.0,
                                num_predict=256, think=False, timeout=60,
                                num_ctx=num_ctx)
    except Exception:
        return None
    obj = _extract_json(text) or {}
    tool = obj.get("tool")
    if tool is not None and tool not in registry:
        tool = None
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
# Public entry point
# -----------------------------------------------------------------
def route(prompt: str, registry: Dict[str, Tool]) -> Action:
    """Route ``prompt`` to a tool, or to nothing with a reason."""
    if not prompt or not prompt.strip():
        return Action(tool=None, source="none", message="Empty prompt.")
    action = _llm_route(prompt, registry)
    if action is not None and action.tool is not None:
        return action
    if action is not None:
        return Action(tool=None, source="llm", confidence=action.confidence,
                      message="I couldn't match that to an analysis I can run. "
                              "Try naming the gene, the platform and what you "
                              "want compared - for example \"compare ALB "
                              "between GPL570 and GPL96\".")
    return Action(
        tool=None, source="none",
        message="I can't reach the language model, so I can't work out which "
                "analysis you mean. Start the model backend and ask again, or "
                "open the analysis window directly.")
