"""
Prompt -> tool router for the GeneVariate assistant.

``route(prompt, registry)`` returns an :class:`Action` naming the tool to run and
the params the model extracted from the prompt. The choice is the model's: it is
handed every tool's name, description, parameters and examples, and is
constrained to emit a single JSON object. It never raises and never runs
anything; the confirmation card is the human gate.

There is deliberately no keyword fallback. A bag-of-words scorer over tool
descriptions cannot tell "compare ALB across platforms" (one gene, several
platforms) from "compare the ALB distribution across tissues" (one platform,
several groups) or from "pool the ALB region across platforms" (a region
question) - the three share almost every word. Every hand-written rule added to
separate one pair silently captured prompts belonging to a third tool, and the
result was not a wrong answer the user could see but the *wrong analysis run
under the right name*, with the parameters it was not given quietly filled from
that tool's defaults. When the model cannot pick a tool, saying so is the
correct output.
"""
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
    # The tool list *is* the prompt, and it grows with the registry. Past the
    # backend's default window a model handed a truncated tool list does not
    # route badly, it emits a couple of characters and stops - which arrives
    # below as "no JSON" and is reported to the user as the backend being
    # unreachable, while the backend is running and answering. Size the window
    # to the text instead of hoping it fits.
    need = (len(system) + len(prompt)) // 3 + 512
    num_ctx = max(4096, 1 << (need - 1).bit_length())
    try:
        text = llm_backend.chat(messages, model=model, temperature=0.0,
                                num_predict=256, think=False, timeout=60,
                                num_ctx=num_ctx)
    except Exception:
        return None
    # Past this point the backend has answered. A reply we cannot use is the
    # model declining, not the backend being down, and the two must not be
    # reported to the user with the same sentence: one is fixed by rephrasing
    # and the other by starting a server.
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
        # The model read the request and the tool list and declined. Its own
        # words, not a guess dressed up as one.
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
