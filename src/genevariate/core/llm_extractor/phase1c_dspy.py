"""DSPy-backed Phase 1c curator — prompt-optimizable alternative.

Public API is intentionally narrow so a driver script can swap this in
place of ``phase1c.Phase1cCurator``:

    from phase1c_dspy import DspyPhase1cCurator
    c = DspyPhase1cCurator()
    r = c.curate(gse_context, samples)   # returns dict like phase1c

The novelty is that the *system instruction* is no longer hand-written:
it is declared as a DSPy ``Signature`` docstring, and optimizers
(``BootstrapFewShot``, ``MIPROv2``) can mutate both the instruction and
the few-shot demonstrations using a metric driven by ground truth.

Design mirrors phase1c.py:
  * per-label agents (Tissue / Condition / Treatment)
  * full-GSE value distribution + rare-values table in the user content
  * quote-substring guard retained as a validator (DSPy can't hallucinate
    its way past it)

Backend: Ollama gemma4:e2b via litellm ('ollama_chat/gemma4:e2b'). Set
PHASE1_MODEL / OLLAMA_URL to override.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import dspy


# ──────────────────────────────────────────────────────────────────────
# LM config.  temperature=0 + fixed seed on the server side.
# ──────────────────────────────────────────────────────────────────────
_MODEL     = os.environ.get("PHASE1_MODEL", "gemma4-e2b-text:latest")
_OLLAMA    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_NUM_CTX   = int(os.environ.get("PHASE1C_NUM_CTX", "8192"))


def configure_lm(think: bool = True, max_tokens: int = 16384):
    """Idempotent DSPy LM setup for Ollama gemma4:e2b."""
    lm = dspy.LM(
        model=f"ollama_chat/{_MODEL}",
        api_base=_OLLAMA,
        temperature=0.0,
        max_tokens=max_tokens,
        num_ctx=_NUM_CTX,
        seed=42,
        # litellm passes unknown kwargs as 'options' to Ollama.
        think=think,
    )
    dspy.configure(lm=lm)
    return lm


# ──────────────────────────────────────────────────────────────────────
# Signature.  The docstring IS the optimizable instruction.
# ──────────────────────────────────────────────────────────────────────
class CurateField(dspy.Signature):
    """Find WRONG {field} values in a GEO series and propose fixes.

    You review EVERY sample of ONE GSE for ONE label at a time. Flag
    a value wrong only if it matches one of:
      (a) PARSER ARTIFACT   contains '[[' ']]' '## completed' '## done'
      (b) FIELD-NAME LEAK   value is a metadata key that appears as
                            'key:' in the raw sample text (e.g.
                            'diagnosis', 'Diagnosis', 'group',
                            'disease', 'status').
      (c) SUBJECT/DONOR ID  value looks like a barcode / donor-id
                            that ALSO appears as 'donor_id: X' or in
                            the title.
      (d) NOISE TAIL        value ends with '; X' where X is a
                            metadata key (appears as 'X:' elsewhere
                            in the sample's raw text).
      (e) CONSENSUS GAP     value is 'Not Specified' AND >=70 % of
                            siblings share a specific V AND V
                            appears verbatim in THIS sample's raw
                            metadata.

    Rules:
      * Return ONE item per (gsm, field) that needs a change; never
        deduplicate near-identical bugs — if CXMA-01-111 and
        CXMA-01-112 both leak as tissue, emit BOTH.
      * Every suggestion must have a `quote` that is a verbatim
        substring of the inputs. If you can't find one, skip.
      * Legend expansions ('AD' -> "Alzheimer's Disease") are NOT
        bugs — don't flag them.
      * Real comorbidities joined with ';' are NOT bugs, but a '; X'
        where X is a metadata KEY IS a noise-tail bug — strip X.

    Output a JSON array. Each item: {"gsm":"...", "current":"...",
    "suggest":"...", "quote":"...", "reason":"..."}. If nothing is
    wrong, return []."""

    field:         str = dspy.InputField(desc="Tissue | Condition | Treatment")
    gse_context:   str = dspy.InputField(desc="GSE title / summary / design")
    distribution:  str = dspy.InputField(desc="value frequency table for this field across the GSE")
    rare_values:   str = dspy.InputField(desc="pre-computed outliers — inspect each")
    samples:       str = dspy.InputField(desc="one line per GSM with raw metadata")
    corrections:   str = dspy.OutputField(
        desc='JSON array of {"gsm","current","suggest","quote","reason"} objects, or []')


# ──────────────────────────────────────────────────────────────────────
# Input-block builders (shared shape with phase1c.py for fair A/B).
# ──────────────────────────────────────────────────────────────────────
def _compact_sample(s: Dict) -> Dict:
    return {
        "gsm":                s.get("gsm") or "",
        "title":              s.get("title", "") or "",
        "source_name":        s.get("source_name_ch1", "") or "",
        "characteristics":    s.get("characteristics_ch1", "") or "",
        "treatment_protocol": s.get("treatment_protocol_ch1", "") or "",
        "description":        s.get("description", "") or "",
        "phase1":             s.get("phase1", {}),
    }


def _distribution(samples: List[Dict], field: str) -> Counter:
    c: Counter = Counter()
    for s in samples:
        v = (s.get("phase1") or {}).get(field, "") or "Not Specified"
        c[v] += 1
    return c


def _fmt_dist(d: Counter, field: str) -> str:
    lines = [f"{field.upper()} DIST (n={sum(d.values())}):"]
    for v, n in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])):
        vshow = v if len(v) <= 60 else v[:57] + "..."
        lines.append(f"  {n:>4}  {vshow!r}")
    return "\n".join(lines)


def _fmt_rare(d: Counter, field: str) -> str:
    total = sum(d.values())
    if total == 0:
        return ""
    thr = max(3, total // 10)
    rare = sorted(((v, n) for v, n in d.items() if n <= thr),
                  key=lambda kv: (kv[1], kv[0]))
    if not rare:
        return ""
    lines = [f"RARE {field.upper()} (count<={thr}/{total}):"]
    for v, n in rare:
        vshow = v if len(v) <= 60 else v[:57] + "..."
        lines.append(f"  n={n:<3} {vshow!r}")
    return "\n".join(lines)


def _fmt_samples(samples: List[Dict], field: str) -> str:
    rows: List[str] = []
    for s in samples:
        cs = _compact_sample(s)
        cur = (cs["phase1"] or {}).get(field, "")
        bits = [f"{cs['gsm']} cur={cur!r}"]
        if cs["title"]:              bits.append(f"t={cs['title']}")
        if cs["source_name"]:        bits.append(f"src={cs['source_name']}")
        if cs["characteristics"]:    bits.append(f"ch={cs['characteristics']}")
        if cs["treatment_protocol"]: bits.append(f"tr={cs['treatment_protocol']}")
        if cs["description"]:        bits.append(f"d={cs['description']}")
        rows.append(" | ".join(bits))
    return "\n".join(rows)


def _fmt_gse_ctx(ctx: Dict) -> str:
    parts = []
    for k in ("gse", "title", "summary", "overall_design"):
        v = ctx.get(k) or ""
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Robust JSON extraction (ported + expanded from phase1c).
# Accepts also `current='val'` / `current="val"` equal-sign glitches.
# ──────────────────────────────────────────────────────────────────────
_FENCE_RX = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_KEY_EQ_SQ = re.compile(r'("?[A-Za-z_]\w*)\s*=\s*\'([^\']*)\'')
_KEY_EQ_DQ = re.compile(r'("?[A-Za-z_]\w*)\s*=\s*"([^"]*)"')
_MISSING_OPEN_Q = re.compile(r'(?P<pre>[,{\s])(?P<k>[A-Za-z_][A-Za-z0-9_]*)":')


def _iter_objs(text: str):
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        d, j = 0, i
        while j < n:
            ch = text[j]
            if ch == "{":
                d += 1
            elif ch == "}":
                d -= 1
                if d == 0:
                    yield (i, j + 1)
                    break
            j += 1
        i = j + 1


def _repair(obj_src: str) -> Optional[Dict]:
    try:
        return json.loads(obj_src)
    except Exception:  # noqa: BLE001
        pass
    s = obj_src
    # Fix  key='val'  ->  "key":"val"
    s = _KEY_EQ_SQ.sub(lambda m: '"%s":"%s"' % (m.group(1).strip('"'),
                                                 m.group(2)), s)
    # Fix  key="val"  (equals)  ->  "key":"val"
    s = _KEY_EQ_DQ.sub(lambda m: '"%s":"%s"' % (m.group(1).strip('"'),
                                                 m.group(2)), s)
    # Missing opening quote on keys:  X":  ->  "X":
    s = _MISSING_OPEN_Q.sub(r'\g<pre>"\g<k>":', s)
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def _extract_json(text: str) -> List[Dict]:
    if not text:
        return []
    for m in _FENCE_RX.finditer(text):
        body = m.group(1).strip()
        if body.startswith("["):
            try:
                v = json.loads(body)
                if isinstance(v, list):
                    return v
            except Exception:  # noqa: BLE001
                pass
    # Try the whole string; fall back to object-by-object with repair.
    try:
        v = json.loads(text.strip())
        if isinstance(v, list):
            return v
    except Exception:  # noqa: BLE001
        pass
    items: List[Dict] = []
    for s, e in _iter_objs(text):
        obj = _repair(text[s:e])
        if isinstance(obj, dict):
            items.append(obj)
    return items


# ──────────────────────────────────────────────────────────────────────
# Quote-substring guard (identical policy to phase1c).
# ──────────────────────────────────────────────────────────────────────
def _haystack(ctx: Dict, samples: List[Dict]) -> str:
    parts: List[str] = []
    for k in ("gse", "title", "summary", "overall_design"):
        v = ctx.get(k) or ""
        if v: parts.append(str(v))
    for s in samples:
        for k in ("title", "source_name_ch1", "characteristics_ch1",
                  "treatment_protocol_ch1", "description"):
            v = s.get(k) or ""
            if v: parts.append(str(v))
    return "\n".join(parts)


def _validate(proposals: List[Dict], samples: List[Dict],
              haystack: str, field: str) -> List[Dict]:
    gsm_set = {(s.get("gsm") or "").lower() for s in samples}
    norm_h = re.sub(r"\s+", " ", haystack)
    out: List[Dict] = []
    for c in proposals or []:
        if not isinstance(c, dict):
            continue
        gsm     = str(c.get("gsm") or "").strip()
        suggest = str(c.get("suggest") or "").strip()
        current = str(c.get("current") or "").strip()
        quote   = str(c.get("quote") or "").strip()
        if gsm.lower() not in gsm_set: continue
        if not suggest:                 continue
        if suggest == current:          continue
        norm_s = re.sub(r"\s+", " ", suggest).strip()
        norm_q = re.sub(r"\s+", " ", quote).strip()
        if not ((norm_s and norm_s in norm_h) or
                (norm_q and norm_q in norm_h)):
            continue
        out.append({"gsm": gsm, "field": field, "current": current,
                    "suggest": suggest, "quote": quote,
                    "reason": str(c.get("reason") or "").strip()})
    return out


def _apply(samples: List[Dict], corrections: List[Dict]) -> List[Dict]:
    by = {(s.get("gsm") or ""): s for s in samples}
    for c in corrections:
        s = by.get(c["gsm"])
        if not s:
            continue
        phase1 = s.setdefault("phase1", {})
        old = phase1.get(c["field"], "")
        phase1[c["field"]] = c["suggest"]
        s.setdefault("phase1c", {})[c["field"]] = {
            "from": old, "to": c["suggest"],
            "reason": c["reason"], "quote": c["quote"],
        }
    return samples


# ──────────────────────────────────────────────────────────────────────
# DSPy module.
# ──────────────────────────────────────────────────────────────────────
class FieldCurator(dspy.Module):
    """One DSPy ChainOfThought wrapped for one field."""

    def __init__(self):
        super().__init__()
        self.pick = dspy.Predict(CurateField)

    def forward(self, field: str, gse_context: str, distribution: str,
                rare_values: str, samples: str):
        return self.pick(field=field, gse_context=gse_context,
                         distribution=distribution, rare_values=rare_values,
                         samples=samples)


# ──────────────────────────────────────────────────────────────────────
# Public facade.
# ──────────────────────────────────────────────────────────────────────
class DspyPhase1cCurator:
    """3 per-label DSPy curators. Same shape of return value as phase1c.py."""

    def __init__(self, program: Optional[FieldCurator] = None,
                 configure: bool = True):
        if configure:
            configure_lm()
        self.program = program or FieldCurator()

    def _one(self, field: str, gse_context: Dict,
             samples: List[Dict]) -> Dict[str, Any]:
        dist = _distribution(samples, field)
        pred = self.program(
            field=field,
            gse_context=_fmt_gse_ctx(gse_context),
            distribution=_fmt_dist(dist, field),
            rare_values=_fmt_rare(dist, field),
            samples=_fmt_samples(samples, field),
        )
        raw = getattr(pred, "corrections", "") or ""
        proposed = _extract_json(raw)
        hay = _haystack(gse_context, samples)
        accepted = _validate(proposed, samples, hay, field)
        return {"field": field, "proposed": proposed, "accepted": accepted,
                "raw_response": raw}

    def curate(self, gse_context: Dict,
               samples: List[Dict]) -> Dict[str, Any]:
        proposed: List[Dict] = []
        accepted: List[Dict] = []
        per_field_raw: Dict[str, str] = {}
        for field in ("Tissue", "Condition", "Treatment"):
            r = self._one(field, gse_context, samples)
            proposed.extend(r["proposed"])
            accepted.extend(r["accepted"])
            per_field_raw[field] = r["raw_response"]
        _apply(samples, accepted)
        return {"samples": samples, "proposed": proposed,
                "accepted": accepted, "rejected": [],
                "raw_response": per_field_raw}


__all__ = ["DspyPhase1cCurator", "FieldCurator", "CurateField",
           "configure_lm", "_fmt_dist", "_fmt_rare", "_fmt_samples",
           "_fmt_gse_ctx", "_distribution", "_extract_json",
           "_validate", "_haystack"]
