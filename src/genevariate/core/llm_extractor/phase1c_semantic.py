"""Two-stage Phase 1c curator: unsupervised anomaly scanner → LLM judge.

Stage 1 — scanner (no LLM, no hardcoded bug patterns):
    For each (field, sample) within a GSE, compute a per-sample suspicion
    score from five domain-agnostic unsupervised signals. Three are
    semantic (computed on BioLORD-2023 embeddings), two are surface-form
    statistics on the raw label string. All are z-normalized per GSE:

      SEMANTIC (content):
        * consensus_distance  = cos-distance of the value's embedding from
                                the centroid of all sibling values for
                                this field in this GSE
        * evidence_gap        = 1 - max cos-sim between the value and any
                                sliced span of the sample's own raw metadata
        * knn_ood             = mean cos-distance to the value's k nearest
                                siblings (Sun et al., ICML 2022 -- SOTA
                                zero-training OOD detector)

      SURFACE (lexical):
        * length_z            = |z-score| of value-length across siblings
        * punct_density_z     = |z-score| of the non-alphanumeric char
                                ratio across siblings
        * ngram_divergence    = Jensen-Shannon divergence between the
                                value's 3-char-gram distribution and the
                                pooled sibling 3-char-gram distribution

    suspicion = max over signals (any single strong signal triggers).
    Flag iff suspicion > max(quantile(q), absolute_floor).

Stage 2 — judge (LLM agent, existing DSPy FieldCurator):
    Only flagged samples are passed to the LLM. Each flagged sample is
    annotated inline with its ANOMALY payload:
      - nearest_sibling : the cos-nearest NON-flagged sibling value
                          (the counterfactual -- "this should probably be X")
      - cluster_size    : count of siblings whose cos-sim to this value
                          exceeds 0.85 (1 = isolated singleton)
      - signals         : the individual z-scores that fired

    The LLM's job becomes a focused binary + extraction: "Given the
    scanner says this value is anomalous *because* <signals> and a
    plausible alternative is <nearest_sibling>, is it actually wrong?
    If yes, emit {suggest, quote}."  This replaces the full-GSE open-ended
    scan that was the baseline failure mode.

Scalability:
    No hardcoded bug patterns anywhere. Any value that is an outlier in
    any of the six unsupervised signals becomes a candidate. The LLM is
    only invoked on candidates and is given structured evidence so that
    decisions do not hinge on LLM pattern-recognition of specific bug
    classes.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import dspy
import numpy as np
from sentence_transformers import SentenceTransformer

from phase1c_dspy import (
    FieldCurator, _compact_sample, _distribution, _extract_json,
    _fmt_dist, _fmt_gse_ctx, _fmt_rare, _haystack, _validate,
    configure_lm,
)

_BIOLORD_MODEL = os.environ.get("BIOLORD_MODEL", "FremyCompany/BioLORD-2023")
_PHASE1_MODEL  = os.environ.get("PHASE1_MODEL",  "gemma4-e2b-text:latest")
_OLLAMA_URL    = os.environ.get("OLLAMA_URL",    "http://localhost:11434")
_NUM_CTX       = int(os.environ.get("PHASE1C_NUM_CTX", "8192"))

_NS = "Not Specified"


# ──────────────────────────────────────────────────────────────────────
# Surface-form feature helpers.
# ──────────────────────────────────────────────────────────────────────
def _raw_text(sample: Dict) -> str:
    parts: List[str] = []
    for k in ("title", "source_name_ch1", "characteristics_ch1",
              "treatment_protocol_ch1", "description"):
        v = sample.get(k) or ""
        if v:
            parts.append(str(v))
    return "\n".join(parts)


def _slice_spans(text: str, min_len: int = 3) -> List[str]:
    spans = set()
    for line in text.split("\n"):
        line = line.strip()
        if len(line) >= min_len:
            spans.add(line)
    for part in re.split(r"[;\n]", text):
        part = part.strip()
        if len(part) >= min_len:
            spans.add(part)
        m = re.match(r"([A-Za-z_][\w ]{0,30})\s*:\s*(.+)", part)
        if m and len(m.group(2).strip()) >= min_len:
            spans.add(m.group(2).strip())
    for part in re.split(r",", text):
        part = part.strip()
        if len(part) >= min_len:
            spans.add(part)
    return list(spans) if spans else [text.strip() or ""]


def _ngrams(text: str, n: int = 3) -> Counter:
    t = f"^{text}$"
    if len(t) < n:
        return Counter([t])
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def _js_div(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence between two Counter distributions.

    Log base 2 so the result is in [0, 1].
    """
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    ptot = sum(p.values()) or 1
    qtot = sum(q.values()) or 1
    js = 0.0
    for k in keys:
        pa = p.get(k, 0) / ptot
        pb = q.get(k, 0) / qtot
        m = 0.5 * (pa + pb)
        if m == 0:
            continue
        if pa > 0:
            js += 0.5 * pa * math.log2(pa / m)
        if pb > 0:
            js += 0.5 * pb * math.log2(pb / m)
    return float(max(0.0, min(1.0, js)))


def _punct_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if not ch.isalnum() and not ch.isspace()) / len(s)


def _zabs_norm(arr: np.ndarray, cap: float = 3.0) -> np.ndarray:
    """|z-score| mapped to [0, 1] by clipping at `cap` sigma."""
    mu = arr.mean()
    sd = arr.std() + 1e-9
    z = np.abs((arr - mu) / sd)
    return np.clip(z / cap, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — unsupervised anomaly scanner.
# ──────────────────────────────────────────────────────────────────────
class SemanticCandidateScreen:
    """Flag (gsm, field) candidates via a 6-signal unsupervised ensemble."""

    _MODEL_CACHE: Dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: str = _BIOLORD_MODEL, k_neighbors: int = 5):
        if model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = self._MODEL_CACHE[model_name]
        self.k = k_neighbors

    # ---------- embeddings --------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False,
                                 convert_to_numpy=True)
        return np.asarray(embs)

    # ---------- main --------------------------------------------------
    def score_field(self, samples: List[Dict], field: str) \
            -> Tuple[Dict[str, Dict[str, float]], np.ndarray, List[str]]:
        """Return (score_table, value_embs, values).

        score_table[gsm] = {
            'consensus', 'evidence', 'knn_ood',
            'length_z', 'punct_z', 'ngram_div',
            'suspicion'
        }
        """
        if not samples:
            return {}, np.zeros((0, 0)), []

        values = [((s.get("phase1") or {}).get(field, _NS) or _NS)
                  for s in samples]
        raw_texts = [_raw_text(s) for s in samples]
        n = len(samples)

        val_embs = self._embed(values)

        # ── semantic: consensus distance ─────────────────────────────
        if n >= 2:
            centroid = val_embs.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-9)
            consensus = 1.0 - (val_embs @ centroid)
        else:
            consensus = np.zeros(n)
        consensus = np.clip(consensus, 0.0, 1.0)

        # ── semantic: evidence gap (value vs its raw-text spans) ─────
        evidence = np.zeros(n)
        for i, raw in enumerate(raw_texts):
            spans = _slice_spans(raw) if raw else []
            if not spans:
                evidence[i] = 1.0
                continue
            span_embs = self._embed(spans)
            if span_embs.shape[0] == 0:
                evidence[i] = 1.0
                continue
            sims = span_embs @ val_embs[i]
            evidence[i] = 1.0 - float(sims.max())
        evidence = np.clip(evidence, 0.0, 1.0)

        # ── semantic: kNN-OOD distance (Sun et al. 2022) ─────────────
        if n >= 2:
            sims_all = val_embs @ val_embs.T
            np.fill_diagonal(sims_all, -np.inf)
            k_eff = min(self.k, n - 1)
            topk = np.sort(sims_all, axis=1)[:, -k_eff:]
            knn_mean_sim = topk.mean(axis=1)
            knn_ood = 1.0 - knn_mean_sim
        else:
            knn_ood = np.zeros(n)
        knn_ood = np.clip(knn_ood, 0.0, 1.0)

        # ── surface: length z ────────────────────────────────────────
        lengths = np.array([len(v) for v in values], dtype=float)
        length_z = _zabs_norm(lengths)

        # ── surface: punctuation density z ───────────────────────────
        puncts = np.array([_punct_ratio(v) for v in values])
        punct_z = _zabs_norm(puncts)

        # ── surface: char-n-gram JS divergence vs pooled siblings ────
        grams = [_ngrams(v) for v in values]
        pooled: Counter = Counter()
        for c in grams:
            pooled += c
        ngram_div_raw = np.zeros(n)
        for i, c in enumerate(grams):
            other = pooled - c
            ngram_div_raw[i] = _js_div(c, other)
        # Scale so the largest div in the GSE maps to 1 (GSE-relative).
        m = ngram_div_raw.max()
        ngram_div = ngram_div_raw / m if m > 1e-9 else ngram_div_raw

        # ── combined suspicion: max across signals ───────────────────
        suspicion = np.max(np.stack([
            consensus, evidence, knn_ood,
            length_z, punct_z, ngram_div,
        ], axis=0), axis=0)

        scores = {
            s["gsm"]: {
                "consensus":  float(consensus[i]),
                "evidence":   float(evidence[i]),
                "knn_ood":    float(knn_ood[i]),
                "length_z":   float(length_z[i]),
                "punct_z":    float(punct_z[i]),
                "ngram_div":  float(ngram_div[i]),
                "suspicion":  float(suspicion[i]),
            }
            for i, s in enumerate(samples)
        }
        return scores, val_embs, values

    # ---------- flag + enrichment ------------------------------------
    def flag_and_enrich(self, samples: List[Dict], field: str,
                        top_quantile: float = 0.70,
                        min_score: float = 0.30,
                        max_flags: Optional[int] = 25,
                        counterfactual_sim_thr: float = 0.85
                        ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        scores, val_embs, values = self.score_field(samples, field)
        if not scores:
            return [], {}

        # Flag set.
        susps = np.array([v["suspicion"] for v in scores.values()])
        thr = max(float(np.quantile(susps, top_quantile)), min_score)
        gsms = list(scores.keys())
        flag_idx = [i for i, g in enumerate(gsms)
                    if scores[g]["suspicion"] > thr]
        flag_idx.sort(key=lambda i: -scores[gsms[i]]["suspicion"])
        if max_flags:
            flag_idx = flag_idx[:max_flags]
        flagged_gsms = [gsms[i] for i in flag_idx]
        flagged_set = set(flag_idx)

        # Counterfactual enrichment per flagged.
        not_flagged_idx = [i for i in range(len(gsms)) if i not in flagged_set]
        enrichment: Dict[str, Dict[str, Any]] = {}
        for i in flag_idx:
            gsm = gsms[i]
            emb_i = val_embs[i] if val_embs.size else None
            nearest = None
            nearest_sim = 0.0
            if emb_i is not None and not_flagged_idx:
                clean_embs = val_embs[not_flagged_idx]
                sims = clean_embs @ emb_i
                j = int(sims.argmax())
                nearest = values[not_flagged_idx[j]]
                nearest_sim = float(sims[j])
                if nearest_sim < 0.35:    # nothing really close -> no CF
                    nearest = None
            if emb_i is not None and val_embs.size:
                all_sims = val_embs @ emb_i
                cluster_size = int((all_sims > counterfactual_sim_thr).sum())
            else:
                cluster_size = 1

            # Pick top-2 signals that fired for WHY payload.
            s = scores[gsm]
            sig_ranked = sorted(
                [(k, v) for k, v in s.items() if k != "suspicion"],
                key=lambda kv: -kv[1])[:3]
            enrichment[gsm] = {
                "suspicion":       s["suspicion"],
                "nearest_sibling": nearest,
                "nearest_sim":     nearest_sim,
                "cluster_size":    cluster_size,
                "top_signals":     sig_ranked,
            }
        return flagged_gsms, enrichment


# ──────────────────────────────────────────────────────────────────────
# Sample-line formatter with inline ANOMALY payload.
# ──────────────────────────────────────────────────────────────────────
def _fmt_samples_with_anomaly(samples: List[Dict], field: str,
                              enrichment: Dict[str, Dict[str, Any]]) -> str:
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
        line = " | ".join(bits)
        e = enrichment.get(cs["gsm"])
        if e is not None:
            sig_str = ",".join(f"{k}={v:.2f}" for k, v in e["top_signals"])
            cf = f"nearest={e['nearest_sibling']!r}" if e["nearest_sibling"] else "nearest=None"
            line += (f" || ANOMALY susp={e['suspicion']:.2f} "
                     f"cluster={e['cluster_size']} {cf} signals[{sig_str}]")
        rows.append(line)
    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────
# Specialist committee (Option B) — 4 narrow DSPy signatures, each
# gated by the unsupervised signals.  Specialists are NOT given
# hardcoded bug examples; they get a narrow task, the sample data,
# and are asked to either propose a fix or abstain.  Their outputs
# are unioned with the general judge's.
# ──────────────────────────────────────────────────────────────────────
class ParserMarkerFix(dspy.Signature):
    """Strip parser/adapter artifacts from label values.

    You receive a batch of flagged (gsm, current) rows for ONE field
    in ONE GEO series.  For each row, decide whether `current`
    contains a TRAILING artifact left over from an upstream parser
    (anything after and including a stray '\\n[[', '## completed',
    '## done', or unmatched bracket junk at the end of the string).

    Return a JSON array of fixes, one per row that needs stripping:
      {"gsm":"...", "current":"...", "suggest":"<stripped>",
       "quote":"<verbatim substring of current>", "reason":"..."}

    Rules:
      * `suggest` MUST be a verbatim PREFIX of `current` (no new chars
        introduced, no abbreviation expansion, no canonicalization).
      * `quote` MUST be a verbatim substring of `current` (usually
        the tail being stripped OR the kept prefix).
      * Skip rows with no trailing artifact.  Return [] if none.
      * DO NOT change abbreviation style (keep 'FTD' as 'FTD',
        keep 'AD' as 'AD').
      * A HALF-OPEN artifact counts — if you see '\\n[[' without a
        matching ']]', or '## completed' without a matching end,
        STRIP everything from the first '\\n' of the marker onward.
      * Do NOT include Python repr quote characters (' or ") around
        your output strings — emit the raw value.
    """
    field:       str = dspy.InputField()
    samples:     str = dspy.InputField(desc="one '<gsm> | <current>' per line")
    corrections: str = dspy.OutputField(desc="JSON array of fixes")


class FieldNameFix(dspy.Signature):
    """Replace metadata-KEY-as-value mistakes with the correct value.

    You receive a batch of flagged rows whose `current` value is a
    metadata KEY (column label: e.g. 'diagnosis', 'Diagnosis',
    'group', 'status') rather than an actual VALUE.  For each row
    you get:  gsm, current, nearest_sibling (majority value in this
    GSE for this field), and raw_text (the sample's source_name,
    characteristics, etc).

    For each row decide: is the correct value extractable as a
    VERBATIM substring of raw_text?  If yes, emit a fix; if no,
    skip the row.

    JSON array items:
      {"gsm":"...", "current":"...", "suggest":"<verbatim substring of raw_text>",
       "quote":"<same verbatim substring>", "reason":"..."}

    Rules:
      * `suggest` and `quote` MUST both be verbatim substrings of
        raw_text for that row.
      * LOOK for 'key: value' patterns in raw_text.  The correct
        value is almost always the phrase right after the colon in
        a line like 'diagnosis: <value>' or 'condition: <value>'
        or 'treatment: <value>'.
      * If `current` is itself a short metadata key (e.g. 5-15
        chars, no spaces, looks like a column name), definitively
        emit a fix — do NOT abstain when a 'key: value' line with
        a plausible value is present in raw_text.
      * If raw_text contains no such line for this row, skip it.
      * Do NOT include Python repr quote characters in your output.
    """
    field:       str = dspy.InputField()
    gse_context: str = dspy.InputField()
    samples:     str = dspy.InputField(
        desc="one '<gsm> | cur=<current> | near=<nearest_sibling> | raw=<raw_text>' per line")
    corrections: str = dspy.OutputField(desc="JSON array of fixes")


class NsPromote(dspy.Signature):
    """Promote 'Not Specified' to the majority sibling when justified.

    You receive flagged rows whose `current` is 'Not Specified' and
    whose nearest sibling value in the GSE is a specific phrase.
    For each row, check whether `nearest_sibling` (or an obvious
    verbatim variant of it) appears in the sample's raw_text.

    If yes → emit a fix:
      {"gsm":"...", "current":"Not Specified",
       "suggest":"<nearest_sibling>", "quote":"<exact span in raw_text>",
       "reason":"..."}
    If no  → skip the row.

    `quote` MUST be a verbatim substring of raw_text.
    Return [] if no row is promotable.
    """
    field:       str = dspy.InputField()
    gse_context: str = dspy.InputField()
    samples:     str = dspy.InputField(
        desc="one '<gsm> | near=<nearest_sibling> | raw=<raw_text>' per line")
    corrections: str = dspy.OutputField(desc="JSON array of fixes")


class CompositeSplit(dspy.Signature):
    """Strip noise-tail tokens from semicolon/comma-joined labels.

    You receive rows whose `current` is a composite like
    'Anxiety; mood scale' or 'Depression; suicide status' where the
    HEAD term before the separator is a real label and the TAIL
    token after the separator is a METADATA KEY (appears as 'key:'
    elsewhere in raw_text).  For each row, strip the tail and emit
    the head as the fix.

    JSON array items:
      {"gsm":"...", "current":"...",
       "suggest":"<head verbatim prefix of current>",
       "quote":"<the tail being stripped, verbatim substring of current>",
       "reason":"..."}

    Rules:
      * `suggest` is the head term (verbatim prefix of `current`,
        trailing whitespace removed).
      * `quote` is the stripped tail, verbatim from `current`.
      * DO NOT strip genuine comorbidities (head AND tail are both
        medical conditions); skip those rows.
      * Return [] if no row is a clear noise-tail.
    """
    field:       str = dspy.InputField()
    gse_context: str = dspy.InputField()
    samples:     str = dspy.InputField(
        desc="one '<gsm> | cur=<current> | near=<nearest_sibling> | raw=<raw_text>' per line")
    corrections: str = dspy.OutputField(desc="JSON array of fixes")


# ---------- gate functions (signal-based, not pattern-based) ---------
def _gate_parser(score: Dict[str, float], enr: Dict[str, Any], val: str) -> bool:
    """High punctuation density OR high n-gram divergence vs siblings."""
    return score.get("punct_z", 0.0) > 0.4 or score.get("ngram_div", 0.0) > 0.5


def _gate_fieldname(score: Dict[str, float], enr: Dict[str, Any], val: str) -> bool:
    """Short value, high evidence gap, has a much longer nearest sibling.

    Signals: length_z high (short outlier), evidence high (doesn't
    match any raw span well).  No regex on specific words like
    'diagnosis' — we rely on the fact that KEY-leak values tend to
    be short, poorly supported, and far from the cluster centroid.
    """
    if val.strip().lower() in {"not specified", ""}:
        return False
    sib = enr.get("nearest_sibling") if enr else None
    if not sib:
        return False
    return (score.get("length_z", 0.0) > 0.4
            and score.get("evidence",  0.0) > 0.3
            and len(sib) > max(len(val) + 2, int(1.3 * len(val))))


def _gate_ns(score: Dict[str, float], enr: Dict[str, Any], val: str) -> bool:
    """Current == 'Not Specified' AND an informative sibling exists."""
    if val.strip().lower() != "not specified":
        return False
    if not enr:
        return False
    sib = enr.get("nearest_sibling")
    sim = float(enr.get("nearest_sim", 0.0))
    return bool(sib) and sim >= 0.25


def _gate_composite(score: Dict[str, float], enr: Dict[str, Any], val: str) -> bool:
    """Current value contains a ';' separator AND there is a sibling."""
    if ";" not in val:
        return False
    if not enr or not enr.get("nearest_sibling"):
        return False
    # Noise-tail heuristic: head (before ';') close-ish to nearest_sibling.
    head = val.split(";", 1)[0].strip().lower()
    sib = str(enr["nearest_sibling"]).strip().lower()
    return len(head) >= 2 and (head == sib or head in sib or sib in head
                                or head[:4] == sib[:4])


# ---------- committee orchestrator ----------------------------------
class SpecialistCommittee:
    """Runs 4 narrow DSPy specialists on signal-gated subsets.

    At large N, the flagged set per field can grow past what gemma4:e2b's
    context (num_ctx=8192) can reason over in one call.  ``chunk_size``
    splits each specialist's gated rows into batches of that size and
    calls the specialist once per chunk.  4 specialists scale as
    ``4 * ceil(|flagged| / chunk_size)`` LLM calls per field.
    """

    def __init__(self, chunk_size: int = 60):
        self.parser_fix = dspy.Predict(ParserMarkerFix)
        self.fname_fix  = dspy.Predict(FieldNameFix)
        self.ns_fix     = dspy.Predict(NsPromote)
        self.split_fix  = dspy.Predict(CompositeSplit)
        self.chunk_size = max(1, int(chunk_size))

    @staticmethod
    def _chunks(seq: List, size: int):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    # ----- sample-block builders (per-specialist) -------------------
    @staticmethod
    def _pm_block(rows: List[Tuple[str, str]]) -> str:
        return "\n".join(f"{g} | {v!r}" for g, v in rows)

    @staticmethod
    def _fn_block(rows: List[Tuple[str, str, str, str]]) -> str:
        # (gsm, cur, near, raw)
        return "\n".join(
            f"{g} | cur={c!r} | near={n!r} | raw={r!r}"
            for g, c, n, r in rows)

    @staticmethod
    def _ns_block(rows: List[Tuple[str, str, str]]) -> str:
        # (gsm, near, raw)
        return "\n".join(
            f"{g} | near={n!r} | raw={r!r}" for g, n, r in rows)

    @staticmethod
    def _unrepr(s: Any) -> Any:
        """Undo Python-repr leakage in specialist output strings.

        Tiny models sometimes copy the `!r` representation (including
        outer quotes and `\\n` escapes) into their JSON output.  Strip
        one layer of matching wrapping quotes and decode common
        escape sequences so the string can match a verbatim substring
        of the sample data.
        """
        if not isinstance(s, str):
            return s
        t = s.strip()
        # Strip matching outer quotes first ('x' or "x").
        if len(t) >= 2 and t[0] in ("'", '"') and t[-1] == t[0]:
            t = t[1:-1]
        # Then strip a single asymmetric leading quote (tiny models
        # sometimes copy the repr's opening " or ' without the close).
        if len(t) >= 1 and t[0] in ("'", '"'):
            t = t[1:]
        # And a single asymmetric trailing quote.
        if len(t) >= 1 and t[-1] in ("'", '"'):
            t = t[:-1]
        # Decode common escapes without invoking full codec machinery.
        for esc, ch in (("\\n", "\n"), ("\\t", "\t"), ("\\'", "'"),
                         ('\\"', '"'), ("\\\\", "\\")):
            t = t.replace(esc, ch)
        return t

    def _run_specialist(self, predictor, **inputs) -> List[Dict]:
        try:
            pred = predictor(**inputs)
        except Exception as e:  # noqa: BLE001
            print(f"    specialist call failed: {e!r}")
            return []
        raw = getattr(pred, "corrections", "") or ""
        items = _extract_json(raw) or []
        # Normalize Python-repr leakage in each proposal's strings.
        for it in items:
            for k in ("current", "suggest", "quote"):
                if k in it:
                    it[k] = self._unrepr(it[k])
        return items

    @staticmethod
    def _mode_sibling(all_samples: List[Dict], field: str,
                      flagged_set: set) -> Optional[str]:
        """Most common non-flagged, non-'Not Specified' value for this field."""
        c: Counter = Counter()
        for s in all_samples:
            if s.get("gsm") in flagged_set:
                continue
            v = (s.get("phase1") or {}).get(field) or ""
            v = v.strip()
            if v and v.lower() != "not specified":
                c[v] += 1
        if not c:
            return None
        return c.most_common(1)[0][0]

    def run(self, field: str, gse_context: Dict,
            all_samples: List[Dict],
            flagged_gsms: List[str],
            enrichment: Dict[str, Dict[str, Any]],
            scores: Dict[str, Dict[str, float]]
            ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """Return (unioned proposals, per-specialist gsm lists).

        Gating philosophy: run EVERY specialist on ALL flagged samples.
        Each specialist is narrow enough that it will return [] on rows
        outside its concern.  This removes the brittle per-gate logic
        and makes the committee robust to new bug shapes.
        """
        if not flagged_gsms:
            return [], {"parser": [], "fieldname": [], "ns": [], "composite": []}

        by_gsm = {s["gsm"]: s for s in all_samples}
        flagged_set = set(flagged_gsms)
        mode_sib = self._mode_sibling(all_samples, field, flagged_set) or ""

        pm_rows: List[Tuple[str, str]] = []
        fn_rows: List[Tuple[str, str, str, str]] = []
        ns_rows: List[Tuple[str, str, str]] = []
        sp_rows: List[Tuple[str, str, str, str]] = []

        for gsm in flagged_gsms:
            s = by_gsm.get(gsm)
            if s is None:
                continue
            val = (s.get("phase1") or {}).get(field) or _NS
            enr = enrichment.get(gsm) or {}
            sc  = scores.get(gsm) or {}
            # Prefer enrichment's nearest_sibling; fall back to mode if NS.
            near = enr.get("nearest_sibling") or mode_sib
            raw = _raw_text(s)

            # Parser: every flagged sample (specialist skips clean rows).
            pm_rows.append((gsm, val))
            # Fieldname gate (signal-based, not pattern-based):
            #   - not NS (NS handled by NsPromote)
            #   - ';'-free (composite handled by CompositeSplit)
            #   - low punctuation density (high punct_z = parser-marker
            #     territory; FieldNameFix would otherwise replace a
            #     parser-artifact value with a raw-text phrase rather
            #     than letting ParserMarkerFix strip the marker)
            _punct    = float(sc.get("punct_z",  0.0))
            _evidence = float(sc.get("evidence", 0.0))
            # Fieldname gate — signal-based (not pattern-based):
            #   not NS; no ';'; low punctuation (else parser territory);
            #   high evidence_gap (value not supported by its own raw
            #   text — KEY-leak signature).  A proper medical phrase
            #   like 'Frontotemporal Dementia' matches its raw spans
            #   closely (low evidence) so it's excluded — preventing
            #   over-correction to abbreviations.
            if (val.strip().lower() != "not specified"
                    and ";" not in val
                    and _punct < 0.5
                    and _evidence > 0.25):
                fn_rows.append((gsm, val, near, raw))
            # NS-promote: every flagged sample whose current IS NS.
            if val.strip().lower() == "not specified" and mode_sib:
                ns_rows.append((gsm, mode_sib, raw))
            # Composite: every flagged sample with ';' in current.
            if ";" in val:
                sp_rows.append((gsm, val, near, raw))

        gated: Dict[str, List[str]] = {
            "parser":    [g for g, _ in pm_rows],
            "fieldname": [g for g, _, _, _ in fn_rows],
            "ns":        [g for g, _, _ in ns_rows],
            "composite": [g for g, _, _, _ in sp_rows],
        }

        proposals: List[Dict] = []

        gse_block = _fmt_gse_ctx(gse_context)

        for chunk in self._chunks(pm_rows, self.chunk_size):
            props = self._run_specialist(
                self.parser_fix,
                field=field,
                samples=self._pm_block(chunk))
            for p in props:
                p["_specialist"] = "parser"
            proposals.extend(props)

        for chunk in self._chunks(fn_rows, self.chunk_size):
            props = self._run_specialist(
                self.fname_fix,
                field=field,
                gse_context=gse_block,
                samples=self._fn_block(chunk))
            for p in props:
                p["_specialist"] = "fieldname"
            proposals.extend(props)

        for chunk in self._chunks(ns_rows, self.chunk_size):
            props = self._run_specialist(
                self.ns_fix,
                field=field,
                gse_context=gse_block,
                samples=self._ns_block(chunk))
            for p in props:
                p["_specialist"] = "ns"
            proposals.extend(props)

        for chunk in self._chunks(sp_rows, self.chunk_size):
            props = self._run_specialist(
                self.split_fix,
                field=field,
                gse_context=gse_block,
                samples=self._fn_block(chunk))
            for p in props:
                p["_specialist"] = "composite"
            proposals.extend(props)

        return proposals, gated


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — existing LLM agent on flagged subset with anomaly payload.
# ──────────────────────────────────────────────────────────────────────
class SemanticPhase1cCurator:
    """Two-stage: 6-signal scanner → LLM judge on flagged candidates."""

    def __init__(self, configure: bool = True,
                 top_quantile: float = 0.70,
                 min_score: float = 0.30,
                 max_flags_per_field: Optional[int] = 25,
                 n_judges: int = 1,
                 judge_configs: Optional[List[Tuple[int, float]]] = None,
                 vote_threshold: Optional[int] = None,
                 specialists: bool = True,
                 specialist_chunk_size: int = 60):
        """
        Scaling knobs for large GSEs (N ≥ 500):
          * ``min_score``             absolute suspicion floor.  At small N
            the quantile threshold dominates; at large N raise this to
            ~0.45 to keep the flagged set tight.  Effective threshold is
            ``max(quantile(top_quantile), min_score)``.
          * ``max_flags_per_field``   hard cap on flagged rows / field.
          * ``specialist_chunk_size`` rows per specialist LLM call.  The
            committee issues ``4 * ceil(|flagged| / chunk_size)`` calls
            per field; Ollama prompt-caches the shared header so each
            chunk amortises well.
        """
        if configure:
            configure_lm()
        self.screen = SemanticCandidateScreen()
        self.agent  = FieldCurator()
        self.top_quantile = top_quantile
        self.min_score    = min_score
        self.max_flags    = max_flags_per_field
        self.specialist_chunk_size = max(1, int(specialist_chunk_size))
        self.n_judges = max(1, int(n_judges))
        # Default diversification: (seed, temperature) pairs.
        # Judge 0 is the deterministic baseline; 1..N add controlled noise
        # so self-consistency has something to aggregate.
        default_cfgs: List[Tuple[int, float]] = [
            (42,   0.0),
            (7,    0.3),
            (99,   0.5),
            (13,   0.2),
            (2025, 0.4),
        ]
        cfgs = judge_configs if judge_configs else default_cfgs
        self.judge_configs: List[Tuple[int, float]] = list(cfgs)[:self.n_judges]
        # Majority threshold: ceil((N+1)/2). Single judge ⇒ 1.
        self.vote_threshold = (int(vote_threshold)
                               if vote_threshold is not None
                               else (self.n_judges // 2) + 1)
        self.use_specialists = bool(specialists)
        self.committee = (SpecialistCommittee(chunk_size=self.specialist_chunk_size)
                          if self.use_specialists else None)

    @staticmethod
    def _build_lm(seed: int, temperature: float,
                  max_tokens: int = 16384, think: bool = True) -> dspy.LM:
        """Build an LM with an explicit (seed, temperature) pair.

        Each ensemble judge runs under ``dspy.context(lm=self._build_lm(...))``
        so the global DSPy LM is not mutated.
        """
        return dspy.LM(
            model=f"ollama_chat/{_PHASE1_MODEL}",
            api_base=_OLLAMA_URL,
            temperature=float(temperature),
            max_tokens=max_tokens,
            num_ctx=_NUM_CTX,
            seed=int(seed),
            think=think,
        )

    def _run_agent(self, field: str, gse_context: Dict,
                   all_samples: List[Dict],
                   flagged_gsms: List[str],
                   enrichment: Dict[str, Dict[str, Any]]
                   ) -> Tuple[List[Dict], str]:
        """Single-judge call against whatever LM is currently configured."""
        if not flagged_gsms:
            return [], ""
        flagged_set = set(flagged_gsms)
        flagged_samples = [s for s in all_samples
                           if s.get("gsm") in flagged_set]

        dist = _distribution(all_samples, field)
        samples_block = _fmt_samples_with_anomaly(
            flagged_samples, field, enrichment)
        pred = self.agent(
            field=field,
            gse_context=_fmt_gse_ctx(gse_context),
            distribution=_fmt_dist(dist, field),
            rare_values=_fmt_rare(dist, field),
            samples=samples_block,
        )
        raw = getattr(pred, "corrections", "") or ""
        proposed = _extract_json(raw)
        hay = _haystack(gse_context, all_samples)
        accepted = _validate(proposed, all_samples, hay, field)
        return accepted, raw

    def _run_agent_ensemble(self, field: str, gse_context: Dict,
                            all_samples: List[Dict],
                            flagged_gsms: List[str],
                            enrichment: Dict[str, Dict[str, Any]]
                            ) -> Tuple[List[Dict], List[Dict[str, Any]]]:
        """Run N judges with diversified (seed, temperature), aggregate.

        Aggregation:  accept (gsm, suggest) iff >= self.vote_threshold of
        the N judges produced that exact (gsm, suggest) pair.  If a gsm
        has multiple passing suggestions, keep the one with the most
        votes (ties broken by first-seen, which is the deterministic
        seed=42,T=0 judge if present).
        """
        if not flagged_gsms:
            return [], []
        per_judge: List[Dict[str, Any]] = []
        for seed, temp in self.judge_configs:
            lm = self._build_lm(seed=seed, temperature=temp)
            with dspy.context(lm=lm):
                acc, raw = self._run_agent(field, gse_context, all_samples,
                                           flagged_gsms, enrichment)
            per_judge.append({
                "seed":     seed,
                "temp":     temp,
                "accepted": acc,
                "raw":      raw,
            })

        votes: Counter = Counter()
        first_seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rec in per_judge:
            for a in rec["accepted"]:
                key = (a["gsm"], a["suggest"])
                votes[key] += 1
                if key not in first_seen:
                    first_seen[key] = dict(a)
        for k, item in first_seen.items():
            item["votes"] = votes[k]
            item["n_judges"] = len(per_judge)

        # Accept items at or above the majority threshold; keep best per gsm.
        passed = [first_seen[k] for k, n in votes.items()
                  if n >= self.vote_threshold]
        by_gsm: Dict[str, Dict[str, Any]] = {}
        for item in sorted(passed, key=lambda x: -x["votes"]):
            if item["gsm"] not in by_gsm:
                by_gsm[item["gsm"]] = item
        voted = list(by_gsm.values())
        return voted, per_judge

    def curate(self, gse_context: Dict,
               samples: List[Dict]) -> Dict[str, Any]:
        accepted: List[Dict] = []
        proposed: List[Dict] = []
        flags: Dict[str, List[str]] = {}
        enrich_all: Dict[str, Dict[str, Dict[str, Any]]] = {}
        scores_all: Dict[str, Dict[str, Dict[str, float]]] = {}
        raw_by_field: Dict[str, str] = {}
        judges_by_field: Dict[str, List[Dict[str, Any]]] = {}
        specialists_by_field: Dict[str, Dict[str, Any]] = {}
        t_screen = 0.0
        t_agent  = 0.0

        for field in ("Tissue", "Condition", "Treatment"):
            t0 = time.time()
            s_table, _embs, _vals = self.screen.score_field(samples, field)
            scores_all[field] = s_table
            flagged, enrichment = self.screen.flag_and_enrich(
                samples, field,
                top_quantile=self.top_quantile,
                min_score=self.min_score,
                max_flags=self.max_flags,
            )
            flags[field] = flagged
            enrich_all[field] = enrichment
            t_screen += time.time() - t0

            t0 = time.time()
            acc, per_judge = self._run_agent_ensemble(
                field, gse_context, samples, flagged, enrichment)
            t_agent += time.time() - t0
            judges_by_field[field] = per_judge
            raw_by_field[field] = per_judge[0]["raw"] if per_judge else ""

            # ── Specialist committee on the same flagged subset ──
            spec_accepted: List[Dict] = []
            if self.committee is not None and flagged:
                t0 = time.time()
                spec_raw, gated = self.committee.run(
                    field, gse_context, samples, flagged,
                    enrichment, s_table)
                t_agent += time.time() - t0
                # Validate through the same _validate used for general judge.
                hay = _haystack(gse_context, samples)
                spec_accepted = _validate(spec_raw, samples, hay, field)
                specialists_by_field[field] = {
                    "gated":    gated,
                    "proposed": spec_raw,
                    "accepted": spec_accepted,
                }

            # Union by gsm (general wins ties; specialists fill misses).
            gsm_seen = {a["gsm"] for a in acc}
            union = list(acc)
            for a in spec_accepted:
                if a["gsm"] not in gsm_seen:
                    union.append(a)
                    gsm_seen.add(a["gsm"])
            accepted.extend(union)
            proposed.extend(union)

        return {
            "samples":   samples,
            "accepted":  accepted, "rejected": [], "proposed": proposed,
            "flags":     flags,
            "enrichment": enrich_all,
            "scores":    scores_all,
            "raw_response": raw_by_field,
            "judges":    judges_by_field,
            "specialists": specialists_by_field,
            "n_judges":  self.n_judges,
            "vote_threshold": self.vote_threshold,
            "elapsed":   {"screen": t_screen, "agent": t_agent},
        }


__all__ = ["SemanticPhase1cCurator", "SemanticCandidateScreen",
           "_fmt_samples_with_anomaly"]
