#!/usr/bin/env python3
"""Phase 1c sibling-canonicalization pass.

Universal post-step that converges phase 1c verbatim surface forms
within a single GSE. Same biological state described two ways
(``former smoker`` vs ``former smoking``, ``never`` vs ``never smoker``)
collapses to ONE canonical surface per GSE — but ``former smoker``
and ``never smoker`` STAY DISTINCT because they encode different
biological states (different polarity / different temporal qualifier).

Algorithm (per (gse, col)):
  1. Collect every distinct phase1c value across the GSE's samples.
  2. Tokenize each value: lowercase alphabetic tokens minus stopwords;
     stem to drop plural/-ing/-er morphology.
  3. Detect the "entity stem" — the stem that appears in ≥ 70% of
     distinct surfaces. The 70% strict-majority threshold prevents
     a 2-surface case ({Control, Depression}) from declaring BOTH
     stems as entity (each would hit 50%) and collapsing to one
     class. Universal: the entity is whatever the corpus itself
     says is dominantly common. No allow-list.
  4. Build a fingerprint per surface:
        fp = (stems − {entity_stem}, digit-tokens)
     i.e. the DISTINGUISHING modifiers (former / never / current /
     post / ex / current / mock / sham / etc.) plus any code-digits.
  5. Group surfaces by identical fingerprint. ``former smoker`` and
     ``former smoking`` and ``former`` all reduce to fp=({"form"},∅)
     → same class; ``never smoker`` reduces to fp=({"neve"},∅) → own
     class; ``smoking (0)`` reduces to fp=(∅,{"0"}) → own class.
  6. Within each class, pick the canonical surface = most-common
     value, ties broken by shortest, then alphabetic.
  7. Rewrite each sample's phase1c[col] to its class canonical.
     Stash the original under phase1c_raw[col] for audit.

Universal: corpus-derived entity detection, no allow-lists, no
GSE-name in the loop, no per-study branches. Same logic runs for
any column on any GSE in any input. Reproducible — pure Python, no
LLM, deterministic.

Usage:
    python run_phase1c_canonicalize.py \
        --input  _extract_results_v5/out_full.json \
        --output _extract_results_v5/out_full.canon_phase1c.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABEL_COLS = ("Tissue", "Condition", "Treatment")
NS = "Not Specified"

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "is", "and", "or", "with", "to",
    "for", "in", "on", "at", "from", "by", "as",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def _tokens(s: str) -> set[str]:
    """Lowercase alphabetic tokens minus stopwords."""
    if not s:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(s)
            if t.lower() not in _STOPWORDS}


def _stem(t: str) -> str:
    """Tiny lemma rule: strip plural -s and agent-noun -er/-ers/-ing."""
    if t.endswith("ers"):    return t[:-3]
    if t.endswith("ing"):    return t[:-3]
    if t.endswith("er"):     return t[:-2]
    if t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def _stems(s: str) -> set[str]:
    return {_stem(t) for t in _tokens(s)}


def _is_ns(v: str | None) -> bool:
    if v is None:
        return True
    return str(v).strip().lower() in ("", "not specified", "ns", "none", "n/a", "na")


_DIGIT_RE = re.compile(r"\d+")


def _equivalence_classes(values: list[str]) -> list[list[str]]:
    """Group surfaces by identical fingerprint = (distinguishing
    stems, digit tokens). Entity stems (≥50% surface coverage) are
    subtracted because they're the shared backdrop, not what
    distinguishes one biological state from another."""
    if not values:
        return []
    n = len(values)
    surface_stems: dict[str, set[str]] = {v: _stems(v) for v in values}

    # Entity stems = stems present in ≥50% of distinct surfaces.
    stem_freq: Counter = Counter()
    for s in surface_stems.values():
        for t in s:
            stem_freq[t] += 1
    entity_stems = {st for st, c in stem_freq.items() if c / n >= 0.70}

    # Fingerprint each surface.
    fp: dict[str, tuple] = {}
    for v in values:
        distinguishing = frozenset(surface_stems[v] - entity_stems)
        digits = frozenset(_DIGIT_RE.findall(v))
        fp[v] = (distinguishing, digits)

    classes: dict[tuple, list[str]] = defaultdict(list)
    for v in values:
        classes[fp[v]].append(v)
    return list(classes.values())


def _canonical_of_class(cls: list[str], counts: Counter) -> str:
    """Pick canonical: highest count, then shortest, then alphabetic."""
    return min(cls, key=lambda v: (-counts[v], len(v), v))


def canonicalize(samples: list[dict]) -> dict:
    """Mutates samples in place; returns stats dict."""
    by_gse: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_gse[s.get("gse", "")].append(s)

    n_samples_changed = 0
    n_surfaces_collapsed = 0
    by_col_examples: dict[str, list[str]] = defaultdict(list)

    for gse, sibs in by_gse.items():
        for col in LABEL_COLS:
            # Per-GSE / per-col distribution from phase1c.
            vals = [(s.get("phase1c") or {}).get(col) or NS for s in sibs]
            counts = Counter(v for v in vals if not _is_ns(v))
            if len(counts) <= 1:
                continue  # nothing to converge

            classes = _equivalence_classes(list(counts.keys()))
            class_canon: dict[str, str] = {}
            for cls in classes:
                if len(cls) == 1:
                    continue
                canon = _canonical_of_class(cls, counts)
                for v in cls:
                    class_canon[v] = canon
                n_surfaces_collapsed += len(cls) - 1
                if len(by_col_examples[col]) < 6:
                    by_col_examples[col].append(
                        f"[{gse}/{col}] " + " | ".join(
                            f"{v!r}({counts[v]})" for v in sorted(
                                cls, key=lambda x: -counts[x])) +
                        f"  →  {canon!r}")

            # Apply per sample. Stash the original under phase1c_raw.
            for s in sibs:
                p1c = s.setdefault("phase1c", {})
                cur = p1c.get(col) or NS
                if cur in class_canon:
                    if cur != class_canon[cur]:
                        s.setdefault("phase1c_raw", {})[col] = cur
                        p1c[col] = class_canon[cur]
                        n_samples_changed += 1

    return {
        "samples_changed": n_samples_changed,
        "surfaces_collapsed": n_surfaces_collapsed,
        "examples": dict(by_col_examples),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.input).read_text())
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if samples is None:
        print(f"[phase1c-canon] no 'samples' key in {args.input}", file=sys.stderr)
        return 1

    stats = canonicalize(samples)

    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))

    print(f"[phase1c-canon] {stats['samples_changed']} samples updated, "
          f"{stats['surfaces_collapsed']} surfaces collapsed")
    for col, exs in stats["examples"].items():
        print(f"  --- {col} ---")
        for ex in exs:
            print(f"  {ex}")
    print(f"[phase1c-canon] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
