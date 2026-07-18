"""GeneVariate CLI for the LLM-GEO-Label-Extractor pipeline.

Wraps the vendored upstream ``run_cli.py`` (now :mod:`upstream_cli`) with a
genevariate-friendly argument surface:

* ``--phases``     pick one or more of {p1, p1b, p1c, p2, all}; default all
* reasoning        permanently ON (gemma think mode, no toggle)
* router           permanently ON (Phase 2 router agent, no toggle)
* ``--semantic``   default ON  (use ``--no-semantic`` to disable)
* ``--scrape``     default ON  (use ``--no-scrape`` to disable)
* ``--resume``     default ON  (use ``--no-resume`` to disable)

All other upstream flags (``--samples``, ``--output``, ``--workers``,
``--model``, ``--backend`` …) are forwarded verbatim. Run ``--help`` for the
full list.

This wrapper never edits the upstream source; it just translates our flags
into the upstream argv vocabulary and delegates to ``upstream_cli.main``.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

# Ensure vendored modules resolve via the package's sys.path tweak.
from . import upstream_cli as _upstream  # noqa: F401  (side-effect import)


PHASES = ("p1", "p1b", "p1c", "p2")
PHASE_TO_DISABLE_FLAG = {
    "p1":  "--no-p1",
    "p1b": "--no-p1b",
    "p1c": "--no-p1c",
    "p2":  "--no-p2",
}


def _parse(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        prog="genevariate-llm-extract",
        description=(
            "Run the LLM-GEO-Label-Extractor pipeline on GEO samples. "
            "Pick which phases to execute. Reasoning and the Phase 2 "
            "router are permanently ON. Semantic curation is on by "
            "default (--no-semantic to disable)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Phase selection ──────────────────────────────────────────────
    p.add_argument(
        "--phases", nargs="+", default=["all"],
        metavar="PHASE",
        help=(
            "Phases to run (space-separated). Choices: "
            "p1, p1b, p1c, p2, all. Default: all."
        ),
    )

    # ── Capability toggles (reasoning + router are permanently ON) ───
    sgrp = p.add_mutually_exclusive_group()
    sgrp.add_argument("--semantic",    dest="semantic", action="store_true",
                      default=True,  help="enable Phase 1c semantic curator (default)")
    sgrp.add_argument("--no-semantic", dest="semantic", action="store_false",
                      help="disable BioLORD semantic curator")

    scgrp = p.add_mutually_exclusive_group()
    scgrp.add_argument("--scrape",    dest="scrape", action="store_true",
                       default=True,  help="scrape GSE metadata from NCBI (default)")
    scgrp.add_argument("--no-scrape", dest="scrape", action="store_false",
                       help="skip NCBI GSE-metadata scrape")

    rsgrp = p.add_mutually_exclusive_group()
    rsgrp.add_argument("--resume",    dest="resume", action="store_true",
                       default=True,  help="resume from checkpoint if present (default)")
    rsgrp.add_argument("--no-resume", dest="resume", action="store_false",
                       help="ignore existing checkpoint")

    # ── I/O ──────────────────────────────────────────────────────────
    p.add_argument("--samples",      required=False)
    p.add_argument("--output",       required=False)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--gpl",          default=None)
    p.add_argument("--search-gpl",   default=None, metavar="QUERY",
                   help="standalone GPL search; exits after printing")
    p.add_argument("--tech",         default=None)
    p.add_argument("--organism",     default=None)
    p.add_argument("--list-gpl-limit", type=int, default=50)
    p.add_argument("--geometadb",    default=None)

    # ── Execution ────────────────────────────────────────────────────
    p.add_argument("--workers",     type=int, default=1,
                   help="parallel collapsers within a GSE (1..32)")
    p.add_argument("--gse-workers", type=int, default=1,
                   help="concurrent GSEs (raise only if Phase 2 disabled)")
    p.add_argument("--limit",       type=int, default=0,
                   help="cap input to first N samples (0 = all)")
    p.add_argument("--backend",     default="ollama",
                   choices=["ollama", "vllm", "sglang", "openai"])
    p.add_argument("--model",       default="gemma4-e2b-text:latest")
    p.add_argument("--ollama-url",  default="http://localhost:11434")
    p.add_argument("--vllm-url",    default="http://localhost:8000/v1")

    args = p.parse_args(list(argv))

    # ── Validate phase selection ─────────────────────────────────────
    phases = {x.lower() for x in args.phases}
    if "all" in phases:
        phases = set(PHASES)
    bad = phases - set(PHASES)
    if bad:
        p.error(f"unknown phase(s): {sorted(bad)}. "
                f"Valid: {list(PHASES)} or 'all'.")
    if not phases:
        p.error("at least one phase is required")
    args.phases_resolved = phases

    # Build the argv that the upstream CLI expects.
    fwd: list[str] = []
    if args.samples:        fwd += ["--samples", str(args.samples)]
    if args.output:         fwd += ["--output", str(args.output)]
    if args.checkpoint:     fwd += ["--checkpoint", str(args.checkpoint)]
    if args.gpl:            fwd += ["--gpl", args.gpl]
    if args.search_gpl is not None:
        fwd += ["--search-gpl", args.search_gpl]
    if args.tech:           fwd += ["--tech", args.tech]
    if args.organism:       fwd += ["--organism", args.organism]
    if args.list_gpl_limit: fwd += ["--list-gpl-limit", str(args.list_gpl_limit)]
    if args.geometadb:      fwd += ["--geometadb", str(args.geometadb)]
    fwd += ["--workers",     str(args.workers)]
    fwd += ["--gse-workers", str(args.gse_workers)]
    fwd += ["--limit",       str(args.limit)]
    fwd += ["--backend",     args.backend]
    fwd += ["--model",       args.model]
    fwd += ["--ollama-url",  args.ollama_url]
    fwd += ["--vllm-url",    args.vllm_url]

    # Phase disablement (only fires for phases NOT selected)
    for phase in PHASES:
        if phase not in args.phases_resolved:
            fwd.append(PHASE_TO_DISABLE_FLAG[phase])

    # Capability negations — reasoning + router are hard-wired ON
    # upstream; only the optional curator and resume/scrape are toggleable.
    if not args.semantic:  fwd.append("--no-semantic-1c")
    if not args.scrape:    fwd.append("--no-scrape")
    if not args.resume:    fwd.append("--no-resume")

    return args, fwd


def main(argv: Sequence[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    args, fwd = _parse(raw)

    print(
        "[llm-extract] phases=" + ",".join(sorted(args.phases_resolved))
        + "  reasoning=ON  router=ON"
        + f"  semantic={args.semantic}"
        + f"  scrape={args.scrape}"
        + f"  resume={args.resume}",
        flush=True,
    )

    # Replace sys.argv so the upstream argparse sees our translated flags.
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["upstream_cli"] + fwd
        _upstream.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
