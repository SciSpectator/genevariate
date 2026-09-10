"""Driver that runs the vendored newest ``geo_label_extractor`` pipeline
(SciSpectator/LLM-GEO-Label-Extractor, origin/main 0bb57cd) from inside
GeneVariate against an OpenAI-compatible server hosting the extractor's own two
models: gemma-4 12B for phase 1 and gemma-4 E2B for Age and phase 2.

Two entry points:

* :func:`extract_labels` - the in-process Stage-1 primitive. Given GEO sample
  metadata dicts, it runs ``Phase1Extractor().extract`` per sample and returns the
  five controlled fields (Sex, Age, Tissue, Condition, Treatment). It needs only
  the LLM backend, no MeSH / Cellosaurus / BioLORD reference databases.

* :func:`run_full_pipeline` - the three-stage driver (extract -> normalize ->
  assemble) exposed by the vendored ``geo_pipeline.main``. Stage 2 requires the
  prebuilt reference artifacts (vocab / index / cellosaurus); when they are
  absent this driver runs Stage 1 (+1b) and assembly only and reports what
  Stage 2 needs, rather than failing.

The vendored tree tracks upstream verbatim except for one patch, carried in
``geo_pipeline.py``: upstream always scraped NCBI for GSE-level context, which
this driver turns into an opt-in flag (``--scrape-gse`` /
:func:`set_gse_scrape`) so no run reaches the network without being asked to.

The vendored package uses absolute (``from geo_label_extractor.x import y``) and
bare sibling (``from mesh_lookup import z``) imports, so both the ``_vendor``
directory and the package directory are placed on ``sys.path`` ahead of any
externally pip-installed copy.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Iterable, List, Optional

# ── vendored package location ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.normpath(os.path.join(_HERE, "..", "_vendor"))
_PKG = os.path.join(_VENDOR, "geo_label_extractor")

ALL_FIELDS = ("Sex", "Age", "Tissue", "Condition", "Treatment")
NS = "Not Specified"

# The models the vendored pipeline is defined against, and the endpoint it
# expects them on. These are geo_pipeline's own defaults, not GeneVariate's
# choice: the phase-1 prompt artifacts were optimized against the 12B with
# reasoning off, and Age/phase-2 against the e2B with reasoning on. Serving
# anything else produces labels that are not the extractor's labels.
DEFAULT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "google/gemma-4-12b-it"
DEFAULT_AGE_MODEL = "google/gemma-4-e2b-it"
DEFAULT_PHASE2_MODEL = "google/gemma-4-e2b-it"


def _ensure_path() -> None:
    """Put the vendored package first so it wins over any external install."""
    for p in (_PKG, _VENDOR):
        if p not in sys.path:
            sys.path.insert(0, p)
        elif sys.path.index(p) != 0:
            sys.path.remove(p)
            sys.path.insert(0, p)


# ── Automatic hardware-aware throttling ──────────────────────────────────
# The vendored pipeline is tuned for datacenter GPUs (extract-workers=64,
# num_ctx=8192, cuda embeddings, large token budgets). Running that as-is on a
# laptop / small consumer GPU can exhaust VRAM/RAM. We detect the machine and
# cap the token / context / worker / embedding knobs the pipeline reads from the
# environment so it stays within the local budget instead of killing the box.
def _detect_vram_gb() -> float:
    """Largest single-GPU VRAM in GiB (best effort); 0.0 if no CUDA GPU."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return max(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())) / (1024 ** 3)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = [float(x) for x in out.stdout.split()
                if x.strip().replace(".", "", 1).isdigit()]
        if vals:
            return max(vals) / 1024.0  # MiB -> GiB
    except Exception:
        pass
    return 0.0


def _detect_ram_gb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def resource_tier(vram_gb: Optional[float] = None,
                  ram_gb: Optional[float] = None) -> Dict[str, object]:
    """Pick a conservative parameter tier from detected GPU VRAM + system RAM.

    Returns ``{"tier", "vram_gb", "ram_gb", "params"}`` where ``params`` holds
    the token/context/worker/embedding caps applied by :func:`regulate_resources`.
    """
    vram = _detect_vram_gb() if vram_gb is None else float(vram_gb)
    ram = _detect_ram_gb() if ram_gb is None else float(ram_gb)
    if vram <= 0:                       # CPU-only: smallest footprint
        name, p = "cpu", dict(
            p1_max=160, ctx=2048, col_workers=1, pool=4, extract_workers=2,
            embed_device="cpu", p2_max=384, p2_ceiling=1536, p2_think=128,
            p2_qbatch=16, p2_embed_batch=32, p2_topk=4)
    elif vram < 8:                      # small consumer GPU (<8 GB)
        name, p = "small", dict(
            p1_max=192, ctx=3072, col_workers=1, pool=8, extract_workers=3,
            embed_device="cpu", p2_max=512, p2_ceiling=2048, p2_think=192,
            p2_qbatch=32, p2_embed_batch=64, p2_topk=6)
    elif vram < 12:                     # e.g. GTX 1080 Ti 11 GB
        name, p = "medium", dict(
            p1_max=256, ctx=4096, col_workers=2, pool=32, extract_workers=4,
            embed_device="cuda", p2_max=768, p2_ceiling=3072, p2_think=256,
            p2_qbatch=64, p2_embed_batch=128, p2_topk=8)
    elif vram < 24:                     # 16 GB class
        name, p = "large", dict(
            p1_max=512, ctx=8192, col_workers=3, pool=128, extract_workers=8,
            embed_device="cuda", p2_max=1024, p2_ceiling=4096, p2_think=384,
            p2_qbatch=128, p2_embed_batch=512, p2_topk=8)
    else:                              # 24 GB+ / datacenter
        name, p = "xlarge", dict(
            p1_max=1024, ctx=8192, col_workers=3, pool=256, extract_workers=16,
            embed_device="cuda", p2_max=1024, p2_ceiling=6144, p2_think=512,
            p2_qbatch=256, p2_embed_batch=1024, p2_topk=8)
    # System RAM caps concurrency regardless of VRAM (each worker holds context).
    if ram and ram < 8:
        p["extract_workers"] = min(p["extract_workers"], 2)
        p["col_workers"], p["pool"] = 1, min(p["pool"], 8)
    elif ram and ram < 16:
        p["extract_workers"] = min(p["extract_workers"], 6)
        p["pool"] = min(p["pool"], 64)
    return {"tier": name, "vram_gb": round(vram, 1),
            "ram_gb": round(ram, 1), "params": p}


def regulate_resources(force: bool = False) -> Dict[str, object]:
    """Set the pipeline's token/context/worker/embed env knobs for this machine.

    Uses ``setdefault`` (so an explicit user-set env always wins) unless
    ``force``. Set ``GEO_NO_AUTOSCALE=1`` to skip entirely. Returns the tier.
    """
    if os.environ.get("GEO_NO_AUTOSCALE", "").lower() in ("1", "true", "yes"):
        return {"tier": "disabled", "vram_gb": 0.0, "ram_gb": 0.0, "params": {}}
    t = resource_tier()
    p = t["params"]

    def sd(key: str, val: object) -> None:
        if force:
            os.environ[key] = str(val)
        else:
            os.environ.setdefault(key, str(val))

    sd("PHASE1_MAX_TOKENS", p["p1_max"])
    sd("LLM_NUM_CTX", p["ctx"])
    sd("LABEL_COL_WORKERS", p["col_workers"])
    sd("LABEL_POOL_SIZE", p["pool"])
    sd("PHASE2_EMBED_DEVICE", p["embed_device"])
    sd("PHASE2_MAX_TOKENS", p["p2_max"])
    sd("PHASE2_TOKEN_CEILING", p["p2_ceiling"])
    sd("PHASE2_THINK_BUDGET", p["p2_think"])
    sd("PHASE2_GPU_QUERY_BATCH", p["p2_qbatch"])
    sd("PHASE2_EMBED_BATCH", p["p2_embed_batch"])
    sd("PHASE2_TOPK", p["p2_topk"])
    # Consumed as the default --extract-workers when the caller passes auto (0).
    os.environ["GEO_EXTRACT_WORKERS"] = str(p["extract_workers"])
    return t


def resolve_backend(url: str = "", model: str = "",
                    age_model: str = "", phase2_model: str = ""
                    ) -> Dict[str, str]:
    """Which server and models a call would use, without changing anything.

    Explicit arguments win; otherwise an endpoint already exported in the
    environment is honoured, and only then the pipeline's own defaults. That
    order is what lets a remote GPU be used by exporting ``VLLM_URL`` and the
    three model names before starting the program.
    """
    return {
        "url": (url or os.environ.get("VLLM_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or DEFAULT_URL).rstrip("/"),
        "model": model or os.environ.get("PHASE1_MODEL") or DEFAULT_MODEL,
        # Age and phase-2 have their own model; they do not inherit phase 1's.
        "age_model": (age_model or os.environ.get("AGE_MODEL")
                      or DEFAULT_AGE_MODEL),
        "phase2_model": (phase2_model or os.environ.get("PHASE2_MODEL")
                         or DEFAULT_PHASE2_MODEL),
    }


def configure_backend(url: str = "", model: str = "",
                      age_model: Optional[str] = None,
                      phase2_model: Optional[str] = None) -> None:
    """Point the vendored LLM backend at an OpenAI-compatible server.

    ``url`` is the ``/v1`` base the backend appends ``/chat/completions`` to.
    Empty arguments fall back to the environment and then to the pipeline's own
    defaults - see :func:`resolve_backend`.
    """
    picked = resolve_backend(url, model, age_model or "", phase2_model or "")
    os.environ["VLLM_URL"] = picked["url"]
    os.environ["OPENAI_BASE_URL"] = picked["url"]
    os.environ["PHASE1_MODEL"] = picked["model"]
    os.environ["AGE_MODEL"] = picked["age_model"]
    os.environ["PHASE2_MODEL"] = picked["phase2_model"]
    os.environ.setdefault("LLM_BACKEND", "vllm")
    # Reasoning is part of the extractor's definition, not a local knob: phase 1
    # runs the 12B with reasoning off, Age runs the e2B with it on. Both are
    # upstream defaults, so nothing is set here.
    #
    # Hardware-aware throttling: cap token/context/worker/embed knobs to the
    # detected GPU VRAM + system RAM so the datacenter-tuned pipeline does not
    # exhaust a local machine (set GEO_NO_AUTOSCALE=1 to disable).
    regulate_resources()


def backend_reachable(url: str = "", timeout: float = 4.0) -> bool:
    """True if an OpenAI-compatible server answers at ``url``."""
    import requests
    base = resolve_backend(url)["url"]
    root = base[:-3] if base.endswith("/v1") else base
    for probe in (base + "/models", root + "/api/tags"):
        try:
            r = requests.get(probe, timeout=timeout)
            if r.status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


# ── Stage 1 in-process primitive ─────────────────────────────────────────
def _raw_from_sample(s: Dict) -> Dict[str, str]:
    """Map a GeneVariate/GEOmetadb sample row to the keys phase1 reads.

    The five published fields are resolved under the alias names the prompt
    artifacts were optimized against. Every other key on the row is carried
    through unchanged, because a non-default column selection is looked up by
    its GEOmetadb name (see ``metadata_fields.describe``).
    """
    def g(*keys: str) -> str:
        for k in keys:
            v = s.get(k)
            if v not in (None, ""):
                return str(v)
        return ""
    raw = {
        "gsm_title":          g("gsm_title", "title", "GSM_title"),
        "source_name":        g("source_name", "source_name_ch1", "source"),
        "characteristics":    g("characteristics", "characteristics_ch1"),
        "treatment_protocol": g("treatment_protocol", "treatment_protocol_ch1"),
        "description":        g("description"),
    }
    for k, v in s.items():
        if k not in raw and v not in (None, ""):
            raw[str(k)] = str(v)
    return raw


def extract_labels(
    samples: Iterable[Dict],
    fields: Iterable[str] = ALL_FIELDS,
    *,
    url: str = "",
    model: str = "",
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, str]]:
    """Run Phase-1 verbatim extraction over ``samples`` via the configured LLM.

    Each input dict provides GEO metadata (``gsm``/``title``/``source_name``/
    ``characteristics``/``treatment_protocol``/``description``; ``_ch1`` variants
    are accepted). Returns one dict per sample carrying the original ``gsm`` (if
    present) plus the requested label fields.
    """
    _ensure_path()
    configure_backend(url=url, model=model)
    from geo_label_extractor.phase1 import Phase1Extractor  # noqa: E402

    agent = Phase1Extractor()
    want = [f for f in fields if f in ALL_FIELDS]
    rows = list(samples)
    out: List[Dict[str, str]] = []
    total = len(rows)
    for i, s in enumerate(rows, 1):
        raw = _raw_from_sample(s)
        rec: Dict[str, str] = {}
        gsm = s.get("gsm") or s.get("GSM")
        if gsm:
            rec["gsm"] = str(gsm)
        for col in want:
            rec[col] = agent.extract_field(raw, col)
        out.append(rec)
        if progress is not None:
            progress(i, total, str(gsm or f"sample {i}"))
    return out


# ── Sample-table input ───────────────────────────────────────────────────
def materialize_samples(table_path: str, out_dir: str) -> str:
    """Write a ``run_cli`` JSON manifest from a GeneVariate sample-meta table.

    The vendored pipeline reads GEOmetadb SQLite or a JSON manifest, but a
    platform downloaded through GeneVariate publishes its metadata as the
    ``*_sample_meta.csv.gz`` table instead. The columns already carry the names
    the extractor resolves (``title``, ``source_name``, ``characteristics``,
    ``treatment_protocol``, ``description``, ``series_id``); only the accession
    is spelled ``GSM``, so that is the one thing renamed. Every other column is
    carried through, which is what makes a non-default ``--fields`` selection
    still resolvable.
    """
    import pandas as pd

    df = pd.read_csv(table_path, dtype=str).fillna("")
    if "gsm" not in df.columns:
        for alias in ("GSM", "gsm_id", "accession"):
            if alias in df.columns:
                df = df.rename(columns={alias: "gsm"})
                break
    if "gsm" not in df.columns:
        raise SystemExit(
            f"{table_path} has no sample-accession column "
            f"(looked for gsm/GSM/gsm_id/accession); columns are "
            f"{', '.join(df.columns)}")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "input_samples.json")
    import json
    with open(target, "w") as fh:
        json.dump(df.to_dict(orient="records"), fh)
    return target


# ── Stage 2 reference-database availability ──────────────────────────────
def phase2_reference_status(vocab: str = "", index: str = "",
                            cellosaurus: str = "",
                            mesh_db: str = "") -> Dict[str, object]:
    """Report whether the Stage-2 reference artifacts are present.

    ``mesh_db`` is optional: the vendored ``mesh_lookup`` falls back to the
    ``mesh.sqlite`` shipped next to it, so an empty value is not a failure - but
    that copy is only reported present when it actually exists, because the
    extractor's own preflight aborts the run without it.
    """
    paths = {"vocab": vocab, "index": index, "cellosaurus": cellosaurus}
    present = {k: bool(v) and os.path.exists(v) for k, v in paths.items()}
    mesh = mesh_db or _vendored_mesh_db()
    paths["mesh_db"] = mesh
    present["mesh_db"] = bool(mesh) and os.path.exists(mesh)
    return {"available": all(present.values()),
            "present": present, "paths": paths}


def _vendored_mesh_db() -> str:
    """Where ``mesh_lookup`` looks for ``mesh.sqlite`` when MESH_DB is unset."""
    return os.environ.get("MESH_DB", os.path.join(_PKG, "mesh.sqlite"))


def reference_paths() -> Dict[str, str]:
    """Resolve the Stage-2 reference artifacts, by environment or convention.

    Callers that want the normalization stage should not each invent their own
    way of naming four large files. ``GEO_VOCAB`` / ``GEO_INDEX`` /
    ``GEO_CELLOSAURUS`` / ``MESH_DB`` win when set; otherwise the conventional
    filenames are looked for in ``GEO_REFS``. With neither, two conventional
    directories are searched: ``refs/`` beside the vendored package, and
    ``rnaseq_labels/refs/`` at the checkout root - the index alone is 1.7 GB, so
    it is normally staged outside the package rather than inside it. A path that
    does not exist is reported as empty rather than passed on, because the
    pipeline would abort on it much later.
    """
    env_refs = os.environ.get("GEO_REFS", "")
    dirs = [env_refs] if env_refs else [
        os.path.join(_PKG, "refs"),
        os.path.normpath(os.path.join(_HERE, "..", "..", "..",
                                      "rnaseq_labels", "refs")),
    ]

    def _pick(env: str, filename: str) -> str:
        chosen = os.environ.get(env, "")
        if not chosen:
            chosen = next((p for p in (os.path.join(d, filename) for d in dirs)
                           if os.path.exists(p)), "")
        return chosen if os.path.exists(chosen) else ""

    return {
        "vocab": _pick("GEO_VOCAB", "vocab.sqlite"),
        "index": _pick("GEO_INDEX", "vocab_index.npz"),
        "cellosaurus": _pick("GEO_CELLOSAURUS", "cellosaurus.sqlite"),
        "mesh_db": _pick("MESH_DB", "mesh.sqlite"),
    }


# ── Selectable metadata columns ──────────────────────────────────────────
def default_metadata_columns() -> List[str]:
    """The published five GEOmetadb columns the extractor reads by default."""
    _ensure_path()
    from geo_label_extractor import metadata_fields  # noqa: E402
    return list(metadata_fields.DEFAULT_COLUMNS)


def set_metadata_columns(columns: Optional[Iterable[str]] = None) -> List[str]:
    """Choose the columns in-process extraction puts in front of the model.

    Applies to :func:`extract_labels` and anything else that goes through the
    vendored phase-1 prompt in this process; ``None`` restores the published
    default. Returns the selection in force.
    """
    _ensure_path()
    from geo_label_extractor import metadata_fields  # noqa: E402
    cols = [str(c).strip() for c in (columns or []) if str(c).strip()]
    if not cols:
        os.environ.pop(metadata_fields.ENV_VAR, None)
        return list(metadata_fields.DEFAULT_COLUMNS)
    metadata_fields.publish(cols)
    return cols


def metadata_columns(input_path: str = "") -> List[Dict[str, object]]:
    """Columns offerable for ``input_path``, in the order the picker shows them.

    Each entry is ``{"column", "prompt", "description", "default"}``. Without an
    input, or when the file is not a readable GEOmetadb, only the default five
    are offered - the pipeline itself falls back the same way.
    """
    _ensure_path()
    from geo_label_extractor import metadata_fields  # noqa: E402
    default = list(metadata_fields.DEFAULT_COLUMNS)
    cols = metadata_fields.available(input_path) if input_path else default
    ordered = default + [c for c in cols if c not in default]
    out: List[Dict[str, object]] = []
    for c in ordered:
        f = metadata_fields.describe(c)
        out.append({"column": f.column, "prompt": f.prompt,
                    "description": f.description, "default": c in default})
    return out


# ── GSE-context scrape (opt-in) ──────────────────────────────────────────
# Phase 1b can fall back to the study-level description when a sample's own
# metadata is silent, and the vendored run_cli fetches that description from
# NCBI. Upstream scrapes unconditionally; here it is off unless the user asks
# for it, because it is one network round-trip per GSE and it writes a sidecar
# cache that a later run would reuse. The GUI exposes the switch.
_SCRAPE_GSE = False


def set_gse_scrape(enabled: bool) -> bool:
    """Turn the NCBI GSE-context scrape on or off for this process."""
    global _SCRAPE_GSE
    _SCRAPE_GSE = bool(enabled)
    return _SCRAPE_GSE


def gse_scrape_enabled() -> bool:
    """Whether :func:`run_full_pipeline` will scrape GSE context by default."""
    return _SCRAPE_GSE


# ── Run staging, shared by the API and the console script ────────────────
def _stage_run(out_dir: str, *, mesh_db: str = "",
               index: str = "") -> tuple[Dict[str, Optional[str]], str]:
    """Prepare ``out_dir`` and the environment the vendored pipeline needs.

    geo_pipeline shells out to run_cli.py / run_phase2.py, so the vendored
    package has to be importable in those subprocesses, and the reference stores
    have to be named by environment because ``mesh_lookup`` resolves them at
    import time. The extractor's preflight also requires the GSE-context cache to
    already exist, while the pipeline only builds it further down, *after* the
    check -- so without this every run on a fresh out-dir aborts. Returns the
    environment to restore afterwards, plus the cache path.
    """
    keys = ("PYTHONPATH", "MESH_DB", "MESH_INDEX", "GSE_CONTEXT_CACHE")
    prev_env: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
    os.makedirs(out_dir, exist_ok=True)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [_PKG, _VENDOR] + ([prev_env["PYTHONPATH"]] if prev_env["PYTHONPATH"] else []))
    if mesh_db:
        os.environ["MESH_DB"] = mesh_db
    if index:
        os.environ["MESH_INDEX"] = index
    ctx_db = os.path.join(out_dir, "gse_context_cache.sqlite")
    os.environ["GSE_CONTEXT_CACHE"] = ctx_db
    from gse_context_cache import GSEContextCache  # noqa: E402
    GSEContextCache()
    return prev_env, ctx_db


# ── Full three-stage pipeline (extract -> normalize -> assemble) ─────────
def run_full_pipeline(
    input_path: str,
    out_dir: str,
    *,
    url: str = "",
    model: str = "",
    labels: Iterable[str] = ALL_FIELDS,
    limit: int = 0,
    vocab: str = "",
    index: str = "",
    cellosaurus: str = "",
    mesh_db: str = "",
    gpls: Optional[Iterable[str]] = None,
    gsms: Optional[Iterable[str]] = None,
    extract_workers: int = 0,
    stop_after: str = "",
    fields: Optional[Iterable[str]] = None,
    tech: str = "",
    organism: str = "",
    scrape_gse: Optional[bool] = None,
) -> Dict[str, object]:
    """Invoke the vendored ``geo_pipeline.main`` for all connected stages.

    ``stop_after`` selects the extraction depth the way the vendored GUI does -
    ``"phase1"`` (verbatim only), ``"phase1b"`` (+ GSE-context recovery), or
    ``"phase2"`` (full extract -> normalize -> assemble). When empty it is chosen
    automatically: ``phase2`` if the Stage-2 reference artifacts are present, else
    ``phase1b`` (normalization/assembly skipped) and the returned dict says so.
    ``extract_workers<=0`` uses the hardware-regulated default. ``input_path`` may
    be a GEOmetadb SQLite file, a ``run_cli``-compatible JSON manifest, or a
    GeneVariate ``*_sample_meta.csv.gz`` table (see :func:`materialize_samples`).

    ``fields`` chooses which GEOmetadb ``gsm`` columns the model reads; ``None``
    keeps the published five-column default that the extractor paper's numbers
    depend on (see :func:`metadata_columns`). ``tech`` and ``organism`` restrict
    a SQLite input by platform technology (``"high-throughput sequencing"`` for
    RNA-seq) and species; both empty means no restriction.

    ``scrape_gse`` fetches each study's title/summary from NCBI for phase-1b
    context; ``None`` follows the process setting (:func:`set_gse_scrape`,
    off by default).

    ``mesh_db`` points the extractor's MeSH store somewhere other than the copy
    beside the vendored package - a run built from freshly rebuilt reference
    databases must not silently fall back to a stale one.
    """
    _ensure_path()
    configure_backend(url=url, model=model)
    picked = resolve_backend(url, model)
    os.makedirs(out_dir, exist_ok=True)

    if extract_workers <= 0:
        extract_workers = int(os.environ.get("GEO_EXTRACT_WORKERS", "4"))

    if input_path.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        input_path = materialize_samples(input_path, out_dir)

    ref = phase2_reference_status(vocab, index, cellosaurus, mesh_db)
    # Phase-2 needs the reference artifacts; downgrade a phase2 request when they
    # are absent so we still produce verbatim + recovery labels instead of erroring.
    want_phase2 = (stop_after == "phase2") or (not stop_after and ref["available"])
    run_stage2 = bool(want_phase2 and ref["available"])
    if stop_after in ("phase1", "phase1b"):
        depth = stop_after
    else:
        depth = "phase2" if run_stage2 else "phase1b"
    argv = [
        "--input", input_path,
        "--out-dir", out_dir,
        "--backend", os.environ.get("LLM_BACKEND", "vllm"),
        "--llm-url", picked["url"],
        # Age and phase-2 run on their own model in the published configuration;
        # collapsing all three onto the phase-1 model would silently change what
        # normalisation was done with.
        "--phase1-model", picked["model"],
        "--age-model", picked["age_model"],
        "--phase2-model", picked["phase2_model"],
        "--extract-workers", str(extract_workers),
    ]
    if labels:
        argv += ["--labels", ",".join(f for f in labels if f in ALL_FIELDS)]
    # Omitting --fields is not the same as passing the default set: the pipeline
    # records "published default" in its manifest only when the flag is absent.
    cols = [str(c).strip() for c in (fields or []) if str(c).strip()]
    if cols and tuple(cols) != tuple(default_metadata_columns()):
        argv += ["--fields", ",".join(cols)]
    if tech:
        argv += ["--tech", tech]
    if organism:
        argv += ["--organism", organism]
    if limit:
        argv += ["--limit", str(limit)]
    for g in (gpls or []):
        argv += ["--gpl", str(g)]
    for g in (gsms or []):
        argv += ["--gsm", str(g)]
    scrape = _SCRAPE_GSE if scrape_gse is None else bool(scrape_gse)
    if scrape:
        argv += ["--scrape-gse"]

    if depth == "phase2":
        argv += ["--vocab", vocab, "--index", index,
                 "--cellosaurus", cellosaurus, "--stop-after", "phase2"]
    else:
        # Verbatim (phase1) or +recovery (phase1b): no MeSH/Cellosaurus/BioLORD
        # normalization -> run extraction only, skip the final assemble merge.
        argv += ["--stop-after", depth, "--no-assemble"]

    prev_env, ctx_db = _stage_run(out_dir, mesh_db=mesh_db, index=index)
    prev_argv = sys.argv[:]
    try:
        from geo_label_extractor import geo_pipeline  # noqa: E402
        sys.argv = ["geo_pipeline"] + argv
        rc = geo_pipeline.main()
    finally:
        sys.argv = prev_argv
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return {"returncode": rc, "out_dir": out_dir,
            "stage2_ran": bool(depth == "phase2"),
            "stop_after": depth, "reference": ref, "scrape_gse": scrape,
            "gse_context_cache": ctx_db, "argv": argv}


def pipeline_progress(out_dir: str) -> int:
    """How many samples the extraction stage has written so far, or -1.

    ``run_cli`` flushes a partial snapshot every ``EXTRACT_SNAPSHOT_EVERY``
    samples, and puts ``samples_done`` first in it, so the count is readable
    from the first few bytes without parsing a file that grows to the size of
    the whole run.
    """
    import re
    snap = os.path.join(out_dir, "extracted.json.partial.json")
    try:
        with open(snap, "rb") as fh:
            head = fh.read(64).decode("utf-8", "ignore")
    except OSError:
        return -1
    m = re.search(r'"samples_done"\s*:\s*(\d+)', head)
    return int(m.group(1)) if m else -1


def read_pipeline_labels(out_dir: str, labels: Iterable[str] = ALL_FIELDS):
    """The pipeline's labels as a GeneVariate label frame (GSM + one col/field).

    Reads the deepest artifact the run actually produced. When normalization and
    assembly ran, that is ``final_labels/LLM_labels_all_samples.csv.gz``, whose
    ``final_*`` columns are the MeSH / Cellosaurus / BioLORD-normalized labels
    - those, not the verbatim spans, are what the extractor publishes. When only
    extraction ran, it is ``extracted.json``, where each record keeps its
    ``phase1`` / ``phase1b`` / ``phase2`` dicts and the deepest non-empty one is
    the label for that field.

    The returned frame carries ``attrs['stage']`` naming which of those it read,
    so a caller can say out loud whether the labels were normalized.

    The accession columns normalization wrote beside each value are carried
    through as well. They are what distinguishes a resolved MeSH concept from a
    catalogued cell line from a locally minted identifier, and that distinction
    cannot be reconstructed from the value alone, so dropping them here would
    make an in-program extraction strictly poorer than the same file loaded
    from disk.
    """
    import json
    import pandas as pd

    fields = [f for f in labels if f in ALL_FIELDS] or list(ALL_FIELDS)
    corpus = os.path.join(out_dir, "final_labels", "LLM_labels_all_samples.csv.gz")
    if os.path.exists(corpus):
        raw = pd.read_csv(corpus, dtype=str).fillna("")
        out = pd.DataFrame({"GSM": raw.get("gsm", pd.Series(dtype=str))})
        for f in fields:
            out[f] = raw.get(f"final_{f}", "").replace("", NS)
            for suffix in ("_id", "_mesh_id", "_cell_id", "_stage", "_curated"):
                col = f"final_{f}{suffix}"
                if col in raw.columns:
                    out[col] = raw[col]
        for src, dst in (("gse", "series_id"), ("gpl", "gpl")):
            if src in raw.columns:
                out[dst] = raw[src]
        out.attrs["stage"] = "phase2"
        return out

    extracted = os.path.join(out_dir, "extracted.json")
    if not os.path.exists(extracted):
        partial = extracted + ".partial.json"
        if not os.path.exists(partial):
            return pd.DataFrame()
        with open(partial, encoding="utf-8") as fh:
            records = json.load(fh).get("samples", [])
        stage = "phase1b (partial)"
    else:
        with open(extracted, encoding="utf-8") as fh:
            records = json.load(fh)
        stage = "phase1b"

    rows = []
    for rec in records:
        row = {"GSM": str(rec.get("gsm") or "").strip().upper(),
               "series_id": rec.get("gse") or "",
               "gpl": rec.get("gpl") or ""}
        for f in fields:
            value = NS
            for key in ("phase2", "phase1b", "phase1"):
                got = str(((rec.get(key) or {}).get(f) or "")).strip()
                if got and got.lower() not in ("not specified", "nan", "none"):
                    value = got
                    break
            row[f] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["stage"] = stage
    return out


def cli(argv: Optional[List[str]] = None) -> object:
    """Console-script entry (``genevariate-llm-extract``).

    Forwards to the vendored ``geo_label_extractor.geo_pipeline`` CLI with the
    local backend configured, so the full vendored pipeline is available from
    the command line. It stages the run the same way :func:`run_full_pipeline`
    does -- otherwise the command line would abort on the extractor's preflight
    where the API call succeeds.
    """
    _ensure_path()
    configure_backend()
    if argv is not None:
        sys.argv = ["geo_pipeline", *argv]

    def _opt(name: str) -> str:
        args = sys.argv[1:]
        if name in args and len(args) > args.index(name) + 1:
            return args[args.index(name) + 1]
        prefix = name + "="
        for a in args:
            if a.startswith(prefix):
                return a[len(prefix):]
        return ""

    out_dir = _opt("--out-dir")
    if out_dir:
        _stage_run(out_dir, index=_opt("--index"))
        source = _opt("--input")
        if source.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
            sys.argv[sys.argv.index(source)] = materialize_samples(source, out_dir)

    from geo_label_extractor import geo_pipeline  # noqa: E402
    return geo_pipeline.main()


__all__ = [
    "ALL_FIELDS", "NS", "DEFAULT_URL", "DEFAULT_MODEL",
    "DEFAULT_AGE_MODEL", "DEFAULT_PHASE2_MODEL",
    "configure_backend", "resolve_backend", "backend_reachable",
    "extract_labels",
    "phase2_reference_status", "reference_paths", "run_full_pipeline",
    "pipeline_progress", "read_pipeline_labels", "cli",
    "default_metadata_columns", "metadata_columns", "set_metadata_columns",
    "set_gse_scrape", "gse_scrape_enabled", "materialize_samples",
    "resource_tier", "regulate_resources",
]
