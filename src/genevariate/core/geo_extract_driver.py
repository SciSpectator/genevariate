"""Driver that runs the vendored ``geo_label_extractor`` pipeline from GeneVariate."""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ── vendored package location ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.normpath(os.path.join(_HERE, "..", "_vendor"))
_PKG = os.path.join(_VENDOR, "geo_label_extractor")

ALL_FIELDS = ("Sex", "Age", "Tissue", "Condition", "Treatment")
NS = "Not Specified"

DEFAULT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "google/gemma-4-12b-it"
DEFAULT_AGE_MODEL = "google/gemma-4-e2b-it"
DEFAULT_PHASE2_MODEL = "google/gemma-4-e2b-it"

_USER_BACKEND = os.environ.get("LLM_BACKEND", "").strip().lower()


def _ensure_path() -> None:
    """Put the vendored package first so it wins over any external install."""
    for p in (_PKG, _VENDOR):
        if p not in sys.path:
            sys.path.insert(0, p)
        elif sys.path.index(p) != 0:
            sys.path.remove(p)
            sys.path.insert(0, p)


# ── Automatic hardware-aware throttling ──────────────────────────────────
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
    """Returns ``{"tier", "vram_gb", "ram_gb", "params"}`` for this machine."""
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
    """Set the pipeline's token/context/worker/embed env knobs; returns the tier."""
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
    sd("GEO_EXTRACT_WORKERS", p["extract_workers"])
    return t


def resolve_backend(url: str = "", model: str = "",
                    age_model: str = "", phase2_model: str = ""
                    ) -> Dict[str, str]:
    """Returns ``{"url", "model", "age_model", "phase2_model"}``, changing nothing."""
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


def _backend_for(base_url: str) -> Tuple[str, str]:
    """Which backend name serves ``base_url``, and that server's root."""
    root = base_url[:-3].rstrip("/") if base_url.endswith("/v1") else base_url
    if _USER_BACKEND:
        return _USER_BACKEND, root
    from .ollama_manager import ollama_server_ok
    return ("ollama" if ollama_server_ok(root, timeout=2) else "vllm"), root


def configure_backend(url: str = "", model: str = "",
                      age_model: Optional[str] = None,
                      phase2_model: Optional[str] = None) -> None:
    """Point the vendored LLM backend at an OpenAI-compatible server."""
    picked = resolve_backend(url, model, age_model or "", phase2_model or "")
    os.environ["VLLM_URL"] = picked["url"]
    os.environ["OPENAI_BASE_URL"] = picked["url"]
    os.environ["PHASE1_MODEL"] = picked["model"]
    os.environ["AGE_MODEL"] = picked["age_model"]
    os.environ["PHASE2_MODEL"] = picked["phase2_model"]
    os.environ.setdefault("LLM_BACKEND", "vllm")
    backend, root = _backend_for(picked["url"])
    if backend == "ollama":
        os.environ["OLLAMA_URL"] = root
    os.environ["PHASE1_BACKEND"] = backend
    regulate_resources()


def backend_reachable(url: str = "", timeout: float = 4.0) -> bool:
    """True if an OpenAI-compatible server answers at ``url``."""
    import requests
    base = resolve_backend(url)["url"]
    root = base[:-3] if base.endswith("/v1") else base
    for probe in (base + "/models", root + "/api/tags"):
        try:
            r = requests.get(probe, timeout=timeout)
            if r.status_code < 300 or r.status_code in (401, 403):
                return True
        except requests.RequestException:
            continue
    return False


# ── Stage 1 in-process primitive ─────────────────────────────────────────
def _raw_from_sample(s: Dict) -> Dict[str, str]:
    """Map a GeneVariate/GEOmetadb sample row to the keys phase1 reads."""
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
    """Run Phase-1 verbatim extraction over ``samples``; one dict per sample out."""
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
    """Write a ``run_cli`` JSON manifest from a GeneVariate sample-meta table."""
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
    """Returns ``{"available", "present", "paths"}`` for the Stage-2 artifacts."""
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
    """Resolve the Stage-2 reference artifacts, by environment or convention."""
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
    """Choose the columns in-process extraction puts in front of the model."""
    _ensure_path()
    from geo_label_extractor import metadata_fields  # noqa: E402
    cols = [str(c).strip() for c in (columns or []) if str(c).strip()]
    if not cols:
        os.environ.pop(metadata_fields.ENV_VAR, None)
        return list(metadata_fields.DEFAULT_COLUMNS)
    metadata_fields.publish(cols)
    return cols


def metadata_columns(input_path: str = "") -> List[Dict[str, object]]:
    """Each entry is ``{"column", "prompt", "description", "default"}``."""
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
    """Prepare ``out_dir``; returns the environment to restore, plus the cache path."""
    keys = ("PYTHONPATH", "MESH_DB", "MESH_INDEX", "GSE_CONTEXT_CACHE")
    prev_env: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "normalized"), exist_ok=True)
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
    """Invoke the vendored ``geo_pipeline.main`` for all connected stages."""
    _ensure_path()
    configure_backend(url=url, model=model)
    picked = resolve_backend(url, model)
    os.makedirs(out_dir, exist_ok=True)

    if extract_workers <= 0:
        extract_workers = int(os.environ.get("GEO_EXTRACT_WORKERS", "4"))

    if input_path.lower().endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        input_path = materialize_samples(input_path, out_dir)

    staged = reference_paths()
    vocab = vocab or staged["vocab"]
    index = index or staged["index"]
    cellosaurus = cellosaurus or staged["cellosaurus"]
    mesh_db = mesh_db or staged["mesh_db"]

    ref = phase2_reference_status(vocab, index, cellosaurus, mesh_db)
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
        "--phase1-model", picked["model"],
        "--age-model", picked["age_model"],
        "--phase2-model", picked["phase2_model"],
        "--extract-workers", str(extract_workers),
    ]
    if labels:
        argv += ["--labels", ",".join(f for f in labels if f in ALL_FIELDS)]
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
    """How many samples the extraction stage has written so far, or -1."""
    import re
    snap = os.path.join(out_dir, "extracted.json.partial.json")
    try:
        with open(snap, "rb") as fh:
            head = fh.read(64).decode("utf-8", "ignore")
    except OSError:
        return -1
    m = re.search(r'"samples_done"\s*:\s*(\d+)', head)
    return int(m.group(1)) if m else -1


def read_pipeline_labels(out_dir: str, labels: Iterable[str] = ALL_FIELDS,
                         depth: str = ""):
    """The pipeline's labels as a label frame (GSM + one col/field), ``attrs['stage']`` set."""
    import json
    import pandas as pd

    fields = [f for f in labels if f in ALL_FIELDS] or list(ALL_FIELDS)
    corpus = os.path.join(out_dir, "final_labels", "LLM_labels_all_samples.csv.gz")
    if os.path.exists(corpus):
        raw = pd.read_csv(corpus, dtype=str).fillna("")
        out = pd.DataFrame({"GSM": raw.get("gsm", pd.Series(dtype=str))})
        for f in fields:
            out[f] = raw.get(f"final_{f}", "").replace("", NS)
            for suffix in ("_id", "_source", "_mesh_id", "_cell_id",
                           "_stage", "_curated"):
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
        stage = f"{depth or 'phase1b'} (partial)"
    else:
        with open(extracted, encoding="utf-8") as fh:
            records = json.load(fh)
        # The extractor wraps the samples in a manifest carrying the counts.
        if isinstance(records, dict):
            records = records.get("samples", [])
        stage = depth or "phase1b"

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
    """Console-script entry (``genevariate-llm-extract``)."""
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
