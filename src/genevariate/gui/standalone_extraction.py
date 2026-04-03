#!/usr/bin/env python3
"""
GeneVariate — Standalone LLM Label Extraction Pipeline
======================================================
Extracts labels for an entire GPL platform, then runs all 4 phases:
  Phase 1    Stateless per-sample LLM extraction (parallel, GPU)
  Phase 1.5  Per-GSE label normalization (majority-vote variant unification)
  Phase 2    Context Re-extraction — NS curation using experiment context
  Phase 3    Cross-experiment harmonization (negation, synonyms, noise, hierarchy)

Usage:
  python standalone_extraction.py --gpl GPL6947 --data-dir ./data
  python standalone_extraction.py --gpl GPL570  --model gemma2:9b --workers 10

Requirements:
  - Ollama running locally (ollama serve)
  - GEOmetadb.sqlite.gz in data directory
  - pip install pandas ollama psutil
"""

import os, sys, re, json, gzip, shutil, time, sqlite3, uuid, subprocess, threading
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

import pandas as pd
import ollama

# ── Deterministic extraction (ported from geo_ns_repair_v2.py) ──
try:
    from .deterministic_extraction import (
        MemoryAgent, GSEContext, GSEWorker,
        build_gse_contexts, is_ns as _is_ns
    )
    _HAS_DET = True
except ImportError:
    try:
        from genevariate.gui.deterministic_extraction import (
            MemoryAgent, GSEContext, GSEWorker,
            build_gse_contexts, is_ns as _is_ns
        )
        _HAS_DET = True
    except ImportError:
        _HAS_DET = False
        print("[WARN] deterministic_extraction.py not found — LLM-only mode")

_MEMORY_AGENT = None
_GSE_CONTEXTS = {}

# ── Thread-local HTTP session (for non-LLM calls only) ──
_tls = threading.local()

def _get_session():
    if not hasattr(_tls, "s") or _tls.s is None:
        _tls.s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        _tls.s.mount("http://", a)
    return _tls.s

_OLLAMA_URL = "http://localhost:11434"

def _ollama_post(prompt, model=None, num_predict=None, num_ctx=None, timeout=120):
    """Call Ollama using ollama.chat() — matching old working code.
    The ollama library handles connection pooling and GPU queuing internally.
    """
    options = {'temperature': 0.0}
    if num_predict is not None:
        options['num_predict'] = num_predict
    if num_ctx is not None:
        options['num_ctx'] = num_ctx

    for attempt in range(1, 4):
        try:
            response = ollama.chat(
                model=model or _OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options=options
            )
            content = response.get('message', {}).get('content', '').strip()
            if not content and attempt < 3:
                time.sleep(2)
                continue
            return content
        except Exception as e:
            err = str(e).lower()
            if 'connection' in err or 'refused' in err or '503' in err:
                if attempt == 3:
                    print(f"[LLM ERROR] Cannot connect to Ollama: {e}")
                    return ""
                time.sleep(min(3 * attempt, 15))
            else:
                print(f"[LLM ERROR] {type(e).__name__}: {e}")
                if attempt == 3:
                    return ""
                time.sleep(2 * attempt)
    return ""

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ═══════════════════════════════════════════════════════════════
#  GPU Detection
# ═══════════════════════════════════════════════════════════════
def detect_gpus():
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({"id": int(parts[0]), "name": parts[1],
                             "vram_gb": round(int(parts[2])/1024, 1),
                             "free_vram_gb": round(int(parts[3])/1024, 1)})
    except Exception:
        pass
    return gpus


def check_ollama_gpu(base_url="http://localhost:11434"):
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url}/api/ps", method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = data.get("models", [])
            if models:
                vram = models[0].get("size_vram", 0)
                total = models[0].get("size", 1)
                if vram > total * 0.5:
                    return "gpu", round(vram / 1e9, 1)
                return "cpu", 0
    except Exception:
        pass
    return "unknown", 0


def _get_vram_usage():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=3)
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        u, t = int(parts[0]), int(parts[1])
        return u, t, 100.0 * u / t if t else 0.0
    except Exception:
        return 0, 0, 0.0


# ═══════════════════════════════════════════════════════════════
#  Resource Watchdog (RAM + VRAM)
# ═══════════════════════════════════════════════════════════════
class ResourceWatchdog:
    RAM_PAUSE = 90
    RAM_RESUME = 85
    VRAM_PAUSE = 90
    VRAM_RESUME = 80

    def __init__(self, log_fn=print):
        self._log = log_fn
        self._gate = threading.Event()
        self._gate.set()
        self._stop = threading.Event()
        self.status = ""

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self._gate.set()

    def wait(self):
        self._gate.wait()

    def _loop(self):
        if not HAS_PSUTIL:
            return
        while not self._stop.is_set():
            ram = psutil.virtual_memory().percent
            vu, vt, vp = _get_vram_usage()
            has_gpu = vt > 0
            state = "running" if self._gate.is_set() else "PAUSED"
            self.status = (f"RAM:{ram:.0f}% | VRAM:{vu}/{vt}MB ({vp:.0f}%) | {state}"
                           if has_gpu else f"RAM:{ram:.0f}% | {state}")

            if ram >= self.RAM_PAUSE and self._gate.is_set():
                self._gate.clear()
                self._log(f"  [Watchdog] RAM {ram:.0f}% — PAUSED")
            elif has_gpu and vp >= self.VRAM_PAUSE and self._gate.is_set():
                self._gate.clear()
                self._log(f"  [Watchdog] VRAM {vp:.0f}% — PAUSED")
            elif not self._gate.is_set():
                if ram < self.RAM_RESUME and ((not has_gpu) or vp < self.VRAM_RESUME):
                    self._gate.set()
                    self._log(f"  [Watchdog] Resources OK — RESUMED")
            self._stop.wait(3)


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════
FIELDS = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']
NS_CURATE_FIELDS = ['Condition', 'Tissue', 'Treatment']

_NOT_SPECIFIED_VALUES = {
    'Not Specified', 'not specified', 'Not specified',
    'N/A', 'n/a', 'NA', 'na', 'nan', 'NaN', 'None', 'none',
    'Unknown', 'unknown', 'UNKNOWN', '', 'Parse Error',
}


# ═══════════════════════════════════════════════════════════════
#  LLM Helpers
# ═══════════════════════════════════════════════════════════════
_OLLAMA_MODEL = None

def detect_model():
    global _OLLAMA_MODEL
    preferred = ['gemma2:9b', 'gemma2', 'llama3:8b', 'mistral', 'phi3', 'qwen2']
    try:
        resp = ollama.list()
        avail = [m.get('name', m.get('model', '')) for m in resp.get('models', [])]
        avail_map = {m.split(':')[0].lower(): m for m in avail}
        avail_map.update({m.lower(): m for m in avail})
        for p in preferred:
            if p.lower() in avail_map:
                _OLLAMA_MODEL = avail_map[p.lower()]
                return _OLLAMA_MODEL
        if avail:
            _OLLAMA_MODEL = avail[0]
            return _OLLAMA_MODEL
    except Exception:
        pass
    return None


def get_sample_text(row):
    """Build metadata text for LLM prompt.
    CRITICAL: characteristics_ch1 is tab/semicolon-separated — split into lines.
    """
    import re as _re_st
    parts = []
    row_dict = dict(row) if hasattr(row, 'items') else {}
    lower_map = {k.lower(): k for k in row_dict.keys()}

    def _get(field):
        val = row.get(field, None)
        if val is None:
            actual = lower_map.get(field.lower())
            if actual:
                val = row.get(actual, None)
        if val and str(val).strip() and str(val).lower() not in ('nan', 'none', ''):
            return str(val).strip()
        return None

    v = _get('title')
    if v: parts.append(f"title: {v}")
    v = _get('source_name_ch1')
    if v: parts.append(f"source_name: {v}")
    v = _get('characteristics_ch1')
    if v:
        items = _re_st.split(r'[\t;]+|\\t', v)
        for item in items:
            item = item.strip()
            if item and len(item) > 1:
                parts.append(f"  {item}")
    v = _get('description')
    if v: parts.append(f"description: {v}")
    v = _get('treatment_protocol_ch1')
    if v: parts.append(f"treatment_protocol: {v}")
    v = _get('organism_ch1')
    if v: parts.append(f"organism: {v}")
    sid = row.get('series_id', row.get('gse', None))
    if sid and str(sid).strip() and str(sid).lower() not in ('nan', 'none'):
        parts.append(f"experiment: {sid}")
    return "\n".join(parts) if parts else "No metadata"


# ═══════════════════════════════════════════════════════════════
#  Phase 1 — classify_sample
# ═══════════════════════════════════════════════════════════════
_GSE_WORKERS_SA = {}  # {gse_id: GSEWorker} cache for standalone

def classify_sample(row, fields=None):
    """Full agent pipeline using GSEWorker from geo_ns_repair_v2.py architecture."""
    gsm_id = row.get('gsm', row.get('GSM', f"UNK_{uuid.uuid4().hex[:6]}"))
    
    if fields is None:
        fields = ['Condition', 'Tissue', 'Age', 'Treatment', 'Treatment_Time']

    if _HAS_DET and _MEMORY_AGENT is not None:
        gse_id = str(row.get('series_id', '')).strip()
        if gse_id and gse_id.lower() not in ('nan', 'none', ''):
            if gse_id not in _GSE_WORKERS_SA:
                ctx = _GSE_CONTEXTS.get(gse_id, GSEContext(gse_id))
                model = _OLLAMA_MODEL or 'gemma2:9b'
                platform = str(row.get('gpl', '')).strip()
                _GSE_WORKERS_SA[gse_id] = GSEWorker(
                    gse_id, ctx, mem_agent=_MEMORY_AGENT,
                    model=model, platform=platform)
            return _GSE_WORKERS_SA[gse_id].classify_sample(row, fields=fields)
    
    return _classify_sample_llm(row, fields) or {'gsm': gsm_id, **{f: 'Not Specified' for f in fields}}


def _classify_sample_llm(row, fields=None):
    """LLM extraction fallback using ollama.chat()."""
    global _OLLAMA_MODEL
    if _OLLAMA_MODEL is None:
        _OLLAMA_MODEL = detect_model()
        if not _OLLAMA_MODEL:
            raise RuntimeError("No Ollama model found")

    gsm_id = row.get('gsm', row.get('GSM', f"UNK_{uuid.uuid4().hex[:6]}"))
    text = get_sample_text(row)

    SCHEMAS = {
        "Condition": "string: specific disease or 'Control'",
        "Tissue": "string: organ, 'Cell Line: X', or 'Cell Type: X'",
        "Age": "string: e.g. '35 years', '72', 'Not specified'",
        "Treatment": "string: e.g. 'LPS', 'Dexamethasone', 'Not specified'",
        "Treatment_Time": "string: e.g. '24h', '6 hours', 'Not specified'",
    }
    if fields is None:
        fields = list(SCHEMAS.keys())
    schema = {f: SCHEMAS[f] for f in fields if f in SCHEMAS}

    if not text or len(text.strip()) < 10:
        return {'gsm': gsm_id, **{k: 'Not Specified' for k in schema}}

    empty_template = {k: "" for k in schema}

    prompt = f"""TASK: Extract biological metadata from this GEO sample.
RULES:
1. Condition: Extract the SPECIFIC disease, disorder, or phenotype.
   "disease: AML" = "Acute Myeloid Leukemia". "downsyndrome.status: DS" = "Down Syndrome".
   "status: control" or "disease: none" or healthy/normal/WT = "Control".
   Knockout mice (e.g. PS-/-) = name the deficiency (e.g. "Prosaposin Deficiency").
   NEVER generic "Cancer" — always specific: "Breast Cancer", "Glioblastoma", etc.

2. Tissue: Extract the anatomical tissue/organ from source_name and characteristics.
   "source_name: PS-/- cerebrum" = "Cerebrum". "tissue: hippocampus" = "Hippocampus".
   "Cell Line: X" ONLY if a specific cell line is named (e.g. MCF-7, HeLa, A549).
   "Cell Type: X" ONLY if cell type named (e.g. Macrophage, T Cell).
   If no tissue stated, infer from disease: Breast Cancer=Breast, Leukemia=Blood, GBM=Brain.

3. Age: Extract numeric age from characteristics.
   "age (y): 25" = "25". "age: 11 months" = "11 months". "25 days old" = "25 days".

4. Treatment: Extract drug, compound, or stimulus applied to the sample.
   "treatment: 10nM estradiol" = "Estradiol". "LPS stimulation" = "LPS".
   Knockout/transgenic is a CONDITION not a treatment.

5. Treatment_Time: Duration of treatment.
   "time: 24h" = "24h". "treated for 6 hours" = "6 hours".

6. CODED VALUES: Use the study_title and study_summary to decode numeric codes.
   If study is about "smokers vs non-smokers" and sample has "smoking: 0" → Condition = "Control".
   If sample has "genotype: transgenic" and study is about Alzheimer → Condition = "Alzheimer Disease Model".

7. FORMAT: Use Title Case. "Not specified" ONLY if information is genuinely absent from ALL metadata.

METADATA:
{text}

Fill in the JSON (replace empty strings with extracted values):
{json.dumps(empty_template)}
"""

    for attempt in range(2):
        try:
            raw = _ollama_post(prompt, timeout=120)
            if not raw:
                if attempt == 0:
                    continue
                return {'gsm': gsm_id, **{k: 'Not Specified' for k in schema}}
            raw = re.sub(r'```json\s*|\s*```', '', raw, flags=re.DOTALL).strip()
            s, e = raw.find('{'), raw.rfind('}')
            if s >= 0 and e > s:
                raw = raw[s:e+1]
            data = json.loads(raw)
            result = {'gsm': gsm_id}
            for k in schema:
                v = str(data.get(k, 'Not Specified')).strip()
                result[k] = v if v else 'Not Specified'
            return result
        except json.JSONDecodeError:
            if attempt == 0:
                prompt = f"Return ONLY valid JSON, no other text.\n{prompt}"
        except Exception as ex:
            if attempt == 1:
                return {'gsm': gsm_id, **{k: 'Not Specified' for k in schema}}
    return {'gsm': gsm_id, **{k: 'Not Specified' for k in schema}}


# ═══════════════════════════════════════════════════════════════
#  Phase 1 — Batch extraction with progress
# ═══════════════════════════════════════════════════════════════
def run_phase1(samples_df, max_workers=10, watchdog=None):
    total = len(samples_df)
    print(f"\n  Phase 1 — Extracting {total:,} samples ({max_workers} workers)...")
    t0 = time.time()
    results = []
    done = 0
    fails = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(classify_sample, row): row
                for _, row in samples_df.iterrows()}
        for fut in as_completed(futs):
            if watchdog:
                watchdog.wait()
            try:
                res = fut.result()
                if res:
                    results.append(res)
                else:
                    fails += 1
            except Exception:
                fails += 1
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                sps = done / max(0.01, elapsed)
                eta = (total - done) / max(0.01, sps)
                wd = f" | {watchdog.status}" if watchdog else ""
                sys.stdout.write(
                    f"\r  [{done:,}/{total:,}] {sps:.2f} smp/s | "
                    f"ETA {timedelta(seconds=int(eta))} | ok={len(results)} fail={fails}{wd}   ")
                sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 1 done: {len(results):,} samples in {timedelta(seconds=int(elapsed))}")

    if not results:
        return None

    p1 = pd.DataFrame(results)
    p1 = p1.rename(columns={'gsm': 'GSM'})
    p1['GSM'] = p1['GSM'].astype(str).str.strip().str.upper()

    # Carry over metadata columns
    for mc in ['title', 'source_name_ch1', 'characteristics_ch1', 'series_id', 'gpl']:
        if mc in samples_df.columns and mc not in p1.columns:
            mapping = samples_df.set_index(
                samples_df['gsm'].astype(str).str.strip().str.upper())[mc]
            p1[mc] = p1['GSM'].map(mapping)

    # NS summary
    for f in FIELDS:
        if f in p1.columns:
            ns = p1[f].astype(str).str.strip().isin(_NOT_SPECIFIED_VALUES).sum()
            print(f"    {f:<20} {ns:>5}/{len(p1)} NS ({ns*100//len(p1)}%)")
    return p1


# ═══════════════════════════════════════════════════════════════
#  Phase 1.5 — Per-GSE Label Normalization
# ═══════════════════════════════════════════════════════════════
def run_phase15(p1):
    print(f"\n  Phase 1.5 — Per-GSE normalization on {len(p1):,} samples...")
    t0 = time.time()
    df = p1.copy()
    SKIP = {'Age', 'Treatment_Time', 'age', 'treatment_time'}

    if 'series_id' not in df.columns:
        print("    No series_id — skipping.")
        return df

    label_cols = [c for c in df.columns
                  if c not in ('GSM', 'gsm', 'series_id', 'gpl', '_platform')
                  and c not in SKIP and df[c].dtype == 'object']
    n_merged = 0
    for gse_id, group in df.groupby('series_id'):
        if len(group) < 2:
            continue
        for col in label_cols:
            vals = group[col].fillna('Not Specified').astype(str).str.strip()
            real = [v for v in vals if v.lower() not in ('not specified', 'n/a', 'unknown', 'nan', '')]
            if len(real) < 2:
                continue
            counter = Counter(real)
            if len(counter) <= 1:
                continue
            canonical = counter.most_common(1)[0][0]
            can_low = canonical.lower().replace(' ', '').replace('-', '').replace('_', '')
            for val, cnt in counter.items():
                if val == canonical:
                    continue
                v_low = val.lower().replace(' ', '').replace('-', '').replace('_', '')

                # CRITICAL: never merge labels with different numbers
                val_nums = re.findall(r'\d+', val)
                can_nums = re.findall(r'\d+', canonical)
                if val_nums != can_nums:
                    continue

                merge = False
                if v_low == can_low:
                    merge = True
                elif (can_low in v_low or v_low in can_low) and len(v_low) > 3 and len(can_low) > 3:
                    merge = True
                elif len(val) > 4 and len(canonical) > 4 and not val_nums:
                    common = sum(1 for a, b in zip(sorted(v_low), sorted(can_low)) if a == b)
                    ratio = common / max(len(v_low), len(can_low))
                    if ratio > 0.85 and abs(len(val) - len(canonical)) <= 3:
                        merge = True
                if merge:
                    idx = group.index
                    mask = df.loc[idx, col] == val
                    n = mask.sum()
                    if n > 0:
                        df.loc[idx[mask], col] = canonical
                        n_merged += n

    elapsed = time.time() - t0
    print(f"  ✓ Phase 1.5 done: {n_merged} variants unified in {timedelta(seconds=int(elapsed))}")
    return df


# ═══════════════════════════════════════════════════════════════
#  Phase 2 — Context Re-extraction (NS Curation)
# ═══════════════════════════════════════════════════════════════
def _fetch_gse(gse_id):
    """Fetch GSE description from NCBI GEO website."""
    import urllib.request
    url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
           f"?acc={gse_id}&targ=self&form=text&view=quick")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GeneVariate/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode('utf-8', errors='replace')
        result = {'title': '', 'summary': '', 'overall_design': ''}
        for line in text.split('\n'):
            if line.startswith('!Series_title'):
                result['title'] = line.split('=', 1)[1].strip() if '=' in line else ''
            elif line.startswith('!Series_summary'):
                val = line.split('=', 1)[1].strip() if '=' in line else ''
                result['summary'] = (result['summary'] + ' ' + val).strip()
            elif line.startswith('!Series_overall_design'):
                val = line.split('=', 1)[1].strip() if '=' in line else ''
                result['overall_design'] = (result['overall_design'] + ' ' + val).strip()
        # Store FULL description — no truncation in cache
        return result if result['title'] else None
    except Exception:
        return None


def run_phase2(df, watchdog=None, cache_path=None):
    """Phase 2: For each NS entry in Condition/Tissue/Treatment, re-extract with context."""
    print(f"\n  Phase 2 — Context Re-extraction on {len(df):,} samples...")
    t0 = time.time()

    # Count NS
    ns_slots = []
    for _, row in df.iterrows():
        gsm = str(row.get('GSM', '')).strip().upper()
        for f in NS_CURATE_FIELDS:
            if f in df.columns:
                v = str(row.get(f, '')).strip()
                if v in _NOT_SPECIFIED_VALUES:
                    ns_slots.append((gsm, f))
    print(f"    {len(ns_slots)} NS slots to curate")
    if not ns_slots:
        print("    Nothing to curate.")
        return df

    # Get unique GSEs
    gses = [str(g).strip() for g in df['series_id'].dropna().unique()
            if str(g).strip() and str(g).strip().upper().startswith('GSE')]

    # Load/build GSE cache
    gse_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                gse_cache = json.load(f)
            print(f"    Loaded {len(gse_cache)} cached GSE descriptions")
        except Exception:
            pass

    need = [g for g in gses if g not in gse_cache]
    if need:
        print(f"    Fetching {len(need)} GSE descriptions from NCBI GEO...")
        for i, gse in enumerate(need):
            desc = _fetch_gse(gse)
            if desc:
                gse_cache[gse] = desc
            if (i + 1) % 20 == 0 or i == len(need) - 1:
                sys.stdout.write(f"\r    GSEs: {i+1}/{len(need)} fetched   ")
                sys.stdout.flush()
            time.sleep(0.35)
        print()
        # Save cache
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(gse_cache, f, indent=2)
            except Exception:
                pass

    # Build consensus per GSE
    consensus = {}
    for gse_id, group in df.groupby('series_id'):
        if str(gse_id).strip() in ('', 'nan'):
            continue
        cons = {}
        for f in NS_CURATE_FIELDS:
            if f in group.columns:
                vals = [str(v).strip() for v in group[f]
                        if str(v).strip() not in _NOT_SPECIFIED_VALUES]
                if vals:
                    cons[f] = Counter(vals).most_common(3)
        if cons:
            consensus[str(gse_id).strip()] = cons

    # Re-extract each NS slot
    corrected = df.copy()
    n_fixed = 0
    total_ns = len(ns_slots)

    # Group by GSM for efficiency
    gsm_ns = {}
    for gsm, field in ns_slots:
        gsm_ns.setdefault(gsm, []).append(field)

    for i, (gsm, fields_to_fix) in enumerate(gsm_ns.items()):
        if watchdog:
            watchdog.wait()

        row = corrected[corrected['GSM'] == gsm]
        if row.empty:
            continue
        row = row.iloc[0]
        gse = str(row.get('series_id', '')).strip()
        text = get_sample_text(row)

        gse_desc = gse_cache.get(gse, {})
        gse_title = gse_desc.get('title', '')
        gse_summary = gse_desc.get('summary', '')
        gse_design = gse_desc.get('overall_design', '')

        cons = consensus.get(gse, {})
        cons_text = "\n".join(f"  {f}: {', '.join(f'{v}({c})' for v, c in pairs)}"
                              for f, pairs in cons.items()) or "  (none)"

        # Sibling examples
        siblings = corrected[corrected['series_id'] == gse].head(5) if gse else pd.DataFrame()
        sib_text = ""
        if not siblings.empty:
            sib_lines = []
            for _, s in siblings.iterrows():
                lbls = {f: str(s.get(f, '')) for f in fields_to_fix
                         if str(s.get(f, '')).strip() not in _NOT_SPECIFIED_VALUES}
                if lbls:
                    sib_lines.append(f"  {s.get('GSM','?')}: {lbls}")
            sib_text = "\n".join(sib_lines[:5])

        cols_str = ", ".join(fields_to_fix)
        schema = {col: "string" for col in fields_to_fix}

        prompt = f"""Extract labels for this GEO sample. Previous pass returned "Not Specified".
Use experiment context to determine correct labels.

SAMPLE: {gsm}
METADATA:
{text[:800]}

EXPERIMENT CONTEXT ({gse}):
Title: {gse_title}
Summary: {gse_summary[:3000]}
Design: {gse_design[:1500]}

CONSENSUS (other samples in {gse}):
{cons_text}

SIBLINGS:
{sib_text}

RULES:
- Condition: SPECIFIC disease name, never generic. Title Case. WT/healthy = "Control".
- Tissue: Infer from disease if no cell line/type. Cell Line: X only if explicit.
- Only "Not Specified" if truly unknowable.

JSON only. Extract: {cols_str}
{json.dumps(schema, indent=2)}
"""
        try:
            resp = ollama.chat(
                model=_OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 100})
            raw = resp['message']['content'].strip()
            raw = re.sub(r'```json\s*|\s*```', '', raw, flags=re.DOTALL).strip()
            s, e = raw.find('{'), raw.rfind('}')
            if s >= 0 and e > s:
                raw = raw[s:e+1]
            data = json.loads(raw)

            idx = corrected[corrected['GSM'] == gsm].index[0]
            for col in fields_to_fix:
                new_val = str(data.get(col, 'Not Specified')).strip()
                if new_val and new_val not in _NOT_SPECIFIED_VALUES:
                    corrected.at[idx, col] = new_val
                    n_fixed += 1
        except Exception:
            pass

        if (i + 1) % 10 == 0 or i == len(gsm_ns) - 1:
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r    Recall: {i+1}/{len(gsm_ns)} GSMs | fixed={n_fixed} | "
                f"{timedelta(seconds=int(elapsed))}   ")
            sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 2 done: {n_fixed} labels recovered in {timedelta(seconds=int(elapsed))}")
    return corrected


# ═══════════════════════════════════════════════════════════════
#  Phase 3 — Cross-Experiment Harmonization
# ═══════════════════════════════════════════════════════════════

# Synonym dictionaries
_CONDITION_SYNONYMS = {
    'alzheimer': 'Alzheimer Disease', 'alzheimers': 'Alzheimer Disease',
    "alzheimer's": 'Alzheimer Disease', "alzheimer's disease": 'Alzheimer Disease',
    'ad': 'Alzheimer Disease', 'parkinson': 'Parkinson Disease',
    "parkinson's": 'Parkinson Disease', 'pd': 'Parkinson Disease',
    'control': 'Control', 'ctrl': 'Control', 'normal': 'Control',
    'healthy': 'Control', 'healthy control': 'Control', 'healthy donor': 'Control',
    'non-diseased': 'Control', 'unaffected': 'Control', 'wild type': 'Control', 'wt': 'Control',
    'aml': 'Acute Myeloid Leukemia', 'cll': 'Chronic Lymphocytic Leukemia',
    'all': 'Acute Lymphoblastic Leukemia', 'cml': 'Chronic Myeloid Leukemia',
    'breast cancer': 'Breast Cancer', 'lung cancer': 'Lung Cancer',
    'nsclc': 'Non-Small Cell Lung Cancer', 'hcc': 'Hepatocellular Carcinoma',
    'colorectal cancer': 'Colorectal Cancer', 'crc': 'Colorectal Cancer',
    'melanoma': 'Melanoma', 'glioblastoma': 'Glioblastoma', 'gbm': 'Glioblastoma',
    'multiple myeloma': 'Multiple Myeloma', 'myeloma': 'Multiple Myeloma',
    'rcc': 'Renal Cell Carcinoma', 'dlbcl': 'Diffuse Large B-Cell Lymphoma',
    't2d': 'Type 2 Diabetes', 'ra': 'Rheumatoid Arthritis',
    'sle': 'Systemic Lupus Erythematosus', 'ms': 'Multiple Sclerosis',
    'copd': 'COPD', 'mdd': 'Major Depressive Disorder',
    'not specified': 'Not Specified', 'unknown': 'Not Specified',
    'na': 'Not Specified', 'n/a': 'Not Specified', 'none': 'Not Specified',
}

_TISSUE_SYNONYMS = {
    'blood': 'Whole Blood', 'whole blood': 'Whole Blood',
    'peripheral blood': 'Peripheral Blood', 'pbmc': 'PBMC', 'pbmcs': 'PBMC',
    'brain': 'Brain', 'hippocampus': 'Hippocampus', 'cerebellum': 'Cerebellum',
    'liver': 'Liver', 'kidney': 'Kidney', 'heart': 'Heart', 'lung': 'Lung',
    'breast': 'Breast', 'colon': 'Colon', 'skin': 'Skin', 'bone marrow': 'Bone Marrow',
    'adipose': 'Adipose Tissue', 'muscle': 'Skeletal Muscle',
    'fibroblast': 'Cell Type: Fibroblast', 'macrophage': 'Cell Type: Macrophage',
    'macrophages': 'Cell Type: Macrophage', 't cell': 'Cell Type: T Cell',
    'monocyte': 'Cell Type: Monocyte', 'neutrophil': 'Cell Type: Neutrophil',
    'hela': 'Cell Line: HeLa', 'mcf7': 'Cell Line: MCF-7', 'mcf-7': 'Cell Line: MCF-7',
    'a549': 'Cell Line: A549', 'huvec': 'Cell Line: HUVEC',
    'hek293': 'Cell Line: HEK293', 'k562': 'Cell Line: K562',
    'thp-1': 'Cell Line: THP-1', 'sh-sy5y': 'Cell Line: SH-SY5Y',
    'hepg2': 'Cell Line: HepG2', 'jurkat': 'Cell Line: Jurkat',
    'not specified': 'Not Specified', 'unknown': 'Not Specified',
}

_NEGATION_PATTERNS = [
    r'\bnon[\s\-]*(cancer|tumor|tumour|malignant|diseased)',
    r'\bno\s+(cancer|tumor|tumour)',
    r'\b(tumor|tumour|cancer)[\s\-]*free\b',
    r'\bnon[\s\-]*(cancerous|tumorous)',
]

_NOISE_PATTERNS = [
    r'\b\d+\.?\d*\s*(nm|nM|uM|µM|mM|ug|µg|mg|ng)\b',
    r'\b\d+\.?\d*\s*(h|hr|hrs|hours?|min|minutes?|days?|weeks?)\b',
    r'\brep(licate)?\s*\d+\b',
    r'\b(sample|patient|donor|subject)\s*#?\s*\d+\b',
    r'\bbatch\s*\d+\b',
]


def run_phase3(df):
    """Phase 3: Negation, noise stripping, synonym normalization, rare merge."""
    print(f"\n  Phase 3 — Harmonizing {len(df):,} samples...")
    t0 = time.time()
    result = df.copy()
    label_cols = [c for c in result.columns
                  if c in FIELDS and result[c].dtype == 'object']
    n_changes = 0

    for col in label_cols:
        before_vals = set(result[col].dropna().unique())

        # 1. Negation detection (Condition only)
        if col == 'Condition':
            for _, row_idx in result.iterrows():
                val = str(result.at[row_idx, col] if hasattr(row_idx, '__int__') else result.loc[_, col])

        # 2. Synonym normalization
        syn_map = _CONDITION_SYNONYMS if col == 'Condition' else \
                  _TISSUE_SYNONYMS if col == 'Tissue' else {}
        if syn_map:
            result[col] = result[col].apply(
                lambda v: syn_map.get(str(v).strip().lower(), str(v).strip())
                if pd.notna(v) else v)

        # 3. Noise stripping (Treatment mainly)
        if col in ('Treatment', 'Condition'):
            for pat in _NOISE_PATTERNS:
                result[col] = result[col].apply(
                    lambda v: re.sub(pat, '', str(v), flags=re.I).strip()
                    if pd.notna(v) else v)
            # Clean up leftover whitespace/punctuation
            result[col] = result[col].apply(
                lambda v: re.sub(r'\s+', ' ', str(v)).strip().strip('- ,;')
                if pd.notna(v) else v)

        # 4. Negation detection
        if col == 'Condition':
            for pat in _NEGATION_PATTERNS:
                mask = result[col].astype(str).str.contains(pat, flags=re.I, na=False)
                if mask.any():
                    result.loc[mask, col] = 'Control'

        after_vals = set(result[col].dropna().unique())
        n_changes += len(before_vals - after_vals)

    elapsed = time.time() - t0
    print(f"  ✓ Phase 3 done: {n_changes} variants harmonized in {timedelta(seconds=int(elapsed))}")

    # Summary
    for col in label_cols:
        before_n = df[col].nunique() if col in df.columns else 0
        after_n = result[col].nunique()
        if before_n != after_n:
            print(f"    {col:<20} {before_n} → {after_n} unique ({before_n - after_n} merged)")

    return result


# ═══════════════════════════════════════════════════════════════
#  Load Platform from Data Directory
# ═══════════════════════════════════════════════════════════════
def load_platform(gpl, data_dir, conn):
    """Load all GSMs for a platform from GEOmetadb."""
    cols = ("gsm, title, source_name_ch1, characteristics_ch1, "
            "description, treatment_protocol_ch1, "
            "organism_ch1, series_id, gpl")
    query = f"SELECT {cols} FROM gsm WHERE UPPER(gpl)=? ORDER BY RANDOM()"
    df = pd.read_sql_query(query, conn, params=[gpl.upper()])
    print(f"  Loaded {len(df):,} samples from {gpl}")
    return df


def load_geometadb(path):
    """Load GEOmetadb into memory."""
    print(f"  Loading GEOmetadb: {path}")
    if path.endswith('.gz'):
        tmp = path.replace('.gz', '.tmp.sqlite')
        if not os.path.exists(tmp):
            with gzip.open(path, 'rb') as fi, open(tmp, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
        disk = sqlite3.connect(tmp)
    else:
        disk = sqlite3.connect(path)
    conn = sqlite3.connect(":memory:")
    disk.backup(conn)
    disk.close()
    return conn


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='GeneVariate Standalone LLM Extraction')
    parser.add_argument('--gpl', required=True, help='GPL platform ID (e.g., GPL6947)')
    parser.add_argument('--data-dir', default='./data', help='Directory with GEOmetadb')
    parser.add_argument('--model', default=None, help='Ollama model (default: auto-detect)')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit samples (0=all)')
    parser.add_argument('--skip-phase2', action='store_true', help='Skip Phase 2 (NS recall)')
    parser.add_argument('--skip-phase3', action='store_true', help='Skip Phase 3 (harmonization)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  GeneVariate — Standalone LLM Extraction Pipeline")
    print("=" * 60)

    # ── Setup ──
    global _OLLAMA_MODEL
    if args.model:
        _OLLAMA_MODEL = args.model
    else:
        detect_model()
    print(f"  Model   : {_OLLAMA_MODEL}")
    print(f"  Platform: {args.gpl}")
    print(f"  Workers : {args.workers}")

    # GPU
    gpus = detect_gpus()
    if gpus:
        print(f"  GPU     : {gpus[0]['name']} ({gpus[0]['vram_gb']}GB, "
              f"{gpus[0]['free_vram_gb']}GB free)")
    gpu_st, gpu_vr = check_ollama_gpu()
    if gpu_st == "gpu":
        print(f"  Ollama  : GPU mode ({gpu_vr}GB VRAM)")
    elif gpu_st == "cpu":
        print(f"  Ollama  : CPU mode — SLOW!")
        print(f"  Fix     : CUDA_VISIBLE_DEVICES=0 OLLAMA_GPU_LAYERS=999 ollama serve")

    # Database
    geo_path = None
    for name in ['GEOmetadb.sqlite.gz', 'GEOmetadb.sqlite', 'geometadb.sqlite.gz']:
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            geo_path = p
            break
    if not geo_path:
        # Search current dir
        for name in ['GEOmetadb.sqlite.gz', 'GEOmetadb.sqlite']:
            if os.path.exists(name):
                geo_path = name
                break
    if not geo_path:
        print(f"[FATAL] GEOmetadb not found in {args.data_dir}")
        sys.exit(1)

    conn = load_geometadb(geo_path)
    samples = load_platform(args.gpl, args.data_dir, conn)

    if args.limit > 0:
        samples = samples.head(args.limit)
        print(f"  Limited to {args.limit} samples")

    out_dir = args.output_dir or os.path.join(args.data_dir, f"{args.gpl}_extraction")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "gse_cache.json")

    # ── Initialize Memory Agent (deterministic extraction) ──
    global _MEMORY_AGENT, _GSE_CONTEXTS
    if _HAS_DET:
        db_path = os.path.join(args.data_dir, "biomedical_memory.db")
        _MEMORY_AGENT = MemoryAgent(db_path)
        print(f"  Memory  : READY (episodic log at {db_path})")
        # Build GSE contexts
        gse_meta = {}
        if 'series_id' in samples.columns:
            try:
                gse_ids = samples['series_id'].dropna().unique().tolist()
                for i in range(0, len(gse_ids), 500):
                    chunk = gse_ids[i:i+500]
                    ph = ",".join("?" * len(chunk))
                    df = pd.read_sql_query(
                        f"SELECT gse, title, summary FROM gse WHERE gse IN ({ph})",
                        conn, params=chunk)
                    for _, r in df.iterrows():
                        gse_meta[str(r['gse']).strip()] = {
                            'title': str(r.get('title', '')).strip(),
                            'summary': str(r.get('summary', '')).strip()[:300]}
            except Exception:
                pass
        _GSE_CONTEXTS = build_gse_contexts(samples, gse_meta, _MEMORY_AGENT)
        print(f"  GSE ctx : {len(_GSE_CONTEXTS)} experiments")

    # ── Watchdog ──
    watchdog = ResourceWatchdog()
    watchdog.start()

    # ── Phase 1 ──
    p1 = run_phase1(samples, max_workers=args.workers, watchdog=watchdog)
    if p1 is None:
        print("[FATAL] Phase 1 produced no results.")
        sys.exit(1)
    p1.to_csv(os.path.join(out_dir, f"{args.gpl}_phase1_raw.csv"), index=False)

    # ── Phase 1.5 ──
    p15 = run_phase15(p1)
    p15.to_csv(os.path.join(out_dir, f"{args.gpl}_phase15.csv"), index=False)

    # ── Ingest into Memory Agent (vocabulary + clusters) ──
    if _HAS_DET and _MEMORY_AGENT is not None:
        print("\n  Ingesting results into Memory Agent...")
        _MEMORY_AGENT.ingest_extraction_results(p15, platform=args.gpl)
        stats = _MEMORY_AGENT.stats()
        print(f"  Vocabulary: {stats.get('vocabulary', {})}")
        print(f"  Clusters:   {stats.get('clusters', {})}")
        # Auto-export cluster files
        cluster_dir = os.path.join(out_dir, "clusters")
        exported = _MEMORY_AGENT.export_clusters_text(cluster_dir)
        if exported:
            print(f"  Exported: {', '.join(exported)} → {cluster_dir}")
        # Save memory DB
        _MEMORY_AGENT.export_db(os.path.join(out_dir, "biomedical_memory.db"))

    # ── Phase 2 ──
    if args.skip_phase2:
        print("\n  Phase 2 — SKIPPED (--skip-phase2)")
        p2 = p15.copy()
    else:
        p2 = run_phase2(p15, watchdog=watchdog, cache_path=cache_path)
        p2.to_csv(os.path.join(out_dir, f"{args.gpl}_phase2.csv"), index=False)

    # ── Phase 3 ──
    if args.skip_phase3:
        print("\n  Phase 3 — SKIPPED (--skip-phase3)")
        p3 = p2.copy()
    else:
        p3 = run_phase3(p2)
        p3.to_csv(os.path.join(out_dir, f"{args.gpl}_phase3_final.csv"), index=False)

    # ── Save clean labels ──
    drop_cols = ['title', 'source_name_ch1', 'characteristics_ch1',
                 'description', 'treatment_protocol_ch1', 'organism_ch1']
    clean = p3.drop(columns=[c for c in drop_cols if c in p3.columns], errors='ignore')
    clean_path = os.path.join(out_dir, f"{args.gpl}_labels.csv")
    clean.to_csv(clean_path, index=False)

    watchdog.stop()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  DONE — {args.gpl}")
    print(f"{'=' * 60}")
    print(f"  Samples  : {len(clean):,}")
    for f in FIELDS:
        if f in clean.columns:
            ns = clean[f].astype(str).str.strip().isin(_NOT_SPECIFIED_VALUES).sum()
            nu = clean[f].nunique()
            print(f"  {f:<20} {nu:>5} unique, {ns:>5} NS ({ns*100//len(clean)}%)")
    print(f"\n  Output: {out_dir}/")
    print(f"    {args.gpl}_phase1_raw.csv   — raw LLM extraction")
    print(f"    {args.gpl}_phase15.csv      — after per-GSE normalization")
    if not args.skip_phase2:
        print(f"    {args.gpl}_phase2.csv       — after NS recovery")
    if not args.skip_phase3:
        print(f"    {args.gpl}_phase3_final.csv — after harmonization")
    print(f"    {args.gpl}_labels.csv       — clean labels (final)")
    print(f"    gse_cache.json              — cached GSE descriptions")
    print()

    conn.close()


if __name__ == "__main__":
    main()
