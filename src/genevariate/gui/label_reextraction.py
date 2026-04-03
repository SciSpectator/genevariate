#!/usr/bin/env python3
"""
Label Re-extraction CLI — Context-enriched re-extraction for "Not Specified" labels.

Uses Ollama (gemma2:9b) with a GSE context cache that:
1. Loads existing extracted labels (with "Not Specified" entries)
2. Groups samples by GSE (experiment) — samples in the same experiment share context
3. For each "Not Specified" sample, builds an enriched context from:
   - The GSE experiment description (scraped from GEO website)
   - Labels already extracted for other samples in the same GSE
   - The sample's own title and characteristics
4. Asks the LLM to re-extract labels using the enriched context
5. Validates consistency: if most samples in a GSE have "Breast Cancer",
   a "Not Specified" in the same GSE is likely also "Breast Cancer"

Cache Architecture:
  - gse_cache: {gse_id: {description, title, known_conditions, known_tissues, ...}}
  - gsm_context: {gsm_id: {title, characteristics, series_id, extracted_labels}}
  - consensus_cache: {gse_id: {Condition: Counter, Tissue: Counter, ...}}

Usage:
  python label_reextraction.py --labels GPL570_labels.csv.gz --metadata GPL570_metadata.csv.gz --output GPL570_corrected.csv.gz
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd
import requests


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:9b"
LABEL_COLUMNS = ["Condition", "Tissue", "Age", "Treatment", "Treatment Time"]
NOT_SPECIFIED = {"Not Specified", "not specified", "NOT SPECIFIED",
                 "N/A", "n/a", "NA", "nan", "NaN", "", "Unknown", "unknown"}
BATCH_SIZE = 1  # Process one sample at a time for memory context
MAX_RETRIES = 3
TIMEOUT = 120  # seconds per LLM call


# ═══════════════════════════════════════════════════════════════════
#  Memory System
# ═══════════════════════════════════════════════════════════════════
class GSEContextCache:
    """Context cache for label extraction context.
    
    Three context tiers:
    1. 1. Core Context: current sample being processed
    2. 2. Recall Context: GSE experiment context + previously extracted labels
    3. Archival Memory: all past extractions (for consistency checking)
    """

    def __init__(self):
        # GSE-level memory: experiment descriptions and consensus labels
        self.gse_memory = {}       # {gse_id: {description, title, overall_design, ...}}
        self.gse_consensus = {}    # {gse_id: {col: Counter({'Cancer': 5, 'Control': 3})}}
        
        # GSM-level memory: per-sample extraction results
        self.gsm_memory = {}       # {gsm_id: {title, characteristics, labels, ...}}
        
        # Archival: global label statistics across all GSEs
        self.global_stats = {col: Counter() for col in LABEL_COLUMNS}
        
        # Processing stats
        self.n_corrected = 0
        self.n_confirmed_ns = 0    # confirmed as truly "Not Specified"
        self.n_failed = 0

    def register_gse(self, gse_id, description="", title="", overall_design="",
                     summary="", gse_type=""):
        """Store experiment-level context in GSE context cache."""
        self.gse_memory[gse_id] = {
            "description": description,
            "title": title,
            "overall_design": overall_design,
            "summary": summary,
            "type": gse_type,
        }
        if gse_id not in self.gse_consensus:
            self.gse_consensus[gse_id] = {col: Counter() for col in LABEL_COLUMNS}

    def register_gsm(self, gsm_id, series_id, title="", characteristics="",
                     source_name="", labels=None):
        """Store sample-level context and extracted labels."""
        self.gsm_memory[gsm_id] = {
            "series_id": series_id,
            "title": title,
            "characteristics": characteristics,
            "source_name": source_name,
            "labels": labels or {},
        }
        # Update consensus counters for non-"Not Specified" labels
        if labels:
            for col, val in labels.items():
                if val and str(val).strip() not in NOT_SPECIFIED:
                    self.gse_consensus.setdefault(series_id, {col: Counter() for col in LABEL_COLUMNS})
                    self.gse_consensus[series_id].setdefault(col, Counter())
                    self.gse_consensus[series_id][col][val] += 1
                    self.global_stats[col][val] += 1

    def get_gse_context(self, gse_id):
        """Retrieve experiment context from GSE context cache."""
        return self.gse_memory.get(gse_id, {})

    def get_gse_consensus(self, gse_id):
        """Get consensus labels for a GSE (what other samples were labeled as)."""
        consensus = {}
        gse_data = self.gse_consensus.get(gse_id, {})
        for col, counter in gse_data.items():
            if counter:
                top = counter.most_common(5)
                consensus[col] = top
        return consensus

    def get_sibling_labels(self, gsm_id, gse_id, limit=10):
        """Get labels from other samples in the same experiment."""
        siblings = []
        for other_gsm, data in self.gsm_memory.items():
            if other_gsm == gsm_id:
                continue
            if data.get("series_id") == gse_id:
                # Only include if they have actual (non-NS) labels
                has_real = any(
                    v and str(v).strip() not in NOT_SPECIFIED
                    for v in data.get("labels", {}).values()
                )
                if has_real:
                    siblings.append({
                        "gsm": other_gsm,
                        "title": data.get("title", ""),
                        "labels": data.get("labels", {}),
                    })
                    if len(siblings) >= limit:
                        break
        return siblings


# ═══════════════════════════════════════════════════════════════════
#  LLM Interface
# ═══════════════════════════════════════════════════════════════════
def call_ollama(prompt, model=MODEL, max_retries=MAX_RETRIES):
    """Call Ollama with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                        "top_p": 0.9,
                    }
                },
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                print(f"  [Ollama] HTTP {resp.status_code}, retry {attempt+1}/{max_retries}")
        except requests.exceptions.Timeout:
            print(f"  [Ollama] Timeout ({TIMEOUT}s), retry {attempt+1}/{max_retries}")
        except requests.exceptions.ConnectionError:
            print(f"  [Ollama] Connection error — is Ollama running? retry {attempt+1}/{max_retries}")
            time.sleep(2)
    return None


def build_correction_prompt(gsm_id, gsm_data, gse_context, consensus,
                            siblings, cols_to_fix):
    """Build a rich prompt with memory context for re-extraction."""
    
    # ── 1. Core Context: current sample ──
    core = f"""SAMPLE: {gsm_id}
Title: {gsm_data.get('title', 'N/A')}
Source: {gsm_data.get('source_name', 'N/A')}
Characteristics: {gsm_data.get('characteristics', 'N/A')}
"""
    
    # ── 2. Recall Context: experiment context ──
    recall = ""
    if gse_context:
        gse_title = gse_context.get("title", "N/A")
        gse_desc = gse_context.get("description", "")
        gse_design = gse_context.get("overall_design", "")
        gse_summary = gse_context.get("summary", "")
        
        # Truncate long descriptions
        desc_text = gse_desc[:800] if gse_desc else ""
        design_text = gse_design[:400] if gse_design else ""
        summary_text = gse_summary[:400] if gse_summary else ""
        
        recall = f"""
EXPERIMENT CONTEXT (from GSE context cache):
  Series: {gsm_data.get('series_id', 'N/A')}
  Title: {gse_title}
  Description: {desc_text}
  Design: {design_text}
  Summary: {summary_text}
"""

    # ── Consensus Memory: what other samples in this experiment were labeled ──
    consensus_text = ""
    if consensus:
        parts = []
        for col, top_vals in consensus.items():
            if top_vals and col in cols_to_fix:
                vals_str = ", ".join(f'"{v}" ({n})' for v, n in top_vals)
                parts.append(f"  {col}: {vals_str}")
        if parts:
            consensus_text = "\nCONSENSUS from other samples in this experiment:\n" + "\n".join(parts) + "\n"

    # ── Sibling Memory: specific examples from same experiment ──
    sibling_text = ""
    if siblings:
        examples = []
        for sib in siblings[:5]:
            lbls = ", ".join(f"{k}={v}" for k, v in sib["labels"].items()
                           if v and str(v).strip() not in NOT_SPECIFIED)
            if lbls:
                examples.append(f"  {sib['gsm']}: title=\"{sib['title'][:80]}\" → {lbls}")
        if examples:
            sibling_text = ("\nOTHER SAMPLES in this experiment (for reference):\n"
                          + "\n".join(examples) + "\n")

    # ── Build final prompt ──
    cols_str = ", ".join(cols_to_fix)
    
    prompt = f"""You are a biomedical metadata extraction tool with GSE context cache.

Your task: extract the following label(s) for this GEO sample: {cols_str}

These labels were previously "Not Specified" — use the experiment context and 
other samples from the same experiment to determine the correct labels.

{core}
{recall}
{consensus_text}
{sibling_text}

CONDITION RULES — be specific, never generic:
- NEVER output just "Cancer" — always the specific type: "Breast Cancer",
  "Hepatocellular Carcinoma", "Multiple Myeloma", "Glioblastoma", etc.
- NEVER output just "Leukemia" — specify: "Acute Myeloid Leukemia", "Chronic Lymphocytic Leukemia"
- NEVER output just "Lymphoma" — specify: "Diffuse Large B-Cell Lymphoma", "Hodgkin Lymphoma"
- Use the FULL disease name from the experiment title/summary, not abbreviations
- Common mappings: AML=Acute Myeloid Leukemia, HCC=Hepatocellular Carcinoma,
  GBM=Glioblastoma, NSCLC=Non-Small Cell Lung Cancer, CLL=Chronic Lymphocytic Leukemia
- For healthy/normal/wild-type: "Control"
- If you encounter a disease not listed above, use the full specific name as written
- Title Case: "Breast Cancer", "Crohn Disease", "Type 2 Diabetes"

TISSUE RULES — cell line/type ONLY when explicitly named:
- ONLY output "Cell Line: NAME" if a cell line is EXPLICITLY named (e.g., "Cell Line: MCF-7")
- ONLY output "Cell Type: NAME" if a specific cell type is EXPLICITLY mentioned
  (e.g., "Cell Type: Macrophages", "Cell Type: CD4+ T Cells")
- If NO cell line and NO cell type is mentioned, INFER tissue from disease context:
  "Breast Cancer" → "Breast", "Glioblastoma" → "Brain", "Liver Cancer" → "Liver",
  "Lung Adenocarcinoma" → "Lung", "Renal Cell Carcinoma" → "Kidney",
  "Colorectal Cancer" → "Colon", "Leukemia" → "Bone Marrow", "Melanoma" → "Skin"
- NEVER output "Cell Line: Not Specified" or "Cell Type: Not Specified"
- PBMC = PBMC, not Whole Blood

OTHER RULES:
- Age: only extract if explicitly stated as a number (years)
- Treatment: drug, compound, or intervention name
- Treatment Time: only if explicitly stated (e.g., "24h", "48 hours")
- If truly unknowable even with context: "Not Specified"
- Match formatting used by sibling samples for consistency

Respond ONLY with a JSON object. Example:
{{"Condition": "Hepatocellular Carcinoma", "Tissue": "Liver"}}
Example with cell line: {{"Condition": "Breast Cancer", "Tissue": "Cell Line: MCF-7"}}
Example without cell line: {{"Condition": "Breast Cancer", "Tissue": "Breast"}}

Extract ONLY the requested columns: {cols_str}
JSON response:"""
    
    return prompt


def parse_llm_response(response_text, cols_to_fix):
    """Parse LLM JSON response, handling common formatting issues."""
    if not response_text:
        return None
    
    # Try to find JSON in the response
    text = response_text.strip()
    
    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    text = text[start:end+1]
    
    try:
        result = json.loads(text)
        # Validate: only return requested columns
        cleaned = {}
        for col in cols_to_fix:
            val = result.get(col, "Not Specified")
            if val is None or str(val).strip() == "":
                val = "Not Specified"
            cleaned[col] = str(val).strip()
        return cleaned
    except json.JSONDecodeError:
        return None


# ═══════════════════════════════════════════════════════════════════
#  Main Processing Pipeline
# ═══════════════════════════════════════════════════════════════════
def load_data(labels_path, metadata_path=None):
    """Load labels and optional metadata."""
    comp = "gzip" if labels_path.endswith(".gz") else None
    labels_df = pd.read_csv(labels_path, compression=comp, low_memory=False)
    
    # Normalize GSM column
    for c in labels_df.columns:
        if c.lower().strip() in ("gsm", "sample", "sample_id", "geo_accession"):
            labels_df = labels_df.rename(columns={c: "GSM"})
            break
    
    if "GSM" not in labels_df.columns:
        first = labels_df.iloc[:, 0].astype(str)
        if first.str.upper().str.startswith("GSM").mean() > 0.5:
            labels_df = labels_df.rename(columns={labels_df.columns[0]: "GSM"})
    
    labels_df["GSM"] = labels_df["GSM"].astype(str).str.strip()
    
    # Load metadata if provided
    meta_df = None
    if metadata_path and os.path.exists(metadata_path):
        comp_m = "gzip" if metadata_path.endswith(".gz") else None
        meta_df = pd.read_csv(metadata_path, compression=comp_m, low_memory=False)
        for c in meta_df.columns:
            if c.lower().strip() in ("gsm", "sample", "geo_accession"):
                meta_df = meta_df.rename(columns={c: "GSM"})
                break
        meta_df["GSM"] = meta_df["GSM"].astype(str).str.strip()
    
    return labels_df, meta_df


def find_not_specified_samples(labels_df):
    """Find all samples with at least one 'Not Specified' label."""
    ns_samples = []
    
    for idx, row in labels_df.iterrows():
        gsm = row.get("GSM", "")
        cols_to_fix = []
        
        for col in LABEL_COLUMNS:
            if col in labels_df.columns:
                val = str(row.get(col, "")).strip()
                if val in NOT_SPECIFIED:
                    cols_to_fix.append(col)
        
        if cols_to_fix:
            ns_samples.append({
                "idx": idx,
                "gsm": gsm,
                "series_id": str(row.get("series_id", "")).strip(),
                "cols_to_fix": cols_to_fix,
            })
    
    return ns_samples


def fetch_gse_from_geo(gse_id):
    """Scrape GSE experiment description from NCBI GEO website.
    Uses the text format endpoint which returns structured metadata.
    """
    import urllib.request
    url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
           f"?acc={gse_id}&targ=self&form=text&view=quick")
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'GeneVariate/1.0 (bioinformatics research tool)'
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode('utf-8', errors='replace')
    
    result = {'title': '', 'summary': '', 'overall_design': ''}
    
    for line in text.split('\n'):
        if line.startswith('!Series_title'):
            val = line.split('=', 1)[1].strip() if '=' in line else ''
            result['title'] = val
        elif line.startswith('!Series_summary'):
            val = line.split('=', 1)[1].strip() if '=' in line else ''
            if result['summary']:
                result['summary'] += ' ' + val
            else:
                result['summary'] = val
        elif line.startswith('!Series_overall_design'):
            val = line.split('=', 1)[1].strip() if '=' in line else ''
            if result['overall_design']:
                result['overall_design'] += ' ' + val
            else:
                result['overall_design'] = val
    
    # Truncate
    result['summary'] = result['summary'][:800]
    result['overall_design'] = result['overall_design'][:500]
    
    return result if result['title'] else None


def build_context(labels_df, meta_df, memory):
    """Build GSE context cache from existing labels + fetch GSE descriptions from GEO."""
    print("\n[Phase 1] Building GSE cache...")
    
    # Register all GSMs with their existing labels
    for _, row in labels_df.iterrows():
        gsm = str(row.get("GSM", "")).strip()
        series_id = str(row.get("series_id", "")).strip()
        
        # Get sample metadata from meta_df if available
        title = ""
        characteristics = ""
        source_name = ""
        if meta_df is not None:
            meta_row = meta_df[meta_df["GSM"] == gsm]
            if not meta_row.empty:
                mr = meta_row.iloc[0]
                title = str(mr.get("title", ""))
                characteristics = str(mr.get("characteristics_ch1", ""))
                source_name = str(mr.get("source_name_ch1", ""))
                if series_id in ("", "nan") and "series_id" in mr.index:
                    series_id = str(mr.get("series_id", ""))
        
        # Current labels
        labels = {}
        for col in LABEL_COLUMNS:
            if col in row.index:
                labels[col] = str(row[col]).strip()
        
        memory.register_gsm(
            gsm, series_id,
            title=title,
            characteristics=characteristics,
            source_name=source_name,
            labels=labels,
        )
    
    # Collect unique GSE IDs from labels
    unique_gses = []
    if "series_id" in labels_df.columns:
        unique_gses = [str(g).strip() for g in labels_df["series_id"].dropna().unique()
                       if str(g).strip() and str(g).strip() != "nan"
                       and str(g).strip().upper().startswith("GSE")]
    
    # Fetch GSE descriptions from GEO website (NOT from GEOmetadb)
    if unique_gses:
        print(f"  Fetching {len(unique_gses)} GSE descriptions from GEO website...")
        t0 = time.time()
        n_ok = 0
        n_fail = 0
        
        for i, gse_id in enumerate(unique_gses):
            try:
                desc = fetch_gse_from_geo(gse_id)
                if desc:
                    memory.register_gse(
                        gse_id,
                        title=desc.get("title", ""),
                        summary=desc.get("summary", ""),
                        overall_design=desc.get("overall_design", ""),
                    )
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    Warning: {gse_id} fetch failed: {e}")
            
            # Progress
            if (i + 1) % 20 == 0 or i == len(unique_gses) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(unique_gses) - i - 1) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r    GSE descriptions: {i+1}/{len(unique_gses)} "
                    f"({n_ok} OK, {n_fail} failed) "
                    f"ETA: {timedelta(seconds=int(eta))}    "
                )
                sys.stdout.flush()
            
            # Rate limit: ~3 requests/sec to be polite to NCBI
            time.sleep(0.35)
        
        print()  # newline
    
    # Print memory stats
    n_gse = len(memory.gse_memory)
    n_gsm = len(memory.gsm_memory)
    print(f"  Memory loaded: {n_gse:,} experiments, {n_gsm:,} samples")
    
    for col in LABEL_COLUMNS:
        n_vals = len(memory.global_stats[col])
        top3 = memory.global_stats[col].most_common(3)
        top_str = ", ".join(f'"{v}" ({n})' for v, n in top3)
        print(f"  {col}: {n_vals} unique values — top: {top_str}")


def process_sample(gsm_id, sample_info, memory):
    """Process one Not Specified sample using memory context."""
    gsm_data = memory.gsm_memory.get(gsm_id, {})
    gse_id = sample_info["series_id"]
    cols_to_fix = sample_info["cols_to_fix"]
    
    # Recall: get experiment context
    gse_context = memory.get_gse_context(gse_id)
    consensus = memory.get_gse_consensus(gse_id)
    siblings = memory.get_sibling_labels(gsm_id, gse_id, limit=8)
    
    # Build prompt with full memory context
    prompt = build_correction_prompt(
        gsm_id, gsm_data, gse_context, consensus, siblings, cols_to_fix)
    
    # Call LLM
    response = call_ollama(prompt)
    if response is None:
        return None
    
    # Parse response
    result = parse_llm_response(response, cols_to_fix)
    if result is None:
        return None
    
    # Update memory with new labels
    if gsm_id in memory.gsm_context:
        for col, val in result.items():
            memory.gsm_memory[gsm_id]["labels"][col] = val
            if val not in NOT_SPECIFIED:
                memory.gse_consensus.setdefault(gse_id, {})
                memory.gse_consensus[gse_id].setdefault(col, Counter())
                memory.gse_consensus[gse_id][col][val] += 1
                memory.global_stats[col][val] += 1
    
    return result


def run_correction(labels_df, meta_df, ns_samples, memory, output_path):
    """Phase 2: Process all Not Specified samples with context-enriched LLM."""
    
    total = len(ns_samples)
    print(f"\n[Phase 2] Processing {total:,} samples with Not Specified labels...")
    print(f"  Model: {MODEL}")
    print(f"  Output: {output_path}")
    print()
    
    # Make a copy for corrections
    corrected_df = labels_df.copy()
    
    # Track corrections for the separate "corrections only" file
    corrections = []
    
    # Timing
    start_time = time.time()
    times_per_sample = []
    
    for i, sample in enumerate(ns_samples):
        gsm_id = sample["gsm"]
        cols = sample["cols_to_fix"]
        idx = sample["idx"]
        
        t0 = time.time()
        
        # Progress bar
        pct = (i + 1) / total * 100
        elapsed = time.time() - start_time
        if times_per_sample:
            avg_time = sum(times_per_sample) / len(times_per_sample)
            remaining = avg_time * (total - i - 1)
            eta_str = str(timedelta(seconds=int(remaining)))
            rate_str = f"{avg_time:.1f}s/sample"
        else:
            eta_str = "calculating..."
            rate_str = "..."
        
        # Display progress
        bar_len = 30
        filled = int(bar_len * (i + 1) / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {i+1}/{total} ({pct:.1f}%) | "
            f"{rate_str} | ETA: {eta_str} | "
            f"Corrected: {memory.n_corrected} | "
            f"Current: {gsm_id}        "
        )
        sys.stdout.flush()
        
        # Process
        result = process_sample(gsm_id, sample, memory)
        
        t1 = time.time()
        times_per_sample.append(t1 - t0)
        
        if result is None:
            memory.n_failed += 1
            continue
        
        # Apply corrections
        any_changed = False
        correction_record = {"GSM": gsm_id, "series_id": sample["series_id"]}
        
        for col in cols:
            new_val = result.get(col, "Not Specified")
            old_val = str(corrected_df.at[idx, col]).strip() if col in corrected_df.columns else "Not Specified"
            
            if new_val not in NOT_SPECIFIED and new_val != old_val:
                corrected_df.at[idx, col] = new_val
                correction_record[f"{col}_old"] = old_val
                correction_record[f"{col}_new"] = new_val
                any_changed = True
                memory.n_corrected += 1
            else:
                memory.n_confirmed_ns += 1
                correction_record[f"{col}_old"] = old_val
                correction_record[f"{col}_new"] = new_val
        
        if any_changed:
            corrections.append(correction_record)
        
        # Save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            _save_checkpoint(corrected_df, corrections, output_path, i + 1, total)
    
    print()  # newline after progress bar
    
    # Final save
    _save_final(corrected_df, corrections, output_path, memory, start_time, total)


def _save_checkpoint(corrected_df, corrections, output_path, n_done, total):
    """Save intermediate checkpoint."""
    ckpt = output_path.replace(".csv", f"_checkpoint_{n_done}.csv")
    comp = "gzip" if ckpt.endswith(".gz") else None
    corrected_df.to_csv(ckpt, index=False, compression=comp)


def _save_final(corrected_df, corrections, output_path, memory, start_time, total):
    """Save final results."""
    elapsed = time.time() - start_time
    
    # Save full corrected labels
    comp = "gzip" if output_path.endswith(".gz") else None
    corrected_df.to_csv(output_path, index=False, compression=comp)
    print(f"\n[Done] Full corrected labels saved: {output_path}")
    
    # Save corrections-only file
    if corrections:
        corr_path = output_path.replace(".csv", "_corrections_only.csv")
        corr_path = corr_path.replace(".gz", "")
        pd.DataFrame(corrections).to_csv(corr_path, index=False)
        print(f"  Corrections-only file: {corr_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  CORRECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples processed: {total:,}")
    print(f"  Labels corrected:        {memory.n_corrected:,}")
    print(f"  Confirmed Not Specified:  {memory.n_confirmed_ns:,}")
    print(f"  LLM failures:            {memory.n_failed:,}")
    print(f"  Total time:              {timedelta(seconds=int(elapsed))}")
    print(f"  Avg time per sample:     {elapsed/max(1,total):.1f}s")
    print(f"{'='*60}")
    
    # Per-column stats
    print(f"\n  Per-column correction counts:")
    if corrections:
        corr_df = pd.DataFrame(corrections)
        for col in LABEL_COLUMNS:
            new_col = f"{col}_new"
            old_col = f"{col}_old"
            if new_col in corr_df.columns:
                changed = corr_df[corr_df[new_col] != corr_df[old_col]]
                n_changed = len(changed)
                if n_changed > 0:
                    top = changed[new_col].value_counts().head(5)
                    top_str = ", ".join(f'"{v}" ({n})' for v, n in top.items())
                    print(f"    {col}: {n_changed} corrected → {top_str}")

    # Save memory to disk for inspection
    mem_dir = os.path.join(os.path.dirname(output_path), "gse_cache")
    os.makedirs(mem_dir, exist_ok=True)
    mem_path = os.path.join(mem_dir, "gse_cache.json")
    try:
        mem_data = {
            "_info": {
                "created": datetime.now().isoformat(),
                "labels_file": output_path,
                "n_experiments": len(memory.gse_memory),
                "n_corrected": memory.n_corrected,
                "n_confirmed_ns": memory.n_confirmed_ns,
                "n_failed": memory.n_failed,
            },
            "gse_descriptions": memory.gse_memory,
            "gse_consensus": {
                gse: {col: dict(counter) for col, counter in cols.items()}
                for gse, cols in memory.gse_consensus.items()
            },
        }
        with open(mem_path, 'w', encoding='utf-8') as f:
            json.dump(mem_data, f, indent=2, ensure_ascii=False)
        print(f"\n  GSE cache saved: {mem_path}")
        print(f"  (Open this JSON to inspect what the agent remembered)")
    except Exception as e:
        print(f"\n  Memory save warning: {e}")


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════
def gui_select_files():
    """Open tkinter file dialogs to select input/output files."""
    import tkinter as tk
    from tkinter import filedialog, messagebox
    
    root = tk.Tk()
    root.withdraw()
    
    # Select labels file
    labels_path = filedialog.askopenfilename(
        title="Select Labels File (with 'Not Specified' entries)",
        filetypes=[("CSV files", "*.csv"), ("Compressed CSV", "*.csv.gz"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    if not labels_path:
        print("No labels file selected. Exiting.")
        sys.exit(0)
    
    # Ask for metadata file (optional)
    use_meta = messagebox.askyesno(
        "Metadata File",
        "Do you have a GEOmetadb metadata file (.csv or .csv.gz)?\n\n"
        "This provides sample titles/characteristics for better context.\n"
        "GSE experiment descriptions will be fetched from the GEO website regardless.",
        parent=root
    )
    metadata_path = None
    if use_meta:
        metadata_path = filedialog.askopenfilename(
            title="Select Metadata File (optional)",
            filetypes=[("CSV files", "*.csv"), ("Compressed CSV", "*.csv.gz"), ("All files", "*.*")],
            initialdir=os.path.dirname(labels_path)
        )
    
    # Output path — same directory as labels, with _corrected suffix
    base = os.path.basename(labels_path)
    if base.endswith('.csv.gz'):
        out_name = base.replace('.csv.gz', '_corrected.csv.gz')
    elif base.endswith('.csv'):
        out_name = base.replace('.csv', '_corrected.csv')
    else:
        out_name = base + '_corrected.csv'
    output_path = os.path.join(os.path.dirname(labels_path), out_name)
    
    # Confirm
    msg = (f"Labels: {labels_path}\n"
           f"Metadata: {metadata_path or 'None'}\n"
           f"Output: {output_path}\n\n"
           f"GSE descriptions will be fetched from the GEO website.\n\n"
           f"Proceed?")
    if not messagebox.askyesno("Confirm", msg, parent=root):
        sys.exit(0)
    
    root.destroy()
    return labels_path, metadata_path, output_path


def main():
    global MODEL, LABEL_COLUMNS
    
    parser = argparse.ArgumentParser(
        description="Label Re-extraction CLI — Context-enriched re-extraction for Not Specified labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with labels only
  python label_reextraction.py --labels GPL570_labels.csv.gz --output GPL570_corrected.csv.gz

  # With metadata for richer context
  python label_reextraction.py \\
    --labels GPL570_labels.csv.gz \\
    --metadata GPL570_metadata.csv.gz \\
    --output GPL570_corrected.csv.gz

  # Custom model
  python label_reextraction.py --labels labels.csv --model gemma2:9b --output corrected.csv
        """
    )
    parser.add_argument("--labels", default=None,
                        help="Path to existing labels CSV (with Not Specified entries)")
    parser.add_argument("--metadata", default=None,
                        help="Path to GEO metadata CSV (for sample titles/characteristics)")
    parser.add_argument("--output", default=None,
                        help="Output path for corrected labels")
    parser.add_argument("--model", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--columns", nargs="+", default=None,
                        help=f"Columns to fix (default: {LABEL_COLUMNS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N Not Specified samples (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without calling LLM")
    parser.add_argument("--gui", action="store_true",
                        help="Open GUI file picker to select files")
    
    args = parser.parse_args()
    
    MODEL = args.model
    if args.columns:
        LABEL_COLUMNS = args.columns
    
    # GUI mode: open file dialogs if --gui or no --labels provided
    if args.gui or args.labels is None:
        labels_path, metadata_path, output_path = gui_select_files()
    else:
        labels_path = args.labels
        metadata_path = args.metadata
        output_path = args.output or labels_path.replace('.csv', '_corrected.csv')
    
    print(f"\n{'='*60}")
    print(f"  Label Re-extraction CLI — MemGPT-style Correction")
    print(f"  Model: {MODEL}")
    print(f"  Labels: {labels_path}")
    print(f"  Metadata: {metadata_path or 'None'}")
    print(f"  Output: {output_path}")
    print(f"  GSE descriptions: fetched from GEO website")
    print(f"  Columns: {LABEL_COLUMNS}")
    print(f"{'='*60}")
    
    # Test Ollama connection
    if not args.dry_run:
        print("\n[Test] Checking Ollama connection...")
        test = call_ollama("Say OK", max_retries=1)
        if test is None:
            print("  ERROR: Cannot connect to Ollama. Is it running?")
            print("  Start with: ollama serve")
            print(f"  Then pull model: ollama pull {MODEL}")
            sys.exit(1)
        print(f"  OK — {MODEL} is responding")
    
    # Load data
    print("\n[Load] Reading data files...")
    labels_df, meta_df = load_data(labels_path, metadata_path)
    print(f"  Labels: {len(labels_df):,} samples, {len(labels_df.columns)} columns")
    if meta_df is not None:
        print(f"  Metadata: {len(meta_df):,} rows")
    
    # Find Not Specified samples
    ns_samples = find_not_specified_samples(labels_df)
    print(f"\n[Scan] Found {len(ns_samples):,} samples with Not Specified labels")
    
    # Per-column breakdown
    col_counts = Counter()
    for s in ns_samples:
        for c in s["cols_to_fix"]:
            col_counts[c] += 1
    for col, cnt in col_counts.most_common():
        pct = cnt / len(labels_df) * 100
        print(f"  {col}: {cnt:,} Not Specified ({pct:.1f}%)")
    
    if args.limit:
        ns_samples = ns_samples[:args.limit]
        print(f"\n  [Limit] Processing only first {args.limit} samples")
    
    if args.dry_run:
        print("\n[Dry Run] Would process the above samples. Exiting.")
        return
    
    # Build memory
    memory = GSEContextCache()
    build_context(labels_df, meta_df, memory)
    
    # Process!
    run_correction(labels_df, meta_df, ns_samples, memory, output_path)


if __name__ == "__main__":
    main()
