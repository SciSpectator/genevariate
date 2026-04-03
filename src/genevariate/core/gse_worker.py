"""
GSEWorker - Autonomous agent that processes NS samples for one GSE experiment.

Architecture:
    Phase 1:   Raw LLM extraction (combined Tissue + Condition + Treatment)
    Phase 1.5: Per-field ReAct collapse agent with 3 tools:
                 SEARCH  - semantic search for cluster candidates
                 PICK    - validate and return a cluster name
                 NEW_CLUSTER - register a novel entity as a new cluster
    Phase 2:   GSE context rescue (dominant sibling labels for remaining NS)

    Treatment: raw labels kept as-is (no cluster vocabulary).
    Episodic memory written after every validated collapse.

Thread-local HTTP sessions for safe parallel usage.
Supports both the `ollama` library and HTTP fallback.
"""

import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests

from genevariate.core.extraction import (
    NS,
    LABEL_COLS,
    LABEL_COLS_SCRATCH,
    EXTRACTION_MODEL,
    EXTRACTION_PROMPT_TEMPLATE,
    parse_json_extraction,
    parse_combined,
    format_raw_block,
    format_sample_for_extraction,
    prompt_extract_combined,
    is_ns,
    phase15_collapse,
    rank_candidates_by_specificity,
)
from genevariate.core.ollama_manager import (
    ollama_server_ok,
    vram_utilisation_pct,
    CPU_OLLAMA_URL,
)

# ---------------------------------------------------------------------------
# Optional ollama library (preferred when available)
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama_lib
    _OLLAMA_LIB_OK = True
except ImportError:
    _ollama_lib = None
    _OLLAMA_LIB_OK = False

# ---------------------------------------------------------------------------
# Thread-local HTTP session management
# ---------------------------------------------------------------------------
_thread_local = threading.local()

DEFAULT_OLLAMA_URL = "http://localhost:11434"
_CPU_OLLAMA_ACTIVE = False


def _get_session() -> requests.Session:
    """Return a thread-local requests.Session (created once per thread)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=8, max_retries=1,
        )
        _thread_local.session.mount("http://", adapter)
    return _thread_local.session


# ---------------------------------------------------------------------------
# GPU / CPU routing helper
# ---------------------------------------------------------------------------

def _pick_ollama_url(
    gpu_url: str = DEFAULT_OLLAMA_URL,
    vram_threshold: float = 92.0,
) -> str:
    """Route request to GPU or CPU Ollama based on current VRAM utilisation."""
    global _CPU_OLLAMA_ACTIVE
    if not _CPU_OLLAMA_ACTIVE:
        return gpu_url
    vpct = vram_utilisation_pct()
    if vpct >= vram_threshold:
        if ollama_server_ok(CPU_OLLAMA_URL):
            return CPU_OLLAMA_URL
    return gpu_url


# ============================================================================
#  GSEWorker - the autonomous per-GSE agent
# ============================================================================

class GSEWorker:
    """
    Processes NS samples for one GSE experiment using a ReAct collapse agent.

    Lifecycle:
        1. Instantiate with GSE id, GSEContext, model config, and MemoryAgent.
        2. Call ``repair_one(gsm, raw_dict)`` for each NS sample.
        3. Or call ``process_all(samples)`` for batch processing.

    The collapse agent has three tools:
        SEARCH       - semantic search against cluster vocabulary
        PICK         - select a known cluster label
        NEW_CLUSTER  - register a new cluster for a unique biomedical entity
    """

    # Class-level constants
    MAX_REACT_TURNS = 3
    EXTRACT_MODEL = EXTRACTION_MODEL          # gemma2:2b for fast extraction
    COLLAPSE_MAX_CANDIDATES = 8
    DOMINANT_THRESHOLD = 0.50                  # GSE rescue threshold

    def __init__(
        self,
        gse_id: str,
        ctx,                                   # GSEContext
        model: str = "gemma2:9b",
        ollama_url: str = DEFAULT_OLLAMA_URL,
        watchdog=None,                         # Watchdog instance (optional)
        mem_agent=None,                        # MemoryAgent instance
        platform: str = "",
    ):
        self.gse_id = gse_id
        self.ctx = ctx
        self.model = model
        self.ollama_url = ollama_url
        self.watchdog = watchdog
        self.mem_agent = mem_agent
        self.platform = platform

        # Pre-build GSE description block for prompt injection
        lines: List[str] = []
        if ctx.title:
            lines.append(f"Experiment title  : {ctx.title}")
        if getattr(ctx, "summary", ""):
            lines.append(f"Experiment summary: {ctx.summary[:500]}")
        if getattr(ctx, "design", ""):
            lines.append(f"Overall design    : {ctx.design[:300]}")
        self._gse_block = "\n".join(lines) + "\n" if lines else ""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg):
        """Log helper - prints to stdout. Override in subclass for GUI logging."""
        print(msg)

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Single-turn LLM call using the configured collapse model."""
        return self._llm_with_model(self.model, prompt, max_tokens)

    def _llm_with_model(
        self, model: str, prompt: str, max_tokens: int = 200
    ) -> str:
        """
        Single-turn LLM generation.

        Tries the ``ollama`` Python library first (keeps connection alive,
        handles streaming internally). Falls back to raw HTTP POST if the
        library is unavailable.
        """
        if self.watchdog:
            self.watchdog.wait_if_paused()
            self.watchdog.record_call()

        url = _pick_ollama_url(self.ollama_url)

        for attempt in range(1, 4):
            try:
                # -- ollama library path --
                # Use HTTP when routed to CPU Ollama (library connects to default)
                if _OLLAMA_LIB_OK and url == self.ollama_url:
                    resp = _ollama_lib.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options={
                            "temperature": 0.0,
                            "num_predict": max_tokens,
                        },
                        keep_alive=-1,
                    )
                    # ollama >=0.2 returns an object; older returns dict
                    if hasattr(resp, "message") and hasattr(resp.message, "content"):
                        return (resp.message.content or "").strip()
                    if isinstance(resp, dict):
                        return resp.get("message", {}).get("content", "").strip()
                    return ""

                # -- HTTP path (required for CPU Ollama on port 11435) --
                sess = _get_session()
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                    },
                }
                r = sess.post(
                    f"{url}/api/chat", json=payload, timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "").strip()

            except Exception as e:
                err = str(e).lower()
                if "out of memory" in err or "cudamalloc" in err:
                    if ollama_server_ok(CPU_OLLAMA_URL):
                        self._log(f"  [OOM] Switching {self.gse_id} to CPU")
                        url = CPU_OLLAMA_URL
                        time.sleep(2)
                    else:
                        time.sleep(8)
                elif "connection refused" in err or "disconnected" in err:
                    time.sleep(5 * attempt)
                else:
                    time.sleep(3 * attempt)
                if attempt == 3:
                    return ""
        return ""

    def _llm_chat(
        self, messages: List[dict], max_tokens: int = 200
    ) -> str:
        """
        Multi-turn chat LLM call (used by the ReAct collapse agent).

        Same retry / OOM logic as ``_llm_with_model``.
        """
        if self.watchdog:
            self.watchdog.wait_if_paused()
            self.watchdog.record_call()

        url = _pick_ollama_url(self.ollama_url)

        for attempt in range(1, 4):
            try:
                # Use ollama library only when on default GPU URL
                if _OLLAMA_LIB_OK and url == self.ollama_url:
                    resp = _ollama_lib.chat(
                        model=self.model,
                        messages=messages,
                        options={
                            "temperature": 0.0,
                            "num_predict": max_tokens,
                            "num_ctx": 4096,
                        },
                        keep_alive=-1,
                    )
                    if hasattr(resp, "message") and hasattr(resp.message, "content"):
                        return (resp.message.content or "").strip()
                    if isinstance(resp, dict):
                        return resp.get("message", {}).get("content", "").strip()
                    return ""

                # HTTP path (required for CPU Ollama on port 11435)
                sess = _get_session()
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                        "num_ctx": 4096,
                    },
                }
                r = sess.post(f"{url}/api/chat", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "").strip()

            except Exception as e:
                err = str(e).lower()
                if "out of memory" in err or "cudamalloc" in err:
                    if ollama_server_ok(CPU_OLLAMA_URL):
                        self._log(f"  [OOM] Switching {self.gse_id} to CPU")
                        url = CPU_OLLAMA_URL
                        time.sleep(2)
                    else:
                        time.sleep(8)
                else:
                    time.sleep(3 * attempt)
                if attempt == 3:
                    return ""
        return ""

    # ------------------------------------------------------------------
    # Agent tools (invoked inside _run_collapse_agent)
    # ------------------------------------------------------------------

    def _tool_gse_context(self, col: str) -> str:
        """
        Tool 1 - GSE Context: return the experiment-level context block
        including sibling label distribution for *col*.
        """
        ctx_block = self.ctx.context_block(col) if hasattr(self.ctx, "context_block") else ""
        if not ctx_block:
            parts: List[str] = []
            if self._gse_block:
                parts.append(self._gse_block)
            counts = self.ctx.label_counts.get(col, Counter())
            if counts:
                parts.append(f"Sibling {col} labels:")
                for lbl, cnt in counts.most_common():
                    parts.append(f"  {lbl!r} ({cnt}x)")
            ctx_block = "\n".join(parts)
        return ctx_block or "(no GSE context available)"

    def _tool_llm_memory(self, col: str, query: str) -> str:
        """
        Tool 2 - LLM Memory (semantic search): query the MemoryAgent
        cluster vocabulary for candidates matching *query*.
        Returns formatted candidate list.
        """
        ma = self.mem_agent
        if ma is None:
            return "OBSERVATION: memory agent not available"

        # Cluster map O(1) lookup first
        cluster = ma.cluster_lookup(col, query)
        if cluster:
            return f"OBSERVATION: cluster_map exact hit -> {cluster!r}"

        # Semantic search
        hits = ma.semantic_search(col, query, k=self.COLLAPSE_MAX_CANDIDATES)
        if not hits:
            return f"OBSERVATION: no semantic matches for {query!r}"

        ranked = rank_candidates_by_specificity(query, hits)
        lines = [f"OBSERVATION: semantic search for {query!r}:"]
        for label, sim, score in ranked[:6]:
            lines.append(f"  {label!r}  sim={sim:.3f}  score={score:.1f}")
        return "\n".join(lines)

    def _tool_episodic(self, col: str, raw_label: str) -> str:
        """
        Tool 3 - Episodic Memory: look up past resolutions for *raw_label*.
        """
        ma = self.mem_agent
        if ma is None:
            return "OBSERVATION: memory agent not available"

        hits = ma.episodic_search(col, raw_label)
        if not hits:
            return f"OBSERVATION: no episodic history for {raw_label!r}"

        lines = [f"OBSERVATION: past resolutions for {raw_label!r}:"]
        for h in hits[:3]:
            lines.append(
                f"  canonical={h['canonical']!r}  count={h['count']}  "
                f"conf={h['confidence']:.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ReAct collapse agent
    # ------------------------------------------------------------------

    def _run_collapse_agent(
        self,
        gsm: str,
        col: str,
        extracted: str,
        gsm_row: Optional[dict] = None,
    ) -> Tuple[str, bool, str]:
        """
        ReAct collapse agent with 3 tools and max 3 turns.

        Actions:
            SEARCH: <query>         - invoke _tool_llm_memory
            PICK: <label>           - finalise with a known cluster name
            NEW_CLUSTER: <name>     - register a new cluster and return it

        Returns:
            (final_label, collapsed_bool, collapse_rule)
        """
        ma = self.mem_agent

        # -- Pre-loaded context segments --
        gse_ctx_text = self._tool_gse_context(col)
        episodic_text = self._tool_episodic(col, extracted)
        init_search_text = self._tool_llm_memory(col, extracted)

        # Sample info
        title = ""
        if gsm_row:
            title = str(gsm_row.get("gsm_title", gsm_row.get("title", "")))[:60]

        system = (
            f"You are a biomedical label normalization agent for GEO field: {col}.\n"
            f"Your job: collapse the extracted label to an approved cluster name.\n\n"
            f"TOOLS (use exactly one per turn):\n"
            f"  SEARCH: <query>       search cluster vocabulary\n"
            f"  PICK: <label>         select a cluster name from results\n"
            f"  NEW_CLUSTER: <name>   register a new cluster for unique entities\n\n"
            f"RULES:\n"
            f"  1. Pick the most specific matching cluster name.\n"
            f"  2. If the entity is genuinely novel, use NEW_CLUSTER with a Title Case name.\n"
            f"  3. If nothing matches and it is not a real entity: PICK: NO_MATCH\n"
            f"  4. Format: THOUGHT: <reason>\\nACTION: SEARCH/PICK/NEW_CLUSTER: <value>\n"
        )

        context = (
            f"[GSE CONTEXT]\n{gse_ctx_text}\n\n"
            f"[EPISODIC MEMORY]\n{episodic_text}\n\n"
            f"[INITIAL SEARCH]\n{init_search_text}\n\n"
            f"SAMPLE {gsm}: {title}\n"
            f"LABEL TO COLLAPSE: {extracted!r}\n\n"
            f"Now reason and act. Start with THOUGHT:"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ]

        for turn in range(self.MAX_REACT_TURNS):
            response = self._llm_chat(messages, max_tokens=150)
            if not response:
                break
            messages.append({"role": "assistant", "content": response})

            # Parse action from response
            action_line = self._parse_action(response)
            if not action_line:
                break

            action_upper = action_line.upper()

            # ---- PICK ----
            if action_upper.startswith("PICK:"):
                val = action_line[5:].strip().strip('"').strip("'")
                if val.upper() in ("NO_MATCH", "NOMATCH", "NO MATCH"):
                    return extracted, False, "react_no_match"
                # Validate against cluster vocabulary
                validated = self._validate_pick(col, val, extracted)
                if validated:
                    return validated, True, "react_pick"
                # Not in vocabulary -- accept as new label
                return val, True, "react_pick_new"

            # ---- NEW_CLUSTER ----
            elif action_upper.startswith("NEW_CLUSTER:"):
                name = action_line[12:].strip().strip('"').strip("'")
                if name and not is_ns(name) and ma is not None:
                    ma.register_new_cluster(col, name, extracted, log_fn=self._log)
                    return name, True, "react_new_cluster"
                return extracted, False, "react_new_cluster_failed"

            # ---- SEARCH ----
            elif action_upper.startswith("SEARCH:"):
                query = action_line[7:].strip().strip('"').strip("'")
                obs = self._tool_llm_memory(col, query)
                messages.append({"role": "user", "content": obs})

            else:
                # Unrecognised action -- break
                break

        return extracted, False, "react_exhausted"

    @staticmethod
    def _parse_action(response: str) -> Optional[str]:
        """Extract the ACTION line from a ReAct-style response."""
        for line in response.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("ACTION:"):
                return stripped[7:].strip()
            # Also accept bare action keywords
            for prefix in ("SEARCH:", "PICK:", "NEW_CLUSTER:"):
                if upper.startswith(prefix):
                    return stripped
        return None

    def _validate_pick(
        self, col: str, picked: str, original: str
    ) -> Optional[str]:
        """
        Validate a PICK action against the cluster vocabulary.
        Returns the canonical cluster name or None.
        """
        ma = self.mem_agent
        if ma is None:
            return picked if picked else None

        # Cluster map lookup
        cluster = ma.cluster_lookup(col, picked)
        if cluster:
            return cluster

        # Check if it is a registered cluster name
        if ma.is_cluster_name(col, picked):
            return picked

        # Fuzzy: check if normalised form matches any cluster
        picked_compact = re.sub(r'[^a-z0-9]', '', picked.lower())
        # Try semantic labels for close match
        hits = ma.semantic_search(col, picked, k=3)
        for label, sim in hits:
            if re.sub(r'[^a-z0-9]', '', label.lower()) == picked_compact:
                return label
            if sim >= 0.95:
                return label

        return None

    # ------------------------------------------------------------------
    # Cluster gate validation
    # ------------------------------------------------------------------

    def _cluster_gate(self, col: str, label: str) -> Tuple[str, bool]:
        """
        Gate check: verify the label is a known cluster name.

        Returns (validated_label, passed_gate).
        If the label is already a cluster name or maps to one, returns the
        canonical form. Otherwise returns the original with passed=False.
        """
        ma = self.mem_agent
        if ma is None:
            return label, True  # no memory agent -> pass through

        # Direct cluster lookup
        cluster = ma.cluster_lookup(col, label)
        if cluster:
            return cluster, True

        if ma.is_cluster_name(col, label):
            return label, True

        return label, False

    # ------------------------------------------------------------------
    # repair_one: the per-sample repair pipeline
    # ------------------------------------------------------------------

    def repair_one(
        self,
        gsm: str,
        raw: dict,
        ns_cols: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Full repair pipeline for one sample.

        Architecture aligned with geo_ns_repair_v2.py:
            1a. Combined raw extraction using EXTRACTION_PROMPT_TEMPLATE (gemma2:2b)
            1b. Fallback with full GSE context for any remaining NS fields
            2.  Per-field resolution (parallel threads):
                - Capitalisation normalisation (cluster_lookup on raw)
                - GSE rescue FIRST (dominant sibling for NS fields)
                - Fast-path cluster_map lookup
                - GSE dominant (>70%) fast-path
                - Phase 1.5 deterministic collapse
                - ReAct collapse agent
                - Cluster gate with new-cluster registration
                - Treatment sibling snap (no cluster vocab)
                - Episodic memory write

        Args:
            gsm: GSM sample identifier
            raw: dict of raw GEO metadata fields
            ns_cols: columns to repair (default: LABEL_COLS)

        Returns:
            dict {col: final_label} for all repaired columns
        """
        if ns_cols is None:
            ns_cols = list(LABEL_COLS)

        result: Dict[str, str] = {col: NS for col in LABEL_COLS_SCRATCH}
        ma = self.mem_agent

        # ── Step 1a: Combined raw extraction with fast model ──

        title_str = str(raw.get("gsm_title", raw.get("title", "")))[:80]
        source_str = str(raw.get("source_name", raw.get("source_name_ch1", "")))[:60]
        chars_str = str(raw.get("characteristics", raw.get("characteristics_ch1", "")))[:250]

        # Use .replace() instead of .format() — template has literal JSON braces
        extraction_prompt = (EXTRACTION_PROMPT_TEMPLATE
            .replace("{TITLE}", title_str)
            .replace("{SOURCE}", source_str)
            .replace("{CHAR}", chars_str))

        # Append GSE context to extraction prompt (v2 approach)
        treat_str = str(raw.get("treatment_protocol", "")).strip()[:200]
        desc_str = str(raw.get("description", "")).strip()[:200]
        if treat_str:
            extraction_prompt += f"\nTreatment protocol: {treat_str}"
        if desc_str:
            extraction_prompt += f"\nDescription: {desc_str}"
        if self._gse_block:
            extraction_prompt += f"\n{self._gse_block[:400]}"

        raw_text = ""
        for _attempt in range(3):
            raw_text = self._llm_with_model(self.EXTRACT_MODEL, extraction_prompt, max_tokens=200)
            if raw_text:
                break
            time.sleep(5 * (_attempt + 1))

        if raw_text:
            parsed = parse_json_extraction(raw_text, LABEL_COLS_SCRATCH)
            for col in LABEL_COLS_SCRATCH:
                if not is_ns(parsed.get(col, NS)):
                    result[col] = parsed[col]

        # ── Step 1b: Fallback with GSE context for still-NS fields ──

        still_ns = [c for c in ns_cols if is_ns(result.get(c, NS))]
        if still_ns and self._gse_block:
            fallback_prompt = prompt_extract_combined(
                gsm, raw, self.ctx, still_ns, gse_block=self._gse_block,
            )
            fb_text = ""
            for _attempt in range(3):
                fb_text = self._llm(fallback_prompt, max_tokens=150)
                if fb_text:
                    break
                time.sleep(5 * (_attempt + 1))
            if fb_text:
                fb_parsed = parse_combined(fb_text, still_ns)
                for col in still_ns:
                    if not is_ns(fb_parsed.get(col, NS)):
                        result[col] = fb_parsed[col]

        # ── Per-field resolution (v2 architecture) ──

        def _run_field(col: str) -> Tuple[str, str, str]:
            """Resolve one field — full v2 pipeline. Returns (col, final, rule)."""
            out1 = result.get(col, NS)

            # --- GSE rescue FIRST (v2: moved before collapse agent) ---
            # If extraction returned NS but GSE has dominant sibling, use it
            ctx_labels = list(self.ctx.label_counts.get(col, Counter()).keys())
            ctx_counts = dict(self.ctx.label_counts.get(col, Counter()))

            if (is_ns(out1) or not out1) and ctx_labels:
                _dom = max(ctx_counts, key=ctx_counts.get)
                _dom_n = ctx_counts[_dom]
                _total_ctx = sum(ctx_counts.values())
                _dom_pct = _dom_n / _total_ctx if _total_ctx else 0
                if _dom_pct >= 0.5 and ma and ma.is_cluster_name(col, _dom):
                    out1 = _dom
                    self._log(f"  [GSE RESCUE] {gsm} {col}: NS rescued "
                              f"from dominant sibling {_dom!r} ({_dom_pct:.0%})")
                elif _dom_pct >= 0.3 and ma:
                    _cand = ma.cluster_lookup(col, _dom)
                    if _cand and ma.is_cluster_name(col, _cand):
                        out1 = _cand
                        self._log(f"  [GSE RESCUE] {gsm} {col}: NS rescued "
                                  f"from sibling {_cand!r} ({_dom_pct:.0%})")

            # Hard skip if truly no evidence anywhere
            if (is_ns(out1) or not out1) and not ctx_labels and not ma:
                return col, NS, ""

            final = out1
            collapsed = False
            collapse_rule = ""

            # --- Fast-path: cluster_map O(1) ---
            if ma and not is_ns(out1):
                direct = ma.cluster_lookup(col, out1)
                if direct and ma.is_cluster_name(col, direct):
                    final = direct
                    collapsed = True
                    collapse_rule = "direct_cluster_map"

            # --- Fast-path: GSE dominant >70% ---
            if ma and not collapsed and ctx_counts:
                top_label, top_count = max(ctx_counts.items(), key=lambda x: x[1])
                total_ctx = sum(ctx_counts.values())
                if total_ctx > 0 and top_count / total_ctx >= 0.70:
                    if ma.is_cluster_name(col, top_label):
                        final = top_label
                        collapsed = True
                        collapse_rule = "gse_dominant"

            # --- Episodic fast-path ---
            if not collapsed and ma is not None:
                ep_hits = ma.episodic_search(col, out1 if not is_ns(out1) else "")
                if ep_hits and ep_hits[0]["count"] >= 2 and ep_hits[0]["confidence"] >= 0.80:
                    final = ep_hits[0]["canonical"]
                    collapsed = True
                    collapse_rule = "episodic"

            # --- ReAct collapse agent ---
            if not collapsed and ma is not None and ma.is_ready(col):
                agent_result, agent_ok, agent_rule = self._run_collapse_agent(
                    gsm, col, out1, gsm_row=raw,
                )
                if agent_ok and not is_ns(agent_result):
                    final = agent_result
                    collapsed = True
                    collapse_rule = agent_rule

            # --- Phase 1.5 deterministic fallback ---
            if not collapsed and ctx_labels:
                matched, det_rule = phase15_collapse(out1, ctx_labels)
                if matched and matched != out1:
                    if ma and not ma.is_cluster_name(col, matched):
                        remapped = ma.cluster_lookup(col, matched)
                        matched = remapped if (remapped and
                                  ma.is_cluster_name(col, remapped)) else None
                    if matched:
                        final = matched
                        collapsed = True
                        collapse_rule = det_rule

            # --- Cluster gate (v2: with new-cluster registration) ---
            _col_has_vocab = ma and ma.is_ready(col)
            if not _col_has_vocab:
                # No cluster vocabulary (e.g. Treatment) — sibling snap
                if final and not is_ns(final) and ctx_counts:
                    _norm_final = final.lower().strip()
                    _best = None
                    _best_count = 0
                    for sib, cnt in ctx_counts.items():
                        _norm_sib = sib.lower().strip()
                        _overlap = sum(
                            1 for w in _norm_final.split()
                            if w in _norm_sib.split())
                        _total_words = max(1, len(_norm_final.split()))
                        if (_norm_sib in _norm_final or
                            _norm_final in _norm_sib or
                            (_overlap / _total_words) >= 0.6):
                            if cnt > _best_count:
                                _best = sib
                                _best_count = cnt
                    if _best and _best != final:
                        final = _best
                        collapse_rule = "treatment_sibling_snap"
                if final and not is_ns(final):
                    collapsed = True
                    collapse_rule = collapse_rule or "raw_extraction"

            elif ma and final != NS and not collapsed:
                # Cluster gate for Tissue/Condition
                if not ma.is_cluster_name(col, final):
                    remapped = ma.cluster_lookup(col, final)
                    if remapped and ma.is_cluster_name(col, remapped):
                        final = remapped
                        collapse_rule = (collapse_rule or "") + "+gate_remap"
                        collapsed = True
                    else:
                        # Check if novel entity deserves new cluster
                        _is_novel = (
                            final and len(final.strip()) >= 4 and
                            not is_ns(final) and
                            not any(w in final.lower()
                                    for w in ("unknown", "unspecified", "n/a",
                                              "na", "none", "other", "not ",
                                              "mixed")))
                        if _is_novel and ma:
                            ma.register_new_cluster(
                                col, final.strip(), out1, log_fn=self._log)
                            collapsed = True
                            collapse_rule = (collapse_rule or "") + "+new_cluster"
                        else:
                            final = NS
                            collapsed = False
                            collapse_rule = (collapse_rule or "") + "+gate_rejected"

            # --- Episodic memory write (post-gate) ---
            if ma and final != out1 and not is_ns(final):
                do_log, conf, _ = ma.should_log(col, out1, final, collapse_rule)
                if do_log:
                    ma.log_resolution(
                        col=col,
                        raw_label=out1,
                        canonical=final,
                        confidence=conf,
                        platform=self.platform,
                        gse=self.gse_id,
                        gsm=gsm,
                        collapse_rule=collapse_rule,
                    )

            return col, final if not is_ns(final) else NS, collapse_rule

        # Run all field agents in parallel threads
        collapse_rules: Dict[str, str] = {}
        all_cols = [c for c in ns_cols if c in LABEL_COLS_SCRATCH]

        if len(all_cols) > 1:
            with ThreadPoolExecutor(max_workers=len(all_cols)) as pool:
                futures = {
                    pool.submit(_run_field, col): col
                    for col in all_cols
                }
                for fut in as_completed(futures):
                    try:
                        col, final, rule = fut.result()
                        result[col] = final
                        collapse_rules[col] = rule
                    except Exception as exc:
                        col = futures[fut]
                        self._log(f"  [WARN] {gsm}/{col} collapse error: {exc}")
        else:
            for col in all_cols:
                col, final, rule = _run_field(col)
                result[col] = final
                collapse_rules[col] = rule

        # Attach metadata for callers that need it
        result["_collapse_rules"] = collapse_rules

        return result

    # ------------------------------------------------------------------
    # process_all: batch processing
    # ------------------------------------------------------------------

    def process_all(
        self,
        samples: List[dict],
        ns_cols: Optional[List[str]] = None,
        max_workers: int = 1,
        progress_fn=None,
    ) -> List[Dict[str, str]]:
        """
        Process a batch of NS samples.

        Args:
            samples:     list of dicts, each with 'gsm' key + raw metadata
            ns_cols:     columns to repair (default: LABEL_COLS)
            max_workers: 1 = sequential, >1 = ThreadPoolExecutor
            progress_fn: optional callback(done_count, total)

        Returns:
            list of result dicts (same order as input)
        """
        total = len(samples)
        if total == 0:
            return []

        results: List[Optional[Dict[str, str]]] = [None] * total

        def _do_one(idx: int) -> Tuple[int, Dict[str, str]]:
            sample = samples[idx]
            gsm = sample.get("gsm", sample.get("GSM", f"UNK_{idx}"))
            try:
                res = self.repair_one(gsm, sample, ns_cols=ns_cols)
            except Exception as exc:
                self._log(f"  [ERROR] {gsm}: {exc}")
                res = {col: NS for col in (ns_cols or LABEL_COLS)}
            return idx, res

        if max_workers <= 1:
            # Sequential mode
            for i in range(total):
                idx, res = _do_one(i)
                results[idx] = res
                if progress_fn:
                    progress_fn(i + 1, total)
        else:
            # Parallel mode
            done_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_do_one, i): i for i in range(total)}
                for fut in as_completed(futures):
                    try:
                        idx, res = fut.result()
                        results[idx] = res
                    except Exception as exc:
                        idx = futures[fut]
                        self._log(f"  [ERROR] sample {idx}: {exc}")
                        results[idx] = {col: NS for col in (ns_cols or LABEL_COLS)}
                    done_count += 1
                    if progress_fn:
                        progress_fn(done_count, total)

        # Fill any Nones (shouldn't happen, but defensive)
        default = {col: NS for col in (ns_cols or LABEL_COLS)}
        return [r if r is not None else dict(default) for r in results]
