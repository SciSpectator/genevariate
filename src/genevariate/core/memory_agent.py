"""
Memory Agent - 4-tier persistent biomedical label memory (SQLite-first).

Architecture:
    Tier 1 - Core memory     : top-N most frequent labels
    Tier 2 - Semantic memory : cluster names embedded, cosine RAG query
    Tier 3 - Episodic memory : log of every past resolution + confidence
    Tier 4 - Knowledge graph : synonym/hierarchy triples

All data stored in SQLite database. No .txt file dependency.
Cluster vocabulary is built directly into SQL tables.

Cross-agent API:
    search(col, query)    -> multi-tier search results
    log_resolution(...)   -> writes episodic tier
    register_new_cluster(...) -> creates new cluster in DB
"""

import os
import re
import threading
import sqlite3
import numpy as np
import requests
from typing import Dict, List, Optional
from collections import Counter

# Constants
NS = "Not Specified"
LABEL_COLS = ["Tissue", "Condition"]
LABEL_COLS_SCRATCH = ["Tissue", "Condition", "Treatment"]

MEM_TOP_K = 10
MEM_MATCH_THRESH = 0.72
MEM_EMBED_BATCH = 32
MEM_CORE_N = 99999
MEM_DB_NAME = "biomedical_memory.db"
MEM_EMBED_MODEL = "nomic-embed-text"

CLUSTER_FILE = {
    "Tissue": "Tissues_clusters.txt",
    "Condition": "Conditions_clusters.txt",
    "Treatment": "Treatments_clusters.txt",
}


class MemoryAgent:
    """
    Persistent 4-tier biomedical label memory agent.

    SQL-first design: all cluster data stored in SQLite.
    Cluster .txt files are optional import sources, not required at runtime.

    Lifecycle:
        1. MemoryAgent(db_path, ollama_url)     open/create DB
        2. build_from_clusters(dir) OR import_clusters(dict)  populate from any source
        3. search(col, text)                     ranked candidates
        4. log_resolution(...)                   episodic write
        5. register_new_cluster(...)             create new cluster in DB only

    Thread-safe: reads use shared connection pool, writes serialised via lock.
    """

    def __init__(self, db_path: str, ollama_url: str = "http://localhost:11434"):
        self.db_path = db_path
        self.ollama_url = ollama_url
        self._lock = threading.Lock()
        self._new_cluster_log: list = []
        self._vec_cache: Dict[str, tuple] = {}
        self._cache_ok: Dict[str, bool] = {c: False for c in LABEL_COLS}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._llm_memory_dir: Optional[str] = None
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init_db(self):
        """Create all memory tables if they don't exist."""
        with self._lock, self._conn() as c:
            c.executescript("""
            -- Tier 1: Core labels (top-N most frequent)
            CREATE TABLE IF NOT EXISTS core_labels (
                col    TEXT NOT NULL,
                label  TEXT NOT NULL,
                freq   INTEGER DEFAULT 1,
                PRIMARY KEY (col, label)
            );

            -- Tier 2: Semantic labels (embedded cluster names)
            CREATE TABLE IF NOT EXISTS semantic_labels (
                col        TEXT NOT NULL,
                label      TEXT NOT NULL,
                embedding  BLOB,
                freq       INTEGER DEFAULT 1,
                PRIMARY KEY (col, label)
            );

            -- Tier 3: Episodic log (every resolution ever made)
            CREATE TABLE IF NOT EXISTS episodic_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                col         TEXT NOT NULL,
                raw_label   TEXT NOT NULL,
                canonical   TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                platform    TEXT DEFAULT '',
                gse         TEXT DEFAULT '',
                gsm         TEXT DEFAULT '',
                ts          TEXT DEFAULT (datetime('now')),
                collapse_rule TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS ep_raw ON episodic_log(col, raw_label);
            CREATE INDEX IF NOT EXISTS ep_can ON episodic_log(col, canonical);

            -- Tier 4: Knowledge graph triples
            CREATE TABLE IF NOT EXISTS kg_triples (
                col      TEXT NOT NULL,
                subject  TEXT NOT NULL,
                relation TEXT NOT NULL,
                object   TEXT NOT NULL,
                weight   REAL DEFAULT 1.0,
                PRIMARY KEY (col, subject, relation, object)
            );
            CREATE INDEX IF NOT EXISTS kg_sub ON kg_triples(col, subject);

            -- Cluster map: raw label -> canonical cluster name (O(1) lookup)
            CREATE TABLE IF NOT EXISTS cluster_map (
                col     TEXT NOT NULL,
                raw     TEXT NOT NULL,
                cluster TEXT NOT NULL,
                PRIMARY KEY (col, raw)
            );
            CREATE INDEX IF NOT EXISTS cm_raw ON cluster_map(col, raw);

            -- Raw labels table: stores all raw extracted labels for reference
            CREATE TABLE IF NOT EXISTS raw_labels (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                col     TEXT NOT NULL,
                raw     TEXT NOT NULL,
                gsm     TEXT DEFAULT '',
                gse     TEXT DEFAULT '',
                platform TEXT DEFAULT '',
                ts      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS rl_col ON raw_labels(col, raw);
            """)

    # ── Cluster file parsing (optional import) ──

    @staticmethod
    def parse_cluster_file(path: str, col: str) -> Dict[str, List[str]]:
        """Parse a cluster file. Returns {cluster_name: [raw_label, ...]}."""
        clusters: Dict[str, List[str]] = {}
        if not os.path.exists(path):
            return clusters

        current_cluster = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.rstrip()
                stripped = line_s.strip()

                if stripped.upper().startswith("CLUSTER:"):
                    name = stripped[len("CLUSTER:"):].strip()
                    name = re.sub(r"\s*\(TOTAL:.*$", "", name, flags=re.IGNORECASE).strip()
                    if name.upper() == "NOT SPECIFIED":
                        current_cluster = None
                        continue
                    if name == name.upper() and len(name) > 1:
                        name = name.title()
                    current_cluster = name
                    if current_cluster not in clusters:
                        clusters[current_cluster] = []

                elif current_cluster is not None:
                    if re.match(r"\s+-\s+", line_s):
                        raw = re.sub(r"\s*\|.*$", "", line_s)
                        raw = re.sub(r"^\s*-\s*", "", raw).strip()
                        if raw:
                            clusters[current_cluster].append(raw)

        return clusters

    # ── Import clusters from dict (SQL-first path) ──

    def import_clusters(self, col: str, clusters: Dict[str, List[str]],
                        log_fn=print) -> None:
        """
        Import cluster vocabulary directly into SQL database.
        This is the primary path - no .txt files needed.

        Args:
            col: Label column (Tissue, Condition, Treatment)
            clusters: {cluster_name: [raw_label, ...]}
        """
        if not clusters:
            log_fn(f"  [MemoryAgent] {col}: no clusters to import")
            return

        log_fn(f"  [MemoryAgent] {col}: importing {len(clusters):,} clusters, "
               f"{sum(len(v) for v in clusters.values()):,} raw label mappings")

        cm_rows = []
        kg_rows = []
        for cluster, raws in clusters.items():
            for raw in raws:
                raw_lower = raw.lower().strip()
                if not raw_lower:
                    continue
                cm_rows.append((col, raw_lower, cluster))
                cm_rows.append((col, raw.strip(), cluster))
                kg_rows.append((col, raw.strip(), "assigned_to", cluster, 1.0))
            cm_rows.append((col, cluster.lower(), cluster))
            cm_rows.append((col, cluster, cluster))

        # Store all normalised forms
        extra_rows = []
        seen = {(c1, r) for c1, r, _ in cm_rows}
        for col_val, raw, cluster in cm_rows:
            for form in self._all_forms(raw):
                if form and (col_val, form) not in seen:
                    extra_rows.append((col_val, form, cluster))
                    seen.add((col_val, form))

        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO cluster_map (col, raw, cluster) "
                "VALUES (?,?,?)", cm_rows + extra_rows)
            c.executemany(
                "INSERT OR REPLACE INTO kg_triples "
                "(col, subject, relation, object, weight) VALUES (?,?,?,?,?)",
                kg_rows)

        log_fn(f"  [MemoryAgent] {col}: {len(cm_rows) + len(extra_rows):,} "
               f"cluster_map entries stored in DB")

        # Embed cluster names
        cluster_names = list(clusters.keys())
        with self._conn() as c:
            existing = {r[0] for r in c.execute(
                "SELECT label FROM semantic_labels WHERE col=?", (col,))}
        new_clusters = [cn for cn in cluster_names if cn not in existing]

        if new_clusters:
            log_fn(f"  [MemoryAgent] {col}: embedding {len(new_clusters):,} cluster names")
            vecs = self._embed_batch(new_clusters, log_fn)
            if vecs is not None:
                with self._lock, self._conn() as c:
                    c.executemany(
                        "INSERT OR REPLACE INTO semantic_labels "
                        "(col, label, embedding, freq) VALUES (?,?,?,?)",
                        [(col, cn, vecs[i].astype(np.float32).tobytes(),
                          len(clusters[cn]))
                         for i, cn in enumerate(new_clusters)])

        # Core labels
        top = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM core_labels WHERE col=?", (col,))
            c.executemany(
                "INSERT INTO core_labels (col, label, freq) VALUES (?,?,?)",
                [(col, cn, len(raws)) for cn, raws in top])

        self._load_cache(col, log_fn)

    # ── Build from cluster files (legacy import path) ──

    def build_from_clusters(self, llm_memory_dir: str, log_fn=print) -> None:
        """
        Populate memory from cluster .txt files.
        Data is stored in SQL DB - the .txt files are just an import source.
        """
        self._llm_memory_dir = llm_memory_dir
        for col in LABEL_COLS:
            fname = CLUSTER_FILE.get(col)
            if not fname:
                continue
            path = os.path.join(llm_memory_dir, fname)
            if not os.path.exists(path):
                log_fn(f"  [MemoryAgent] {col}: cluster file not found: {path}")
                continue

            log_fn(f"  [MemoryAgent] {col}: parsing {fname}")
            clusters = self.parse_cluster_file(path, col)
            if not clusters:
                log_fn(f"  [MemoryAgent] {col}: no clusters parsed")
                continue

            self.import_clusters(col, clusters, log_fn)

    # ── Build from DataFrames ──

    def build(self, all_dfs: Dict[str, "pd.DataFrame"], log_fn=print) -> None:
        """Populate from input DataFrames (non-NS labels only)."""
        import pandas as pd
        for col in LABEL_COLS:
            freq: Counter = Counter()
            for platform, df in all_dfs.items():
                if col not in df.columns:
                    continue
                mask = df[col].notna() & (df[col] != NS) & (df[col].str.strip() != "")
                confirmed = df.loc[mask, col].tolist()
                freq.update(confirmed)
            if not freq:
                continue
            log_fn(f"  [MemoryAgent] {col}: {len(freq):,} unique labels ingested")

            top = freq.most_common()
            with self._lock, self._conn() as c:
                c.execute("DELETE FROM core_labels WHERE col=?", (col,))
                c.executemany(
                    "INSERT INTO core_labels (col, label, freq) VALUES (?,?,?)",
                    [(col, lbl, cnt) for lbl, cnt in top])
            self._load_cache(col, log_fn)

    # ── Normalisation helpers ──

    @staticmethod
    def _norm_raw(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"[-_/]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _strip_cell_prefix(text: str) -> str:
        tl = text.lower().strip()
        for pfx in ("cell line:", "cell type:", "tissue:", "organ:",
                     "cell line :", "cell type :", "tissue :"):
            if tl.startswith(pfx):
                return text[len(pfx):].strip()
        return text

    @staticmethod
    def _all_forms(text: str) -> List[str]:
        t = text.strip()
        forms = [t, t.lower(), MemoryAgent._norm_raw(t)]
        stripped = MemoryAgent._strip_cell_prefix(t)
        if stripped != t:
            forms += [stripped, stripped.lower(), MemoryAgent._norm_raw(stripped)]
        return [f for f in forms if f]

    # ── Cluster lookup (O(1)) ──

    def cluster_lookup(self, col: str, raw_label: str) -> Optional[str]:
        try:
            with self._conn() as c:
                for attempt in self._all_forms(raw_label):
                    row = c.execute(
                        "SELECT cluster FROM cluster_map WHERE col=? AND raw=?",
                        (col, attempt)).fetchone()
                    if row:
                        return row[0]
                row = c.execute(
                    "SELECT label FROM semantic_labels "
                    "WHERE col=? AND LOWER(label)=LOWER(?)",
                    (col, raw_label.strip())).fetchone()
                if row:
                    return row[0]
        except Exception:
            pass
        return None

    # ── Embedding helpers ──

    def _detect_embed_model(self) -> str:
        if getattr(self, "_embed_model_detected", None):
            return self._embed_model_detected
        preferred = ["nomic-embed-text", "mxbai-embed-large",
                     "snowflake-arctic-embed", "all-minilm"]
        try:
            resp = requests.get(
                self.ollama_url.rstrip("/") + "/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"].split(":")[0]
                         for m in resp.json().get("models", [])]
            for p in preferred:
                if any(p in a for a in available):
                    found = next(a for a in available if p in a)
                    self._embed_model_detected = found
                    return found
            embed_models = [a for a in available if "embed" in a.lower()]
            if embed_models:
                self._embed_model_detected = embed_models[0]
                return embed_models[0]
        except Exception:
            pass
        self._embed_model_detected = MEM_EMBED_MODEL
        return MEM_EMBED_MODEL

    def _call_embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        base = self.ollama_url.rstrip("/")
        model = self._detect_embed_model()
        try:
            resp = requests.post(
                base + "/api/embed",
                json={"model": model, "input": texts}, timeout=60)
            if resp.status_code != 404:
                resp.raise_for_status()
                data = resp.json()
                if "embeddings" in data:
                    return data["embeddings"]
        except Exception:
            pass
        results = []
        for text in texts:
            try:
                resp = requests.post(
                    base + "/api/embeddings",
                    json={"model": model, "prompt": text}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "embedding" in data:
                    results.append(data["embedding"])
                else:
                    return None
            except Exception:
                return None
        return results if results else None

    def _embed_batch(self, labels: List[str], log_fn) -> Optional[np.ndarray]:
        all_vecs = []
        for i in range(0, len(labels), MEM_EMBED_BATCH):
            chunk = labels[i:i + MEM_EMBED_BATCH]
            vecs = self._call_embed(chunk)
            if vecs is None:
                model_used = self._detect_embed_model()
                log_fn(f"  [MemoryAgent] embed failed for model '{model_used}'. "
                       f"Fix: ollama pull {model_used}")
                return None
            all_vecs.append(np.array(vecs, dtype=np.float32))
        mat = np.ascontiguousarray(np.vstack(all_vecs), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return np.ascontiguousarray(mat / norms, dtype=np.float32)

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        vecs = self._call_embed([text])
        if vecs is None:
            return None
        vec = np.ascontiguousarray(np.array(vecs[0], dtype=np.float32))
        n = np.linalg.norm(vec)
        result = vec / n if n > 0 else None
        if result is not None and len(self._embed_cache) < 10_000:
            self._embed_cache[text] = result
        return result

    def _load_cache(self, col: str, log_fn=print):
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT label, embedding FROM semantic_labels "
                    "WHERE col=? AND embedding IS NOT NULL ORDER BY label",
                    (col,)).fetchall()
            if not rows:
                return
            labels = [r[0] for r in rows]
            mat = np.ascontiguousarray(np.stack([
                np.frombuffer(r[1], dtype=np.float32).copy()
                for r in rows]), dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            mat = np.ascontiguousarray(mat / norms, dtype=np.float32)
            with self._lock:
                self._vec_cache[col] = (labels, mat)
                self._cache_ok[col] = True
            log_fn(f"  [MemoryAgent] {col}: loaded {len(labels):,} vectors into RAM")
        except Exception as e:
            log_fn(f"  [MemoryAgent] cache load error ({col}): {e}")

    def load_cache_all(self, log_fn=print):
        for col in LABEL_COLS:
            self._load_cache(col, log_fn)

    # ── Tier 2: Semantic search ──

    def _safe_dot(self, mat: np.ndarray, vec: np.ndarray) -> Optional[np.ndarray]:
        try:
            if mat is None or vec is None:
                return None
            if mat.ndim != 2 or vec.ndim != 1:
                return None
            if mat.shape[1] != vec.shape[0]:
                return None
            mat_c = np.ascontiguousarray(mat, dtype=np.float32)
            vec_c = np.ascontiguousarray(vec, dtype=np.float32)
            return mat_c @ vec_c
        except Exception:
            return None

    def semantic_search(self, col: str, text: str,
                        k: int = MEM_TOP_K) -> List[tuple]:
        try:
            if not self._cache_ok.get(col):
                return []
            with self._lock:
                labels, mat = self._vec_cache[col]
            if mat is None or len(labels) == 0:
                return []
            results = {}

            def _search_one(query_text: str):
                vec = self._embed_one(query_text)
                if vec is None:
                    return
                sims = self._safe_dot(mat, vec)
                if sims is None:
                    return
                for i in np.argsort(sims)[::-1][:k]:
                    if sims[i] >= MEM_MATCH_THRESH:
                        lbl = labels[i]
                        results[lbl] = max(results.get(lbl, 0.0), float(sims[i]))

            _search_one(text)
            stripped = self._strip_cell_prefix(text)
            if stripped != text:
                _search_one(stripped)

            return sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]
        except Exception:
            return []

    # ── Tier 3: Episodic ──

    def episodic_search(self, col: str, raw_label: str) -> List[dict]:
        try:
            with self._conn() as c:
                rows = c.execute("""
                    SELECT canonical, AVG(confidence) AS avg_conf,
                           COUNT(*) AS cnt, MAX(ts) AS last_ts
                    FROM episodic_log
                    WHERE col=? AND raw_label=?
                    GROUP BY canonical
                    ORDER BY cnt DESC, avg_conf DESC
                    LIMIT 5
                """, (col, raw_label)).fetchall()
            return [{"canonical": r[0], "confidence": r[1],
                     "count": r[2], "last_ts": r[3]} for r in rows]
        except Exception:
            return []

    def log_resolution(self, col: str, raw_label: str, canonical: str,
                       confidence: float = 1.0, platform: str = "",
                       gse: str = "", gsm: str = "",
                       collapse_rule: str = "") -> None:
        try:
            with self._lock, self._conn() as c:
                c.execute("""
                    INSERT INTO episodic_log
                      (col, raw_label, canonical, confidence,
                       platform, gse, gsm, collapse_rule)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (col, raw_label, canonical, confidence,
                      platform, gse, gsm, collapse_rule))
                if raw_label != canonical:
                    c.execute("""
                        INSERT OR REPLACE INTO kg_triples
                          (col, subject, relation, object, weight)
                        VALUES (?,?,?,?,?)
                    """, (col, raw_label, "variant_of", canonical, confidence))
        except Exception:
            pass

    def store_raw_label(self, col: str, raw: str, gsm: str = "",
                        gse: str = "", platform: str = "") -> None:
        """Store a raw extracted label for reference and analysis."""
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT INTO raw_labels (col, raw, gsm, gse, platform) "
                    "VALUES (?,?,?,?,?)",
                    (col, raw, gsm, gse, platform))
        except Exception:
            pass

    # ── Tier 4: Knowledge graph ──

    def kg_lookup(self, col: str, label: str) -> List[tuple]:
        try:
            with self._conn() as c:
                return c.execute("""
                    SELECT object, relation, weight
                    FROM kg_triples
                    WHERE col=? AND subject=?
                    ORDER BY weight DESC LIMIT 5
                """, (col, label)).fetchall()
        except Exception:
            return []

    # ── Tier 1: Core labels ──

    def core_labels(self, col: str, n: int = MEM_CORE_N) -> List[str]:
        try:
            with self._conn() as c:
                return [r[0] for r in c.execute(
                    "SELECT label FROM core_labels WHERE col=? "
                    "ORDER BY freq DESC LIMIT ?", (col, n))]
        except Exception:
            return []

    # ── Unified search ──

    def search(self, col: str, text: str) -> dict:
        return {
            "cluster": self.cluster_lookup(col, text),
            "episodic": self.episodic_search(col, text),
            "semantic": self.semantic_search(col, text),
            "kg": self.kg_lookup(col, text),
        }

    def is_cluster_name(self, col: str, label: str) -> bool:
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT 1 FROM semantic_labels WHERE col=? AND label=?",
                    (col, label)).fetchone()
                return row is not None
        except Exception:
            return False

    def is_ready(self, col: str) -> bool:
        return self._cache_ok.get(col, False)

    # ── Register new cluster (SQL-only, no .txt writes) ──

    def register_new_cluster(self, col: str, cluster_name: str,
                             raw_label: str, log_fn=print) -> None:
        """
        Create a new cluster directly in the SQL database.
        No .txt file writes - SQL DB is the single source of truth.
        """
        cluster_name = cluster_name.strip()
        raw_lower = raw_label.lower().strip()
        if not cluster_name or not raw_lower:
            return

        # Write to cluster_map
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cluster_map (col, raw, cluster) VALUES (?,?,?)",
                (col, raw_lower, cluster_name))
            c.execute(
                "INSERT OR REPLACE INTO cluster_map (col, raw, cluster) VALUES (?,?,?)",
                (col, cluster_name.lower(), cluster_name))
            c.execute(
                "INSERT OR REPLACE INTO cluster_map (col, raw, cluster) VALUES (?,?,?)",
                (col, cluster_name, cluster_name))
            c.execute(
                "INSERT OR REPLACE INTO kg_triples "
                "(col, subject, relation, object, weight) VALUES (?,?,?,?,?)",
                (col, raw_label, "assigned_to", cluster_name, 1.0))

        # Track for run report
        from datetime import datetime
        self._new_cluster_log.append({
            "col": col,
            "cluster_name": cluster_name,
            "raw_label": raw_label,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Embed and add to semantic_labels
        vecs = self._embed_batch([cluster_name], log_fn)
        if vecs is not None:
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT OR REPLACE INTO semantic_labels "
                    "(col, label, embedding, freq) VALUES (?,?,?,?)",
                    (col, cluster_name, vecs[0].astype("float32").tobytes(), 1))
            self._load_cache(col, log_fn)
            log_fn(f"  [NEW CLUSTER] {col}: '{cluster_name}' registered + embedded (SQL)")
        else:
            log_fn(f"  [NEW CLUSTER] {col}: '{cluster_name}' registered (cluster_map only)")

    def get_new_cluster_log(self) -> list:
        return list(self._new_cluster_log)

    # ── Memory system prompt ──

    def memory_system_prompt(self, col: str) -> str:
        stats = self.stats()
        n_ep = stats.get("episodic", {}).get(col, 0)
        n_sem = stats.get("semantic", {}).get(col, 0)
        n_kg = stats.get("kg_triples", 0)

        return (
            "=== MEMORY AWARE AGENT - SYSTEM INSTRUCTIONS ===\n"
            f"You are a biomedical metadata normalization agent for GEO field: {col}.\n"
            "Your ONLY job: map the extracted label to one approved CLUSTER NAME.\n\n"
            "CRITICAL RULE: Cluster names are human-approved canonical labels.\n"
            "  Output ONLY a cluster name returned by your tools.\n"
            "  NEVER output a raw label, abbreviation, or free-form text.\n"
            "  NEVER invent or modify a cluster name.\n\n"
            "MEMORY STORE INVENTORY:\n"
            "  Cluster map  (Tier 0) : direct raw->cluster lookups\n"
            f"  Tier 2 - Semantic     : {n_sem:,} cluster names, vector-indexed\n"
            f"  Tier 3 - Episodic     : {n_ep:,} past resolutions logged\n"
            f"  Tier 4 - KG triples   : {n_kg:,} assignment records\n\n"
            "=== END SYSTEM INSTRUCTIONS ==="
        )

    def should_log(self, col: str, raw: str, canonical: str,
                   collapse_rule: str) -> tuple:
        if raw == canonical:
            return False, 0.0, "identity"
        if collapse_rule in ("", "vocab_exact"):
            return False, 0.0, "no_change"
        if collapse_rule == "episodic":
            return True, 0.98, "episodic_confirmed"
        if collapse_rule.startswith("kg_"):
            return True, 0.95, "kg_verified"
        if collapse_rule == "semantic_vocab":
            return True, 0.88, "semantic_llm"
        if collapse_rule in ("exact_match", "abbreviation"):
            return True, 0.92, "deterministic"
        return True, 0.80, "other"

    # ── Stats ──

    def stats(self) -> dict:
        try:
            with self._conn() as c:
                sem = {r[0]: r[1] for r in c.execute(
                    "SELECT col, COUNT(*) FROM semantic_labels GROUP BY col")}
                epi = {r[0]: r[1] for r in c.execute(
                    "SELECT col, COUNT(*) FROM episodic_log GROUP BY col")}
                kg = c.execute("SELECT COUNT(*) FROM kg_triples").fetchone()[0]
                cm = {r[0]: r[1] for r in c.execute(
                    "SELECT col, COUNT(DISTINCT cluster) FROM cluster_map GROUP BY col")}
            return {"semantic": sem, "episodic": epi,
                    "kg_triples": kg, "clusters": cm}
        except Exception:
            return {}

    # ── Export clusters from DB to dict ──

    def export_clusters(self, col: str) -> Dict[str, List[str]]:
        """Export cluster vocabulary from DB as {cluster_name: [raw_labels]}."""
        clusters: Dict[str, List[str]] = {}
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT raw, cluster FROM cluster_map WHERE col=? "
                    "ORDER BY cluster", (col,)).fetchall()
            for raw, cluster in rows:
                if cluster not in clusters:
                    clusters[cluster] = []
                if raw != cluster and raw != cluster.lower():
                    clusters[cluster].append(raw)
        except Exception:
            pass
        return clusters
