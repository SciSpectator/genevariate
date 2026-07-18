"""Shared MeSH + out-of-distribution-mesh lookup library for the Phase 2
redesign.

Owns:
  - SQLite handle to mesh.sqlite (mesh_terms / mesh_synonyms / mesh_tree /
    mesh_parent / ood_mesh_clusters / ood_mesh_synonyms / resolutions)
  - BioLORD-2023 model (lazy, class-cached)
  - Pre-built embedding index over all MeSH descriptors (one vector per
    descriptor; built once via ``python mesh_lookup.py --build-index``,
    cached as ``mesh_embeddings.npz`` next to mesh.sqlite).

Used by both:
  - mcp_mesh_server.py (thin stdio MCP wrapper)
  - phase2_mesh.py     (in-process Phase 2 driver)

Column → MeSH category gating (used to filter both exact and semantic
hits to the right MeSH branch):
  Tissue    → A           (Anatomy)
  Condition → C, F        (Diseases + Psychiatry)
  Treatment → D, E        (Drugs + Therapeutics)
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Paths + constants
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH    = os.environ.get(
    "MESH_DB",
    str(Path(__file__).resolve().parent / "mesh.sqlite"),
)
INDEX_PATH = os.environ.get(
    "MESH_INDEX",
    str(Path(DB_PATH).with_suffix(".embeddings.npz")),
)
BIOLORD_MODEL = os.environ.get("BIOLORD_MODEL", "FremyCompany/BioLORD-2023")

# col → allowed MeSH categories
COL_CATS: dict[str, tuple[str, ...]] = {
    "Tissue":    ("A",),
    "Condition": ("C", "F"),
    "Treatment": ("D", "E"),
}
# col → OOD-mesh id prefix (used to allocate ART-{T,C,X}-##### IDs)
COL_PREFIX = {"Tissue": "T", "Condition": "C", "Treatment": "X"}


_RESOLUTIONS_DDL = """
CREATE TABLE IF NOT EXISTS resolutions (
    input_label  TEXT NOT NULL COLLATE NOCASE,
    col          TEXT NOT NULL,
    output_id    TEXT NOT NULL,
    output_name  TEXT NOT NULL,
    source       TEXT NOT NULL,   -- 'mesh' | 'ood-mesh-existing' | 'ood-mesh-minted'
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_res_input ON resolutions(input_label);
CREATE INDEX IF NOT EXISTS idx_res_col   ON resolutions(col);
-- Cluster-ready: dedupe episodic recall by (input_label, col); upsert
-- semantics on record_resolution. Concurrent collapsers writing the
-- same (raw, col) collapse to one row instead of bloating the log.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_res_label_col
    ON resolutions(input_label COLLATE NOCASE, col);

-- Cluster-ready: enforce one OOD-mesh cluster per (col, label) so two
-- concurrent minters racing on the same novel raw cannot both succeed.
-- The first INSERT wins; the loser hits IntegrityError and re-lookups.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_ood_mesh_col_label
    ON ood_mesh_clusters(col, label COLLATE NOCASE);

-- Tier 4.5 verifier verdict cache: skip re-verifying the same
-- (raw, col, picked_id) under the same prompt version. Keyed for
-- distributed read fan-out so every CollapserAgent in the fleet
-- reuses any verifier verdict already produced.
CREATE TABLE IF NOT EXISTS verifier_decisions (
    raw_lc          TEXT NOT NULL,
    col             TEXT NOT NULL,
    picked_id       TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    verdict         TEXT NOT NULL,    -- 'KEEP' | 'REJECT'
    created_at      TEXT NOT NULL,
    PRIMARY KEY (raw_lc, col, picked_id, prompt_version)
);

-- Tier 0.5 polarity verdict cache: a grammatical ASSERT/NEGATE
-- judgement over a raw label. NEGATE raws short-circuit to NS
-- (no separate "absence" MeSH lattice — a negated label is, by
-- universal rule, a missing observation = Not Specified). The
-- table is keyed by (raw_lc, col, prompt_version) so cluster-wide
-- reuse is safe and bumping the prompt version invalidates stale
-- verdicts without wiping the row history.
CREATE TABLE IF NOT EXISTS polarity_decisions (
    raw_lc          TEXT NOT NULL,
    col             TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    polarity        TEXT NOT NULL,    -- 'ASSERT' | 'NEGATE'
    created_at      TEXT NOT NULL,
    PRIMARY KEY (raw_lc, col, prompt_version)
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Library
# ─────────────────────────────────────────────────────────────────────────────
class MeshDB:
    """All MeSH + out-of-distribution (OOD) mesh lookups and minting
    against ``mesh.sqlite``."""

    _MODEL_CACHE: dict = {}     # BIOLORD_MODEL -> SentenceTransformer
    _INDEX_CACHE: dict = {}     # INDEX_PATH    -> (vectors, meta)

    # ── lifecycle ────────────────────────────────────────────────────────
    def __init__(self, db_path: str = DB_PATH):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"MeSH DB not found: {db_path}")
        self.db_path = db_path
        self._con: sqlite3.Connection | None = None

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None,
                timeout=30.0,
            )
            self._con.row_factory = sqlite3.Row
            # WAL + NORMAL sync: many concurrent readers, one writer at a
            # time — required for in-process A2A fan-out and for safe
            # multi-process access on shared NFS / cluster storage.
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute("PRAGMA busy_timeout=30000")
            self._con.execute("PRAGMA foreign_keys=ON")
            self._con.executescript(_RESOLUTIONS_DDL)
            self._validate_ood_mesh_integrity(self._con)
        return self._con

    # ── shared pipeline health helper ──────────────────────────────────
    @staticmethod
    def verify_pipeline_health(
        *, ollama_url: str, model_name: str,
        gse_cache_db: str = "gse_context_cache.sqlite",
        require_biolord: bool = True, strict: bool = True,
    ) -> dict:
        """Inline guardrail invoked by every extractor (run_cli /
        run_phase2_standalone / llm_label_extractor GUI) BEFORE the
        sample loop begins. Verifies Ollama backend, both SQLite caches,
        and the BioLORD index. ``strict=True`` raises on first failure;
        ``strict=False`` returns the issue dict for the caller to log.
        """
        import json as _j, urllib.request as _u, urllib.error as _ue
        issues: list[str] = []

        # 1. Ollama backend
        try:
            r = _u.urlopen(f"{ollama_url}/api/tags", timeout=5)
            names = {m.get("name","") for m in _j.loads(r.read()).get("models",[])}
            if model_name not in names:
                issues.append(f"ollama at {ollama_url}: model {model_name!r} not loaded "
                              f"(found {len(names)} models). Fix: ollama pull {model_name}")
        except (_ue.URLError, OSError, _j.JSONDecodeError) as e:
            issues.append(f"ollama unreachable at {ollama_url}: {e!r}. "
                          f"Fix: start ollama serve")

        # 2. mesh.sqlite (OOD-mesh integrity — the 2026-04-28 corruption check)
        if not Path(DB_PATH).exists():
            issues.append(f"mesh.sqlite missing at {DB_PATH}. Fix: build_mesh_db.py")
        else:
            con = sqlite3.connect(DB_PATH)
            try:
                tabs = {r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                need = {"ood_mesh_clusters","ood_mesh_synonyms","resolutions","mesh_terms"}
                miss = need - tabs
                if miss:
                    issues.append(f"mesh.sqlite missing tables {sorted(miss)}")
                else:
                    orphans = con.execute(
                        "SELECT COUNT(*) FROM ood_mesh_synonyms "
                        "WHERE ood_mesh_id NOT IN (SELECT id FROM ood_mesh_clusters)"
                    ).fetchone()[0]
                    bad_label = con.execute(
                        "SELECT COUNT(*) FROM ood_mesh_synonyms s "
                        "JOIN ood_mesh_clusters a ON a.id=s.ood_mesh_id "
                        "WHERE LOWER(s.synonym)<>LOWER(a.label)"
                    ).fetchone()[0]
                    bad_cond = con.execute(
                        "SELECT COUNT(*) FROM ood_mesh_clusters "
                        "WHERE col='Condition' AND LOWER(label) IN "
                        "('wt','wild-type','wildtype','wild type')"
                    ).fetchone()[0]
                    if orphans or bad_label or bad_cond:
                        issues.append(
                            f"mesh.sqlite OOD-mesh corruption: "
                            f"orphans={orphans} mismatches={bad_label} "
                            f"genotype-Condition-OOD-entries={bad_cond}. "
                            f"Fix: MeshDB.repair_ood_mesh()")
            finally:
                con.close()

        # 3. gse_context_cache.sqlite
        if not Path(gse_cache_db).exists():
            issues.append(f"gse_context_cache.sqlite missing at {gse_cache_db}")

        # 4. MeSH embedding index (only if BioLORD path will be used)
        if require_biolord and not Path(INDEX_PATH).exists():
            issues.append(f"MeSH embedding index missing at {INDEX_PATH}. "
                          f"Fix: python mesh_lookup.py --build-index")

        if issues and strict:
            raise RuntimeError("[health] " + " | ".join(issues))
        return {"ok": not issues, "issues": issues}

    @staticmethod
    def repair_ood_mesh(mesh_db: str | None = None,
                        gse_cache_db: str = "gse_context_cache.sqlite") -> dict:
        """Idempotent in-place repair for the orphan-synonyms /
        cross-pollinated / genotype-Condition corruption pattern. Backs
        up both DBs before touching them."""
        import shutil, time
        mp = mesh_db or DB_PATH
        ts = int(time.time())
        if Path(mp).exists():
            shutil.copy2(mp, f"{mp}.bak.{ts}")
        if Path(gse_cache_db).exists():
            shutil.copy2(gse_cache_db, f"{gse_cache_db}.bak.{ts}")
        M = sqlite3.connect(mp); M.execute("BEGIN")
        try:
            M.execute("DELETE FROM ood_mesh_synonyms WHERE ood_mesh_id IN ("
                      "SELECT id FROM ood_mesh_clusters WHERE col='Condition' "
                      "AND LOWER(label) IN ('wt','wild-type','wildtype','wild type'))")
            M.execute("DELETE FROM ood_mesh_clusters WHERE col='Condition' "
                      "AND LOWER(label) IN ('wt','wild-type','wildtype','wild type')")
            M.execute("DELETE FROM ood_mesh_synonyms "
                      "WHERE ood_mesh_id NOT IN (SELECT id FROM ood_mesh_clusters)")
            M.execute("DELETE FROM ood_mesh_synonyms WHERE rowid IN ("
                      "SELECT s.rowid FROM ood_mesh_synonyms s "
                      "JOIN ood_mesh_clusters a ON a.id=s.ood_mesh_id "
                      "WHERE LOWER(s.synonym)<>LOWER(a.label))")
            M.execute("DELETE FROM resolutions")
            M.commit()
        except Exception:
            M.rollback(); raise
        M.execute("VACUUM"); M.close()
        if Path(gse_cache_db).exists():
            C = sqlite3.connect(gse_cache_db); C.execute("BEGIN")
            try:
                C.execute("DELETE FROM gse_phase2_canon WHERE col='Condition'")
                C.commit()
            except Exception:
                C.rollback(); raise
            C.execute("VACUUM"); C.close()
        return {"backup_suffix": ts}

    @staticmethod
    def _validate_ood_mesh_integrity(con: sqlite3.Connection) -> None:
        """Warn-only integrity audit on the OOD-mesh tables. Catches the
        2026-04-28 corruption pattern (orphan synonyms / synonym surface
        form drifted from parent label) before phase2 reads stale rows.
        Promote to ``raise`` once one full clean run confirms zero warnings.
        """
        try:
            orphans = con.execute(
                "SELECT COUNT(*) FROM ood_mesh_synonyms "
                "WHERE ood_mesh_id NOT IN (SELECT id FROM ood_mesh_clusters)"
            ).fetchone()[0]
            mismatches = con.execute(
                "SELECT COUNT(*) FROM ood_mesh_synonyms s "
                "JOIN ood_mesh_clusters a ON a.id = s.ood_mesh_id "
                "WHERE LOWER(s.synonym) <> LOWER(a.label)"
            ).fetchone()[0]
            if orphans or mismatches:
                import sys
                print(
                    f"[mesh_lookup] WARN cache-integrity: "
                    f"{orphans} orphan synonyms, {mismatches} label-mismatched synonyms. "
                    f"Run --repair-ood-mesh to clean.",
                    file=sys.stderr, flush=True,
                )
        except sqlite3.OperationalError:
            # tables not yet built (fresh DB) — silently OK
            pass

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # ── 1. exact MeSH lookup (name + synonym) ────────────────────────────
    def lookup_mesh(self, label: str, col: str | None = None) -> list[dict]:
        """Return all MeSH hits whose name OR synonym == label (NOCASE)."""
        label = (label or "").strip()
        if not label:
            return []
        cats = COL_CATS.get(col or "", ())
        cat_clause = ""
        params: list = [label]
        if cats:
            placeholders = ",".join("?" * len(cats))
            cat_clause = f" AND t.category IN ({placeholders})"
            params.extend(cats)

        # Names
        name_rows = self.con.execute(
            f"SELECT t.id, t.name, t.category, t.scope, 'name' AS via "
            f"FROM mesh_terms t WHERE t.name = ? COLLATE NOCASE{cat_clause}",
            params,
        ).fetchall()

        # Synonyms (re-bind label as first param)
        syn_rows = self.con.execute(
            f"SELECT t.id, t.name, t.category, t.scope, 'synonym' AS via "
            f"FROM mesh_synonyms s "
            f"JOIN mesh_terms t ON t.id = s.mesh_id "
            f"WHERE s.synonym = ? COLLATE NOCASE{cat_clause}",
            params,
        ).fetchall()

        # Dedupe by id, prefer 'name' over 'synonym'
        seen: dict[str, dict] = {}
        for r in list(name_rows) + list(syn_rows):
            d = dict(r)
            if d["id"] in seen:
                continue
            seen[d["id"]] = d
        return list(seen.values())

    # ── 2. semantic MeSH search (BioLORD) ───────────────────────────────
    def find_similar_mesh(
        self, label: str, col: str | None = None, k: int = 20
    ) -> list[dict]:
        """BioLORD top-K MeSH descriptors. Returns dicts with score in [0,1]."""
        label = (label or "").strip()
        if not label:
            return []
        vectors, meta = self._load_index()
        model = self._load_model()
        q = model.encode([label], normalize_embeddings=True)[0].astype(np.float32)
        sims = vectors @ q                          # cosine (vectors are L2-norm)

        # Apply col category gate.
        cats = set(COL_CATS.get(col or "", ()))
        if cats:
            mask = np.array([m["category"] in cats for m in meta], dtype=bool)
            if mask.any():
                sims = np.where(mask, sims, -1.0)

        # Top-K via argpartition
        k_eff = min(k, len(sims))
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        out: list[dict] = []
        for i in idx:
            if sims[i] < 0:                          # filtered out
                continue
            m = meta[i]
            out.append({
                "id":       m["id"],
                "name":     m["name"],
                "category": m["category"],
                "scope":    m["scope"],
                "score":    float(sims[i]),
            })
        return out

    # ── 3. tree walk ────────────────────────────────────────────────────
    def get_mesh_tree(
        self, mesh_id: str, direction: str = "ancestors", depth: int = 2
    ) -> list[dict]:
        """Walk parent or child edges from a given descriptor."""
        if direction not in ("ancestors", "descendants"):
            return []
        edge_col, key_col = ("child_id", "parent_id") if direction == "ancestors" \
                            else ("parent_id", "child_id")
        seen = {mesh_id}
        frontier = {mesh_id}
        layers: list[list[dict]] = []
        for _ in range(max(1, depth)):
            if not frontier:
                break
            qmarks = ",".join("?" * len(frontier))
            rows = self.con.execute(
                f"SELECT DISTINCT {key_col} AS id FROM mesh_parent "
                f"WHERE {edge_col} IN ({qmarks})",
                tuple(frontier),
            ).fetchall()
            new = {r["id"] for r in rows} - seen
            if not new:
                break
            seen.update(new)
            qmarks = ",".join("?" * len(new))
            term_rows = self.con.execute(
                f"SELECT id, name, category FROM mesh_terms WHERE id IN ({qmarks})",
                tuple(new),
            ).fetchall()
            layers.append([dict(r) for r in term_rows])
            frontier = new
        return [t for layer in layers for t in layer]

    # ── 4. OOD-mesh lookup + mint ───────────────────────────────────────
    def lookup_ood_mesh(self, label: str, col: str) -> dict | None:
        label = (label or "").strip()
        if not label or col not in COL_PREFIX:
            return None
        # Exact label OR exact synonym, scoped to col.
        row = self.con.execute(
            "SELECT id, label, col, source FROM ood_mesh_clusters "
            "WHERE col = ? AND label = ? COLLATE NOCASE LIMIT 1",
            (col, label),
        ).fetchone()
        if row:
            return dict(row)
        row = self.con.execute(
            "SELECT a.id, a.label, a.col, a.source FROM ood_mesh_synonyms s "
            "JOIN ood_mesh_clusters a ON a.id = s.ood_mesh_id "
            "WHERE a.col = ? AND s.synonym = ? COLLATE NOCASE LIMIT 1",
            (col, label),
        ).fetchone()
        return dict(row) if row else None

    def create_ood_mesh(self, label: str, col: str) -> dict:
        """Mint a new ART-{T,C,X}-##### cluster into the out-of-distribution
        (OOD) mesh (idempotent on existing label).

        Cluster-safe: the existence-check, mint, and occurrence-bump are
        all done inside one ``BEGIN IMMEDIATE`` transaction so concurrent
        minters serialize on the SQLite write lock. The
        ``uniq_ood_mesh_col_label`` index is the cross-process safety net —
        if a peer process raced past us, the INSERT raises IntegrityError
        and we recover by incrementing occurrences on the existing row.
        """
        label = (label or "").strip()
        if not label or col not in COL_PREFIX:
            raise ValueError(f"create_ood_mesh: invalid label/col {label!r}/{col!r}")

        prefix = COL_PREFIX[col]
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

        self.con.execute("BEGIN IMMEDIATE")
        try:
            # Existence check INSIDE the transaction so once we hold the
            # write lock no peer can slip a row in before our decision.
            existing = self.lookup_ood_mesh(label, col)
            if existing:
                self.con.execute(
                    "UPDATE ood_mesh_clusters SET occurrences = occurrences + 1 "
                    "WHERE id = ?", (existing["id"],),
                )
                self.con.execute("COMMIT")
                return existing

            row = self.con.execute(
                "SELECT id FROM ood_mesh_clusters WHERE col = ? "
                "ORDER BY id DESC LIMIT 1", (col,),
            ).fetchone()
            next_n = 1 if not row else int(row["id"].rsplit("-", 1)[1]) + 1
            new_id = f"ART-{prefix}-{next_n:05d}"
            self.con.execute(
                "INSERT INTO ood_mesh_clusters "
                "(id, label, col, source, created_at, occurrences) "
                "VALUES (?, ?, ?, 'minted', ?, 1)",
                (new_id, label, col, now),
            )
            self.con.execute(
                "INSERT OR IGNORE INTO ood_mesh_synonyms (ood_mesh_id, synonym) VALUES (?, ?)",
                (new_id, label),
            )
            self.con.execute("COMMIT")
            return {"id": new_id, "label": label, "col": col, "source": "minted"}
        except sqlite3.IntegrityError:
            # Cross-process race: a peer minted the same (col, label) just
            # before our INSERT. Recover by bumping occurrences on the
            # winning row so the call still counts as one observation.
            self.con.execute("ROLLBACK")
            existing = self.lookup_ood_mesh(label, col)
            if not existing:
                raise
            self.con.execute(
                "UPDATE ood_mesh_clusters SET occurrences = occurrences + 1 "
                "WHERE id = ?", (existing["id"],),
            )
            return existing

    # ── 4b. verifier verdict cache (Tier 4.5) ───────────────────────────
    def get_verifier_verdict(
        self, raw_lc: str, col: str, picked_id: str, prompt_version: str,
    ) -> str | None:
        """Return cached verdict ('KEEP'|'REJECT') or None if uncached."""
        row = self.con.execute(
            "SELECT verdict FROM verifier_decisions "
            "WHERE raw_lc = ? AND col = ? AND picked_id = ? "
            "AND prompt_version = ? LIMIT 1",
            (raw_lc, col, picked_id, prompt_version),
        ).fetchone()
        return row["verdict"] if row else None

    def cache_verifier_verdict(
        self, raw_lc: str, col: str, picked_id: str,
        prompt_version: str, verdict: str,
    ) -> None:
        """Persist verifier verdict for distributed reuse (idempotent)."""
        if verdict not in ("KEEP", "REJECT"):
            raise ValueError(f"verdict must be KEEP|REJECT, got {verdict!r}")
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.con.execute(
            "INSERT OR REPLACE INTO verifier_decisions "
            "(raw_lc, col, picked_id, prompt_version, verdict, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (raw_lc, col, picked_id, prompt_version, verdict, now),
        )

    # ── 4c. polarity verdict cache (Tier 0.5) ───────────────────────────
    def get_polarity(
        self, raw_lc: str, col: str, prompt_version: str,
    ) -> str | None:
        """Return cached polarity ('ASSERT'|'NEGATE') or None if uncached."""
        row = self.con.execute(
            "SELECT polarity FROM polarity_decisions "
            "WHERE raw_lc = ? AND col = ? AND prompt_version = ? LIMIT 1",
            (raw_lc, col, prompt_version),
        ).fetchone()
        return row["polarity"] if row else None

    def cache_polarity(
        self, raw_lc: str, col: str, prompt_version: str, polarity: str,
    ) -> None:
        """Persist polarity verdict for distributed reuse (idempotent)."""
        if polarity not in ("ASSERT", "NEGATE"):
            raise ValueError(f"polarity must be ASSERT|NEGATE, got {polarity!r}")
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.con.execute(
            "INSERT OR REPLACE INTO polarity_decisions "
            "(raw_lc, col, prompt_version, polarity, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (raw_lc, col, prompt_version, polarity, now),
        )

    # ── 5. resolution log ───────────────────────────────────────────────
    def record_resolution(
        self, input_label: str, col: str, output_id: str,
        output_name: str, source: str,
    ) -> None:
        """Upsert episodic recall for ``(input_label, col)``. The UNIQUE
        index ``uniq_res_label_col`` collapses duplicates so the table
        stays bounded as the corpus grows into millions of samples.
        Latest decision wins (REPLACE)."""
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.con.execute(
            "INSERT OR REPLACE INTO resolutions "
            "(input_label, col, output_id, output_name, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (input_label, col, output_id, output_name, source, now),
        )

    def get_resolution_history(
        self, input_label: str, col: str | None = None, k: int = 5,
    ) -> list[dict]:
        params: list = [input_label]
        clause = ""
        if col:
            clause = " AND col = ?"
            params.append(col)
        params.append(k)
        rows = self.con.execute(
            f"SELECT * FROM resolutions WHERE input_label = ? COLLATE NOCASE{clause} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ── BioLORD plumbing ────────────────────────────────────────────────
    @classmethod
    def _load_model(cls):
        if BIOLORD_MODEL not in cls._MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            cls._MODEL_CACHE[BIOLORD_MODEL] = SentenceTransformer(BIOLORD_MODEL)
        return cls._MODEL_CACHE[BIOLORD_MODEL]

    def _load_index(self):
        if INDEX_PATH in self._INDEX_CACHE:
            return self._INDEX_CACHE[INDEX_PATH]
        if not Path(INDEX_PATH).exists():
            raise FileNotFoundError(
                f"MeSH embedding index not found at {INDEX_PATH}. "
                f"Build it with: python mesh_lookup.py --build-index"
            )
        npz = np.load(INDEX_PATH, allow_pickle=True)
        vectors = npz["vectors"].astype(np.float32)
        ids       = npz["ids"]
        names     = npz["names"]
        cats      = npz["categories"]
        scopes    = npz["scopes"]
        meta = [
            {"id": str(ids[i]), "name": str(names[i]),
             "category": str(cats[i]), "scope": str(scopes[i])}
            for i in range(len(ids))
        ]
        self._INDEX_CACHE[INDEX_PATH] = (vectors, meta)
        return vectors, meta


# ─────────────────────────────────────────────────────────────────────────────
# Index builder (CLI)
# ─────────────────────────────────────────────────────────────────────────────
def _build_index(db_path: str = DB_PATH, index_path: str = INDEX_PATH,
                 batch: int = 128, max_synonyms: int = 5) -> None:
    """One-shot: encode every MeSH descriptor into a single vector and dump
    to ``mesh_embeddings.npz``. Embedded text combines the canonical name,
    its scope note, and up to ``max_synonyms`` concept synonyms — gives
    BioLORD enough lexical variation to bridge abbreviations like
    'PBC' → 'Liver Cirrhosis, Biliary'.
    """
    import time
    from sentence_transformers import SentenceTransformer

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT id, name, scope, category FROM mesh_terms ORDER BY id"
    ).fetchall()
    print(f"  descriptors: {len(rows)}")

    # Pull synonyms in one pass.
    syn_map: dict[str, list[str]] = {}
    for r in con.execute("SELECT mesh_id, synonym FROM mesh_synonyms"):
        syn_map.setdefault(r["mesh_id"], []).append(r["synonym"])

    texts: list[str] = []
    ids:   list[str] = []
    names: list[str] = []
    cats:  list[str] = []
    scopes: list[str] = []
    for r in rows:
        ids.append(r["id"])
        names.append(r["name"])
        cats.append(r["category"] or "")
        scope = (r["scope"] or "").strip()
        scopes.append(scope)
        syns = syn_map.get(r["id"], [])[:max_synonyms]
        parts = [r["name"]]
        if scope:
            parts.append(scope)
        if syns:
            parts.append("; ".join(syns))
        texts.append(" | ".join(parts))

    model = SentenceTransformer(BIOLORD_MODEL)
    print(f"  encoding {len(texts)} strings, batch={batch} ...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  encoded in {time.time()-t0:.1f}s; shape={vectors.shape}")

    np.savez(
        index_path,
        vectors=vectors,
        ids=np.array(ids,    dtype=object),
        names=np.array(names, dtype=object),
        categories=np.array(cats, dtype=object),
        scopes=np.array(scopes, dtype=object),
    )
    sz = Path(index_path).stat().st_size / 1024 / 1024
    print(f"  wrote {index_path} ({sz:.1f} MB)")


def _split_csv(value: Iterable[str] | None, default: tuple) -> tuple:
    """Helper for env-var category lists (unused stub for forward-compat)."""
    return default


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=INDEX_PATH)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    if args.build_index:
        _build_index(args.db, args.out, batch=args.batch)
    else:
        ap.print_help()
