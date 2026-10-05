"""What a keyword matches in the GEO metadata, decided in one place."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set

import pandas as pd

_LEFT = r"(?<![0-9a-z])"
_RIGHT = r"(?![0-9a-z])"

GSE_EXCLUDED = {"gse", "status", "submission_date", "last_update_date",
                "pubmed_id", "contributor"}
GSM_EXCLUDED = {"gsm", "contact", "supplementary_file", "data_row_count",
                "channel_count", "organism_ch1", "status", "series_id",
                "submission_date", "last_update_date", "data_processing",
                "gpl"}


def table_columns(conn, table: str) -> List[str]:
    """Every column of ``table``, or nothing when the table is absent."""
    try:
        return [r[1] for r in conn.execute(
            f"PRAGMA table_info({table})").fetchall()]
    except Exception:
        return []


def searchable_columns(conn, table: str) -> List[str]:
    """The columns of ``table`` a keyword is looked for in."""
    excluded = GSE_EXCLUDED if table == "gse" else GSM_EXCLUDED
    return [c for c in table_columns(conn, table)
            if str(c).lower() not in excluded]


def search_tokens(raw: str) -> Set[str]:
    """The tokens a typed query stands for."""
    out: Set[str] = set()
    for term in {t.strip().lower() for t in str(raw or "").split(",") if t.strip()}:
        out.add(term)
        cleaned = re.sub(r"[^a-z0-9\s]", "", term).strip()
        if cleaned:
            out.add(cleaned)
        if cleaned.endswith("s") and len(cleaned) > 3:
            out.add(cleaned[:-1])
        no_space = cleaned.replace(" ", "")
        if no_space and no_space != cleaned and len(no_space) > 3:
            out.add(no_space)
        for word in cleaned.split():
            if len(word) >= 4:
                out.add(word)
                if word.endswith("s") and len(word) > 4:
                    out.add(word[:-1])
    return out


def word_patterns(tokens: Iterable[str]) -> Dict[str, Pattern[str]]:
    """One boundary-anchored pattern per token."""
    pats: Dict[str, Pattern[str]] = {}
    for t in tokens:
        tail = "" if t.endswith("s") else "s?"
        pats[t] = re.compile(_LEFT + re.escape(t) + tail + _RIGHT)
    return pats


def tokens_present(blob_lower: str, blob_clean: str, tokens: Iterable[str],
                   patterns: Dict[str, Pattern[str]],
                   whole_words: bool = True) -> List[str]:
    """Which tokens the text carries, under the rule currently in force."""
    if not whole_words:
        return [t for t in tokens if t in blob_lower or t in blob_clean]
    found = []
    for t in tokens:
        pat = patterns.get(t)
        if pat is not None and (pat.search(blob_lower) or pat.search(blob_clean)):
            found.append(t)
    return found


def keep_whole_word_rows(df: Optional[pd.DataFrame], cols: Sequence[str],
                         tokens: Iterable[str],
                         patterns: Dict[str, Pattern[str]],
                         whole_words: bool = True) -> Optional[pd.DataFrame]:
    """Rows where a keyword stands as a word, not inside a longer one."""
    if not whole_words or df is None or getattr(df, "empty", True):
        return df
    use = [c for c in cols if c in df.columns]
    if not use:
        return df
    blob = pd.Series("", index=df.index)
    for c in use:
        col = df[c]
        blob = blob + (col.astype(str).str.lower() + " ").where(col.notna(), "")
    blob_clean = blob.str.replace(r"[^a-z0-9\s]", "", regex=True)
    keep = pd.Series(False, index=df.index)
    for t in tokens:
        pat = patterns.get(t)
        if pat is None:
            continue
        keep |= (blob.str.contains(pat, regex=True)
                 | blob_clean.str.contains(pat, regex=True))
    return df[keep]


def like_clause(cols: Sequence[str], tokens: Sequence[str]):
    """The scan both entry points issue: one LIKE per column per token."""
    parts, params = [], []
    for tok in tokens:
        for col in cols:
            parts.append(f"LOWER({col}) LIKE ?")
            params.append(f"%{tok}%")
    return " OR ".join(parts), params


def search_corpus(conn, tokens, patterns, *, whole_words: bool = True,
                  log=lambda _m: None, stop=None) -> Dict[str, object]:
    """Experiments a keyword reaches, by their own text and by their samples'."""
    def _cancelled():
        return stop is not None and stop.is_set()

    out: Dict[str, object] = {
        "series_rows": None, "sample_rows": None,
        "gse_from_series": set(), "gse_from_samples": set(),
        "experiments": set(), "matched_samples": set(),
        "series_columns": [], "sample_columns": [],
    }
    toks = sorted(tokens)
    if not toks:
        return out

    gse_cols = searchable_columns(conn, "gse")
    out["series_columns"] = gse_cols
    if gse_cols:
        where, params = like_clause(gse_cols, toks)
        sel = ", ".join(dict.fromkeys(["gse"] + gse_cols))
        try:
            rows = pd.read_sql_query(f"SELECT {sel} FROM gse WHERE {where}",
                                     conn, params=params)
            before = len(rows)
            rows = keep_whole_word_rows(rows, gse_cols, toks, patterns,
                                        whole_words)
            out["series_rows"] = rows
            out["gse_from_series"] = set(rows["gse"].tolist())
            log(f"[search] {len(rows)} experiment(s) by their own description")
            if whole_words and before > len(rows):
                log(f"[search]   whole-word matching dropped {before - len(rows):,}")
        except Exception as exc:
            log(f"[search] series query failed: {exc}")
    if _cancelled():
        return out

    gsm_cols = searchable_columns(conn, "gsm")
    out["sample_columns"] = gsm_cols
    if gsm_cols:
        where, params = like_clause(gsm_cols, toks)
        sel = ", ".join(dict.fromkeys(["gsm", "series_id"] + gsm_cols))
        try:
            rows = pd.read_sql_query(f"SELECT {sel} FROM gsm WHERE {where}",
                                     conn, params=params)
            before = len(rows)
            rows = keep_whole_word_rows(rows, gsm_cols, toks, patterns,
                                        whole_words)
            out["sample_rows"] = rows
            out["matched_samples"] = {str(g).upper()
                                      for g in rows["gsm"].tolist()}
            log(f"[search] {len(rows):,} sample(s) matched")
            if whole_words and before > len(rows):
                log(f"[search]   whole-word matching dropped {before - len(rows):,}")
        except Exception as exc:
            log(f"[search] sample query failed: {exc}")
    if _cancelled():
        return out

    out["gse_from_samples"] = experiments_of(
        conn, out["matched_samples"], out.get("sample_rows"), log=log)
    out["experiments"] = set(out["gse_from_series"]) | set(out["gse_from_samples"])
    log(f"[search] {len(out['experiments'])} experiment(s) in total")
    return out


def experiments_of(conn, gsm_ids, sample_rows=None,
                   log=lambda _m: None) -> set:
    """Every experiment the given samples belong to."""
    found: set = set()
    if sample_rows is not None and "series_id" in getattr(
            sample_rows, "columns", []):
        for sid in sample_rows["series_id"].dropna():
            for part in str(sid).split(","):
                part = part.strip()
                if part:
                    found.add(part)
    ids = [str(g) for g in (gsm_ids or [])]
    if not ids:
        return found
    before = len(found)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join(["?"] * len(chunk))
        try:
            r = pd.read_sql_query(
                f"SELECT DISTINCT gse FROM gse_gsm WHERE gsm IN ({ph})",
                conn, params=chunk)
            found.update(r["gse"].tolist())
        except Exception as exc:
            log(f"[search] gse_gsm lookup failed: {exc}")
            break
    if len(found) > before:
        log(f"[search]   {len(found) - before} experiment(s) named by gse_gsm alone")
    return found
