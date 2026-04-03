#!/usr/bin/env python3
"""
GeneVariate — LLM Agentic Extraction (ported from geo_ns_repair_v2_8_.py)
=========================================================================
Architecture:
  Phase 1:   Raw LLM extraction (Tissue, Condition, Treatment via ollama.chat)
  Phase 1.5: ReAct collapse agent — reason + act loop with tools:
               SEARCH: <query>      — search accumulated vocabulary
               PICK: <label>        — finalise with an existing label
             Deterministic fallback: exact match, abbreviation, number guard
  Phase 2:   GSE context rescue — sibling label consensus for remaining NS

  Age / Treatment_Time: deterministic regex parsers (no LLM, no GSE context)
  
  Memory: episodic log (SQLite) — persists past resolutions across runs.
  NO LLM_memory/cluster files — collapse against ALL accumulated results.
  Each phase can be enabled/disabled independently by the user.
"""

import os, re, json, sqlite3, threading, time
from collections import Counter
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import ollama as _ollama_lib
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

NS = "Not Specified"
LABEL_COLS = ["Tissue", "Condition"]
LABEL_COLS_FULL = ["Tissue", "Condition", "Treatment"]

# ── Extraction prompt (same format as proven working in geo_ns_repair_v2_8_.py)
_EXTRACTION_PROMPT = (
    "TASK: Read the metadata below and extract exactly what is written.\n"
    "Do NOT normalise, generalise, or map to any vocabulary — copy the specific term.\n"
    "FIELDS:\n"
    "  Tissue    : anatomical tissue, organ, cell type, or cell line as written\n"
    "  Condition : disease, phenotype, or health status as written\n"
    "  Treatment : drug or stimulus as written. None/vehicle = Untreated.\n"
    "  Age       : numeric age with units as written\n"
    "  Treatment_Time : duration of treatment as written\n"
    "RULES:\n"
    "  - Copy the most specific term present (e.g. Alveolar Macrophages not Lung)\n"
    "  - If a cell type is named, use the cell type (e.g. NK cells not PBMC)\n"
    "  - Healthy/normal/WT/control = Control\n"
    "  - Unknown or absent field = Not Specified\n"
    "  - Title Case. Output JSON only.\n"
    "METADATA: Title: {TITLE}\nSource: {SOURCE}\nCharacteristics: {CHAR}\n"
    'JSON SCHEMA: {{"Tissue":"", "Condition":"", "Treatment":"", "Age":"", "Treatment_Time":""}}'
)

def is_ns(text):
    if not text: return True
    return text.lower().strip() in {
        "not specified","n/a","none","unknown","na","not available",
        "not applicable","unclear","unspecified","missing","undetermined",
        "","nan","null"}

# ══════════════════════════════════════════════════════════════════════════════
#  JSON PARSER (same greedy regex as proven working code)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json_extraction(text, cols):
    result = {c: NS for c in cols}
    if not text: return result
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            for col in cols:
                for key in [col, col.replace(' ','_'), col.lower(), col.replace(' ','_').lower()]:
                    if key in data:
                        val = str(data[key]).strip()
                        if val and val.lower() not in ('none','null','','not specified'):
                            result[col] = val
                        break
            return result
    except: pass
    # Fallback: parse "Field: value" lines
    for line in (text or "").splitlines():
        line = line.strip()
        for col in cols:
            pfx = f"{col}:"
            if line.lower().startswith(pfx.lower()):
                val = line[len(pfx):].strip().strip('"').strip("'")
                if val and not is_ns(val): result[col] = val
                break
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1.5 — Deterministic collapse
# ══════════════════════════════════════════════════════════════════════════════

def _norm(t): return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]',' ',t.lower())).strip()
def _compact(t): return _norm(t).replace(' ','')
def _initials(t): return ''.join(w[0] for w in _norm(t).split() if w)
def _numbers(t): return re.findall(r'\d+',t)
def _numeric_guard_ok(a,b):
    na,nb = _numbers(a),_numbers(b)
    if not na or not nb: return True
    return sorted(na)==sorted(nb)

def phase15_collapse(extracted, ctx_labels):
    """Deterministic collapse: exact match + abbreviation + number guard."""
    if not extracted or not ctx_labels: return None,None
    ce,ie = _compact(extracted),_initials(extracted)
    for ex in ctx_labels:
        if not ex or is_ns(ex): continue
        if not _numeric_guard_ok(extracted,ex): continue
        cx,ix = _compact(ex),_initials(ex)
        if ce==cx: return ex,"exact_match"
        if 2<=len(ce)<=6 and len(cx)>len(ce) and ce==ix and len(ix)>=2: return ex,"abbreviation"
        if 2<=len(cx)<=6 and len(ce)>len(cx) and cx==ie and len(ie)>=2: return ex,"abbreviation"
    return None,None

# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY AGENT — episodic log + accumulated vocabulary (NO cluster files)
# ══════════════════════════════════════════════════════════════════════════════

class MemoryAgent:
    """
    Persistent memory: episodic log + KG + vocabulary + clusters.
    Context window approach: ALL extracted labels accumulate in vocabulary.
    Clusters auto-built from normalization rules.
    String fields: Tissue, Condition, Treatment + custom user fields
    Numeric fields (Age, Treatment_Time): stored but NOT clustered.
    """
    NUMERIC_FIELDS = {"Age", "Treatment_Time"}

    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._vocabulary = {}  # {col: Counter(label -> count)}
        self._init_db()
        self._load_vocabulary()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL"); return c

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS episodic_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT, col TEXT, raw_label TEXT,
                canonical TEXT, confidence REAL DEFAULT 1.0, platform TEXT DEFAULT '',
                gse TEXT DEFAULT '', gsm TEXT DEFAULT '',
                ts TEXT DEFAULT(datetime('now')), collapse_rule TEXT DEFAULT '');
            CREATE INDEX IF NOT EXISTS ep_raw ON episodic_log(col,raw_label);
            CREATE TABLE IF NOT EXISTS kg_triples(
                col TEXT, subject TEXT, relation TEXT, object TEXT,
                weight REAL DEFAULT 1.0, PRIMARY KEY(col,subject,relation,object));
            CREATE TABLE IF NOT EXISTS vocabulary(
                col TEXT NOT NULL, label TEXT NOT NULL, count INTEGER DEFAULT 1,
                platform TEXT DEFAULT '', last_seen TEXT DEFAULT(datetime('now')),
                PRIMARY KEY(col, label));
            CREATE TABLE IF NOT EXISTS clusters(
                col TEXT NOT NULL, canonical TEXT NOT NULL,
                member TEXT NOT NULL, rule TEXT DEFAULT 'exact',
                PRIMARY KEY(col, canonical, member));
            CREATE INDEX IF NOT EXISTS cl_member ON clusters(col, member);
            CREATE TABLE IF NOT EXISTS field_meta(
                col TEXT PRIMARY KEY, field_type TEXT DEFAULT 'string');
            """)

    def _load_vocabulary(self):
        try:
            with self._conn() as c:
                for col, label, count in c.execute("SELECT col,label,count FROM vocabulary"):
                    if col not in self._vocabulary: self._vocabulary[col] = Counter()
                    self._vocabulary[col][label] = count
        except: pass

    def _is_numeric_field(self, col):
        return col in self.NUMERIC_FIELDS

    def register_field(self, col, field_type="string"):
        try:
            with self._lock, self._conn() as c:
                c.execute("INSERT OR REPLACE INTO field_meta(col,field_type) VALUES(?,?)", (col, field_type))
            if field_type == "numeric": self.NUMERIC_FIELDS.add(col)
        except: pass

    def add_to_vocabulary(self, col, label, platform=""):
        if not label or is_ns(label): return
        if col not in self._vocabulary: self._vocabulary[col] = Counter()
        self._vocabulary[col][label] += 1
        try:
            with self._lock, self._conn() as c:
                c.execute("INSERT INTO vocabulary(col,label,count,platform) VALUES(?,?,1,?) "
                          "ON CONFLICT(col,label) DO UPDATE SET count=count+1,last_seen=datetime('now')",
                          (col, label, platform))
        except: pass

    def search_vocabulary(self, col, query, k=10):
        if not query or is_ns(query): return []
        vocab = self._vocabulary.get(col, Counter())
        if not vocab: return []
        qn = _norm(query); results = []
        for label, count in vocab.most_common():
            ln = _norm(label)
            if qn == ln: score = 1.0
            elif qn in ln or ln in qn: score = 0.9
            elif _compact(query) == _initials(label): score = 0.85
            else:
                qw, lw = set(qn.split()), set(ln.split())
                score = len(qw & lw) / max(1, len(qw | lw)) * 0.8
            if score >= 0.3: results.append((label, score, count))
        results.sort(key=lambda x: (-x[1], -x[2]))
        return results[:k]

    def get_vocabulary(self, col):
        return dict(self._vocabulary.get(col, {}))

    def get_all_fields(self):
        return sorted(self._vocabulary.keys())

    def build_clusters_from_vocabulary(self, col, log_fn=print):
        if self._is_numeric_field(col):
            return 0
        vocab = self._vocabulary.get(col, Counter())
        if not vocab: return 0
        labels_sorted = [lbl for lbl, _ in vocab.most_common()]
        groups = {}; assigned = set()
        for label in labels_sorted:
            if label in assigned: continue
            cl, il = _compact(label), _initials(label)
            group = [label]; assigned.add(label)
            for other in labels_sorted:
                if other in assigned: continue
                if not _numeric_guard_ok(label, other): continue
                co, io = _compact(other), _initials(other)
                if cl == co or (2<=len(co)<=6 and len(cl)>len(co) and co==il and len(il)>=2) or \
                   (2<=len(cl)<=6 and len(co)>len(cl) and cl==io and len(io)>=2):
                    group.append(other); assigned.add(other)
            groups[group[0]] = group
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM clusters WHERE col=?", (col,))
            rows = []
            for canonical, members in groups.items():
                for member in members:
                    rule = "exact" if _compact(member)==_compact(canonical) else "abbreviation"
                    rows.append((col, canonical, member, rule))
                    if member != canonical:
                        c.execute("INSERT OR REPLACE INTO kg_triples VALUES(?,?,?,?,?)",
                                  (col, member, "variant_of", canonical, 0.9))
            c.executemany("INSERT OR REPLACE INTO clusters VALUES(?,?,?,?)", rows)
        log_fn(f"  [Clusters] {col}: {len(groups)} clusters from {len(vocab)} labels")
        return len(groups)

    def build_all_clusters(self, log_fn=print):
        total = 0
        for col in sorted(self._vocabulary.keys()):
            if not self._is_numeric_field(col):
                total += self.build_clusters_from_vocabulary(col, log_fn)
        return total

    def get_clusters(self, col):
        clusters = {}
        try:
            with self._conn() as c:
                for can, mem in c.execute("SELECT canonical,member FROM clusters WHERE col=? ORDER BY canonical", (col,)):
                    clusters.setdefault(can, []).append(mem)
        except: pass
        return clusters

    def cluster_lookup(self, col, raw_label):
        if not raw_label or is_ns(raw_label): return None
        try:
            with self._conn() as c:
                for form in [raw_label, raw_label.lower(), _norm(raw_label)]:
                    row = c.execute("SELECT canonical FROM clusters WHERE col=? AND member=?", (col, form)).fetchone()
                    if row: return row[0]
        except: pass
        return None

    def ingest_extraction_results(self, df, fields=None, platform="", log_fn=print):
        if fields is None:
            skip = {'gsm','GSM','series_id','gpl','platform','_agents','_audit'}
            fields = [c for c in df.columns if c not in skip]
        for col in fields:
            if col not in df.columns: continue
            vals = df[col].dropna().astype(str)
            vals = vals[~vals.isin(['Not Specified','Not specified','','nan','None'])]
            if vals.empty: continue
            try:
                npct = pd.to_numeric(vals, errors='coerce').notna().sum() / max(1, len(vals))
                if npct > 0.7:
                    self.register_field(col, "numeric")
                    log_fn(f"  [Ingest] {col}: numeric ({len(vals)} vals)")
                    for label in vals: self.add_to_vocabulary(col, str(label).strip(), platform)
                    continue
            except: pass
            self.register_field(col, "string")
            counts = Counter(vals.str.strip())
            for label, count in counts.items():
                if label and not is_ns(label):
                    if col not in self._vocabulary: self._vocabulary[col] = Counter()
                    self._vocabulary[col][label] += count
                    try:
                        with self._lock, self._conn() as c:
                            c.execute("INSERT INTO vocabulary(col,label,count,platform) VALUES(?,?,?,?) "
                                      "ON CONFLICT(col,label) DO UPDATE SET count=count+?,last_seen=datetime('now')",
                                      (col, label, count, platform, count))
                    except: pass
            log_fn(f"  [Ingest] {col}: {len(counts)} unique labels")
        log_fn(f"  [Ingest] Building clusters...")
        self.build_all_clusters(log_fn)

    def export_clusters_text(self, output_dir, log_fn=print):
        os.makedirs(output_dir, exist_ok=True)
        exported = []
        for col in sorted(self._vocabulary.keys()):
            if self._is_numeric_field(col): continue
            clusters = self.get_clusters(col)
            if not clusters: continue
            fname = f"{col}_clusters.txt"
            path = os.path.join(output_dir, fname)
            vocab = self._vocabulary.get(col, Counter())
            sorted_cl = sorted(clusters.items(), key=lambda x: sum(vocab.get(m,0) for m in x[1]), reverse=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {col} Clusters ({len(clusters)} clusters)\n\n")
                for canonical, members in sorted_cl:
                    total = sum(vocab.get(m,0) for m in members)
                    f.write(f"CLUSTER: {canonical} (TOTAL: {total})\n")
                    for member in sorted(members, key=lambda m: -vocab.get(m,0)):
                        f.write(f"  - {member}    | {vocab.get(member,0)}\n")
                    f.write("\n")
            exported.append(fname)
            log_fn(f"  [Export] {fname}: {len(clusters)} clusters")
        return exported

    def export_db(self, output_path, log_fn=print):
        import shutil
        try:
            with self._conn() as c: c.execute("PRAGMA wal_checkpoint(FULL)")
            shutil.copy2(self.db_path, output_path)
            log_fn(f"  [Export] Saved: {output_path}")
            return True
        except Exception as e:
            log_fn(f"  [Export] Error: {e}"); return False

    def episodic_search(self, col, raw_label):
        try:
            with self._conn() as c:
                rows = c.execute("SELECT canonical,AVG(confidence),COUNT(*),MAX(ts) FROM episodic_log WHERE col=? AND raw_label=? GROUP BY canonical ORDER BY COUNT(*) DESC LIMIT 5", (col, raw_label)).fetchall()
            return [{"canonical":r[0],"confidence":r[1],"count":r[2]} for r in rows]
        except: return []

    def log_resolution(self, col, raw_label, canonical, confidence=1.0, platform="", gse="", gsm="", collapse_rule=""):
        try:
            with self._lock, self._conn() as c:
                c.execute("INSERT INTO episodic_log(col,raw_label,canonical,confidence,platform,gse,gsm,collapse_rule) VALUES(?,?,?,?,?,?,?,?)", (col,raw_label,canonical,confidence,platform,gse,gsm,collapse_rule))
                if raw_label!=canonical: c.execute("INSERT OR REPLACE INTO kg_triples VALUES(?,?,?,?,?)", (col,raw_label,"variant_of",canonical,confidence))
        except: pass

    def kg_lookup(self, col, label):
        try:
            with self._conn() as c: return c.execute("SELECT object,relation,weight FROM kg_triples WHERE col=? AND subject=? ORDER BY weight DESC LIMIT 5", (col,label)).fetchall()
        except: return []

    def should_log(self, col, raw, canonical, rule):
        if raw==canonical: return False, 0.0, "identity"
        if rule in ("",): return False, 0.0, "no_change"
        if rule=="episodic": return True, 0.98, "episodic"
        if rule in ("exact_match","abbreviation"): return True, 0.92, "deterministic"
        if "react" in rule: return True, 0.85, "agent"
        return True, 0.80, "other"

    def stats(self):
        try:
            with self._conn() as c:
                ep = {r[0]:r[1] for r in c.execute("SELECT col,COUNT(*) FROM episodic_log GROUP BY col")}
                cl = {r[0]:r[1] for r in c.execute("SELECT col,COUNT(DISTINCT canonical) FROM clusters GROUP BY col")}
            vocab = {c: len(v) for c,v in self._vocabulary.items() if v}
            return {"episodic": ep, "vocabulary": vocab, "clusters": cl}
        except: return {}


class GSEContext:
    def __init__(self, gse_id):
        self.gse_id = gse_id; self.title = ""; self.summary = ""; self.design = ""
        self.label_counts = {c: Counter() for c in LABEL_COLS_FULL}
        self.total = 0
    def add_sample(self, labels):
        for col in self.label_counts:
            val = labels.get(col, NS)
            if val and not is_ns(val): self.label_counts[col][val] += 1
        self.total += 1
    def set_meta(self, title, summary, design=""):
        self.title = (title or"").strip()
        self.summary = (summary or"").strip()
        self.design = (design or"").strip()

# ══════════════════════════════════════════════════════════════════════════════
#  GSEWorker — THE AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class GSEWorker:
    """
    Processes samples for one GSE experiment.
    Full agentic pipeline from geo_ns_repair_v2_8_.py:
      Phase 1:   Raw LLM extraction (ollama.chat)
      Phase 1.5: ReAct collapse agent (SEARCH/PICK) + deterministic fallback
      Phase 2:   GSE context rescue for remaining NS
    """

    def __init__(self, gse_id, ctx, mem_agent=None, model="gemma2:9b",
                 platform="", log_fn=None,
                 enable_phase15=True, enable_phase2=True,
                 ollama_url=None):
        self.gse_id = gse_id
        self.ctx = ctx
        self.mem_agent = mem_agent
        self.model = model
        self.platform = platform
        self._log = log_fn or (lambda m: None)
        self.enable_phase15 = enable_phase15
        self.enable_phase2 = enable_phase2
        self.ollama_url = ollama_url or DEFAULT_URL
        # Pre-build GSE description block
        lines = []
        if ctx.title: lines.append(f"Experiment title: {ctx.title}")
        if ctx.summary: lines.append(f"Summary: {ctx.summary[:500]}")
        if ctx.design: lines.append(f"Design: {ctx.design[:300]}")
        self._gse_block = "\n".join(lines) + "\n" if lines else ""

    def _llm(self, prompt, max_tokens=200):
        if not _HAS_OLLAMA: return ""
        # Try with configured URL (GPU or CPU)
        for attempt in range(1, 4):
            try:
                resp = _ollama_lib.chat(
                    model=self.model,
                    messages=[{"role":"user","content":prompt}],
                    options={"temperature":0.0,"num_predict":max_tokens},
                    keep_alive=-1)
                if hasattr(resp,'message') and hasattr(resp.message,'content'):
                    return (resp.message.content or "").strip()
                elif isinstance(resp,dict):
                    return resp.get("message",{}).get("content","").strip()
                return ""
            except Exception as e:
                err = str(e).lower()
                if "out of memory" in err or "cudamalloc" in err:
                    # VRAM exhausted — try CPU fallback
                    if _CPU_OLLAMA_ACTIVE and ollama_server_ok(CPU_OLLAMA_URL):
                        self._log(f"  [OOM] Switching {self.gse_id} to CPU Ollama")
                        self.ollama_url = CPU_OLLAMA_URL
                        time.sleep(2)
                    else:
                        time.sleep(8)
                elif "connection refused" in err or "disconnected" in err:
                    time.sleep(5 * attempt)
                else:
                    time.sleep(3 * attempt)
                if attempt == 3: return ""
        return ""

    def _llm_chat(self, messages, max_tokens=200):
        if not _HAS_OLLAMA: return ""
        for attempt in range(1, 4):
            try:
                resp = _ollama_lib.chat(
                    model=self.model, messages=messages,
                    options={"temperature":0.0,"num_predict":max_tokens,"num_ctx":4096},
                    keep_alive=-1)
                if hasattr(resp,'message') and hasattr(resp.message,'content'):
                    return (resp.message.content or "").strip()
                elif isinstance(resp,dict):
                    return resp.get("message",{}).get("content","").strip()
                return ""
            except Exception as e:
                err = str(e).lower()
                if "out of memory" in err or "cudamalloc" in err:
                    if _CPU_OLLAMA_ACTIVE and ollama_server_ok(CPU_OLLAMA_URL):
                        self._log(f"  [OOM] Switching {self.gse_id} to CPU Ollama")
                        self.ollama_url = CPU_OLLAMA_URL
                        time.sleep(2)
                    else:
                        time.sleep(8)
                else:
                    time.sleep(3 * attempt)
                if attempt == 3: return ""
        return ""

    # ── PHASE 1: Raw LLM extraction ──────────────────────────────────────

    def _extract_raw(self, gsm_row):
        """Phase 1: Extract all fields from raw metadata via LLM."""
        title = str(gsm_row.get('title','') or '')[:80]
        source = str(gsm_row.get('source_name_ch1','') or '')[:80]
        chars = str(gsm_row.get('characteristics_ch1','') or '').replace('\t',' ')[:300]

        prompt = (_EXTRACTION_PROMPT
            .replace("{TITLE}", title)
            .replace("{SOURCE}", source)
            .replace("{CHAR}", chars))

        for attempt in range(3):
            raw_text = self._llm(prompt, max_tokens=200)
            if raw_text: break
            time.sleep(3 * (attempt + 1))

        all_cols = ['Tissue','Condition','Treatment','Age','Treatment_Time']
        result = _parse_json_extraction(raw_text, all_cols)

        # Deterministic Age/Treatment_Time override (more reliable than LLM)
        det_age = _extract_age_regex(gsm_row)
        if not is_ns(det_age): result['Age'] = det_age
        det_tt = _extract_treatment_time_regex(gsm_row)
        if not is_ns(det_tt): result['Treatment_Time'] = det_tt

        return result

    # ── PHASE 1.5: ReAct collapse agent ──────────────────────────────────

    def _run_collapse_agent(self, gsm, col, out1, ctx_labels, ctx_counts, gsm_row=None):
        """ReAct agent loop: THOUGHT → ACTION → OBSERVATION, max 3 turns."""
        MAX_TURNS = 3
        ma = self.mem_agent

        # Build context
        gse_ctx_text = ""
        if self._gse_block: gse_ctx_text += self._gse_block
        if ctx_labels:
            gse_ctx_text += "Sibling labels:\n"
            for lbl in sorted(ctx_counts, key=ctx_counts.get, reverse=True):
                gse_ctx_text += f"  {lbl!r} ({ctx_counts[lbl]}x)\n"

        # Episodic hits
        epi_text = ""
        if ma:
            hits = ma.episodic_search(col, out1)
            if hits:
                epi_text = "Past resolutions:\n"
                for h in hits[:3]:
                    epi_text += f"  {h['canonical']!r} ({h['count']}x, conf={h['confidence']:.2f})\n"

        # Initial vocabulary search
        init_search = ""
        if ma:
            search_seed = out1 if not is_ns(out1) else (max(ctx_counts, key=ctx_counts.get) if ctx_counts else "")
            if search_seed:
                hits = ma.search_vocabulary(col, search_seed)
                if hits:
                    init_search = f"Vocabulary search for {search_seed!r}:\n"
                    for lbl, score, cnt in hits[:6]:
                        init_search += f"  {lbl!r}  score={score:.2f} count={cnt}\n"

        system = (
            f"You are a biomedical label normalization agent for: {col}.\n"
            f"Collapse the extracted label to match existing vocabulary.\n\n"
            f"TOOLS (one per turn):\n"
            f"  SEARCH: <query>   search vocabulary for matching labels\n"
            f"  PICK: <label>     select a label from vocabulary\n\n"
            f"RULES:\n"
            f"1. Pick the most specific match from vocabulary.\n"
            f"2. If nothing matches: PICK: NO_MATCH\n"
            f"3. Format: THOUGHT: <reason>\\nACTION: SEARCH/PICK: <value>\n"
        )

        title = str(gsm_row.get('title','') or '')[:60] if gsm_row else ""
        context = (
            f"EXPERIMENT:\n{gse_ctx_text}\n"
            f"{epi_text}\n"
            f"SAMPLE {gsm}: {title}\n"
            f"LABEL TO COLLAPSE: {out1!r}\n\n"
            f"{init_search}\n"
            f"Now reason and act. Start with THOUGHT:"
        )

        messages = [
            {"role":"system","content":system},
            {"role":"user","content":context},
        ]

        for turn in range(MAX_TURNS):
            response = self._llm_chat(messages, max_tokens=120)
            if not response: break
            messages.append({"role":"assistant","content":response})

            # Parse ACTION
            action_line = ""
            for line in response.splitlines():
                lu = line.strip().upper()
                if lu.startswith("ACTION:"):
                    action_line = line.strip()[7:].strip(); break
                elif lu.startswith("SEARCH:") or lu.startswith("PICK:"):
                    action_line = line.strip(); break

            if not action_line: break
            au = action_line.upper()

            if au.startswith("PICK:"):
                val = action_line[5:].strip()
                if val.upper() in ("NO_MATCH","NOMATCH"):
                    return out1, False, "react_no_match"
                # Validate against vocabulary
                if ma:
                    vocab = ma._vocabulary.get(col, Counter())
                    # Exact or normalized match
                    for existing in vocab:
                        if _compact(val) == _compact(existing):
                            return existing, True, "react_pick"
                # Accept as-is if not in vocab (new label)
                return val, True, "react_pick_new"

            elif au.startswith("SEARCH:"):
                query = action_line[7:].strip()
                if ma:
                    hits = ma.search_vocabulary(col, query)
                    if hits:
                        obs = f"OBSERVATION: results for {query!r}:\n"
                        for lbl, score, cnt in hits[:6]:
                            obs += f"  {lbl!r}  score={score:.2f} count={cnt}\n"
                    else:
                        obs = f"OBSERVATION: no matches for {query!r}"
                else:
                    obs = "OBSERVATION: vocabulary not available"
                messages.append({"role":"user","content":obs})
            else:
                break

        return out1, False, "react_exhausted"

    # ── FULL PIPELINE: classify one sample ────────────────────────────────

    def classify_sample(self, gsm_row, fields=None):
        """
        Full agent pipeline:
          Phase 1:   Raw LLM extraction
          Phase 1.5: Collapse via ReAct agent + deterministic rules
          Phase 2:   GSE context rescue for remaining NS
        """
        gsm_id = gsm_row.get('gsm', gsm_row.get('GSM', '?'))
        if fields is None:
            fields = ['Condition','Tissue','Age','Treatment','Treatment_Time']

        # ── PHASE 1: Raw extraction ──────────────────────────────────────
        result = self._extract_raw(gsm_row)
        result['gsm'] = gsm_id
        ma = self.mem_agent

        # Add resolved labels to vocabulary
        if ma:
            for col in fields:
                val = result.get(col, NS)
                if not is_ns(val): ma.add_to_vocabulary(col, val)

        if not self.enable_phase15:
            return result

        # ── PHASE 1.5: Collapse for Tissue, Condition, Treatment ─────────
        for col in ['Tissue','Condition','Treatment']:
            if col not in fields: continue
            out1 = result.get(col, NS)
            if is_ns(out1): continue  # nothing to collapse

            ctx_labels = list(self.ctx.label_counts.get(col, {}).keys())
            ctx_counts = dict(self.ctx.label_counts.get(col, {}))
            final, collapsed, rule = out1, False, ""

            # Episodic fast-path
            if ma:
                hits = ma.episodic_search(col, out1)
                if hits and hits[0]["count"] >= 2 and hits[0]["confidence"] >= 0.8:
                    final, collapsed, rule = hits[0]["canonical"], True, "episodic"

            # GSE dominant fast-path (70%+)
            if not collapsed and ctx_counts and col != 'Treatment':
                top_label, top_count = max(ctx_counts.items(), key=lambda x: x[1])
                total = sum(ctx_counts.values())
                if total > 0 and top_count / total >= 0.70:
                    final, collapsed, rule = top_label, True, "gse_dominant"

            # Deterministic collapse against siblings
            if not collapsed and ctx_labels:
                matched, r = phase15_collapse(out1, ctx_labels)
                if matched:
                    final, collapsed, rule = matched, True, r

            # ReAct agent (only if vocabulary has entries to search)
            if not collapsed and ma and _HAS_OLLAMA:
                vocab_size = len(ma._vocabulary.get(col, {}))
                if vocab_size >= 5:  # enough vocabulary to be useful
                    final, collapsed, rule = self._run_collapse_agent(
                        gsm_id, col, out1, ctx_labels, ctx_counts, gsm_row)

            if final != out1 and not is_ns(final):
                result[col] = final
                if ma:
                    do, conf, _ = ma.should_log(col, out1, final, rule)
                    if do:
                        ma.log_resolution(col=col, raw_label=out1, canonical=final,
                            confidence=conf, platform=self.platform,
                            gse=self.gse_id, gsm=gsm_id, collapse_rule=rule)
                    ma.add_to_vocabulary(col, final)

        if not self.enable_phase2:
            return result

        # ── PHASE 2: GSE context rescue for remaining NS ─────────────────
        # Only for Tissue, Condition (NOT Age, Treatment_Time)
        for col in ['Tissue','Condition']:
            if col not in fields: continue
            if not is_ns(result.get(col, NS)): continue

            ctx_counts = dict(self.ctx.label_counts.get(col, {}))
            if not ctx_counts: continue

            # Step 2a: LLM with GSE context
            sibling_text = "\n".join(f"  {lbl} ({cnt}x)"
                for lbl, cnt in sorted(ctx_counts.items(), key=lambda x: -x[1]))
            p2 = (
                f"{self._gse_block}\n"
                f"{col} labels in this experiment:\n{sibling_text}\n\n"
                f"Sample {gsm_id}: "
                f"Title: {str(gsm_row.get('title',''))[:60]}\n"
                f"Source: {str(gsm_row.get('source_name_ch1',''))[:60]}\n\n"
                f"What is the {col} for this sample? Match sibling labels if appropriate.\n"
                f"If not clear: Not Specified\n\n{col}:"
            )
            llm_out = self._llm(p2, max_tokens=60)
            if llm_out:
                val = llm_out.split('\n')[0].strip().strip('"').strip("'")
                val = re.sub(rf"^{col}\s*:\s*","",val,flags=re.I).strip()
                if val and not is_ns(val):
                    result[col] = val
                    if ma: ma.add_to_vocabulary(col, val)

            # Step 2b: Dominant sibling rescue (50%+)
            if is_ns(result.get(col, NS)) and ctx_counts:
                dom = max(ctx_counts, key=ctx_counts.get)
                dom_pct = ctx_counts[dom] / sum(ctx_counts.values())
                if dom_pct >= 0.50:
                    result[col] = dom
                    if ma: ma.add_to_vocabulary(col, dom)

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC REGEX PARSERS (Age, Treatment_Time)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_chars(text):
    if not text: return []
    items = re.split(r'[\t;]+|\\t', str(text))
    pairs = []
    for item in items:
        item = item.strip()
        if ':' in item:
            k, _, v = item.partition(':'); pairs.append((k.strip(), v.strip()))
        elif item: pairs.append(('', item))
    return pairs

_AGE_RE = re.compile(r'^\s*(age|age\s*\([^)]*\)|patient[_ ]?age|donor[_ ]?age|age[_ ]?years?)\s*[:=]\s*(.+)', re.I)
_TT_RE = re.compile(r'^\s*(time|duration|time[_ ]?point|treatment[_ ]?time|timepoint|hours?[_ ]?post)\s*[:=]\s*(.+)', re.I)

def _extract_age_regex(r):
    for k,v in _parse_chars(str(r.get('characteristics_ch1','') or'')):
        m = _AGE_RE.match(f"{k}: {v}" if k else v)
        if m:
            a = m.group(2).strip()
            if a and a.lower() not in ('na','n/a','unknown','nan','none',''): return a
    return NS

def _extract_treatment_time_regex(r):
    for k,v in _parse_chars(str(r.get('characteristics_ch1','') or'')):
        m = _TT_RE.match(f"{k}: {v}" if k else v)
        if m:
            val = m.group(2).strip()
            if val and val.lower() not in ('na','n/a','none','nan',''): return val
    return NS

# ══════════════════════════════════════════════════════════════════════════════
#  BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_gse_contexts(samples_df, gse_meta=None, mem_agent=None):
    contexts = {}
    if 'series_id' not in samples_df.columns: return contexts
    for gse_id, group in samples_df.groupby('series_id'):
        gse_id = str(gse_id).strip()
        if not gse_id or gse_id.lower() in ('nan','none',''): continue
        ctx = GSEContext(gse_id)
        if gse_meta and gse_id in gse_meta:
            m = gse_meta[gse_id]
            ctx.set_meta(m.get('title',''), m.get('summary',''))
        for _, row in group.iterrows():
            labels = {}
            for col in LABEL_COLS_FULL:
                v = row.get(col, NS)
                labels[col] = NS if (pd.isna(v) or str(v).strip()=='') else str(v).strip()
            ctx.add_sample(labels)
        contexts[gse_id] = ctx
    return contexts

def find_llm_memory_dir(data_dir):
    """Not used — kept for import compatibility."""
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  GPU / CPU HYBRID INFRASTRUCTURE (ported from geo_ns_repair_v2_9_.py)
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, platform as _platform_mod
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

MODEL_RAM_GB = {
    "gemma2:2b": 2.0, "gemma2:9b": 5.4, "gemma2:9b-q4_0": 5.0,
    "gemma2:27b": 18.0, "llama3:8b": 5.5, "llama3.1:8b": 5.5,
    "mistral:7b": 4.8, "qwen2.5:7b": 4.4,
}
DEFAULT_MODEL_GB = 5.4
DEFAULT_URL = "http://localhost:11434"
CPU_OLLAMA_URL = "http://localhost:11435"
_CPU_OLLAMA_ACTIVE = False
_cpu_server_proc = None


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


def ollama_server_ok(base_url=DEFAULT_URL, timeout=3):
    try:
        import requests
        return requests.get(f"{base_url}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def compute_ollama_parallel(model="gemma2:9b", reserve_gb=4.0, extra_vram_gb=0.0):
    """
    Compute HYBRID worker count: GPU workers + CPU workers.
    Returns (total, gpu_workers, cpu_workers).
    """
    try:
        gpus = detect_gpus()
        model_key = model.strip().lower()
        slot_gb = MODEL_RAM_GB.get(model_key, DEFAULT_MODEL_GB)
        free_gb = psutil.virtual_memory().available / 1e9 if _HAS_PSUTIL else 8.0

        # GPU workers
        if gpus:
            total_vram = sum(g["vram_gb"] for g in gpus)
            try:
                import requests
                ps = requests.get(f"{DEFAULT_URL}/api/ps", timeout=2).json()
                loaded_vram = sum(m.get("size_vram", 0)/1e9 for m in ps.get("models", []))
            except Exception:
                loaded_vram = slot_gb
            kv_per_slot = max(0.3, slot_gb * 0.15)
            headroom = total_vram - loaded_vram - 1.0 - extra_vram_gb
            gpu_workers = max(1, min(8, int(headroom / kv_per_slot)))
        else:
            gpu_workers = 0

        # CPU workers
        ram_after = free_gb - reserve_gb
        ram_slots = max(0, int(ram_after / slot_gb))
        cpu_count = os.cpu_count() or 4
        usable_cpu = max(1, cpu_count - 2)
        cpu_workers = min(ram_slots, usable_cpu)

        total = max(1, gpu_workers + cpu_workers)
        return total, gpu_workers, cpu_workers
    except Exception:
        return 1, 0, 1


def start_ollama_cpu_server(log_fn=print, num_parallel=2):
    """Launch second Ollama on port 11435, CPU only. Returns Popen or None."""
    global _cpu_server_proc
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OLLAMA_HOST"] = "0.0.0.0:11435"
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_FLASH_ATTENTION"] = "0"
    env["OLLAMA_MODELS"] = os.path.expanduser("~/.ollama/models")

    log_fn(f"  Starting CPU Ollama on port 11435 ({num_parallel} workers)...")
    try:
        if _platform_mod.system().lower() == "windows":
            proc = subprocess.Popen(["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            proc = subprocess.Popen(["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, preexec_fn=os.setsid)
        for i in range(30):
            time.sleep(1)
            if ollama_server_ok(CPU_OLLAMA_URL):
                log_fn(f"  CPU Ollama ready ({i+1}s) — port 11435")
                _cpu_server_proc = proc
                return proc
            if i % 5 == 4:
                log_fn(f"    waiting ({i+1}s)...")
        log_fn("  [WARN] CPU Ollama did not start — CPU workers disabled")
        proc.terminate()
        return None
    except Exception as e:
        log_fn(f"  [WARN] Could not start CPU Ollama: {e}")
        return None


def stop_cpu_server():
    global _cpu_server_proc, _CPU_OLLAMA_ACTIVE
    if _cpu_server_proc:
        try:
            _cpu_server_proc.terminate()
        except Exception:
            pass
        _cpu_server_proc = None
    _CPU_OLLAMA_ACTIVE = False


def vram_utilisation_pct():
    """Current VRAM % used (0-100)."""
    try:
        gpus = detect_gpus()
        if not gpus: return 0.0
        total = sum(g["vram_gb"] for g in gpus)
        free = sum(g["free_vram_gb"] for g in gpus)
        return 100.0 * (total - free) / total if total else 0.0
    except Exception:
        return 0.0


def _pick_ollama_url(gpu_url=DEFAULT_URL, vram_threshold=92.0):
    """Route worker to GPU or CPU Ollama based on VRAM load."""
    global _CPU_OLLAMA_ACTIVE
    if not _CPU_OLLAMA_ACTIVE:
        return gpu_url
    vpct = vram_utilisation_pct()
    if vpct >= vram_threshold:
        if ollama_server_ok(CPU_OLLAMA_URL):
            return CPU_OLLAMA_URL
    return gpu_url
