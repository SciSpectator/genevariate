"""
GSEContext - MemGPT-style rolling memory for one GEO experiment (GSE).

Seeded once at startup from the full platform DataFrame. Updated live
as the GSEWorker resolves NS samples, so every subsequent sample sees
the freshly assigned labels.
"""

import threading
from typing import Dict, List
from collections import Counter

NS = "Not Specified"
LABEL_COLS = ["Tissue", "Condition"]
LABEL_COLS_SCRATCH = ["Tissue", "Condition", "Treatment"]


class GSEContext:
    """
    Complete memory block for one GEO experiment (GSE).

    Thread-safe via a per-instance lock.
    """

    def __init__(self, gse_id: str):
        self.gse_id = gse_id
        self.title = ""
        self.summary = ""
        self.design = ""
        self._samples: List[Dict] = []
        self.label_counts: Dict[str, Counter] = {c: Counter() for c in LABEL_COLS_SCRATCH}
        self._ns_count: Dict[str, int] = {c: 0 for c in LABEL_COLS_SCRATCH}
        self.total = 0
        self._lock = threading.Lock()

    def add_sample(self, gsm: str, labels: Dict[str, str], mem_agent=None):
        """
        Load one sample into context.
        If mem_agent provided, normalise label casing via cluster_lookup.
        """
        rec = {"gsm": gsm}
        for col in self.label_counts:
            val = labels.get(col, NS)
            if val != NS and mem_agent is not None:
                cased = mem_agent.cluster_lookup(col, val)
                if cased:
                    val = cased
            rec[col] = val
            if val != NS:
                self.label_counts[col][val] += 1
            else:
                self._ns_count[col] += 1
        self._samples.append(rec)
        self.total += 1

    def set_meta(self, title: str, summary: str, design: str = ""):
        self.title = (title or "").strip()
        self.summary = (summary or "").strip()
        self.design = (design or "").strip()

    def update_label(self, gsm: str, col: str, new_val: str):
        """No-op: label_counts is a static snapshot loaded at startup."""
        pass

    def labeled_count(self, col: str) -> int:
        return sum(self.label_counts[col].values())

    def diverse_examples(self, col: str, n: int = 5) -> List[Dict]:
        """Return up to n examples covering distinct labels."""
        seen, examples = set(), []
        for s in self._samples:
            v = s.get(col, NS)
            if v != NS and v not in seen:
                examples.append(s)
                seen.add(v)
            if len(examples) >= n:
                break
        return examples

    def context_block(self, col: str) -> str:
        """Context block injected into every LLM prompt."""
        lc = self.labeled_count(col)
        ns = self._ns_count[col]

        lines = []
        if self.title:
            lines.append(f"Experiment title  : {self.title}")
        if self.summary:
            lines.append(f"Experiment summary: {self.summary}")
        if self.design:
            lines.append(f"Overall design    : {self.design}")
        lines.append(
            f"Total samples    : {self.total}"
            f"  |  Labeled {col}: {lc}"
            f"  |  Still NS     : {ns}"
        )

        if self.label_counts[col]:
            lines.append(f"\nKnown {col} labels in this experiment:")
            for label, count in self.label_counts[col].most_common():
                lines.append(f"  [{count:>4}]  {label}")
        else:
            lines.append(f"\nNo {col} labels assigned yet in this experiment.")

        examples = self.diverse_examples(col, n=5)
        if examples:
            other = [c for c in LABEL_COLS if c != col][0]
            lines.append(f"\nExample labeled samples:")
            for ex in examples:
                lines.append(
                    f"  {ex['gsm']}    {col}: {ex.get(col, NS)}"
                    f"  |  {other}: {ex.get(other, NS)}"
                )
        return "\n".join(lines)
