"""
Tests for the per-run reproducibility manifest (``core.reproducibility``).
All offline, pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genevariate.core.reproducibility import (
    build_manifest, hash_data, manifest_to_markdown,
)


def _df(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"GSM": [f"G{i}" for i in range(5)],
                         "A": rng.normal(size=5), "B": rng.normal(size=5)})


def test_hash_is_deterministic_and_content_sensitive():
    a, b = _df(0), _df(0)
    assert hash_data(a) == hash_data(b)          # same content -> same hash
    assert hash_data(a) != hash_data(_df(1))     # different content -> different
    # robust across input kinds, never raises
    assert isinstance(hash_data(np.arange(10)), str)
    assert isinstance(hash_data({"x": _df(0)}), str)
    assert isinstance(hash_data([1, 2, 3]), str)


def test_build_manifest_captures_four_pillars():
    man = build_manifest("condition_enrichment",
                         params={"platform": "GPL570", "case_label": "tumor"},
                         inputs={"platform": _df(0)},
                         seed=42)
    assert man["tool"] == "condition_enrichment"
    assert man["seed"] == 42
    assert man["params"]["platform"] == "GPL570"
    assert "platform" in man["data_hashes"]
    # versions include the packages that shape analysis output
    assert "numpy" in man["versions"] and "pandas" in man["versions"]
    assert man["environment"]["python"]
    assert "timestamp" in man


def test_manifest_params_are_jsonable():
    man = build_manifest("x", params={"df": _df(0), "n": np.int64(3),
                                       "f": np.float64(1.5), "libs": ("a", "b")})
    p = man["params"]
    assert isinstance(p["df"], str) and p["df"].startswith("<DataFrame")
    assert p["n"] == 3 and p["f"] == 1.5
    assert p["libs"] == ["a", "b"]


def test_manifest_markdown_renders():
    man = build_manifest("run_ngs_de", params={"min_count": 10},
                         inputs={"counts": _df(0)}, seed=7)
    md = manifest_to_markdown(man)
    assert "Reproducibility" in md
    assert "run_ngs_de" in md
    assert "seed" in md and "7" in md
