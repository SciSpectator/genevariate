"""
Offline tests for the conversational-assistant core (``core.chatbot``).

No GUI, no network: ollama is monkeypatched off (keyword fallback), and the LLM
transport is monkeypatched to return canned JSON to exercise the parse/validate
and malformed-fallback paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genevariate.core.chatbot import build_registry, route
from genevariate.core.chatbot import router as router_mod
from genevariate.core.chatbot.tools import Action


class FakeApp:
    def __init__(self, platforms=None):
        self.gpl_datasets = platforms or {}


def _platform_df():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(5)]
    rows = []
    for i in range(6):
        row = {"GSM": f"GSM{i}",
               "Classified_condition": "tumor" if i < 3 else "normal"}
        for g in genes:
            row[g] = float(rng.normal())
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _ollama_off(monkeypatch):
    """Force keyword mode unless a test opts back in."""
    import genevariate.core.ollama_manager as om
    monkeypatch.setattr(om, "ollama_server_ok", lambda *a, **k: False)


def test_registry_has_core_tools():
    reg = build_registry(FakeApp())
    for name in ("list_platforms", "condition_enrichment",
                 "variability_enrichment", "rank_genes", "run_ngs_de",
                 "classify_distributions", "meta_enrichment",
                 "gene_distribution", "compare_gene"):
        assert name in reg


def test_keyword_route_condition_enrichment():
    reg = build_registry(FakeApp())
    action = route("run condition enrichment on GPL570 tumor vs normal", reg)
    assert action.tool == "condition_enrichment"
    assert action.source == "keyword"
    assert action.params.get("case_label") == "tumor"
    assert action.params.get("control_label") == "normal"


def test_keyword_route_list_platforms():
    reg = build_registry(FakeApp())
    action = route("what platforms are loaded", reg)
    assert action.tool == "list_platforms"


def test_keyword_extracts_gene_and_platform():
    reg = build_registry(FakeApp())
    a = route("analyze the distribution of TP53 on GPL570", reg)
    assert a.tool == "gene_distribution"
    assert a.params.get("gene") == "TP53"
    assert a.params.get("platform") == "GPL570"
    # platform id must survive even when it differs from the first loaded one
    a2 = route("rank genes tumor vs normal on GPL96", reg)
    assert a2.tool == "rank_genes"
    assert a2.params.get("platform") == "GPL96"


def test_keyword_extracts_multiple_platforms_for_compare():
    reg = build_registry(FakeApp())
    a = route("compare EGFR across GPL570 and GPL96", reg)
    assert a.tool == "compare_gene"
    assert a.params.get("gene") == "EGFR"
    assert a.params.get("platforms") == ["GPL570", "GPL96"]


def test_keyword_route_ngs_path():
    reg = build_registry(FakeApp())
    action = route("run deseq2 on data/counts.csv treated vs control", reg)
    assert action.tool == "run_ngs_de"
    assert action.params.get("counts_path") == "data/counts.csv"


def test_llm_route_valid_json(monkeypatch):
    reg = build_registry(FakeApp())
    import genevariate.core.ollama_manager as om
    monkeypatch.setattr(om, "ollama_server_ok", lambda *a, **k: True)
    monkeypatch.setattr(om, "model_available", lambda *a, **k: True)

    def fake_chat(messages, **kwargs):
        return ('Sure! {"tool": "rank_genes", "params": '
                '{"case_label": "a", "control_label": "b"}, "confidence": 0.9}')

    monkeypatch.setattr(
        "genevariate.core.llm_extractor.llm_backend.chat", fake_chat)
    action = route("rank the genes please", reg)
    assert action.tool == "rank_genes"
    assert action.source == "llm"
    assert action.params.get("case_label") == "a"


def test_llm_malformed_falls_back(monkeypatch):
    reg = build_registry(FakeApp())
    import genevariate.core.ollama_manager as om
    monkeypatch.setattr(om, "ollama_server_ok", lambda *a, **k: True)
    monkeypatch.setattr(om, "model_available", lambda *a, **k: True)
    monkeypatch.setattr(
        "genevariate.core.llm_extractor.llm_backend.chat",
        lambda messages, **kw: "no json here at all")
    # should not raise; falls back to keyword router
    action = route("list platforms", reg)
    assert isinstance(action, Action)
    assert action.tool == "list_platforms"


def test_condition_executor_runs():
    """The condition tool resolves + executes headlessly on a platform."""
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["condition_enrichment"]
    resolved = tool.resolver(app, {"platform": "GPLX"})
    resolved = tool.coerce(resolved)
    assert resolved["platform"] == "GPLX"
    assert resolved["condition_column"] == "Classified_condition"
    # execute with a no-op progress callback; ranking path needs no gseapy
    result = tool.executor(app, resolved, lambda v, t: None)
    assert result.ok
    assert "ranked" in result.payload
    assert not result.payload["ranked"].empty
    # condition tool now carries a markdown description/analysis
    assert result.report


def test_classify_distributions_tool_runs():
    """Modality-landscape tool: excludes Classified_* cols and reports."""
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["classify_distributions"]
    resolved = tool.coerce(tool.resolver(app, {"platform": "GPLX"}))
    result = tool.executor(app, resolved, lambda v, t: None)
    assert result.ok
    tags = result.payload["tags"]
    # only the five gene columns are classified, not the metadata columns
    assert set(tags.index) == {f"G{i}" for i in range(5)}
    assert result.report


def test_gsea_term_count_excludes_error_rows():
    from genevariate.core.chatbot.registry import _gsea_term_count
    assert _gsea_term_count(None) == 0
    assert _gsea_term_count(pd.DataFrame()) == 0
    # a gseapy failure frame carries an 'error' column — not real terms
    errs = pd.DataFrame({"library": ["A", "B"], "error": ["no overlap", "boom"]})
    assert _gsea_term_count(errs) == 0
    ok = pd.DataFrame({"Term": ["t1", "t2"], "NES": [1.0, -1.0]})
    assert _gsea_term_count(ok) == 2


def test_gene_distribution_tool_reports():
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["gene_distribution"]
    resolved = tool.coerce(tool.resolver(app, {"gene": "G0", "platform": "GPLX"}))
    result = tool.executor(app, resolved, lambda v, t: None)
    assert result.ok
    assert "G0" in result.report


def test_run_ngs_de_missing_file_is_graceful():
    """A non-existent count path must return ok=False, never raise."""
    app = FakeApp()
    reg = build_registry(app)
    tool = reg["run_ngs_de"]
    resolved = tool.coerce(tool.resolver(app, {
        "counts_path": "/tmp/gv_does_not_exist_1234.csv",
        "condition_column": "cond", "case_label": "a", "control_label": "b"}))
    result = tool.executor(app, resolved, lambda v, t: None)
    assert result.ok is False
    assert "not found" in result.summary.lower()


def test_planner_reads_gene_not_verb():
    """The offline planner must not read the word 'load' as gene 'LOAD'."""
    from genevariate.core.chatbot import agent as agent_mod
    app = FakeApp()
    reg = build_registry(app)
    plan = agent_mod.plan(
        "load GPL570 and GPL96 then compare TP53 across them", app, reg)
    tools = [s.tool for s in plan.steps]
    assert tools.count("load_geo_platform") == 2      # both GPLs loaded
    cmp = [s for s in plan.steps if s.tool == "compare_gene"]
    assert cmp and cmp[0].params.get("gene") == "TP53"
    # no step should carry a bogus gene extracted from an English verb
    for s in plan.steps:
        assert s.params.get("gene") in (None, "TP53")


def test_planner_routes_analytical_intent():
    """Offline goal for enrichment must plan the real analysis tool."""
    from genevariate.core.chatbot import agent as agent_mod
    app = FakeApp()
    reg = build_registry(app)
    plan = agent_mod.plan(
        "run condition enrichment on GPL570 tumor vs normal", app, reg)
    tools = [s.tool for s in plan.steps]
    assert "condition_enrichment" in tools
    ce = [s for s in plan.steps if s.tool == "condition_enrichment"][0]
    assert ce.params.get("case_label") == "tumor"
    assert ce.params.get("control_label") == "normal"
    # it should NOT degrade to profiling a bogus gene like 'CONDITION'
    assert "gene_distribution" not in tools
