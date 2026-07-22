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
                 "variability_enrichment", "rank_genes",
                 "classify_distributions", "meta_enrichment",
                 "gene_distribution", "compare_gene",
                 "compare_modalities", "gene_connections",
                 "activity_inference", "run_analysis_code"):
        assert name in reg


def test_run_analysis_code_tool_executes_snippet():
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["run_analysis_code"]
    code = ("means = platforms['GPLX'][['G0','G1']].mean()\n"
            "print('n_samples', platforms['GPLX'].shape[0])\n"
            "result = float(means['G0'])")
    resolved = tool.coerce(tool.resolver(app, {"code": code}))
    res = tool.executor(app, resolved, lambda v, t: None)
    assert res.ok
    assert "n_samples 6" in res.payload["stdout"]
    assert isinstance(res.payload["result"], float)


def test_run_analysis_code_blocks_imports_and_dunder():
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["run_analysis_code"]
    for bad in ("import os\nresult = 1",
                "result = (1).__class__.__bases__",
                "result = open('/etc/passwd').read()"):
        resolved = tool.coerce(tool.resolver(app, {"code": bad}))
        res = tool.executor(app, resolved, lambda v, t: None)
        assert res.ok is False
        assert "blocked" in res.summary.lower() or "not run" in res.summary.lower()


def test_run_analysis_code_reports_runtime_error():
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["run_analysis_code"]
    resolved = tool.coerce(tool.resolver(app, {"code": "result = 1/0"}))
    res = tool.executor(app, resolved, lambda v, t: None)
    assert res.ok is False
    assert "ZeroDivisionError" in res.summary


def test_save_learned_tool_roundtrip(tmp_path):
    """Agent can promote a snippet into a persisted named tool that then
    appears in a rebuilt registry and runs with typed params."""
    app = FakeApp({"GPLX": _platform_df()})
    app.data_dir = str(tmp_path)
    reg = build_registry(app)
    assert "save_learned_tool" in reg

    save = reg["save_learned_tool"]
    code = ("g = params.get('gene', 'G0')\n"
            "df = list(platforms.values())[0]\n"
            "result = float(df[g].mean())")
    resolved = save.resolver(app, {
        "name": "Gene Mean!", "description": "mean of a gene", "code": code,
        "params": '[{"name":"gene","type":"str","required":false,"default":"G0"}]',
        "examples": "mean of a gene, average gene value"})
    res = save.executor(app, resolved, lambda v, t: None)
    assert res.ok
    assert res.payload.get("_registry_dirty") is True

    # rebuilt registry now exposes the learned tool (sanitized name)
    reg2 = build_registry(app)
    assert "gene_mean" in reg2
    lt = reg2["gene_mean"]
    assert lt.description.startswith("[learned]")
    out = lt.executor(app, lt.resolver(app, {"gene": "G1"}), lambda v, t: None)
    assert out.ok
    assert isinstance(out.payload.get("result"), float)


def test_save_learned_tool_refuses_unsafe_code(tmp_path):
    app = FakeApp({"GPLX": _platform_df()})
    app.data_dir = str(tmp_path)
    reg = build_registry(app)
    save = reg["save_learned_tool"]
    res = save.executor(app, save.resolver(
        app, {"name": "evil", "code": "import os\nresult = 1"}),
        lambda v, t: None)
    assert res.ok is False
    assert "unsafe" in res.summary.lower() or "import" in res.summary.lower()
    # nothing persisted -> not present in a rebuilt registry
    assert "evil" not in build_registry(app)


def _coexpr_platform(seed, n=40, scale=1.0, offset=7.0):
    rng = np.random.default_rng(seed)
    tp53 = rng.normal(offset, scale, n)
    return pd.DataFrame({
        "GSM": [f"GSM{seed}_{i}" for i in range(n)],
        "Classified_condition": ["a"] * n,
        "TP53": tp53,
        "MDM2": tp53 * 0.9 + rng.normal(0, scale * 0.2, n),
        "NOISE": rng.normal(0, 1, n),
    })


def test_compare_modalities_tool_runs():
    app = FakeApp({"GPL570": _coexpr_platform(1, scale=1.0, offset=7.0),
                   "RNAseq_x": _coexpr_platform(2, scale=3.0, offset=2.0)})
    reg = build_registry(app)
    tool = reg["compare_modalities"]
    resolved = tool.coerce(tool.resolver(app, {"gene": "TP53"}))
    assert resolved["method"] == "zscore"
    result = tool.executor(app, resolved, lambda v, t: None)
    assert result.ok
    assert set(result.table["modality"]) == {"microarray", "rna-seq"}
    assert result.report


def test_gene_connections_single_and_consensus():
    app = FakeApp({"GPL570": _coexpr_platform(1, scale=1.0, offset=7.0),
                   "RNAseq_x": _coexpr_platform(2, scale=3.0, offset=2.0)})
    reg = build_registry(app)
    tool = reg["gene_connections"]
    # single source
    one = tool.executor(app, tool.coerce(tool.resolver(
        app, {"gene": "TP53", "platforms": ["GPL570"]})), lambda v, t: None)
    assert one.ok and "MDM2" in one.table.index
    # cross-modality consensus (both sources)
    both = tool.executor(app, tool.coerce(tool.resolver(
        app, {"gene": "TP53"})), lambda v, t: None)
    assert both.ok and "MDM2" in both.table.index
    assert "NOISE" not in both.table.index


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


@pytest.mark.parametrize("prompt, expected", [
    # list / status
    ("what platforms are loaded", "list_platforms"),
    ("list my datasets", "list_platforms"),
    ("which datasets do I have", "list_platforms"),
    # enrichment family
    ("run condition enrichment on GPL570 tumor vs normal", "condition_enrichment"),
    ("condition enrichment tumor versus normal", "condition_enrichment"),
    ("run variability enrichment on GPL96", "variability_enrichment"),
    ("which genes are most variable and enriched", "variability_enrichment"),
    ("run meta enrichment across GPL570 and GPL96 tumor vs normal", "meta_enrichment"),
    ("combine enrichment across platforms tumor vs normal", "meta_enrichment"),
    # ranking
    ("rank genes tumor vs normal", "rank_genes"),
    ("top differential genes on GPL570", "rank_genes"),
    ("show me the most differentially expressed genes", "rank_genes"),
    # single-gene distribution vs whole-platform classification
    ("analyze the distribution of TP53", "gene_distribution"),
    ("show TP53 distribution on GPL570", "gene_distribution"),
    ("plot histogram of EGFR", "gene_distribution"),
    ("is BRCA1 bimodal", "gene_distribution"),
    ("classify the gene distributions on GPL570", "classify_distributions"),
    ("what distribution shapes are in GPL96", "classify_distributions"),
    # compare (plain) vs modalities (harmonised)
    ("compare TP53 across GPL570 and GPL96", "compare_gene"),
    ("how does EGFR differ across platforms", "compare_gene"),
    ("side by side MYC on GPL570 and GPL96", "compare_gene"),
    ("compare TP53 across microarray and rna-seq modalities", "compare_modalities"),
    ("harmonize TP53 across modalities with z-score", "compare_modalities"),
    ("batch correct and compare MYC across platforms", "compare_modalities"),
    # connections
    ("what genes are connected to TP53 on GPL570", "gene_connections"),
    ("co-expression network for EGFR", "gene_connections"),
    ("which genes correlate with BRCA1", "gene_connections"),
    # loading
    ("load GPL570", "load_geo_platform"),
    ("load the GEO platform GPL96", "load_geo_platform"),
    ("download dataset GPL571", "load_geo_platform"),
    # activity + single-cell
    ("infer TF activity on GPL570", "activity_inference"),
    ("transcription factor activity for GPL96", "activity_inference"),
    ("fetch single cell data for TP53 in lung", "fetch_single_cell"),
    ("get single-cell expression of EGFR in brain", "fetch_single_cell"),
])
def test_keyword_router_prompt_battery(prompt, expected):
    """Broad natural-language battery on the deterministic (ollama-off) router."""
    reg = build_registry(FakeApp({"GPL570": _platform_df(),
                                  "GPL96": _platform_df()}))
    assert route(prompt, reg).tool == expected


def test_strong_intent_overrides_bag_of_words():
    """The high-precision deterministic overrides fire on their target shapes."""
    reg = build_registry(FakeApp({"GPL570": _platform_df(),
                                  "GPL96": _platform_df()}))
    # load verb + GPL id must never fall through to no-tool
    assert route("download dataset GPL571", reg).tool == "load_geo_platform"
    # a single gene + shape word is a profile, not a modality comparison
    assert route("is BRCA1 bimodal", reg).tool == "gene_distribution"
    # side-by-side of one gene across 2 GPLs is a plain compare, not modalities
    assert route("side by side MYC on GPL570 and GPL96", reg).tool == "compare_gene"
    # a fetch verb WITHOUT a GPL id (single-cell) is not a platform load
    assert route("fetch single cell data for TP53 in lung", reg).tool \
        == "fetch_single_cell"


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


def test_find_leaked_tool_call_parses_llama_native_format():
    """Llama sometimes writes a tool call as text; we must recover it."""
    from genevariate.core.chatbot.langchain_agent import _find_leaked_tool_call
    # both '/' and '=' separators, with surrounding prose
    a = _find_leaked_tool_call(
        'Sure. <function/condition_enrichment>{"platform": "GPL570", '
        '"case_label": "treated", "control_label": "control"}</function>')
    assert a is not None
    name, params = a
    assert name == "condition_enrichment"
    assert params["platform"] == "GPL570"
    b = _find_leaked_tool_call(
        '<function=list_platforms>{}</function>')
    assert b is not None and b[0] == "list_platforms" and b[1] == {}
    # plain text / malformed JSON must not match
    assert _find_leaked_tool_call("no function call here") is None
    assert _find_leaked_tool_call(
        "<function/x>{not json}</function>") is None


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


def test_charts_descriptor_is_data_grounded():
    """describe_values extracts the numbers a reader takes off a chart."""
    import numpy as np
    from genevariate.core.chatbot import charts
    v = np.concatenate([np.full(50, 2.0), np.full(50, 8.0)])  # clear two modes
    d = charts.describe_values(v, dist_class="Bimodal")
    assert d["n"] == 100
    assert d["n_modes"] == 2
    assert abs(d["mean"] - 5.0) < 1e-6
    block = charts.describe_distribution_block(d)
    assert "What the chart shows" in block and "two peaks" in block


def test_gene_distribution_emits_figure_and_chart_block():
    import matplotlib
    matplotlib.use("Agg")
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["gene_distribution"]
    resolved = tool.coerce(tool.resolver(app, {"gene": "G0", "platform": "GPLX"}))
    res = tool.executor(app, resolved, lambda v, t: None)
    assert res.ok
    assert res.figure is not None                       # embeddable plot
    assert "What the chart shows" in res.report          # LLM-narratable descriptor
    assert "chart" in res.payload and res.payload["chart"]["n"] == 6


def test_enrichment_bar_reports_top_genes():
    import matplotlib
    matplotlib.use("Agg")
    app = FakeApp({"GPLX": _platform_df()})
    reg = build_registry(app)
    tool = reg["rank_genes"]
    resolved = tool.coerce(tool.resolver(
        app, {"platform": "GPLX", "case_label": "tumor",
              "control_label": "normal"}))
    res = tool.executor(app, resolved, lambda v, t: None)
    assert res.ok
    assert res.figure is not None
    assert "What the chart shows" in res.report
    assert res.payload["chart"]["top"]
