"""
Check the analyses against the truth planted by ``make_synthetic_platform.py``.

Every assertion here corresponds to a line in the generated GROUND_TRUTH.md, so
a failure names the specific claim that broke rather than a stack trace in the
middle of a plotting routine. Run it after regenerating the data, or whenever
the statistics change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genevariate.core.analysis.enrichment import benjamini_hochberg   # noqa: E402
from genevariate.core.analysis.overdispersion import (                # noqa: E402
    enrichment_diagnostics,
)

HI = 7.5

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, claim: str, detail: str = ""):
    _results.append((bool(ok), claim, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {claim}"
          + (f"\n          {detail}" if detail else ""))


def load(folder: Path):
    expr = pd.read_csv(folder / "GPL99999_synthetic_expression.csv.gz")
    lab = pd.read_csv(folder / "GPL99999_synthetic_labels.csv")
    df = expr.merge(lab, on="GSM", how="inner")
    if len(df) != len(expr):
        raise SystemExit("expression and labels do not line up on GSM")
    return df


# ── 1. single-gene enrichment, and the honesty layer ──────────────────────

def test_enrichment(df):
    print("\nEnrichment - GENE_C brushed high")
    reg = (df["GENE_C"] > HI).to_numpy()
    d = enrichment_diagnostics(reg, df["Tissue"], df["series_id"], seed=0)

    brain = d["Brain"]
    inside = brain["a"] / reg.sum()
    outside = brain["c"] / (~reg).sum()
    check(inside > 4 * outside,
          "Brain is enriched in the GENE_C slab",
          f"{inside:.1%} inside vs {outside:.1%} outside "
          f"(lift {inside / outside:.1f}x)")
    check(brain["ci_low"] > 1.0,
          "Brain's study-bootstrapped CI excludes lift 1",
          f"CI [{brain['ci_low']:.2f}, {brain['ci_high']:.2f}]")

    print("\nOverdispersion - Blood lives in 4 studies and no gene")
    blood = d["Blood"]
    check(blood["n_gse"] is not None and blood["n_gse"] <= 5,
          "Blood is reported as confined to a handful of studies",
          f"n_GSE = {blood['n_gse']}")
    check(blood["rho"] > 0.5,
          "Blood's clumping rho is high",
          f"rho = {blood['rho']:.3f}")
    check(blood["n_eff_sel"] < 0.5 * reg.sum(),
          "effective sample size is discounted, not the raw count",
          f"n_eff = {blood['n_eff_sel']:.1f} vs n = {int(reg.sum())}")

    print("\nOverdispersion - Brain is spread across many studies")
    check(brain["rho"] < blood["rho"],
          "a genuinely gene-driven label clumps less than a study artefact",
          f"Brain rho {brain['rho']:.3f} < Blood rho {blood['rho']:.3f}")


# ── 2. FDR negative control ───────────────────────────────────────────────

def test_fdr_negative_control(df):
    print("\nBH-FDR - Condition is noise and must not survive")
    from scipy.stats import fisher_exact

    pvals, names = [], []
    for gene in ["GENE_A", "GENE_B", "GENE_C", "XIST", "GENE_NOISE"]:
        reg = (df[gene] > HI).to_numpy()
        for val in ("tumor", "normal"):
            hit = (df["Condition"] == val).to_numpy()
            a = int((hit & reg).sum())
            c = int((hit & ~reg).sum())
            pvals.append(fisher_exact([[a, int(reg.sum()) - a],
                                       [c, int((~reg).sum()) - c]],
                                      alternative="greater")[1])
            names.append(f"{gene}/{val}")
    q = benjamini_hochberg(pvals)
    survivors = [f"{n} q={v:.3f}" for n, v in zip(names, q) if v < 0.05]
    check(not survivors,
          "no Condition association survives BH-FDR at q<0.05",
          f"{len(pvals)} tests, min q = {q.min():.3f}"
          + (f"; survivors: {survivors}" if survivors else ""))


# ── 3. box model ──────────────────────────────────────────────────────────

def test_box_model(df):
    from genevariate.core.analysis.box_model import (
        fit_label_model, integrate_box, relaxation_attribution,
    )

    print("\nBox model - Liver on (GENE_A, GENE_B)")
    X = df[["GENE_A", "GENE_B"]].to_numpy()
    y = (df["Tissue"] == "Liver").to_numpy()
    m = fit_label_model(X, y, df["series_id"], feature_names=["GENE_A", "GENE_B"])

    check(m.grouped and m.n_splits >= 2,
          "folds are split by study, not by sample",
          f"{m.n_splits} grouped folds")
    check(m.auc > 0.80,
          "cross-fitted AUC recovers the AND gate",
          f"AUC = {m.auc:.3f}")
    check(m.ece_cal <= m.ece_raw + 1e-9,
          "isotonic calibration does not make calibration worse",
          f"ECE {m.ece_raw:.4f} -> {m.ece_cal:.4f}")

    hot = integrate_box(m, [(HI, X[:, 0].max()), (HI, X[:, 1].max())], seed=0)
    cold = integrate_box(m, [(X[:, 0].min(), 5.0), (X[:, 1].min(), 5.0)], seed=0)
    check(hot["p_uniform"] > 0.5,
          "integrating the hot box approaches the planted rate of 0.85",
          f"p_uniform = {hot['p_uniform']:.3f} "
          f"(folds {hot['fold_low']:.3f}-{hot['fold_high']:.3f})")
    check(cold["p_uniform"] < 0.10,
          "the cold corner returns the planted background of 0.02",
          f"p_uniform = {cold['p_uniform']:.3f}")

    attr = relaxation_attribution(
        m, [(HI, X[:, 0].max()), (HI, X[:, 1].max())], seed=0)
    check(attr["GENE_A"]["drop"] > 0.15 and attr["GENE_B"]["drop"] > 0.15,
          "an AND gate cannot spare either constraint",
          f"GENE_A -{attr['GENE_A']['drop']:.3f}, "
          f"GENE_B -{attr['GENE_B']['drop']:.3f}")

    print("\nBox model - Brain on (GENE_C, GENE_NOISE)")
    X2 = df[["GENE_C", "GENE_NOISE"]].to_numpy()
    y2 = (df["Tissue"] == "Brain").to_numpy()
    m2 = fit_label_model(X2, y2, df["series_id"],
                         feature_names=["GENE_C", "GENE_NOISE"])
    attr2 = relaxation_attribution(
        m2, [(HI, X2[:, 0].max()), (HI, X2[:, 1].max())], seed=0)
    check(attr2["GENE_C"]["drop"] > 0.15,
          "GENE_C carries the Brain box",
          f"drop {attr2['GENE_C']['drop']:.3f}")
    check(abs(attr2["GENE_NOISE"]["drop"]) < 0.10,
          "GENE_NOISE is charged nothing",
          f"drop {attr2['GENE_NOISE']['drop']:+.3f}")


# ── 5. the clean single marker ────────────────────────────────────────────

def test_sex_marker(df):
    print("\nSex - XIST is a clean single-gene marker")
    reg = (df["XIST"] > HI).to_numpy()
    d = enrichment_diagnostics(reg, df["Sex"], df["series_id"], seed=0)
    fem = d["female"]
    frac = fem["a"] / reg.sum()
    check(frac > 0.95,
          "the XIST slab is almost entirely female",
          f"{frac:.1%} of {int(reg.sum())} samples")

    masks = {"XIST": reg, "GENE_NOISE": (df["GENE_NOISE"] > HI).to_numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--dir", default=str(Path.home() / "Desktop" /
                                               "genevariate_synthetic"))
    args = ap.parse_args()

    df = load(Path(args.dir))
    print(f"{len(df):,} samples / {df['series_id'].nunique()} studies / "
          f"{df['Tissue'].nunique()} tissue values")

    test_enrichment(df)
    test_fdr_negative_control(df)
    test_box_model(df)
    test_sex_marker(df)

    n_pass = sum(ok for ok, _, _ in _results)
    print(f"\n{'=' * 66}\n{n_pass}/{len(_results)} claims recovered")
    failed = [c for ok, c, _ in _results if not ok]
    if failed:
        print("FAILED:")
        for c in failed:
            print(f"  - {c}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
