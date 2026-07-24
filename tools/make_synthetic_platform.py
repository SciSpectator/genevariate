"""
Generate a synthetic platform whose answers are known in advance.

The point of this file is not to make plausible-looking data. It is to plant a
handful of specific structures - an AND gate, a single-gene marker, a label that
lives in four studies and nowhere else, and a gene that means nothing - so that
every analysis in the app can be checked against an answer that was decided
before the analysis ran. A test that only says "it produced a number" is not a
test.

Writes three files into the output directory:

``GPL99999_synthetic_expression.csv.gz``  GSM, series_id, five gene columns
``GPL99999_synthetic_labels.csv``         GSM, Tissue, Sex, Condition
``GROUND_TRUTH.md``                       what each analysis should recover

Run ``validate_synthetic.py`` afterwards to check the claims automatically, or
load the two CSVs in the GUI and read them off the tabs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_STUDIES = 80
STUDY_MIN, STUDY_MAX = 12, 60
GENES = ["GENE_A", "GENE_B", "GENE_C", "XIST", "GENE_NOISE"]

# "high" cut for every gene; the generator plants labels against this exact value
HI = 7.5

# how many studies the clumped label is confined to
N_BLOOD_STUDIES = 4


def build(seed: int = 0):
    rng = np.random.default_rng(seed)

    sizes = rng.integers(STUDY_MIN, STUDY_MAX + 1, size=N_STUDIES)
    series = np.repeat([f"GSE9{i:04d}" for i in range(N_STUDIES)], sizes)
    n = int(sizes.sum())
    gsm = np.array([f"GSM{900000 + i}" for i in range(n)])

    # Per-study offsets are what make rho non-zero: two samples from the same
    # study are more alike than two samples picked at random, which is the whole
    # reason the honesty layer exists.
    study_idx = np.repeat(np.arange(N_STUDIES), sizes)
    batch = rng.normal(0.0, 0.8, size=(N_STUDIES, len(GENES)))[study_idx]
    expr = rng.normal(6.0, 2.0, size=(n, len(GENES))) + batch
    E = {g: expr[:, i] for i, g in enumerate(GENES)}

    hi_a = E["GENE_A"] > HI
    hi_b = E["GENE_B"] > HI
    hi_c = E["GENE_C"] > HI
    hi_x = E["XIST"] > HI

    # ── Tissue ────────────────────────────────────────────────────────────
    tissue = np.full(n, "Other", dtype=object)

    # Brain: one gene is enough (planted 0.80, background 0.02)
    p_brain = np.where(hi_c, 0.80, 0.02)
    tissue[rng.random(n) < p_brain] = "Brain"

    # Liver: an AND gate (planted 0.85, background 0.02). Written after Brain so
    # the conjunction wins where the two overlap.
    p_liver = np.where(hi_a & hi_b, 0.85, 0.02)
    tissue[rng.random(n) < p_liver] = "Liver"

    # Blood: confined to four studies and driven by no gene at all. This is the
    # clumping control - counted naively it looks like a real finding.
    blood_studies = rng.choice(N_STUDIES, size=N_BLOOD_STUDIES, replace=False)
    blood = np.isin(study_idx, blood_studies)
    tissue[blood] = "Blood"

    # ── Sex: a clean single marker, 99% faithful ──────────────────────────
    sex = np.where(hi_x, "female", "male").astype(object)
    flip = rng.random(n) < 0.01
    sex[flip] = np.where(sex[flip] == "female", "male", "female")

    # ── Condition: pure noise, the negative control for FDR ───────────────
    condition = np.where(rng.random(n) < 0.4, "tumor", "normal").astype(object)

    expr_df = pd.DataFrame({"GSM": gsm, "series_id": series})
    for g in GENES:
        expr_df[g] = np.round(E[g], 4)

    labels_df = pd.DataFrame({"GSM": gsm, "Tissue": tissue,
                              "Sex": sex, "Condition": condition})
    return expr_df, labels_df, blood_studies


def truth_markdown(expr_df, labels_df, blood_studies) -> str:
    n = len(expr_df)
    hi_a = expr_df["GENE_A"] > HI
    hi_b = expr_df["GENE_B"] > HI
    hi_c = expr_df["GENE_C"] > HI
    t = labels_df["Tissue"]
    box_ab = hi_a & hi_b

    def pct(x):
        return f"{100 * float(x):.1f}%"

    lines = [
        "# Ground truth for the synthetic platform",
        "",
        f"{n:,} samples across {expr_df['series_id'].nunique()} studies, "
        f"five genes, cut at **{HI}** for every gene.",
        "",
        "Load `GPL99999_synthetic_expression.csv.gz` as the platform and "
        "`GPL99999_synthetic_labels.csv` as the labels, then brush each gene "
        f"from {HI} upward.",
        "",
        "## What was planted",
        "",
        "| Label | Rule | Planted rate | Background |",
        "|---|---|---|---|",
        "| Tissue = Liver | GENE_A high **AND** GENE_B high | 0.85 | 0.02 |",
        "| Tissue = Brain | GENE_C high (one gene, no partner) | 0.80 | 0.02 |",
        "| Tissue = Blood | confined to 4 studies, **no gene involved** | - | - |",
        "| Sex = female | XIST high | 0.99 | 0.01 |",
        "| Condition | nothing at all | - | - |",
        "",
        "GENE_NOISE is independent of every label and is never part of any rule.",
        "",
        "## What each tab must report",
        "",
        "### Enrichment (single gene)",
        f"- Brushing **GENE_C > {HI}** must show **Brain** enriched, "
        f"observed {pct((t[hi_c] == 'Brain').mean())} inside vs "
        f"{pct((t[~hi_c] == 'Brain').mean())} outside.",
        "- **Condition** must survive nothing: every tumor/normal q-value should "
        "fail BH-FDR. If it passes, the FDR column is wrong.",
        f"- **Blood** must show a high raw count but `n_GSE = {len(blood_studies)}`, "
        "`rho` near 1 and `n_eff` collapsed to roughly the number of studies, "
        "not the number of samples. This is the honesty layer's whole job.",
        "",
        "### Gene Synergy (GENE_A AND GENE_B)",
        f"- Box holds {int(box_ab.sum()):,} samples, of which "
        f"{int((box_ab & (t == 'Liver')).sum()):,} are Liver "
        f"({pct((t[box_ab] == 'Liver').mean())}).",
        "- **Liver** must come back with synergy odds ratio clearly above 1 and a "
        "bootstrap CI whose lower bound stays above 1 - the conjunction beats the "
        "product of the two marginal lifts.",
        "- **Brain** must come back with synergy well **below** 1. A sample "
        "carries one tissue, so inside the A-and-B corner the Liver rule "
        "displaces Brain - the interaction term has to report that antagonism, "
        "not just fail to find synergy.",
        "",
        "### Gene Synergy (GENE_C AND GENE_NOISE)",
        "- **Brain** enrichment must survive, but the synergy term must sit near 1. "
        "A conjunction with a meaningless gene is not a conjunction.",
        "",
        "### Box Model (GENE_A, GENE_B)",
        "- Cross-fitted AUC for Liver should land around 0.85-0.92, and the "
        "calibration error must not get worse after the isotonic layer.",
        f"- Integrating the box [{HI}, max] x [{HI}, max] should return "
        "`p_uniform` in the neighbourhood of the planted 0.85.",
        "- Integrating the cold corner (both genes below 5) should return "
        "well under 0.10, near the planted background of 0.02.",
        "- Relaxation attribution must charge **both** GENE_A and GENE_B a large "
        "drop. An AND gate cannot spare either one.",
        "",
        "### Box Model (GENE_C, GENE_NOISE)",
        "- GENE_C must carry a large drop; **GENE_NOISE must be near zero**. "
        "This is the false-positive check on attribution.",
        "",
        "### Sex",
        f"- Brushing XIST > {HI} must recover female at roughly "
        f"{pct((labels_df.loc[expr_df['XIST'] > HI, 'Sex'] == 'female').mean())}, "
        "with no synergy against any other gene.",
        "",
        "## Studies holding Blood",
        "",
        "`" + ", ".join(sorted(f"GSE9{i:04d}" for i in blood_studies)) + "`",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(Path.home() / "Desktop" /
                                               "genevariate_synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    expr_df, labels_df, blood_studies = build(args.seed)

    expr_path = out / "GPL99999_synthetic_expression.csv.gz"
    lab_path = out / "GPL99999_synthetic_labels.csv"
    expr_df.to_csv(expr_path, index=False, compression="gzip")
    labels_df.to_csv(lab_path, index=False)
    (out / "GROUND_TRUTH.md").write_text(
        truth_markdown(expr_df, labels_df, blood_studies))

    print(f"{len(expr_df):,} samples / {expr_df['series_id'].nunique()} studies")
    print(f"  expression -> {expr_path}")
    print(f"  labels     -> {lab_path}")
    print(f"  truth      -> {out / 'GROUND_TRUTH.md'}")
    print()
    print(labels_df["Tissue"].value_counts().to_string())


if __name__ == "__main__":
    main()
