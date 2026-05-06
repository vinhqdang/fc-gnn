"""
Statistical significance testing and paper-ready results analysis.
Wilcoxon signed-rank test (non-parametric) across datasets.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare
from itertools import combinations


def load_results(path="results/raw_results.json"):
    with open(path) as f:
        return json.load(f)


def run_significance_tests(df, metric="macro_f1"):
    """
    Wilcoxon signed-rank test: FC-GNN vs each baseline across datasets.
    Returns DataFrame of p-values.
    """
    models = df["Model"].unique()
    datasets = df["Dataset"].unique()
    fc_vals = []
    results = {}

    fc_data = df[df["Model"] == "fc_gnn"][["Dataset", metric]].set_index("Dataset")

    for model in models:
        if model == "fc_gnn":
            continue
        model_data = df[df["Model"] == model][["Dataset", metric]].set_index("Dataset")
        common = fc_data.index.intersection(model_data.index)
        if len(common) < 3:
            continue
        fc_vals = fc_data.loc[common, metric].values
        bl_vals = model_data.loc[common, metric].values
        valid = ~(np.isnan(fc_vals) | np.isnan(bl_vals))
        if valid.sum() < 3:
            results[model] = {"p_value": np.nan, "significant": False, "fc_better": None}
            continue
        try:
            stat, p = wilcoxon(fc_vals[valid], bl_vals[valid], alternative="two-sided")
            fc_better = fc_vals[valid].mean() > bl_vals[valid].mean()
            results[model] = {"p_value": float(p), "significant": p < 0.05, "fc_better": fc_better}
        except Exception as e:
            results[model] = {"p_value": np.nan, "significant": False, "fc_better": None}

    return results


def print_paper_table(df):
    """Print LaTeX-formatted results table for the paper."""
    print("\n=== PAPER TABLE (Point Prediction) ===")
    print(r"\begin{tabular}{llrrrrr}")
    print(r"\hline")
    print(r"Dataset & Model & Accuracy & Macro-F1 & AUPRC & FAR & MCC \\")
    print(r"\hline")

    for ds in df["Dataset"].unique():
        ds_df = df[df["Dataset"] == ds].sort_values("Macro-F1", ascending=False)
        first = True
        for _, row in ds_df.iterrows():
            ds_label = ds if first else ""
            first = False
            bold = r"\textbf{" if row["Model"] == "fc_gnn" else ""
            endbold = r"}" if row["Model"] == "fc_gnn" else ""
            print(f"{ds_label} & {bold}{row['Model']}{endbold} & "
                  f"{bold}{row['Accuracy']:.4f}{endbold} & "
                  f"{bold}{row['Macro-F1']:.4f}{endbold} & "
                  f"{bold}{row['AUPRC']:.4f}{endbold} & "
                  f"{bold}{row['FAR']:.4f}{endbold} & "
                  f"{bold}{row['MCC']:.4f}{endbold} \\\\")
        print(r"\hline")
    print(r"\end{tabular}")

    print("\n=== PAPER TABLE (CP Metrics — CP models only) ===")
    cp_df = df[df["Coverage"].notna()]
    print(r"\begin{tabular}{llrrrr}")
    print(r"\hline")
    print(r"Dataset & Model & Coverage & APSS & SHP & Cov-Gap \\")
    print(r"\hline")
    for ds in cp_df["Dataset"].unique():
        ds_df = cp_df[cp_df["Dataset"] == ds].sort_values("APSS")
        first = True
        for _, row in ds_df.iterrows():
            ds_label = ds if first else ""
            first = False
            bold = r"\textbf{" if row["Model"] == "fc_gnn" else ""
            endbold = r"}" if row["Model"] == "fc_gnn" else ""
            print(f"{ds_label} & {bold}{row['Model']}{endbold} & "
                  f"{bold}{row['Coverage']:.4f}{endbold} & "
                  f"{bold}{row['APSS']:.4f}{endbold} & "
                  f"{bold}{row['SHP']:.4f}{endbold} & "
                  f"{bold}{row['Cov-Gap']:.4f}{endbold} \\\\")
        print(r"\hline")
    print(r"\end{tabular}")


def print_summary_stats(df):
    """Print summary statistics across all datasets."""
    print("\n=== SUMMARY: FC-GNN vs Baselines (averaged across all datasets) ===")

    cp_models = ["fc_gnn", "cfgnn", "daps", "rrgnn", "snaps"]
    cp_df = df[df["Model"].isin(cp_models)]

    metrics = ["Accuracy", "Macro-F1", "AUPRC", "Coverage", "APSS", "Cov-Gap"]
    summary = cp_df.groupby("Model")[metrics].mean()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Conditional Coverage Advantage ===")
    mondrian_models = ["fc_gnn", "rrgnn"]
    m_df = df[df["Model"].isin(mondrian_models) & df["Cov-Gap"].notna()]
    if not m_df.empty:
        avg_gap = m_df.groupby("Model")["Cov-Gap"].mean()
        print(f"Average coverage gap (lower is better):")
        for model, gap in avg_gap.items():
            print(f"  {model}: {gap:.4f}")

    print("\n=== FC-GNN Interpretability ===")
    interp_df = df[df["Model"] == "fc_gnn"][
        ["Dataset", "Rule-Fidelity", "Rule-Complexity", "Expl-Stability"]
    ].dropna()
    if not interp_df.empty:
        print(interp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nMean rule fidelity: {interp_df['Rule-Fidelity'].mean():.4f}")
        print(f"Mean expl. stability: {interp_df['Expl-Stability'].mean():.4f}")


def main():
    df = pd.read_csv("results/results_table.csv")
    df = df.rename(columns={"coverage": "Coverage", "apss": "APSS",
                              "shp": "SHP", "coverage_gap": "Cov-Gap",
                              "accuracy": "Accuracy", "macro_f1": "Macro-F1",
                              "auprc": "AUPRC", "far": "FAR", "mcc": "MCC",
                              "model": "Model", "dataset": "Dataset",
                              "rule_fidelity": "Rule-Fidelity",
                              "rule_complexity": "Rule-Complexity",
                              "explanation_stability": "Expl-Stability"})

    print_summary_stats(df)

    # Significance tests
    print("\n=== SIGNIFICANCE TESTS (FC-GNN vs baselines, Wilcoxon, Macro-F1) ===")
    sig_f1 = run_significance_tests(df, metric="Macro-F1")
    for model, res in sig_f1.items():
        direction = "FC-GNN >" if res["fc_better"] else "FC-GNN <"
        sig = "* p<0.05" if res["significant"] else "ns"
        print(f"  FC-GNN vs {model:8s}: p={res['p_value']:.4f} {sig} ({direction} baseline)")

    print("\n=== SIGNIFICANCE TESTS (FC-GNN vs baselines, Wilcoxon, Coverage) ===")
    sig_cov = run_significance_tests(df[df["Coverage"].notna()], metric="Coverage")
    for model, res in sig_cov.items():
        if res["p_value"] is np.nan:
            continue
        direction = "FC-GNN >" if res["fc_better"] else "FC-GNN <"
        sig = "* p<0.05" if res["significant"] else "ns"
        print(f"  FC-GNN vs {model:8s}: p={res['p_value']:.4f} {sig} ({direction} baseline)")

    print_paper_table(df)


if __name__ == "__main__":
    main()
