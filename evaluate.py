"""
Full evaluation script: runs all models on all datasets, produces results table.
Usage: python evaluate.py [--datasets D1 D2] [--models M1 M2] [--epochs N]
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from fc_gnn.data.synthetic import ALL_DATASETS
from train import train_and_evaluate, MODELS, CP_MODELS


def run_evaluation(datasets, models, epochs=80, alpha=0.1, hidden=64,
                   n_rules=16, seed=42, output_dir="results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        all_results[dataset] = {}

        for model_name in models:
            print(f"\n  >> Model: {model_name}")
            try:
                res = train_and_evaluate(
                    dataset, model_name, epochs=epochs, alpha=alpha,
                    hidden=hidden, n_rules=n_rules, seed=seed, verbose=True
                )
                all_results[dataset][model_name] = res
                print(f"  Done. acc={res.get('accuracy', 0):.4f} "
                      f"f1={res.get('macro_f1', 0):.4f} "
                      f"coverage={res.get('coverage', '-')}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[dataset][model_name] = {"error": str(e)}

    return all_results


def format_results_table(all_results, alpha=0.1):
    """Format results into a pandas DataFrame for display."""
    rows = []
    for dataset, model_results in all_results.items():
        for model, res in model_results.items():
            if "error" in res:
                continue
            row = {
                "Dataset": dataset,
                "Model": model,
                "Accuracy": res.get("accuracy", np.nan),
                "Macro-F1": res.get("macro_f1", np.nan),
                "AUPRC": res.get("auprc", np.nan),
                "FAR": res.get("far", np.nan),
                "MCC": res.get("mcc", np.nan),
                "Coverage": res.get("coverage", np.nan),
                "APSS": res.get("apss", np.nan),
                "SHP": res.get("shp", np.nan),
                "Cov-Gap": res.get("coverage_gap", np.nan),
                "Time(s)": res.get("train_time_s", np.nan),
            }
            # Add interpretability if present
            if "rule_fidelity" in res:
                row["Rule-Fidelity"] = res["rule_fidelity"]
                row["Rule-Complexity"] = res["rule_complexity"]
                row["Expl-Stability"] = res["explanation_stability"]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_table(df):
    """Print formatted results table."""
    if df.empty:
        print("No results to display.")
        return

    float_cols = [c for c in df.columns if df[c].dtype == float]
    formatters = {c: lambda x: f"{x:.4f}" if not np.isnan(x) else "  -  "
                  for c in float_cols}

    print("\n" + "="*120)
    print("RESULTS TABLE")
    print("="*120)
    print(df.to_string(index=False, formatters=formatters))
    print("="*120)

    # Summary: FC-GNN vs best baseline on CP metrics
    if "fc_gnn" in df["Model"].values and "Coverage" in df.columns:
        fc_rows = df[df["Model"] == "fc_gnn"]
        baseline_rows = df[df["Model"] != "fc_gnn"]

        print("\nFC-GNN vs Best Baseline Summary:")
        for dataset in df["Dataset"].unique():
            fc = df[(df["Dataset"] == dataset) & (df["Model"] == "fc_gnn")]
            baselines = df[(df["Dataset"] == dataset) & (df["Model"] != "fc_gnn")]
            if fc.empty or baselines.empty:
                continue
            fc_apss = fc["APSS"].values[0] if not fc["APSS"].isna().all() else np.nan
            best_baseline_apss = baselines["APSS"].min()
            if not np.isnan(fc_apss) and not np.isnan(best_baseline_apss):
                gain = (best_baseline_apss - fc_apss) / (best_baseline_apss + 1e-8) * 100
                print(f"  {dataset}: APSS={fc_apss:.3f} vs best-baseline={best_baseline_apss:.3f} "
                      f"(efficiency gain: {gain:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS[:3])
    parser.add_argument("--models", nargs="+",
                        default=["fc_gnn", "cfgnn", "daps", "rrgnn", "snaps",
                                 "flgnn", "fgat", "gcn", "sage"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_rules", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    print(f"Running evaluation on datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs} | Alpha: {args.alpha} | Seed: {args.seed}")

    all_results = run_evaluation(
        args.datasets, args.models, epochs=args.epochs, alpha=args.alpha,
        hidden=args.hidden, n_rules=args.n_rules, seed=args.seed,
        output_dir=args.output_dir
    )

    # Save raw results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/raw_results.json", "w") as f:
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, default=convert, indent=2)

    df = format_results_table(all_results, alpha=args.alpha)
    df.to_csv(f"{args.output_dir}/results_table.csv", index=False)
    print_table(df)
    print(f"\nResults saved to {args.output_dir}/")
