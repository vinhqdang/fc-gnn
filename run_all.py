"""
Master script: Run full FC-GNN evaluation on all 7 datasets.
Produces complete results table and figures for the paper.
Usage: python run_all.py [--quick] [--datasets D1 D2 ...]
"""

import argparse
import sys
from pathlib import Path
from fc_gnn.data.synthetic import ALL_DATASETS
from evaluate import run_evaluation, format_results_table, print_table
from fc_gnn.utils.visualization import plot_results, plot_coverage_gap


def main():
    parser = argparse.ArgumentParser(description="Run full FC-GNN evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 3 datasets, 40 epochs")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Datasets to evaluate")
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

    if args.quick:
        args.datasets = ALL_DATASETS[:3]
        args.epochs = 40
        print("Quick mode: 3 datasets, 40 epochs")

    print("\n" + "="*70)
    print("FC-GNN: Fuzzy-Conformal Graph Neural Networks")
    print("Cybersecurity Anomaly Detection — Full Evaluation")
    print("="*70)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Epochs={args.epochs} | Alpha={args.alpha} | Seed={args.seed}")
    print(f"Output: {args.output_dir}/")

    # Run evaluation
    all_results = run_evaluation(
        args.datasets, args.models,
        epochs=args.epochs, alpha=args.alpha,
        hidden=args.hidden, n_rules=args.n_rules,
        seed=args.seed, output_dir=args.output_dir
    )

    # Format and display table
    df = format_results_table(all_results, alpha=args.alpha)
    df.to_csv(f"{args.output_dir}/results_table.csv", index=False)
    print_table(df)

    # Generate figures
    try:
        # Convert for plotting
        plot_data = {}
        for ds, model_results in all_results.items():
            plot_data[ds] = {}
            for model, res in model_results.items():
                if "error" not in res:
                    plot_data[ds][model] = res
        if plot_data:
            plot_results(plot_data, output_dir=f"{args.output_dir}/figures")
            plot_coverage_gap(plot_data, output_dir=f"{args.output_dir}/figures")
            print(f"\nFigures saved to {args.output_dir}/figures/")
    except Exception as e:
        print(f"Warning: figure generation failed: {e}")

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Results saved to: {args.output_dir}/results_table.csv")
    return all_results


if __name__ == "__main__":
    main()
