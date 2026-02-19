#!/usr/bin/env python3
"""
Load benchmark CSV(s) from one or more C++ autodiff engines and plot speed/time comparisons.
Usage:
  python plot_benchmarks.py [file1.csv file2.csv ...]
  If no files given, uses benchmark_results.csv in the project root.
Output: Saves graphs under benchmarks/plots/
"""
from pathlib import Path
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV(s)")
    parser.add_argument("csv_files", nargs="*", default=["benchmark_results.csv"],
                        help="CSV file(s) (same schema with 'engine' column for comparison)")
    parser.add_argument("-o", "--out-dir", default="benchmarks/plots",
                        help="Output directory for plots")
    parser.add_argument("--no-pandas", action="store_true", help="Use only stdlib (no pandas)")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        if not args.no_pandas:
            print("Install pandas and matplotlib: pip install pandas matplotlib", file=sys.stderr)
            sys.exit(1)
        pd = None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Resolve paths: look in project root if path is just filename
    root = Path(__file__).resolve().parent.parent
    csv_paths = []
    for p in args.csv_files:
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            csv_paths.append(path)
        else:
            print(f"Warning: not found {path}", file=sys.stderr)

    if not csv_paths:
        print("No CSV files found. Run the C++ benchmark first to create benchmark_results.csv", file=sys.stderr)
        sys.exit(1)

    # Load and merge
    dfs = []
    for p in csv_paths:
        d = pd.read_csv(p)
        if "engine" not in d.columns:
            d["engine"] = p.stem
        if "dataset" in d.columns:
            d["dataset"] = d["dataset"].astype(str).str.strip()
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip()
    if "samples" in df.columns:
        df["samples"] = pd.to_numeric(df["samples"], errors="coerce").fillna(0).astype(int)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = df["engine"].unique().tolist()
    colors = plt.cm.tab10.colors[: max(len(engines), 1)]
    engine_color = {e: colors[i % len(colors)] for i, e in enumerate(engines)}

    # 1) Training time vs samples (regression sweep: reg_n*)
    reg = df[df["dataset"].apply(lambda x: str(x).startswith("reg_n"))].copy()
    if not reg.empty:
        reg = reg.sort_values("samples")
        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in reg["engine"].unique():
            sub = reg[reg["engine"] == engine]
            ax.plot(sub["samples"], sub["train_sec"], "o-", label=engine, color=engine_color.get(engine, "gray"))
        ax.set_xlabel("Samples")
        ax.set_ylabel("Training time (s)")
        ax.set_title("Regression: training time vs dataset size (10 feat, 30 epochs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "train_sec_vs_samples_regression.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / 'train_sec_vs_samples_regression.png'}")

    # 2) Training time vs samples (classification sweep: clf_n*)
    clf = df[df["dataset"].apply(lambda x: str(x).startswith("clf_n"))].copy()
    if not clf.empty:
        clf = clf.sort_values("samples")
        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in clf["engine"].unique():
            sub = clf[clf["engine"] == engine]
            ax.plot(sub["samples"], sub["train_sec"], "s-", label=engine, color=engine_color.get(engine, "gray"))
        ax.set_xlabel("Samples")
        ax.set_ylabel("Training time (s)")
        ax.set_title("Classification: training time vs dataset size (5 feat, 3 classes, 30 epochs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "train_sec_vs_samples_classification.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / 'train_sec_vs_samples_classification.png'}")

    # 3) Time per epoch (ms) by dataset – bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = list(df["dataset"].unique())
    n_ds = len(datasets)
    n_eng = max(len(engines), 1)
    x = range(n_ds)
    w = 0.8 / n_eng
    for i, engine in enumerate(engines):
        sub = df[df["engine"] == engine]
        vals = [sub[sub["dataset"] == d]["epoch_ms"].values[0] if len(sub[sub["dataset"] == d]) else 0 for d in datasets]
        ax.bar([xi + i * w for xi in x], vals, width=w, label=engine, color=engine_color.get(engine, "gray"))
    ax.set_xticks([xi + (n_eng - 1) * w / 2 for xi in x])
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Time per epoch (ms)")
    ax.set_title("Time per epoch by dataset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_ms_by_dataset.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'epoch_ms_by_dataset.png'}")

    # 4) Samples per second by dataset
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, engine in enumerate(engines):
        sub = df[df["engine"] == engine]
        vals = [sub[sub["dataset"] == d]["samples_per_sec"].values[0] if len(sub[sub["dataset"] == d]) else 0 for d in datasets]
        ax.bar([xi + i * w for xi in x], vals, width=w, label=engine, color=engine_color.get(engine, "gray"))
    ax.set_xticks([xi + (n_eng - 1) * w / 2 for xi in x])
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Samples / sec")
    ax.set_title("Throughput by dataset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "samples_per_sec_by_dataset.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'samples_per_sec_by_dataset.png'}")

    # 5) Total training time (s) by dataset
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, engine in enumerate(engines):
        sub = df[df["engine"] == engine]
        vals = [sub[sub["dataset"] == d]["train_sec"].values[0] if len(sub[sub["dataset"] == d]) else 0 for d in datasets]
        ax.bar([xi + i * w for xi in x], vals, width=w, label=engine, color=engine_color.get(engine, "gray"))
    ax.set_xticks([xi + (n_eng - 1) * w / 2 for xi in x])
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Total training time by dataset")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "train_sec_by_dataset.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'train_sec_by_dataset.png'}")

    # 6) Final loss by dataset (where applicable)
    has_loss = "final_loss" in df.columns and df["final_loss"].notna().any()
    if has_loss:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, engine in enumerate(engines):
            sub = df[df["engine"] == engine]
            vals = [sub[sub["dataset"] == d]["final_loss"].values[0] if len(sub[sub["dataset"] == d]) else 0 for d in datasets]
            ax.bar([xi + i * w for xi in x], vals, width=w, label=engine, color=engine_color.get(engine, "gray"))
        ax.set_xticks([xi + (n_eng - 1) * w / 2 for xi in x])
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel("Final loss")
        ax.set_title("Final loss by dataset")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "final_loss_by_dataset.png", dpi=150)
        plt.close()
        print(f"Saved {out_dir / 'final_loss_by_dataset.png'}")

    # 7) Accuracy by dataset (classification)
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        clf_ds = [d for d in datasets if df[df["dataset"] == d]["accuracy"].notna().any()]
        if clf_ds:
            fig, ax = plt.subplots(figsize=(10, 5))
            xx = range(len(clf_ds))
            ww = 0.8 / max(len(engines), 1)
            for i, engine in enumerate(engines):
                sub = df[df["engine"] == engine]
                vals = [sub[sub["dataset"] == d]["accuracy"].values[0] * 100 if len(sub[sub["dataset"] == d]) else 0 for d in clf_ds]
                ax.bar([xi + i * ww for xi in xx], vals, width=ww, label=engine, color=engine_color.get(engine, "gray"))
            ax.set_xticks([xi + (len(engines) - 1) * ww / 2 for xi in xx])
            ax.set_xticklabels(clf_ds, rotation=45, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy by dataset (classification)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out_dir / "accuracy_by_dataset.png", dpi=150)
            plt.close()
            print(f"Saved {out_dir / 'accuracy_by_dataset.png'}")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
