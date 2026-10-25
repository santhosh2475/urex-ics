"""
Generate the seaborn diagnostic plots for one dataset: the scaling effect
on all 51 sensors, the episode-aware split timeline, a cleaned-data
preview, and per-model feature-importance / loss-curve / score plots.

    python src/make_visuals.py --dataset swat

Saves everything to results/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import visualize as viz
from config import RESULTS
from data import load, load_swat
from experiment import fit_and_score
from models import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat")
    ap.add_argument("--sensor", default="LIT101", help="sensor shown in the split timeline")
    ap.add_argument(
        "--models", nargs="+", default=["random_forest", "naive_bayes", "lstm_ae"],
        help="which registered models to generate feature-importance/loss/score plots for",
    )
    args = ap.parse_args()

    out_dir = RESULTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading raw + split data...")
    df_raw = load_swat()
    split = load(args.dataset)
    print(split.summary())

    # --- scaling effect: table + plot -------------------------------------
    summary = viz.scaling_summary(df_raw, split)
    summary.to_csv(out_dir / f"{args.dataset}-scaling-summary.csv", index=False)
    print(f"\nStandardScaler summary, all {len(summary)} sensors (first 10 shown):")
    print(summary.head(10).to_string(index=False))
    viz.plot_scaling_effect(summary, out_dir / f"{args.dataset}-scaling-effect.png")

    # --- cleaned (scaled) data, shown as an actual dataframe ---------------
    cleaned = pd.DataFrame(split.X_train[:8], columns=split.feature_names)
    cleaned.insert(0, "label", split.y_train[:8])
    cleaned.to_csv(out_dir / f"{args.dataset}-cleaned-preview.csv", index=False)
    print("\nCleaned (scaled) training data -- this is what every model actually sees, first 8 rows:")
    print(cleaned.round(3).to_string(index=False))

    # --- split timeline ------------------------------------------------------
    viz.plot_split_timeline(
        df_raw, split, args.sensor, out_dir / f"{args.dataset}-split-timeline.png"
    )

    # --- per-model plots -----------------------------------------------------
    for name in args.models:
        if name not in REGISTRY:
            print(f"  skipping unknown model {name!r}")
            continue
        print(f"\n--- {name} ---")
        detector = REGISTRY[name]()
        threshold, test_scores = fit_and_score(split, detector)

        if hasattr(getattr(detector, "model", None), "feature_importances_"):
            viz.plot_feature_importance(
                detector, split.feature_names, out_dir / f"{args.dataset}-{name}-importance.png"
            )
        if getattr(detector, "loss_history", None):
            viz.plot_loss_curve(detector, out_dir / f"{args.dataset}-{name}-loss.png")

        viz.plot_score_timeline(
            split.y_test, test_scores, threshold,
            out_dir / f"{args.dataset}-{name}-scores.png",
            f"{detector.name}: test score vs. real attacks (shaded)",
        )
        viz.plot_score_distribution(
            split.y_test, test_scores,
            out_dir / f"{args.dataset}-{name}-dist.png",
            f"{detector.name}: does the score separate normal from attack?",
        )

    print(f"\nAll figures + CSVs saved under {out_dir}")


if __name__ == "__main__":
    main()
