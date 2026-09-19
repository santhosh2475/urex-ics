"""
Experiment runner.

One entry point for every result in the project, so that no number in the
final report was produced by an ad-hoc notebook cell nobody can reproduce.

    python src/experiment.py                                  # swat, all models
    python src/experiment.py --dataset swat --models xgboost lstm_ae random

The threshold is chosen on VALIDATION and applied unchanged to TEST.
Choosing it on test would be tuning on the test set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from config import RESULTS, THRESHOLD_PERCENTILE
from data import load
from evaluate import evaluate_all
from models import REGISTRY


def run_one(split, detector, threshold_percentile: float = THRESHOLD_PERCENTILE):
    """Fit one detector on one split and evaluate it on the test window."""
    print(f"  {detector.name} ...")
    detector.fit(split)

    val_scores = detector.score(split.X_val)
    threshold = float(np.percentile(val_scores, threshold_percentile))

    test_scores = detector.score(split.X_test)
    result = evaluate_all(
        split.y_test,
        test_scores,
        threshold=threshold,          # from validation, deliberately
        model=detector.name,
        dataset=split.name,
    )
    print(result.summary())
    print()
    return result


def run(split, model_names: list[str], tag: str = "") -> list:
    print(split.summary())
    print()

    results = []
    for name in model_names:
        if name not in REGISTRY:
            print(f"  skipping unknown model {name!r}")
            continue
        try:
            results.append(run_one(split, REGISTRY[name]()))
        except Exception as exc:  # keep going: one failure should not lose the rest
            print(f"  FAILED {name}: {type(exc).__name__}: {exc}\n")

    if results:
        _save(results, split.name, tag)
        _print_table(results)
    return results


def _save(results, dataset: str, tag: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{dataset}{'-' + tag if tag else ''}-{stamp}.json"
    payload = [
        {
            "dataset": r.meta.get("dataset"),
            "model": r.meta.get("model"),
            "threshold": r.threshold,
            "anomaly_rate": r.anomaly_rate,
            "roc_auc": r.roc_auc,
            "pr_auc": r.pr_auc,
            "pointwise": r.pointwise.as_dict(),
            "point_adjusted": r.point_adjusted.as_dict(),
            "f1_inflation": r.f1_inflation,
        }
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {path.relative_to(RESULTS.parent)}")


def _print_table(results) -> None:
    print()
    print(f"{'model':<26} {'PR-AUC':>8} {'F1 pw':>8} {'F1 adj':>8} {'inflation':>10}")
    print("-" * 64)
    for r in results:
        print(
            f"{r.meta.get('model', '?'):<26} {r.pr_auc:>8.3f} "
            f"{r.pointwise.f1:>8.3f} {r.point_adjusted.f1:>8.3f} "
            f"{r.f1_inflation:>+10.3f}"
        )
    print()
    print("Large inflation means the model is essentially guessing and getting")
    print("lucky once per attack window. Read the pointwise column first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat", help="registered dataset name")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["random", "xgboost", "xgboost_lags", "lstm_ae", "lstm_forecast"],
        help=f"any of: {', '.join(REGISTRY)}",
    )
    args = ap.parse_args()

    run(load(args.dataset), args.models)


if __name__ == "__main__":
    main()
