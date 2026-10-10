"""
Validation-based hyperparameter selection.

Every detector picks its decision THRESHOLD from validation scores
(experiment.py's run_one) -- but until this file, nothing picked a MODEL
hyperparameter from validation data. Every number baked into a detector's
__init__ (n_estimators=100, n_neighbors=5, SVM's default C=1.0, ...) was an
engineering judgment call, not something chosen because it scored better on
held-out data.

    python src/tune.py --dataset swat --grid 0.1 1 3 10 30

Scores each candidate on validation PR-AUC -- threshold-free, which matters
specifically on this dataset because SWaT's train/val/test anomaly rates
differ so much (0.93% / 22.95% / 3.76%) that any single fixed threshold is
an unreliable way to rank candidates -- then refits the winning
hyperparameter and reports it on test exactly the way experiment.py does
for every other model, so the tuned result is directly comparable to the
rest of results/swat-*.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from functools import partial

import numpy as np
from sklearn.metrics import average_precision_score

from config import RESULTS, THRESHOLD_PERCENTILE
from data import Split, load
from evaluate import evaluate_all
from models import SVMDetector


def grid_search(factory, param_grid: list[dict], split: Split, verbose: bool = True):
    """
    `factory(**params)` must return an unfit Detector -- the same
    `fit`/`score` contract every entry in models.REGISTRY already has, so
    this works for any detector, not just SVM, by passing a different
    factory and a grid over whichever of its constructor arguments you want
    to search.

    Each candidate is fit on split.X_train (the detector's own fit()
    decides how -- e.g. SVM's stratified subsampling runs unchanged) and
    scored on split.X_val via PR-AUC. Returns (best_result, every_result):
    the full grid is printed and returned, not just the winner, because a
    flat or noisy grid is itself something worth seeing.
    """
    results = []
    for params in param_grid:
        model = factory(**params).fit(split)
        val_scores = model.score(split.X_val)
        pr_auc = float(average_precision_score(split.y_val, val_scores))
        results.append({"params": params, "pr_auc": pr_auc, "model": model})
        if verbose:
            print(f"    {params} -> val PR-AUC {pr_auc:.4f}")

    best = max(results, key=lambda r: r["pr_auc"])
    if verbose:
        print(f"  best: {best['params']} (val PR-AUC {best['pr_auc']:.4f})")
    return best, results


def tune_svm_C(split: Split, C_grid: list[float]):
    param_grid = [{"C": c} for c in C_grid]
    print(f"  grid search: svm C in {C_grid}, scored on validation PR-AUC")
    return grid_search(SVMDetector, param_grid, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat")
    ap.add_argument(
        "--grid", nargs="+", type=float, default=[0.1, 1.0, 3.0, 10.0, 30.0],
        help="candidate values for SVM's C",
    )
    args = ap.parse_args()

    split = load(args.dataset)
    print(split.summary())
    print()

    best, all_results = tune_svm_C(split, args.grid)
    model = best["model"]
    chosen_C = best["params"]["C"]

    val_scores = model.score(split.X_val)
    threshold = float(np.percentile(val_scores, THRESHOLD_PERCENTILE))
    test_scores = model.score(split.X_test)
    result = evaluate_all(
        split.y_test, test_scores, threshold=threshold,
        model=f"svm(C={chosen_C},tuned)", dataset=split.name,
    )
    print()
    print("Final result on TEST with the validation-chosen C:")
    print(result.summary())

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{split.name}-svm-tuned-{stamp}.json"
    payload = [{
        "dataset": result.meta.get("dataset"),
        "model": result.meta.get("model"),
        "threshold": result.threshold,
        "anomaly_rate": result.anomaly_rate,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "pointwise": result.pointwise.as_dict(),
        "point_adjusted": result.point_adjusted.as_dict(),
        "f1_inflation": result.f1_inflation,
        "grid_search": [
            {"params": r["params"], "val_pr_auc": r["pr_auc"]} for r in all_results
        ],
    }]
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
