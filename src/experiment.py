"""
Experiment runner.

One entry point for every result in the project, so that no number in the
final report was produced by an ad-hoc notebook cell nobody can reproduce.

    python src/experiment.py --protocol a1a2 [protocol.py options]   # PRIMARY
    python src/experiment.py --protocol chronological|stratified     # secondary

    python src/experiment.py                                  # swat, all models
    python src/experiment.py --dataset swat --models xgboost lstm_ae random

The threshold is chosen on VALIDATION and applied unchanged to TEST.
Choosing it on test would be tuning on the test set.

The percentile is taken over validation's NORMAL-labelled rows only, not
every row in validation. A percentile taken over a validation window that
is itself partly attacks (SWaT's is 22.95% attack) sits artificially high
-- "the top 3% most unusual score, including a quarter of rows that are
genuinely unusual" is a different, stricter number than "the top 3% most
unusual score among known-normal operation," which is the actual question
a deployment threshold is supposed to answer. Using every row silently
conflated the two and was plausibly why so many models landed at test
F1 = 0.000 this session.
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


def score_part(detector, split, part: str) -> np.ndarray:
    """
    Score one part block by block, in time order, and concatenate.

    A chronological split has one block per part, so this is exactly one
    detector.score() call. The stratified split joins several non-adjacent
    stretches of time; scoring them in one call would let a windowed model
    build windows across the join.
    """
    X = getattr(split, f"X_{part}")
    return np.concatenate([detector.score(X[s:e]) for s, e in split.bounds(part)])


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising pointwise F1 on (validation) labels."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def fit_and_score(
    split,
    detector,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    threshold_mode: str = "percentile",
):
    """
    Fit one detector, calibrate its threshold on validation, score test.

    threshold_mode:
      "percentile"  the `threshold_percentile` of validation's NORMAL-row
                    scores (default; needs no attack labels, so it is the
                    rule an unsupervised deployment could actually use).
      "val_f1"      the threshold that maximises F1 on validation, which
                    uses validation's attack labels. Still never looks at
                    test. The percentile rule fixes the false-positive rate
                    at (100 - percentile)% of normal rows: 3% of ~180k
                    normal test rows is ~5,000 false alarms against ~7,000
                    attack rows, which caps precision near 0.57 even for a
                    perfect ranking. Reporting both separates "the model
                    ranks badly" from "the threshold rule is too loose".

    Returns (threshold, test_scores) -- the raw score array, for callers
    that need it directly (diagnostic plots in visualize.py / notebooks).
    run_one() below wraps this to produce a full metrics Result instead.
    """
    detector.fit(split)

    val_scores = score_part(detector, split, "val")
    normal_mask = split.y_val == 0
    if threshold_mode == "val_f1" and 0 < split.y_val.sum() < len(split.y_val):
        threshold = best_f1_threshold(split.y_val, val_scores)
    elif normal_mask.any():
        threshold = float(np.percentile(val_scores[normal_mask], threshold_percentile))
    else:
        # No known-normal rows in validation at all -- can't calibrate
        # against "normal" the way this is meant to work. Fall back to all
        # of validation rather than crash, but say so loudly: this is a
        # degenerate split, not a case this protocol was designed for.
        print(
            f"    NOTE {detector.name}: validation split has no normal rows; "
            "falling back to a threshold over all of validation."
        )
        threshold = float(np.percentile(val_scores, threshold_percentile))

    test_scores = score_part(detector, split, "test")
    return threshold, test_scores


def run_one(
    split,
    detector,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    threshold_mode: str = "percentile",
):
    """Fit one detector on one split and evaluate it on the test window."""
    print(f"  {detector.name} ...")
    threshold, test_scores = fit_and_score(
        split, detector, threshold_percentile, threshold_mode
    )
    result = evaluate_all(
        split.y_test,
        test_scores,
        threshold=threshold,          # from validation, deliberately
        model=detector.name,
        dataset=split.name,
        threshold_mode=threshold_mode,
    )
    result.test_scores = test_scores  # kept for plots; not saved to JSON
    print(result.summary())
    print()
    return result


def run(
    split,
    model_names: list[str],
    tag: str = "",
    threshold_mode: str = "percentile",
    overrides: dict | None = None,
) -> list:
    """`overrides` maps a model name to constructor kwargs, e.g.
    {"lstm_forecast": {"train_stride": 5}}."""
    print(split.summary())
    print()

    overrides = overrides or {}
    results = []
    for name in model_names:
        if name not in REGISTRY:
            print(f"  skipping unknown model {name!r}")
            continue
        try:
            detector = REGISTRY[name](**overrides.get(name, {}))
            results.append(run_one(split, detector, threshold_mode=threshold_mode))
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
            "threshold_mode": r.meta.get("threshold_mode", "percentile"),
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
    import sys

    # The PRIMARY protocol (nominal A1 fit/val/calibration -> attacked A2
    # test, no attack labels) lives in protocol.py; `--protocol a1a2` hands
    # every remaining argument to it. chronological / stratified below are
    # the SECONDARY protocols, where attack-label models are allowed.
    if "--protocol" in sys.argv and sys.argv[sys.argv.index("--protocol") + 1] == "a1a2":
        import protocol

        i = sys.argv.index("--protocol")
        sys.argv = [sys.argv[0]] + sys.argv[1:i] + sys.argv[i + 2 :]
        return protocol.main()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat", help="registered dataset name")
    ap.add_argument("--protocol", default="chronological",
                    choices=["chronological", "stratified", "a1a2"])
    ap.add_argument("--threshold", default="percentile",
                    choices=["percentile", "val_f1"])
    ap.add_argument(
        "--models",
        nargs="+",
        default=["random", "xgboost", "xgboost_lags", "lstm_ae", "lstm_forecast"],
        help=f"any of: {', '.join(REGISTRY)}",
    )
    args = ap.parse_args()

    run(load(args.dataset, protocol=args.protocol), args.models,
        threshold_mode=args.threshold)


if __name__ == "__main__":
    main()
