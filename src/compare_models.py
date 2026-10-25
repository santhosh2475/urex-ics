"""
Plot saved experiment results side by side.

Reads every results/<dataset>-*.json written by experiment.py, keeps the
most recent run per model, and draws PR-AUC / ROC-AUC / pointwise F1 /
point-adjusted F1 as grouped horizontal bars on one figure.

    python src/compare_models.py --dataset swat
    python src/compare_models.py --dataset swat --models naive_bayes random_forest oner j48 nnge_stub_knn svm random xgboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS


def _matches(saved_name: str, registry_name: str) -> bool:
    """A saved model name carries its parameters -- 'cnn_classifier(w=60,f=64)'
    for registry key 'cnn_classifier', 'xgboost+lags[1, 5, 30]' for
    'xgboost_lags' -- so an exact-string filter silently drops every
    parameterised detector. Match on the bare prefix instead."""
    if saved_name == registry_name:
        return True
    return saved_name.startswith(registry_name) and saved_name[len(registry_name):len(registry_name) + 1] in "(+"


def load_latest(dataset: str, models: list[str] | None = None) -> dict[str, dict]:
    """One entry per model: the most recent results/<dataset>-*.json that
    contains it, keyed by the file's mtime so a later rerun always wins.
    Returned in `models` order when given, so callers control display order
    instead of getting whatever order the files happened to load in."""
    latest: dict[str, tuple[float, dict]] = {}
    for path in sorted(RESULTS.glob(f"{dataset}-*.json")):
        mtime = path.stat().st_mtime
        for row in json.loads(path.read_text()):
            name = row["model"]
            if models is not None and not any(_matches(name, m) for m in models):
                continue
            if name not in latest or mtime > latest[name][0]:
                latest[name] = (mtime, row)
    if not latest:
        raise FileNotFoundError(
            f"no results/{dataset}-*.json found -- run "
            f"`python src/experiment.py --dataset {dataset} --models ...` first"
        )
    if models is None:
        order = list(latest)
    else:
        order = [
            name for m in models for name in latest if _matches(name, m)
        ]
    return {name: latest[name][1] for name in order}


def plot(results: dict[str, dict], dataset: str, out_path: Path | None = None):
    order = list(results)
    n = len(order)

    pr_auc = [results[m]["pr_auc"] for m in order]
    roc_auc = [results[m]["roc_auc"] for m in order]
    f1_pw = [results[m]["pointwise"]["f1"] for m in order]
    f1_adj = [results[m]["point_adjusted"]["f1"] for m in order]

    fig, axes = plt.subplots(1, 4, figsize=(16, 0.5 * n + 1.5), sharey=True)
    y = np.arange(n)

    panels = [
        (axes[0], pr_auc, "PR-AUC", None),
        (axes[1], roc_auc, "ROC-AUC", 0.5),   # 0.5 = coin flip, worth marking
        (axes[2], f1_pw, "F1 (pointwise)", None),
        (axes[3], f1_adj, "F1 (point-adjusted)", None),
    ]

    for ax, values, title, ref_line in panels:
        ax.barh(y, values, color="#3c6e91", height=0.6)
        for yi, v in zip(y, values):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
        if ref_line is not None:
            ax.axvline(ref_line, color="#999999", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=10)
    axes[0].invert_yaxis()  # first model at the top

    fig.suptitle(f"{dataset}: model comparison, same split, same threshold rule", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_path or RESULTS / f"{dataset}-comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path.relative_to(RESULTS.parent)}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    results = load_latest(args.dataset, args.models)
    plot(results, args.dataset)


if __name__ == "__main__":
    main()
