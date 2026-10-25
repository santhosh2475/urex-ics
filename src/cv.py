"""
Grouped cross-validation over attack incidents.

    python src/cv.py                                   # default model set, 5 folds
    python src/cv.py --models naive_bayes logistic_regression --select pearson
    python src/cv.py --summarise results/swat-cv-<stamp>.jsonl

A single split leaves ~6 incidents in test, and on SWaT the model ranking
flips with which 6 they are. Here every incident is tested exactly once
(`data.episode_folds`), every number is reported per fold, and the summary
is mean +- std across folds, so a ranking only counts if it survives the
rotation.

Everything that is SELECTED -- threshold, number of Pearson features -- is
selected on that fold's validation incidents and applied unchanged to its
test incidents. Test folds are never used to choose anything.

Results are appended to a .jsonl file one (fold, model) at a time, so a
long run that dies halfway keeps what it finished and `--resume` skips it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import RESULTS, THRESHOLD_PERCENTILE
from data import LOADERS, episode_folds
from evaluate import evaluate_all
from experiment import best_f1_threshold, score_part
from features import pearson_select, subset
from models import REGISTRY

DEFAULT_MODELS = [
    "random", "naive_bayes", "logistic_regression", "logreg_adam", "mlp_bce",
    "adaboost", "xgboost", "random_forest", "cnn_classifier", "lstm_forecast", "lstm_ae",
]
# consecutive windows overlap in 59 of 60 rows; every 5th is plenty and 5x faster
DEFAULT_OVERRIDES = {"lstm_forecast": {"train_stride": 5}, "lstm_ae": {"train_stride": 5}}
PEARSON_GRID = (10, 20, 30, 40, None)


def _select_pearson(split, model_name, overrides):
    """Top-k Pearson features (episode relevance, no redundancy pruning --
    pruning removed exactly the sensors that decouple under attack), with k
    chosen on THIS fold's validation PR-AUC."""
    best = None
    for k in PEARSON_GRID:
        keep, _ = pearson_select(split, top_k=k, relevance="episode", max_redundancy=1.01)
        fs = subset(split, keep)
        det = REGISTRY[model_name](**overrides.get(model_name, {})).fit(fs)
        ap = float(average_precision_score(fs.y_val, score_part(det, fs, "val")))
        if best is None or ap > best[0]:
            best = (ap, k, fs, det)
    return best


def run_fold(split, model_name, overrides, select=None) -> tuple[dict, pd.Series]:
    t0 = time.time()
    n_features = split.n_features
    chosen_k = None
    if select == "pearson":
        _, chosen_k, split, det = _select_pearson(split, model_name, overrides)
        n_features = split.n_features
    else:
        det = REGISTRY[model_name](**overrides.get(model_name, {})).fit(split)

    val = score_part(det, split, "val")
    test = score_part(det, split, "test")
    normal = split.y_val == 0
    thresholds = {
        "percentile": float(np.percentile(val[normal], THRESHOLD_PERCENTILE)),
        "val_f1": best_f1_threshold(split.y_val, val),
    }
    row = {
        "fold": split.fold, "key": model_name, "model": det.name, "select": select,
        "n_features": n_features, "pearson_k": chosen_k,
        "test_incidents": int((split.incidents["part"] == 2).sum()),
        "test_attack_rate": float(split.y_test.mean()),
        "best_epoch": getattr(det, "best_epoch", None),
        "loss_history": list(getattr(det, "loss_history", []) or []),
        "val_loss_history": list(getattr(det, "val_loss_history", []) or []),
    }
    for mode, thr in thresholds.items():
        r = evaluate_all(split.y_test, test, threshold=thr)
        row.update({
            "roc_auc": r.roc_auc, "pr_auc": r.pr_auc,
            f"{mode}_f1": r.pointwise.f1, f"{mode}_precision": r.pointwise.precision,
            f"{mode}_recall": r.pointwise.recall, f"{mode}_f1_adjusted": r.point_adjusted.f1,
        })
    # per-incident detection at the validation-F1 threshold, keyed by the
    # incident's global id so all folds line up into one 22-column table
    flagged = test >= thresholds["val_f1"]
    rows_t = split.row_index["test"]
    det_frac = pd.Series({
        f"#{r.incident}": float(flagged[(rows_t >= r.start) & (rows_t < r.end) & (split.y_test == 1)].mean())
        for r in split.incidents[split.incidents["part"] == 2].itertuples()
    })
    row["per_incident"] = det_frac.to_dict()
    row["seconds"] = time.time() - t0
    return row, det_frac


def summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["name"] = df["key"] + np.where(df["select"].fillna("") == "pearson", "+pearson", "")
    metrics = ["pr_auc", "roc_auc", "val_f1_f1", "percentile_f1"]
    g = df.groupby("name")[metrics]
    out = pd.DataFrame({
        m: g[m].mean().map("{:.3f}".format) + " ± " + g[m].std().map("{:.3f}".format)
        for m in metrics
    })
    out["folds"] = g.size()
    # how often the model beats random on PR-AUC, fold by fold
    rnd = df[df["key"] == "random"].set_index("fold")["pr_auc"]
    if len(rnd):
        out["beats_random"] = df.groupby("name").apply(
            lambda d: f"{int((d.set_index('fold')['pr_auc'] > rnd.reindex(d['fold']).to_numpy()).sum())}/{len(d)}"
        )
    order = g["pr_auc"].mean().sort_values(ascending=False).index
    return out.loc[order].rename(columns={"val_f1_f1": "f1 (val-F1 thr)", "percentile_f1": "f1 (p97 thr)"})


def per_incident_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["name"] = df["key"] + np.where(df["select"].fillna("") == "pearson", "+pearson", "")
    table = {}
    for name, d in df.groupby("name"):
        merged = {}
        for pi in d["per_incident"]:
            merged.update(pi)
        table[name] = merged
    t = pd.DataFrame(table).T
    return t[sorted(t.columns, key=lambda c: int(c[1:]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="swat")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select", choices=["pearson"], default=None,
                    help="per-fold Pearson top-k, k chosen on that fold's validation")
    ap.add_argument("--resume", type=Path, default=None, help="append to an existing .jsonl")
    ap.add_argument("--summarise", type=Path, default=None, help="only print a saved run")
    ap.add_argument("--cache", type=Path, default=None,
                    help="pickled load_<dataset>() frame, to skip re-parsing the xlsx")
    args = ap.parse_args()

    if args.summarise:
        rows = [json.loads(l) for l in args.summarise.read_text().splitlines() if l.strip()]
        print(summarise(rows).to_string())
        return

    df = pd.read_pickle(args.cache) if args.cache else LOADERS[args.dataset]()
    folds = episode_folds(df, n_folds=args.folds, name=args.dataset)

    path = args.resume or RESULTS / f"{args.dataset}-cv-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["fold"], r["key"], r["select"]))
    print(f"writing -> {path}")

    rows = []
    for sp in folds:
        print(f"\n=== fold {sp.fold}: {sp.summary().splitlines()[3].strip()}")
        for m in args.models:
            if (sp.fold, m, args.select) in done:
                continue
            try:
                row, _ = run_fold(sp, m, DEFAULT_OVERRIDES, args.select)
            except Exception as exc:  # one failure should not lose the rest
                print(f"  FAILED {m}: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(row)
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {m:<22} PR-AUC {row['pr_auc']:.3f}  F1(valF1) {row['val_f1_f1']:.3f}  "
                  f"({row['seconds']:.0f}s)", flush=True)

    all_rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print("\n" + summarise(all_rows).to_string())


if __name__ == "__main__":
    main()
