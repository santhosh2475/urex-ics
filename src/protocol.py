"""
A1 -> A2 primary protocol runner.

    python src/protocol.py                                  # default suite, seed 42, shared preprocessing
    python src/protocol.py --models xgb_cusum iforest --seeds 5 --preprocess robust_winsor
    python src/protocol.py --summarise results/swat-a1a2-shared-<stamp>.json

Order of operations, per model and seed, and what each step may read:

    fit          A1-fit rows                (model parameters)
    select       A1-validation rows         (early stopping, actuator->sensor
                                             mapping, residual statistics)
    calibrate    A1-calibration rows        (alarm thresholds, every profile)
    test once    A2 rows + A2 labels        (metrics only)

Threshold profiles (config.THRESHOLD_PROFILES), all chosen without A2:
    budget_0.1/0.2/0.5   lowest threshold whose eventized alarms on the
                         calibration stream stay within N events/hour and a
                         25% duty cap -- the PRIMARY operating points
    q995_fit             0.995 quantile of A1-fit scores, min30 events
    ksigma_cal           mean + 3 sd of A1-calibration scores

Every results file stores the data report (source files, rows, timestamps,
gaps, partition boundaries), the profile values, each detector's settings,
and per-attack outcomes. Primary metrics: event miss rate (with an interval-
bootstrap CI), detected attacks, detection delay, achieved nominal alarm
rate and duty, false alarm events on A2. Pointwise P/R/F1, AP, ROC-AUC,
UDR and FAR are complementary; point-adjusted F1 is diagnostic only.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # xgboost + torch in one process
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from datetime import datetime

import numpy as np

from config import (CUSUM, DATA_PROFILES, EVENT_PROFILES, PREPROCESS_PROFILES, RESULTS,
                    SEEDS, THRESHOLD_PROFILES)
from data import load_swat_streams, nominal_split
from evaluate import (alarm_load, attack_event_metrics, binary_metrics, calibrate_budget,
                      eventize, point_adjust, udr_far_rates)
from nominal_models import NOMINAL_REGISTRY, UNSUPERVISED_CLASSICAL

DEFAULT_MODELS = [
    "linear_cusum", "rf_cusum", "knn_cusum", "xgb_cusum", "lstm_cusum", "linear_maxz", "xgb_maxz",
    "pca_spe", "iforest", "kmeans", "elliptic_envelope", "lof", "dbscan", "ocsvm",
    "dense_ae", "lstm_ae",
]


def thresholds_for(scores: dict) -> dict:
    """Every operating point, from A1 scores only. Returns
    {profile_name: (threshold, event_profile_name, achieved_load_on_cal)}."""
    out = {}
    b = THRESHOLD_PROFILES["budget"]
    for rate in b["events_per_hour"]:
        thr, load = calibrate_budget(scores["cal"], rate, b["duty_cap"], EVENT_PROFILES[b["events"]])
        out[f"budget_{rate}"] = (thr, b["events"], load)
    q = THRESHOLD_PROFILES["q995_fit"]
    thr = float(np.quantile(scores[q["on"]], q["quantile"]))
    out["q995_fit"] = (thr, q["events"], alarm_load(scores["cal"], thr, EVENT_PROFILES[q["events"]]))
    k = THRESHOLD_PROFILES["ksigma_cal"]
    thr = float(scores["cal"].mean() + k["k"] * scores["cal"].std())
    out["ksigma_cal"] = (thr, k["events"], alarm_load(scores["cal"], thr, EVENT_PROFILES[k["events"]]))
    return out


def evaluate_operating_point(y: np.ndarray, test_scores: np.ndarray, thr: float, ev_name: str) -> dict:
    ev_params = EVENT_PROFILES[ev_name]
    events = eventize(test_scores >= thr, **ev_params)       # identical rule to calibration
    pred = np.zeros(len(y), dtype=int)
    for s, e in events:
        pred[s:e] = 1
    pw = binary_metrics(y, pred)
    return {
        "threshold": thr,
        "event_profile": ev_name,
        **attack_event_metrics(y, events),
        "pointwise": pw.as_dict(),
        **udr_far_rates(y, pred),
        "point_adjusted_f1_diagnostic": binary_metrics(y, point_adjust(y, pred)).f1,
    }


def run_model(split, key: str, seed: int, preprocess: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    kw = {}
    if key in UNSUPERVISED_CLASSICAL and PREPROCESS_PROFILES[preprocess]["pca_unsupervised"]:
        kw["pca"] = PREPROCESS_PROFILES[preprocess]["pca_unsupervised"]
    t0 = time.time()
    det = NOMINAL_REGISTRY[key](seed=seed, **kw).fit(split)
    t_fit = time.time() - t0
    scores = {"fit": det.score(split.X_fit), "cal": det.score(split.X_cal)}
    test = det.score(split.X_test)                            # A2, scored once
    y = split.y_test
    ops = thresholds_for(scores)
    return {
        "key": key, "seed": seed, "detector": det.describe(),
        "fit_seconds": t_fit, "total_seconds": time.time() - t0,
        "roc_auc": float(roc_auc_score(y, test)),
        "average_precision": float(average_precision_score(y, test)),
        "operating_points": {
            name: {"calibration_load": load, **evaluate_operating_point(y, test, thr, ev)}
            for name, (thr, ev, load) in ops.items()
        },
        "loss_history": getattr(det, "loss_history", None) or getattr(getattr(det, "det", None), "loss_history", None),
        "val_loss_history": getattr(det, "val_loss_history", None) or getattr(getattr(det, "det", None), "val_loss_history", None),
    }, test


def summarise(path) -> str:
    import pandas as pd

    res = json.loads(open(path).read())
    rows = []
    for r in res["runs"]:
        d = r["detector"]
        row = {"model": r["key"], "objective": d["objective"], "family": d["family"],
               "seed": r["seed"], "AP": r["average_precision"], "ROC": r["roc_auc"]}
        for op in ("budget_0.1", "budget_0.5", "q995_fit", "ksigma_cal"):
            o = r["operating_points"][op]
            row[f"{op}:det"] = f"{o['detected']}/{o['n_intervals']}"
            row[f"{op}:FA/h"] = o["false_alarm_events_per_normal_hour"]
            row[f"{op}:duty"] = o["normal_duty"]    # share of normal A2 seconds under alarm
            row[f"{op}:cal/h"] = o["calibration_load"]["events_per_hour"]
            row[f"{op}:F1"] = o["pointwise"]["f1"]
        rows.append(row)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "seed"]
    agg = df.groupby(["model", "objective", "family"])[num].mean().reset_index()
    dets = df.groupby("model")[[c for c in df.columns if c.endswith(":det")]].first().reset_index()
    out = agg.merge(dets, on="model").sort_values(["objective", "family", "AP"], ascending=[True, True, False])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    return out.round(3).to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", type=int, default=1, help=f"first N of {SEEDS}")
    ap.add_argument("--preprocess", default="shared", choices=list(PREPROCESS_PROFILES))
    ap.add_argument("--data", default="swat2015", choices=list(DATA_PROFILES))
    ap.add_argument("--save-scores", action="store_true", help="also save A2 score arrays (.npy)")
    ap.add_argument("--summarise", default=None)
    args = ap.parse_args()

    if args.summarise:
        print(summarise(args.summarise))
        return

    split, report = nominal_split(args.data, args.preprocess)
    print(split.summary())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"swat-a1a2-{args.preprocess}-{stamp}.json"
    payload = {
        "protocol": "A1->A2 primary",
        "data_report": report,
        "profiles": {"data": DATA_PROFILES[args.data], "preprocess": PREPROCESS_PROFILES[args.preprocess],
                     "events": EVENT_PROFILES, "thresholds": THRESHOLD_PROFILES, "cusum": CUSUM},
        "runs": [], "failures": [],
    }
    score_dir = RESULTS / f"a1a2-scores-{stamp}"
    for seed in SEEDS[: args.seeds]:
        for key in args.models:
            print(f"\n--- {key} (seed {seed})", flush=True)
            try:
                run, test = run_model(split, key, seed, args.preprocess)
            except Exception as exc:                 # failures to fit are reported, not hidden
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                payload["failures"].append({"key": key, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            payload["runs"].append(run)
            if args.save_scores:
                score_dir.mkdir(exist_ok=True)
                np.save(score_dir / f"{key}-seed{seed}.npy", test.astype(np.float32))
            o = run["operating_points"]["budget_0.1"]
            print(f"  AP {run['average_precision']:.3f}  ROC {run['roc_auc']:.3f}  "
                  f"budget 0.1/h: detected {o['detected']}/{o['n_intervals']}, "
                  f"false alarms {o['false_alarm_events_per_normal_hour']:.2f}/h  ({run['total_seconds']:.0f}s)",
                  flush=True)
            path.write_text(json.dumps(payload, indent=1, default=str))   # incremental
    print(f"\nsaved -> {path}")
    print(summarise(path))


if __name__ == "__main__":
    main()
