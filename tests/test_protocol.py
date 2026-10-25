"""Unit tests for the A1 -> A2 primary protocol, on synthetic streams.
Run with: python -m pytest tests/ -v   (real-data checks: test_protocol_real.py)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from config import EVENT_PROFILES
from data import nominal_split
from evaluate import alarm_load, attack_event_metrics, calibrate_budget, eventize
from nominal_models import NOMINAL_REGISTRY, cusum_score
from protocol import run_model, thresholds_for

COLS = ["FIT101", "LIT101", "MV101", "P101", "AIT201", "FIT201", "P201", "P205"]


def _streams(n1=6000, n2=3000, seed=0, a2_shift=0.0):
    """A1 nominal and A2 with three attacks; FIT101 follows P101, P201 is
    constant in A1 but switches in A2 (like SWaT's never-used pumps)."""
    rng = np.random.default_rng(seed)

    def frame(n, start, attacks=()):
        p101 = (np.sin(np.arange(n) / 300) > 0).astype(float) + 1
        d = {
            "FIT101": 2.0 * p101 + rng.normal(0, 0.05, n),
            "LIT101": np.cumsum(rng.normal(0, 0.1, n)) * 0.01 + 500,
            "MV101": p101.copy(),
            "P101": p101,
            "AIT201": 260 + rng.normal(0, 0.5, n),
            "FIT201": 1.5 * p101 + rng.normal(0, 0.05, n),
            "P201": np.ones(n),
            "P205": 2.0 + (rng.random(n) > 0.5),
        }
        df = pd.DataFrame(d)
        df["label"] = 0
        for s, e in attacks:
            df.loc[s : e - 1, "FIT101"] += 3.0
            df.loc[s : e - 1, "label"] = 1
        df.insert(0, "timestamp", pd.date_range(start, periods=n, freq="1s"))
        return df

    a1 = frame(n1, "2015-12-22 16:30:00")
    a2 = frame(n2, "2015-12-28 10:00:00", attacks=[(500, 600), (1500, 1550), (2500, 2700)])
    a2["P201"] = np.where(np.arange(n2) > 1000, 2.0, 1.0)
    a2["AIT201"] += a2_shift
    return a1, a2, {"synthetic": True}


@pytest.fixture(scope="module")
def split():
    sp, _ = nominal_split(streams=_streams())
    return sp


# --------------------------------------------------------------------------
# A2 isolation and train-only preprocessing
# --------------------------------------------------------------------------

def test_partitions_are_contiguous_a1_and_a2_is_only_test(split):
    b = split.boundaries
    assert b["fit"]["end"] < b["val"]["start"] < b["val"]["end"] < b["cal"]["start"]
    assert b["cal"]["end"] < b["test"]["start"]
    assert b["fit"]["rows"] + b["val"]["rows"] + b["cal"]["rows"] == 6000


def test_preprocessing_is_fit_on_a1_fit_rows_only():
    a1, a2, rep = _streams()
    sp1, _ = nominal_split(streams=(a1, a2, rep))
    a2_mod = a2.copy()
    a2_mod[COLS] = a2_mod[COLS] * 100 + 7          # wildly different A2
    sp2, _ = nominal_split(streams=(a1, a2_mod, rep))
    for part in ("X_fit", "X_val", "X_cal"):
        assert np.array_equal(getattr(sp1, part), getattr(sp2, part)), part
    fit_rows = a1.iloc[: sp1.boundaries["fit"]["rows"]][sp1.feature_names].to_numpy()
    assert np.allclose(sp1.preprocessor.scaler_.mean_, fit_rows.mean(axis=0))


def test_columns_constant_in_fit_are_dropped_even_if_a2_varies(split):
    assert "P201" in split.preprocessor.dropped_
    assert "P201" not in split.feature_names


def test_thresholds_never_depend_on_a2():
    a1, a2, rep = _streams()
    sp1, _ = nominal_split(streams=(a1, a2, rep))
    a2_other = a2.copy()
    a2_other["label"] = 1 - a2_other["label"]       # different A2 labels
    a2_other["FIT101"] += 50                         # and different A2 values
    sp2, _ = nominal_split(streams=(a1, a2_other, rep))
    r1, _ = run_model(sp1, "pca_spe", 42, "shared")
    r2, _ = run_model(sp2, "pca_spe", 42, "shared")
    for op in r1["operating_points"]:
        assert r1["operating_points"][op]["threshold"] == r2["operating_points"][op]["threshold"]


# --------------------------------------------------------------------------
# eventization and alarm budgets
# --------------------------------------------------------------------------

def test_eventize_dilate_merge_min_duration_cooldown():
    a = np.zeros(80, bool)
    a[[5, 6, 7, 20, 30, 31, 70]] = True
    assert eventize(a) == [(5, 8), (20, 21), (30, 32), (70, 71)]
    assert eventize(a, min_duration=2) == [(5, 8), (30, 32)]
    assert eventize(a, merge_gap=12) == [(5, 32), (70, 71)]     # gaps 12 and 9 both <= 12
    # operator profile: dilate 2 -> (3,10)(18,23)(28,34)(68,73); merge<=5 joins
    # 23..28; cooldown 10 folds (18,34) into (3,10); (68,73) is 34 s later: new event
    assert eventize(a, **EVENT_PROFILES["operator"]) == [(3, 34), (68, 73)]


def test_budget_calibration_meets_budget_and_duty_on_calibration():
    rng = np.random.default_rng(1)
    cal = rng.normal(size=36_000)                    # 10 h of 1 Hz scores
    cal[5000:5100] += 6
    ev = EVENT_PROFILES["operator"]
    thresholds = []
    for budget in (0.1, 0.5, 2.0):
        thr, load = calibrate_budget(cal, budget, 0.25, ev)
        assert load["events_per_hour"] <= budget and load["duty"] <= 0.25
        assert load == alarm_load(cal, thr, ev)      # same rule, same numbers
        thresholds.append(thr)
    assert thresholds[0] >= thresholds[1] >= thresholds[2]   # looser budget, lower threshold


def test_attack_event_metrics_detection_delay_and_false_alarms():
    y = np.zeros(1000, int)
    y[100:200] = 1
    y[500:520] = 1
    events = [(150, 160), (700, 710)]               # hits attack 1 at +50 s; one false event
    m = attack_event_metrics(y, events)
    assert m["n_intervals"] == 2 and m["detected"] == 1 and m["miss_rate"] == 0.5
    assert m["per_interval"][0]["delay_s"] == 50
    assert m["false_alarm_events"] == 1


# --------------------------------------------------------------------------
# detectors: alignment, statelessness across blocks
# --------------------------------------------------------------------------

FAST = {"lstm_cusum": {"epochs": 1, "window": 20}, "dense_ae": {"epochs": 1},
        "lstm_ae": {"epochs": 1}}


@pytest.mark.parametrize("key", sorted(NOMINAL_REGISTRY))
def test_every_nominal_detector_returns_one_score_per_row(split, key):
    det = NOMINAL_REGISTRY[key](seed=0, **FAST.get(key, {}))
    if key == "lstm_ae":
        det.kw["epochs"] = 1
    det.fit(split)
    s = det.score(split.X_test)
    assert s.shape == (len(split.X_test),) and np.isfinite(s).all()


def test_scoring_carries_no_state_between_blocks(split):
    det = NOMINAL_REGISTRY["linear_cusum"](seed=0).fit(split)
    a, b = split.X_test[:1000], split.X_test[1000:2000]
    det.score(a)                                     # an earlier block
    assert np.array_equal(det.score(b), NOMINAL_REGISTRY["linear_cusum"](seed=0).fit(split).score(b))


def test_cusum_matches_the_textbook_recursion():
    z = np.array([[0.0], [1.0], [2.0], [-3.0]])
    # b = 0.5: S+ = 0, 0.5, 2.0, 0 ; S- = 0, 0, 0, 2.5
    assert np.allclose(cusum_score(z, 0.5), [0.0, 0.5, 2.0, 2.5])


def test_predictive_arm_uses_actuators_to_predict_explained_sensors(split):
    det = NOMINAL_REGISTRY["linear_cusum"](seed=0).fit(split)
    m = det.info["mapping"]
    assert m["FIT101"]["included"] and m["FIT101"]["fit_r2"] > 0.9
    assert not m["AIT201"]["included"]               # pure noise: not actuator-explained
    assert all(n.startswith(("MV", "P")) for n in m["FIT101"]["input_names"])


def test_operating_points_cover_every_threshold_profile(split):
    r, _ = run_model(split, "iforest", 0, "shared")
    assert set(r["operating_points"]) == {"budget_0.1", "budget_0.2", "budget_0.5",
                                          "q995_fit", "ksigma_cal"}
    ops = thresholds_for({"fit": np.arange(100.0), "cal": np.arange(100.0)})
    assert ops["q995_fit"][0] == pytest.approx(np.quantile(np.arange(100.0), 0.995))
