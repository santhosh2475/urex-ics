"""Unit tests for src/baseline.py (A1 -> A2 process-prediction vs unsupervised baseline)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import baseline as B


def test_metrics_definitions():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    m = B.metrics(y, p)
    assert m["precision"] == 0.75 and m["recall"] == 0.75
    assert m["udr"] == 0.25 and abs(m["far"] - 1 / 6) < 1e-12
    assert abs(m["recall"] + m["udr"] - 1) < 1e-12          # UDR = 1 - recall


def test_pointwise_statistic_tests_each_sample_on_its_own():
    r = np.array([[0.0], [5.0], [0.0], [0.0]])
    sh, sl = B.residual_statistic(r, np.zeros(1), np.full(1, 0.5), "pointwise")
    # x_{i-1}: the spike at row 1 shows up at row 2 only, then is gone
    assert np.allclose(sh[:, 0], [0, 0, 4.5, 0]) and np.allclose(sl, 0)
    sh_acc, _ = B.residual_statistic(np.full((5, 1), 1.0), np.zeros(1), np.full(1, 0.5), "accumulating")
    assert np.allclose(sh_acc[:, 0], [0.5, 1.0, 1.5, 2.0, 2.5])   # a constant offset accumulates


def test_process_predictor_raises_no_alarm_on_its_training_data():
    rng = np.random.default_rng(0)
    p101 = (rng.random(3000) > 0.5).astype(float)
    X = np.column_stack([2 * p101 + rng.normal(0, 0.05, 3000), p101])
    cols = ["FIT101", "P101"]
    mapping = B.select_mapping(X, cols)
    assert mapping["FIT101"]["input_names"] == ["P101"] and mapping["FIT101"]["r2"] > 0.99
    det = B.ProcessPredictor("LinearR", mapping, 0).fit(X)
    for mode in ("pointwise", "accumulating"):
        assert det.predict(X, mode).sum() == 0
    X_att = X.copy()
    X_att[1000:1100, 0] += 3                                   # spoof FIT101
    assert det.predict(X_att)[1001:1100].mean() > 0.9


def test_mapping_considers_the_stage_actuators_together():
    rng = np.random.default_rng(1)
    a, b = rng.integers(1, 3, 5000).astype(float), rng.integers(1, 3, 5000).astype(float)
    X = np.column_stack([a + b + rng.normal(0, 0.01, 5000), a, b])
    m = B.select_mapping(X, ["FIT501", "P501", "P502"])
    assert sorted(m["FIT501"]["input_names"]) == ["P501", "P502"]


def test_attack_matching_finds_a_spoofed_value_and_a_forced_state():
    n = 4000
    df = pd.DataFrame({"AIT202": np.full(n, 8.4), "MV304": np.full(n, 2.0),
                       "label": np.zeros(n, int)})
    df.loc[1000:1199, ["AIT202", "label"]] = [6.0, 1]          # "AIT-202 set to 6"
    df.loc[2500:2699, ["MV304", "label"]] = [1.0, 1]           # "MV-304 closed"
    train = df[df.label == 0]
    actions = {k: B.ATTACK_ACTIONS[k] for k in ("AIT-202 set to 6", "MV-304 closed")}
    m = B.match_attacks(df, train, actions)
    assert m.loc["AIT-202 set to 6", "start"] == 1000 and m.loc["MV-304 closed", "start"] == 2500


def test_attack_type_counts_stages_and_points():
    assert B.attack_type({"AIT202": 6.0}) == "single stage, single point"
    assert B.attack_type({"P203": 1, "P205": 1}) == "single stage, multi point"
    assert B.attack_type({"AIT402": 260.0, "AIT502": 260.0}) == "multi stage, single point"
    assert B.attack_type({"UV401": 1, "AIT502": 150.0, "P501": 2}) == "multi stage, multi point"
    # three points, all stage 1
    assert B.attack_type(B.ATTACK_ACTIONS["P-101, MV-101 on, LIT-101 set to 700"]) == "single stage, multi point"


def test_exclusion_rules_use_only_training_quantities():
    mapping = {"FIT101": {"r2": 0.95}, "AIT201": {"r2": 0.05}, "AIT203": {"r2": 0.6}}
    drift = pd.Series({"FIT101": 0.01, "AIT201": 0.40, "AIT203": 0.27})
    rules = B.exclusion_rules(mapping, drift)
    assert rules[B.ALL] == set()
    assert rules["R1: mapping R^2 < 0.5"] == {"AIT201"}
    assert rules["R2: normal-level drift > 10% of range"] == {"AIT201", "AIT203"}


def test_lag_features_only_look_back():
    rng = np.random.default_rng(2)
    A = rng.integers(1, 3, (200, 2)).astype(float)
    F = B.lag_features(A, [1, 5])
    assert F.shape == (200, 2 * 3 * 2)                         # (shift, roll, diff) x lags x columns
    A2 = A.copy()
    A2[150:] = 9.0                                             # change the future only
    assert np.array_equal(B.lag_features(A2, [1, 5])[:150], F[:150])


def test_lagged_mapping_points_at_each_inputs_own_lag_columns():
    n_cols, lagged = 4, [2, 3]
    mapping = {"FIT101": {"target": 0, "inputs": [3]}}
    inp = B.lagged_mapping(mapping, lagged, n_cols, n_lags=1)["FIT101"]["inputs"]
    # input 3 is the 2nd lagged column: its shift / roll / diff copies sit at
    # n_cols + block * len(lagged) + 1
    assert inp == [3, 4 + 0 * 2 + 1, 4 + 1 * 2 + 1, 4 + 2 * 2 + 1]


def test_state_predictor_flags_a_state_its_sensors_contradict():
    rng = np.random.default_rng(3)
    p101 = rng.integers(1, 3, 4000)
    fit101 = 2 * (p101 == 2) + rng.normal(0, 0.05, 4000)
    S = np.column_stack([fit101, p101]).astype(float)
    X = (S - S.min(axis=0)) / np.ptp(S, axis=0)
    det = B.StatePredictor(["FIT101", "P101"], set(), 0).fit(X, S)
    assert det.predict(X, S).mean() < 0.01
    S_att = S.copy()
    S_att[1000:1100, 1] = 3 - S_att[1000:1100, 1]              # reported pump state flipped
    S_att[2000:2010, 1] = 0                                    # a state never seen in TRAIN
    pred = det.predict(X, S_att)
    assert pred[1000:1100].mean() > 0.9 and pred[2000:2010].all()


def test_cnn_predictor_gives_zero_residual_without_a_full_window():
    rng = np.random.default_rng(4)
    p101 = (rng.random(1500) > 0.5).astype(float)
    X = np.column_stack([2 * p101 + rng.normal(0, 0.05, 1500), p101])
    mapping = {"FIT101": {"target": 0, "inputs": [1]}}
    det = B.CNNPredictor(mapping, 0, {"window": 10, "epochs": 1, "train_stride": 1}).fit(X)
    r = det.residuals(X)
    assert r.shape == (1500, 1) and np.all(r[:9] == 0) and np.any(r[9:] != 0)


def test_deep_scorer_scores_every_row_and_uses_context():
    rng = np.random.default_rng(5)
    t = np.arange(3000)
    X = np.column_stack([np.sin(t / 20), np.cos(t / 30)]) + rng.normal(0, 0.01, (3000, 2))
    for name in B.DEEP_UNSUPERVISED:
        score, info = B.deep_scorer(name, X[:2000])
        s = score(X[2000:], context=X[:2000])
        assert s.shape == (1000,) and np.isfinite(s).all() and info["fit_rows"] == 1600
        # with context, the first rows get their own window instead of the first score
        assert not np.allclose(s[:59], s[59])


def test_pipeline_reads_no_published_numbers():
    src = (Path(B.__file__)).read_text()
    assert "references" not in src.split('"""', 2)[2]          # code, not the docstring, never imports them
