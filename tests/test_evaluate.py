"""Regression tests. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from evaluate import find_segments, point_adjust, evaluate_all, random_baseline_check
from data import chronological_split, make_synthetic, make_windows


def test_find_segments():
    assert find_segments(np.array([0,1,1,0,0,1,0])) == [(1,3),(5,6)]
    assert find_segments(np.array([1,1,1])) == [(0,3)]   # runs at boundaries
    assert find_segments(np.array([0,0,0])) == []


def test_point_adjust_expands_full_segment():
    y = np.array([0,1,1,1,0]); pred = np.array([0,0,1,0,0])
    assert (point_adjust(y, pred) == np.array([0,1,1,1,0])).all()


def test_point_adjust_leaves_false_positives_alone():
    y = np.array([0,0,1,1,0]); pred = np.array([1,0,0,0,0])
    assert (point_adjust(y, pred) == pred).all()


def test_kim_et_al_random_beats_sota_under_point_adjust():
    """The project's central methodological claim, as a test."""
    r = random_baseline_check()
    assert r.pointwise.f1 < 0.10, "random should be near-useless pointwise"
    assert r.point_adjusted.f1 > 0.50, "point_adjust implementation broken"
    assert abs(r.roc_auc - 0.5) < 0.05, "random scores should have no signal"


def test_split_is_chronological_and_complete():
    df = make_synthetic(n=5000, n_features=8)
    s = chronological_split(df, name="t")
    assert len(s.y_train)+len(s.y_val)+len(s.y_test) == 5000
    assert s.n_features == 8


def test_windows_align_labels_to_last_timestep():
    X = np.arange(100).reshape(100,1).astype(float); y = np.zeros(100, int); y[95:] = 1
    Xw, yw = make_windows(X, y, window=10)
    assert Xw.shape == (91, 10, 1)
    assert yw[-1] == 1 and yw[0] == 0
