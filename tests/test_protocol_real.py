"""Real-data integration checks for the A1 -> A2 protocol. Skipped when the
iTrust files (or their parsed cache) are not on this machine. Kept apart
from the synthetic unit tests so a missing download never hides a unit
failure and a passing unit suite is never mistaken for a data check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from config import DATA_PROCESSED, DATA_RAW

HAVE_DATA = (DATA_PROCESSED / "swat2015_A1_raw.pkl").exists() or (
    DATA_RAW / "swat" / "SWaT_Dataset_Normal_v0.xlsx").exists()

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="SWaT files not present")


def test_swat_streams_are_our_v0_files_with_the_expected_shape():
    from data import contiguous_runs, load_swat_streams

    a1, a2, rep = load_swat_streams()
    assert rep["A1"]["source"].endswith("SWaT_Dataset_Normal_v0.xlsx")
    assert rep["A2"]["source"].endswith(("SWaT_Dataset_Attack_v0.xlsx", "swat-2015-data.csv"))
    # A1 2015-12-22 16:30 -> 2015-12-28 09:59, A2 -> 2016-01-02 14:59
    assert rep["A1"]["start"] == "2015-12-22 16:30:00"
    assert rep["A1"]["end"] == "2015-12-28 09:59:59"
    assert rep["A2"]["end"] == "2016-01-02 14:59:59"
    assert rep["A1"]["attack_rows"] == 0 and rep["A1"]["gaps_not_1s"] == 0
    assert rep["A1"]["duplicate_timestamps"] == rep["A2"]["duplicate_timestamps"] == 0
    # 35 contiguous labelled attack intervals in the attack file
    assert len(contiguous_runs(a2["label"].to_numpy().astype(bool))) == 35
    assert rep["A2"]["attack_rows"] == 54_621


def test_full_normal_profile_keeps_the_start_up_hours():
    from data import load_swat_streams

    a1, _, rep = load_swat_streams("full_normal")
    assert rep["A1"]["start"] == "2015-12-22 16:00:00" and len(a1) == 496_800


def test_nominal_split_keeps_a2_out_of_every_a1_partition():
    from data import nominal_split

    sp, rep = nominal_split()
    b = rep["boundaries"]
    assert b["cal"]["end"] < b["test"]["start"]
    assert b["fit"]["rows"] + b["val"]["rows"] + b["cal"]["rows"] == rep["A1"]["rows"]
    assert len(sp.X_test) == rep["A2"]["rows"]
