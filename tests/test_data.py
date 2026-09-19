"""Integration test for load_swat(). Run with: python -m pytest tests/ -v

This runs against the real iTrust SWaT files, not a synthetic stand-in --
the project trains and evaluates on SWaT only, so a fabricated fixture would
verify nothing about whether the real download actually parses. Skipped
when the files aren't present locally (they're gitignored, per iTrust's
redistribution terms).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from config import DATA_RAW
from data import chronological_split, load_swat

_NORMAL = DATA_RAW / "swat" / "SWaT_Dataset_Normal_v0.xlsx"
_ATTACK = DATA_RAW / "swat" / "SWaT_Dataset_Attack_v0.xlsx"

pytestmark = pytest.mark.skipif(
    not (_NORMAL.exists() and _ATTACK.exists()),
    reason=f"real SWaT files not found at {_NORMAL.parent} -- download from iTrust to run this",
)


def test_load_swat_parses_the_real_download():
    df = load_swat()

    assert "label" in df.columns
    assert set(df["label"].unique()) <= {0, 1}
    assert df["label"].sum() > 0, "attack file should contribute at least one anomalous row"
    assert not df.drop(columns=["label"]).isna().any().any()

    s = chronological_split(df, name="swat")
    assert s.n_features == df.shape[1] - 1
