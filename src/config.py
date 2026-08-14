"""Project paths and global constants.

Import paths from here rather than hardcoding them, so the repo works
unchanged on a laptop and on any lab machine.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
NOTES = ROOT / "notes"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# experiment defaults
# --------------------------------------------------------------------------

SEED = 42

# Chronological split fractions. Applied in time order, never shuffled.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# test is the remainder

# Threshold is selected on the validation set at this percentile of scores,
# then applied unchanged to test. Never tuned on test.
THRESHOLD_PERCENTILE = 97.0

# Sequence length for windowed models. Held constant across model families
# so the comparison is not confounded by different context lengths.
WINDOW_SIZE = 60  # 60 seconds at 1 Hz sampling


# --------------------------------------------------------------------------
# dataset registry
# --------------------------------------------------------------------------
# temporal_dependency is the hypothesised position on the project's central
# axis. It is a PRIOR to be tested empirically in the EDA, not a finding.

DATASET_INFO = {
    "hai": {
        "source": "NSR / ETRI, South Korea",
        "domain": "power generation (turbine + boiler + water treatment via HIL)",
        "temporal_dependency": "high",
        "role": "primary, cross-domain replication",
    },
    "swat": {
        "source": "iTrust, SUTD",
        "domain": "water treatment",
        "temporal_dependency": "high",
        "role": "primary, literature comparability",
    },
    "wadi": {
        "source": "iTrust, SUTD",
        "domain": "water distribution",
        "temporal_dependency": "high",
        "role": "primary, high-dimensional stress case",
    },
    "morris": {
        "source": "Mississippi State University",
        "domain": "gas pipeline SCADA",
        "temporal_dependency": "low (TO VERIFY)",
        "role": "primary, contrast case where boosting should win",
    },
    "batadal": {
        "source": "iTrust, SUTD",
        "domain": "water distribution",
        "temporal_dependency": "moderate",
        "role": "secondary, fast smoke test",
    },
}

ATTRIBUTION = (
    "The SWaT, WADI, and BATADAL datasets were provided by iTrust, Centre for "
    "Research in Cyber Security, Singapore University of Technology and Design."
)
