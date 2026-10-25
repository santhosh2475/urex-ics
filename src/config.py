"""Project paths and global constants.

Import paths from here rather than hardcoding them, so the repo works
unchanged on a laptop and on any lab machine.

References for the threshold and event profiles below:
  Heydari & Nyarko (2026): V. Heydari and K. Nyarko, "Alarm-budgeted event-
                           level evaluation of ICS anomaly detection: Lessons
                           from SWaT and WADI," IEEE Access, vol. 14, pp.
                           17585-17609, 2026, doi: 10.1109/ACCESS.2026.3659034.
  Yu et al. (2026): R. Yu, M. Wang, J. Yun, J. Du, L. Wang, and Y. Suo,
                    "Unified reproduction and event-level evaluation of
                    industrial multivariate time-series anomaly detection
                    methods," in Proc. 7th Int. Conf. Comput., Netw. Internet
                    Things (CNIoT), 2026, pp. 1-5, doi:
                    10.1109/CNIoT69547.2026.11582255.
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


# --------------------------------------------------------------------------
# A1 -> A2 pipelines: named profiles
# --------------------------------------------------------------------------
# protocol.py (fit / validate / calibrate on the normal stream A1, test once
# on the attacked stream A2) and baseline.py (train on all of A1, test on A2)
# read their settings from the profiles below, and every results file stores
# the values it used. Several choices follow published SWaT studies, cited
# where they apply; where a value is ours, the comment says so.

DATA_PROFILES = {
    # protocol.py: A1 from 16:30 (the first 30 minutes are plant start-up;
    # the A1 range used by Heydari & Nyarko 2026), cut in time order into
    # fit / validation / calibration
    "swat2015": {
        "normal_file": "swat/SWaT_Dataset_Normal_v0.xlsx",
        # the A2 xlsx parses slowly; the csv export of the same file is used
        # when present (see swat-preprocess/data/README.md)
        "attack_file": "swat/SWaT_Dataset_Attack_v0.xlsx",
        "attack_csv": "../../swat-preprocess/data/swat-2015-data.csv",
        # The "drop 6 hours" convention is available as a1_start=None +
        # startup_rows=21600, but is not the default: on this file it removes
        # every stage-5 pump switch, so stage-5 sensors become unexplainable.
        "a1_start": "2015-12-22 16:30:00",
        "startup_rows": 0,
        "fit_frac": 0.60,
        "val_frac": 0.20,   # model/training choices only
        "cal_frac": 0.20,   # alarm thresholds only
    },
    # baseline.py: the whole normal file from 16:00:00 as training data. The
    # start-up hours are the only period in which the stage-5 pumps switch, so
    # the stage-5 actuator -> sensor regressions need them.
    "full_normal": {
        "normal_file": "swat/SWaT_Dataset_Normal_v0.xlsx",
        "attack_file": "swat/SWaT_Dataset_Attack_v0.xlsx",
        "attack_csv": "../../swat-preprocess/data/swat-2015-data.csv",
        "a1_start": None,
        "startup_rows": 0,
    },
}

PREPROCESS_PROFILES = {
    # default: drop columns constant in the fit rows, keep actuator states,
    # one StandardScaler fit on A1-fit rows only
    "shared": {"scaler": "standard", "winsorize": None, "pca_unsupervised": None},
    # min-max on fit rows; PCA keeping 99% of variance in front of the
    # unsupervised models only (process prediction needs named sensors)
    "minmax_pca": {"scaler": "minmax", "winsorize": None, "pca_unsupervised": 0.99},
    # robust scaling and winsorisation at fit-row quantiles (clip quantiles
    # are our choice)
    "robust_winsor": {"scaler": "robust", "winsorize": (0.001, 0.999), "pca_unsupervised": None},
}

EVENT_PROFILES = {
    # operator-facing alarm events: dilate alarms 2 s, merge runs <= 5 s
    # apart, drop runs shorter than 2 s, 10 s cooldown between events
    # (after Heydari & Nyarko, IEEE Access 2026)
    "operator": {"dilate": 2, "merge_gap": 5, "min_duration": 2, "cooldown": 10},
    # minimum event length 30 s, no merging (after Yu et al., CNIOT 2026,
    # whose duration unit is unstated; 30 samples at 1 Hz assumed)
    "min30": {"dilate": 0, "merge_gap": 0, "min_duration": 30, "cooldown": 0},
}

THRESHOLD_PROFILES = {
    # primary: lowest threshold on the A1 calibration stream whose eventized
    # alarms stay within N events/hour AND a 25% duty cap
    "budget": {"events_per_hour": (0.1, 0.2, 0.5), "duty_cap": 0.25, "events": "operator"},
    # 0.995 quantile of the A1-fit scores
    "q995_fit": {"quantile": 0.995, "on": "fit", "events": "min30"},
    # mean + k * sd of the A1-calibration scores, k = 3
    "ksigma_cal": {"k": 3.0, "on": "cal", "events": "operator"},
}

# CUSUM on standardised residuals for protocol.py's predictive detectors.
# Allowance b = 0.5 sd is the textbook default and is our choice.
CUSUM = {"drift": 0.5}

SEEDS = (42, 43, 44, 45, 46)  # runs default to the first; --seeds 5 for spread
