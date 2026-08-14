# UREx CDE2605R — Gradient boosting vs deep learning for ICS anomaly detection

Benchmarking gradient boosting (XGBoost, LightGBM, CatBoost) against deep
sequence models (LSTM autoencoder) for anomaly detection in Industrial
Control Systems, in order to build a decision framework mapping operating
conditions to an appropriate model family.

Santhosh · Year 2 Industrial and Systems Engineering · National University of Singapore
AY2026/27 Semester 1

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v          # should be all green
python src/experiment.py --smoke    # end-to-end run on synthetic data
```

The smoke test needs no datasets. Run it first: if it works, the pipeline is
sound and the only remaining unknown is the real data.

---

## Structure

```
src/config.py       paths, seeds, split fractions, dataset registry
src/data.py         loading, chronological splitting, scaling, windowing
src/models.py       detectors, all behind one interface
src/evaluate.py     dual-protocol metrics (pointwise + point-adjust)
src/experiment.py   the single entry point for every result

tests/              regression tests, including the Kim et al. check
notes/datasets.md   glossary, dataset inventory, selection rationale
notes/papers.md     reading log
configs/            experiment configurations
results/            JSON output, timestamped
data/raw/           datasets, gitignored, never committed
```

---

## Four methodological commitments

These are the difference between a benchmark and a pile of numbers. Each is
enforced in code, not just written down.

**1. Every metric is reported under both protocols.**
Most published ICS anomaly detection work evaluates under *point-adjust*,
where flagging any single timestep inside an attack window credits the model
with the whole window. Kim et al. (AAAI 2022) showed a random score beats
published state of the art under that protocol. `random_baseline_check()`
in `evaluate.py` reproduces this, and `tests/` asserts it. Read the
pointwise column first; treat the gap as a finding.

**2. Splits are chronological. Always.**
No shuffling, no `random_state` on the split. A random split on time-series
data leaks the future into training. Asserted in `_assert_split_valid`.

**3. Scalers fit on training data only.**
Fitting on the full series leaks the test distribution's moments into
training. Enforced in `chronological_split`.

**4. Thresholds come from validation, never test.**
`experiment.py` selects the threshold at a percentile of *validation* scores
and applies it unchanged to test. ROC-AUC and PR-AUC are reported alongside
so no conclusion rests on that single choice.

Accuracy is deliberately never reported. At anomaly rates near 1 percent it
carries no information.

---

## Fair-comparison rules

The benchmark is only meaningful if neither family is quietly advantaged.

- Both families train on the **same training window**. Boosting uses labels
  (supervised); the autoencoder uses the normal rows only (unsupervised).
  Neither sees data the other could not have seen.
- Windowed and pointwise models produce score vectors of **identical
  length**, aligned to the same timeline via `unwindow_scores`, so both are
  evaluated against the same `y_true`.
- `WINDOW_SIZE` is held constant across model families so results are not
  confounded by different context lengths.
- Missing-value handling is decided once per dataset and applied identically
  to all models.

---

## Adding a dataset

1. Write a loader in `data.py` returning a time-sorted DataFrame with one
   `label` column (1 = anomaly) and numeric feature columns. All the messy,
   dataset-specific cleaning goes here so nothing downstream needs a special
   case.
2. Register it in `LOADERS` and add an entry to `DATASET_INFO` in
   `config.py`.
3. `python src/experiment.py --dataset <name>`

---

## Status

- [x] Evaluation harness, dual protocol, tested
- [x] Chronological splitting with guardrails
- [x] XGBoost detector, with optional lag features
- [x] LSTM autoencoder detector
- [x] End-to-end smoke test on synthetic data
- [ ] HAI loader
- [ ] SWaT loader (awaiting iTrust)
- [ ] WADI loader (awaiting iTrust)
- [ ] Morris SCADA loader — the contrast case, load-bearing for the framework
- [ ] Reproduce ESORICS 2022 autoencoder numbers on SWaT before claiming anything

---

## Attribution

The SWaT, WADI, and BATADAL datasets were provided by iTrust, Centre for
Research in Cyber Security, Singapore University of Technology and Design.

HAI: Shin, Lee, Yun and Kim, *HAI 1.0: HIL-based Augmented ICS Security
Dataset*, USENIX CSET 2020.

See `notes/datasets.md` for full terms and the selection rationale.
