# UREx CDE2605R — Gradient boosting vs deep learning for ICS anomaly detection

Benchmarking Machine Learning against Deep
sequence models (LSTM autoencoder,forecaster) for anomaly detection in Industrial
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
python src/baseline.py              # MAIN: A1 -> A2, R1 -> R2 -> R3 iterations, every model (~30 min)
python src/baseline.py --only baseline       # ... the label-free models only (~26 min)
python src/baseline.py --only attack-label   # ... the attack-label classifiers only (~5 min)
python src/protocol.py              # A1 -> A2 with alarm budgets and event-level metrics
python src/experiment.py            # secondary: chronological split, attack-label models allowed
```

Results land in `results/baseline/`; `notebooks/baseline.ipynb` is the main
notebook and only loads and shows them.

---

## Structure

```
src/baseline.py     MAIN pipeline: A1 -> A2, R1 -> R2 -> R3, label-free models + attack-label table
src/config.py       paths, seeds, split fractions, dataset registry
src/data.py         loading, chronological splitting, scaling, windowing
src/models.py       detectors, all behind one interface
src/evaluate.py     dual-protocol metrics (pointwise + point-adjust)
src/features.py     Pearson feature selection and drift checks, train-only
src/nominal_models.py  alarm-budget protocol detectors: process prediction + unsupervised
src/protocol.py     alarm-budget A1 -> A2 runner: fit / select / calibrate on A1, test on A2
src/experiment.py   secondary-protocol runner (chronological, stratified)
src/cv.py           secondary: grouped cross-validation over attack incidents
src/tune.py         validation-based hyperparameter selection
src/visualize.py, src/make_visuals.py, src/compare_models.py   plotting helpers

notebooks/baseline.ipynb       MAIN notebook: shows results/baseline/
notebooks/a1a2_protocol.ipynb  alarm-budget protocol
notebooks/protocols.ipynb      secondary protocols

tests/              regression tests, including the Kim et al. [1] check
notes/datasets.md   glossary, dataset inventory, selection rationale
notes/papers.md     reading log
configs/            experiment configurations
results/            JSON output, timestamped; results/baseline/ for the main pipeline
data/raw/           datasets, gitignored, never committed
```

---

## Main pipeline: process prediction vs unsupervised, A1 -> A2 (`src/baseline.py`)

The research question is how **supervised** and **unsupervised** approaches
compare on **previously unseen** SWaT attacks, and how **classical** and
**deep** models differ. The main pipeline trains on the whole normal file
(from 16:00: the start-up hours are the only period in which the stage-5 pumps
switch) and tests on the attack file, pointwise metrics. No attack label is
used for training or thresholds. Every number comes from our own SWaT v0 files;
the results record which file each stream was read from.

| objective | models | alarm |
|---|---|---|
| process prediction (actuators -> sensors) | LinearR, RF, KNN, XGBoost; XGBoost+lags (1 / 5 / 30 s lags, rolling means, differences); 1D-CNN (last 60 s of every actuator); under R3 also each sensor's 90 s change | any kept sensor's (or change target's) residual statistic outside its control limits (training extremes) |
| state prediction (sensors -> actuators) | LogisticR, one per actuator | standardised 1 - P(reported state) above mu + 3 sigma of training scores |
| unsupervised | iForest, K-Means, Elliptic Envelope, LOF, DBSCAN, One-Class SVM (PCA, 99%); LSTM-AE, LSTM-forecast (60 s windows) | above mu + 3 sigma of training scores |

* **Iteration 1, R1:** exclude the sensors the actuators explain poorly
  (mapping R^2 < 0.5, 9 sensors). LinearR, RF and XGBoost reach F1 0.46 at a
  9% false alarm rate.
* **Iteration 2, R2:** exclude the sensors whose daily normal level moves by
  more than 10% of its range within the normal file (6 chemical analysers).
  R1 kept the drifting AIT203, which alone alarms on 8.9% of attack-free rows,
  and dropped four stable sensors, losing the attack on AIT202. Both rules
  read only the normal file; the diagnosis that led to R2 used the attack
  file's attack-free rows.
* **R2 results:** XGBoost, RF and LinearR F1 0.81 at 0.2-0.3% false alarms;
  best unsupervised iForest F1 0.75 at 1.4%. LSTM-forecast scores F1 0.53 at
  14.5% false alarms but catches "LIT-101 set to 700" (98%), which no
  supervised model flags more than 4% of. XGBoost alarms in only 12 of the 35
  labelled attack intervals.
* **Iteration 3, R3:** keep R2's sensors and add, as further process-prediction
  targets, each sensor's change over the last 90 s, predicted from the
  actuators' mean states over those 90 s. A tank level integrates in- minus
  outflow, so the actuators explain its change, not its level. A sensor gets a
  change target when the actuators explain its change with R^2 >= 0.5 (R1's
  bar): 11 sensors, the three tank levels included. 90 s is the shortest
  window at which every tank level passes. The rule reads only the normal
  file; the diagnosis (which attacks R2 misses) used the attack file, and
  five other variants were scored on it and set aside, so the gain is
  somewhat optimistic. LogisticR and the unsupervised models keep R2's sensors.
* **R3 results:** XGBoost and XGBoost+lags F1 0.87 (recall 0.82-0.83),
  alarming in 28 of 35 intervals; LinearR 0.83, RF 0.82, KNN 0.80; the 1D-CNN
  does not gain (0.79). The cost is false alarms: XGBoost 1.0% (from 0.2%),
  269 false alarm events against 37. XGBoost now catches "LIT-101 set to 700"
  (100%). No model flags more than 2% of "MV-304 closed".
* **Attack-label classifiers** (`xgboost_lags`, `logistic_regression`,
  `cnn_classifier`) need attack rows to train, so they run separately on the
  episode-stratified split: F1 0.002 / 0.32 / 0.19 on held-out incidents. Not
  comparable row for row with the table above.
* RF's parallel predictions differ by ~1e-15 between runs, enough to move its
  numbers in the third decimal.

## Alarm-budget protocol: nominal A1 -> attacked A2 (`src/protocol.py`)

The same question at event level, with alarm budgets. The design draws on two
published SWaT studies -- Yu et al. [2] and Heydari & Nyarko [3] -- but every number here comes from our own pipeline and our own
copies of the iTrust v0 files:

| step | data | what it may set |
|---|---|---|
| fit | A1, first 60% (2015-12-22 16:30 -> 12-26 02:59) | model parameters, scaler |
| select | A1, next 20% | early stopping, actuator->sensor mapping, residual statistics |
| calibrate | A1, last 20% (-> 12-28 09:59) | alarm thresholds |
| test, once | A2 (2015-12-28 10:00 -> 2016-01-02 14:59) | metrics only |

No A2 row or label reaches scaling, feature choice, early stopping, model
selection or thresholds; `tests/test_protocol.py` asserts it.

**"Supervised" here means process prediction, not attack labels.** The
supervised arm regresses sensor readings on actuator states using nominal
data, and scores CUSUM over standardised residuals. Sensors the actuators do
not explain (fit R^2 < 0.5: mostly the chemical analysers) are left to the
unsupervised arm. Every detector is tagged with its objective and family:

| objective | classical | deep |
|---|---|---|
| process prediction | linear, random forest, k-NN, XGBoost regression (+CUSUM / max-z) | LSTM predictor, same inputs and targets |
| unsupervised | PCA-SPE, Isolation Forest, K-means, Elliptic Envelope, LOF, DBSCAN, One-Class SVM | dense AE, LSTM AE |

The attack-label classifiers in `models.py` (xgboost, cnn_classifier, ...) are
a separate **attack-label-trained extension**: secondary protocols and the
main pipeline's separate attack-label table only, never in this protocol's table.

**Operating points and metrics.** Thresholds come from A1 calibration under
named profiles (`config.py`): alarm budgets of 0.1 / 0.2 / 0.5 eventized
alarm events per hour with a 25% duty cap (`budget`, primary); the 0.995
quantile of A1-fit scores (`q995_fit`); mean + 3 sd of A1-calibration scores
(`ksigma_cal`). Alarms are eventized with one shared rule (`operator`: dilate
2 s, merge gaps <= 5 s, drop runs < 2 s, 10 s cooldown) at calibration and at
test. Primary metrics: attacks detected out of 35, miss rate with an
attack-interval bootstrap CI, detection delay, achieved nominal alarm rate and
duty, false alarm events and alarm duty on normal A2 time. Pointwise P/R/F1,
AP, ROC-AUC, UDR and FAR are complementary; point-adjusted F1 is diagnostic
only.

**Data facts checked on every run** (stored in the results file): A1 495,000
rows at 1 Hz with no gaps, duplicates, missing values or attack rows; A2
449,919 rows, 54,621 attack rows in 35 contiguous intervals, one 82 s
timestamp gap. 11 actuators are constant in the fit rows and dropped.

**Not implemented:** DAGMM, USAD, OmniAnomaly, TranAD, aCVAE (no reproducible
settings in the material held). iTrust's official attack list is not in this
repo, so attack stages elsewhere are inferred from the data and labelled so.

The chronological, episode-stratified and cross-validation experiments remain
as named **secondary** protocols. Their results are never combined with the
main pipeline's or this protocol's tables.

---

## Four methodological commitments

These are the difference between a benchmark and a pile of numbers. Each is
enforced in code, not just written down.

**1. Every metric is reported under both protocols.**
Most published ICS anomaly detection work evaluates under *point-adjust*,
where flagging any single timestep inside an attack window credits the model
with the whole window. Kim et al. [1] showed a random score beats
published state of the art under that protocol. `random_baseline_check()`
in `evaluate.py` reproduces this, and `tests/` asserts it. Read the
pointwise column first; treat the gap as a finding.

**2. Splits never shuffle rows.**
No `random_state` on the split. A random split on time-series data leaks the
future into training. Two protocols, both reported:

- `chronological` (default): one cut in time. Deployment-realistic, but on
  SWaT it gives validation the one 10-hour attack (78% of attack rows) and
  train no stage-4 attack at all, at 0.93% / 22.95% / 3.76% attack rate.
- `stratified` (`stratified_episode_split`): whole attack incidents assigned
  to train/val/test, stratified by the stage they target, with 300 purged
  rows at every boundary and no window built across a join. Measures
  coverage of attack types, at the cost of some temporal realism.

`python src/experiment.py --protocol stratified`. Asserted in
`_assert_split_valid` and `tests/test_stratified.py`.

**3. Scalers fit on training data only.**
Fitting on the full series leaks the test distribution's moments into
training. Enforced in `chronological_split`.

**4. Thresholds and feature selection come from validation, never test.**
`experiment.py` selects the threshold at a percentile of *validation* scores
(or, with `--threshold val_f1`, the F1-maximising one on validation labels)
and applies it unchanged to test. `features.pearson_select` ranks features on
training rows only. ROC-AUC and PR-AUC are reported alongside
so no conclusion rests on that single choice.

Accuracy is deliberately never reported. At anomaly rates near 1 percent it
carries no information.

---

## Fair-comparison rules

The benchmark is only meaningful if neither family is quietly advantaged.

- Both families train on the **same training window**. Boosting uses labels
  (supervised); the deep models use the normal rows only (unsupervised).
  Neither sees data the other could not have seen.
- The deep family is represented by **two** detectors, and both are always
  reported. `lstm_ae` scores by reconstruction and `lstm_forecast` scores by
  one-step prediction error. This is not redundancy — see "The deep baseline
  is two models" below. Reporting only the autoencoder would credit the
  entire deep-learning family with a limitation of one scoring rule.
- Sequence models window each **contiguous run** of normal training rows.
  Masking the attack rows out and windowing the result splices non-adjacent
  timesteps together and trains the model on transitions that never occur.
  Enforced by `make_windows_segmented` in `data.py`.
- Engineered lag features fill their NaN head with 0, never `bfill`.
  Backfilling reaches forward in time and is a leak, however small.
- Windowed and pointwise models produce score vectors of **identical
  length**, aligned to the same timeline via `unwindow_scores`, so both are
  evaluated against the same `y_true`.
- `WINDOW_SIZE` is held constant across model families so results are not
  confounded by different context lengths.
- Missing-value handling is decided once per dataset and applied identically
  to all models.

---

## The deep baseline is two models

The autoencoder scores a timestep by how well it can **reconstruct** it — but
that timestep is part of the encoder's own input. The score therefore measures
compressibility, not predictability, and that distinction decides what the
model can detect at all.

An attack that shifts a sensor's level is less compressible, and the
autoencoder finds it. An attack that leaves the marginal distribution intact
and only breaks the temporal correlation is invisible to it. For an AR(0.95)
process, the last timestep's deviation from its window mean has variance
0.904 under normal dynamics and 0.905 under an i.i.d. replacement — no gap for
a reconstruction score to exploit. On `make_synthetic(temporal=True)` the
autoencoder measures **ROC-AUC 0.38–0.46, below chance**, at every bottleneck
size and training budget tried. Training it longer makes it worse.

`lstm_forecast` scores the same windows on one-step prediction error, so the
scored timestep is withheld from the input. On the same data it reaches
**ROC-AUC ~0.97**. This is the Malhotra et al. [4] / Hundman et al. [5]
formulation and the standard deep baseline in the telemetry and ICS
literature.

The methodological point for the report: had the benchmark shipped only the
autoencoder, it would have concluded "gradient boosting beats deep learning on
temporally-defined attacks" — the project's headline claim — when the true
cause was a scoring rule structurally blind to the anomalies in question. That
is the same category of error as evaluating under point-adjust alone, and it
gets the same treatment: report both, and treat the gap as a finding.


## References

[1] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon, "Towards a rigorous
evaluation of time-series anomaly detection," in *Proc. AAAI Conf. Artif.
Intell.*, vol. 36, no. 7, 2022, pp. 7194–7201, doi:
[10.1609/aaai.v36i7.20680](https://doi.org/10.1609/aaai.v36i7.20680).

[2] R. Yu, M. Wang, J. Yun, J. Du, L. Wang, and Y. Suo, "Unified reproduction
and event-level evaluation of industrial multivariate time-series anomaly
detection methods," in *Proc. 7th Int. Conf. Comput., Netw. Internet Things
(CNIoT)*, 2026, pp. 1–5, doi:
[10.1109/CNIoT69547.2026.11582255](https://doi.org/10.1109/CNIoT69547.2026.11582255).

[3] V. Heydari and K. Nyarko, "Alarm-budgeted event-level evaluation of ICS
anomaly detection: Lessons from SWaT and WADI," *IEEE Access*, vol. 14,
pp. 17585–17609, 2026, doi:
[10.1109/ACCESS.2026.3659034](https://doi.org/10.1109/ACCESS.2026.3659034).

[4] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long short term memory
networks for anomaly detection in time series," in *Proc. Eur. Symp. Artif.
Neural Netw. (ESANN)*, Bruges, Belgium, 2015.
[PDF](https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf)

[5] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom,
"Detecting spacecraft anomalies using LSTMs and nonparametric dynamic
thresholding," in *Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discov. Data
Mining*, 2018, pp. 387–395, doi:
[10.1145/3219819.3219845](https://doi.org/10.1145/3219819.3219845).

[6] C. Fung, S. Srinarasi, K. Lucas, H. B. Phee, and L. Bauer, "Perspectives
from a comprehensive evaluation of reconstruction-based anomaly detection in
industrial control systems," in *Computer Security – ESORICS 2022*, ser.
Lecture Notes in Computer Science. Cham, Switzerland: Springer, 2022,
pp. 493–513, doi:
[10.1007/978-3-031-17143-7_24](https://doi.org/10.1007/978-3-031-17143-7_24).

[7] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, "A dataset to support
research in the design of secure water treatment systems," in *Critical
Information Infrastructures Security (CRITIS 2016)*, ser. Lecture Notes in
Computer Science, vol. 10242. Cham, Switzerland: Springer, 2017, pp. 88–99,
doi: [10.1007/978-3-319-71368-7_8](https://doi.org/10.1007/978-3-319-71368-7_8).
