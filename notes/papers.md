

## Primary-protocol sources (read 25 Oct 2026)

**Yu et al. — Unified Reproduction and Event-Level Evaluation of Industrial Multivariate Time-Series Anomaly Detection Methods** — CNIoT 2026 (doi 10.1109/CNIoT69547.2026.11582255)
- Claim: point-level F1 and event-level detection disagree (iForest best point F1, hit ratio 0.14 on SWaT).
- Method: normal-train standardisation, 0.995 quantile of normal-training scores, merge gap 0, min duration 30 (unit unstated).
- So what for my project: the yu26_q995 threshold profile and yu26 event profile.

**Heydari, Nyarko — Alarm-Budgeted Event-Level Evaluation of ICS Anomaly Detection: Lessons from SWaT and WADI** — IEEE Access 14, 2026
- Claim: compare detectors at matched nominal alarm budgets, by missed incidents and delay.
- Method: thresholds on the nominal A1 stream for 0.1/0.2/0.5 events/hour with a 25% duty cap; eventization dilate 2 s / merge 5 s / min 2 s / cooldown 10 s; five seeds; 35 valid A2 intervals.
- So what for my project: the primary operating points and eventization (config.THRESHOLD_PROFILES["budget"], EVENT_PROFILES["hey26"]). Not read in full here -- values taken from the study protocol document.

**Mesca, Pricop, Stamatescu — Rigorous Evaluation of ML Intrusion Detection for Water Treatment Systems on SWaT Network Traffic** — Appl. Sci. 16, 7206, 2026
- Claim: random stratified splits inflate supervised IDS results (XGBoost AUROC 0.884 random -> 0.315 leave-one-attack-out); the unsupervised LSTM-AE generalises best but is bimodal per attack.
- Method: network-traffic modality, attack-held-out and LOO protocols, per-attack AUROC.
- So what for my project: independent support for holding whole attacks out (stratified split, cv.py) and for reporting per-attack outcomes rather than means.

---

## Priority queue 

**Kim, Choi, Choi, Lee, Yoon — Towards a Rigorous Evaluation of Time-series Anomaly Detection** — AAAI 2022
- Claim:
- Method:
- So what for my project:


**Grinsztajn, Oyallon, Varoquaux — Why do tree-based models still outperform deep learning on typical tabular data?** — NeurIPS 2022
- Claim:
- Method:
- So what for my project:


**Fung et al. — Perspectives from a Comprehensive Evaluation of Reconstruction-based Anomaly Detection in ICS** — ESORICS 2022
  (code: github.com/pwwl/ics-anomaly-detection)
- Claim:
- Method:
- So what for my project:


---

## Dataset papers

**Shin, Lee, Yun, Kim — HAI 1.0: HIL-based Augmented ICS Security Dataset** — USENIX CSET 2020


**Goh, Adepu, Junejo, Mathur — A dataset to support research in the design of secure water treatment systems** — CRITIS 2016


**Ahmed, Palleti, Mathur — WADI: a water distribution testbed for research in the design of secure cyber physical systems** — CySWATER 2017


**Morris & Gao — Industrial control system traffic data sets for intrusion detection research** — ICCIP 2014


---

## Surveys, for the background section

**Conti, Donadel, Turrin — A survey on industrial control system testbeds and datasets for security research** — IEEE COMST 2021


**Malhotra et al. — LSTM-based encoder-decoder for multi-sensor anomaly detection** — arXiv 2016

