# Datasets: glossary, inventory, and selection rationale

UREx CDE2605R — Benchmarking gradient boosting against deep learning for anomaly detection in industrial control systems
Santhosh, Year 2 Industrial and Systems Engineering, National University of Singapore
Last updated 14 August 2026

---

## 1. Glossary

Terms are defined once here and used without expansion elsewhere in the project.

### Domain

| Term | Expansion | Meaning |
|---|---|---|
| **ICS** | Industrial Control System | The computers, controllers, and networks that operate a physical industrial process: a water plant, a power station, a production line. Distinct from ordinary IT because a fault has physical consequences. |
| **CPS** | Cyber-Physical System | Broader term for any system where computation and physical processes are tightly coupled. An ICS is a type of CPS. Papers use the two almost interchangeably. |
| **SCADA** | Supervisory Control and Data Acquisition | The supervisory layer of an ICS: the software and hardware that collects readings from remote equipment, displays them to an operator, and issues control commands. Often used loosely as a synonym for the whole control system. |
| **PLC** | Programmable Logic Controller | The industrial computer that directly controls equipment. Reads sensors, applies control logic, drives actuators. Where the actual decisions happen. |
| **HIL** | Hardware-In-the-Loop | A simulation technique where real physical hardware is connected to a simulated model of the rest of the system. Lets researchers study a large plant without building one. |
| **Sensor / Actuator** | — | A sensor measures (flow, pressure, level, voltage). An actuator acts (pump, valve, motor). Both appear as columns in these datasets. |
| **Testbed** | — | A physical scaled-down plant built for research, so that attacks can be launched safely and real data collected. |
| **Modbus** | — | A widely used industrial communication protocol. Appears in network-traffic captures. |
| **pcap** | Packet capture | Raw network traffic file format. Some datasets ship these alongside the sensor readings. |

### Method

| Term | Expansion | Meaning |
|---|---|---|
| **GBDT** | Gradient Boosted Decision Trees | The traditional ML family under test. Concrete implementations: XGBoost, LightGBM, CatBoost. Builds many shallow trees sequentially, each correcting the previous ones. |
| **LSTM** | Long Short-Term Memory | A recurrent neural network architecture designed to carry information across long time gaps. The standard deep baseline for sequential sensor data. |
| **Autoencoder** | — | A network trained to reconstruct its own input. Trained only on normal data, it reconstructs normal behaviour well and anomalies badly, so reconstruction error becomes the anomaly score. |
| **Reconstruction-based detection** | — | The above principle in general. Contrasted with forecasting-based detection, which predicts the next value and flags large prediction errors. |
| **Point-adjust** | — | An evaluation protocol where flagging any single timestep inside an attack window credits the model with detecting the whole window. Dominant in the literature and known to be badly inflationary. |
| **MCC** | Matthews Correlation Coefficient | A single-figure classification metric that stays meaningful under heavy class imbalance, unlike accuracy. |
| **AUC / ROC-AUC / PR-AUC** | Area Under the Curve | Threshold-free performance measures. PR-AUC is the more informative of the two when anomalies are rare. |

### Institutional

| Term | Expansion |
|---|---|
| **iTrust** | Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD). Operates the testbeds and distributes most of the datasets used here. |
| **SUTD** | Singapore University of Technology and Design |
| **NSR / ETRI** | National Security Research Institute, an affiliated institute of ETRI, South Korea. Produced the HAI dataset. |
| **PUB** | Singapore's national water agency, which advised on the SWaT testbed. |
| **EMA** | Energy Market Authority of Singapore, which advised on the EPIC testbed. |

---

## 2. Why dataset choice is the central design decision

The research question is *when* gradient boosting matches or beats deep learning for ICS anomaly detection, and where it breaks down. That question can only be answered by datasets that differ along the axes hypothesised to drive the answer.

The dominant axis is **temporal dependency**: how much of the signal that distinguishes an attack from normal operation lives in the *sequence* rather than in any single timestep.

- Where anomalies are visible in a single row of sensor readings, the problem is effectively tabular. Gradient boosting is expected to be strong here, and cheaper and more interpretable than any deep model.
- Where anomalies are only visible as a departure from the recent trajectory, or as a broken relationship between correlated signals over time, a sequence model has a structural advantage that boosting must approximate through hand-built lag features.

A benchmark drawn only from high-temporal-dependency testbeds can conclude "use deep learning" and nothing else. That is not a decision framework. **The portfolio must span both ends of the axis**, and this is the single most important constraint on dataset selection.

Secondary axes, varied experimentally rather than by dataset choice: class imbalance (anomaly rate), label availability (supervised versus unsupervised regimes), and process type (water treatment versus power generation, to test whether findings generalise beyond one physical domain).

---

## 3. Inventory: what I have

Requested from iTrust on 14 August 2026 and approved. Processing time is up to three working days.

| Dataset | Source | Access | Role in this project |
|---|---|---|---|
| **SWaT** | iTrust, SUTD | Requested 14 Aug 2026 | **Primary.** Field-standard sequential benchmark |
| **WADI** | iTrust, SUTD | Requested 14 Aug 2026 | **Primary.** High-dimensional stress case |
| **BATADAL** | iTrust, SUTD | Requested 14 Aug 2026 | Secondary. Small, clean, useful for fast iteration |
| **EPIC** | iTrust, SUTD | Requested 14 Aug 2026 | Reserve. Cross-domain generalisation check |
| **CISS** | iTrust, SUTD | Requested 14 Aug 2026 | Not planned for use |
| **Blaq_0** | iTrust, SUTD | Requested 14 Aug 2026 | Not planned for use |
| **IoT** | iTrust, SUTD | Requested 14 Aug 2026 | Not planned for use |
| **HAI** | NSR / ETRI, South Korea | Open on GitHub, no application | **Primary.** Already familiar from prior work |
| **Morris SCADA** | Mississippi State University | Open | **Primary.** The low-temporal-dependency contrast case |

All seven iTrust datasets were requested in a single submission because the form is per-organisation rather than per-project, and a second request would cost another three days. Holding a dataset is not a commitment to analyse it. The working set is deliberately four: SWaT, HAI, Morris SCADA, and WADI.

---

## 4. The datasets in detail

### SWaT — Secure Water Treatment
**iTrust, SUTD. Primary dataset.**

A scaled-down but high-fidelity, industry-compliant emulation of a modern six-stage water treatment facility, operating at roughly 19 litres per minute. It was the first testbed built by iTrust, funded by Singapore's Ministry of Defence and guided by PUB, and commissioned in March 2015.

The core release covers eleven days of continuous operation, seven under normal conditions and four with attack scenarios, recording all values from 51 sensors and actuators at one-second intervals plus the network traffic, with every row labelled normal or abnormal. Attacks were designed from attack models developed by the iTrust team rather than injected synthetically.

**Why it is in the study.** It is the de facto standard benchmark in ICS anomaly detection, which means results here are directly comparable against a large published literature. That comparability is what makes the point-adjust critique demonstrable rather than merely asserted. It also sits firmly at the high-temporal-dependency end of the axis: many SWaT attacks are stealthy manipulations that look unremarkable in any single row and are only detectable as a violation of the process dynamics.

**Versions.** iTrust distributes many runs, from the original December 2015 collection through to releases in 2026. The December 2015 A1 and A2 physical data is the version used by almost all published work, so it is the one to use for comparability. Record the exact version in the report.

### WADI — Water Distribution
**iTrust, SUTD. Primary dataset.**

An extension of SWaT representing a water *distribution* network rather than a treatment plant. Sixteen days of continuous operation, fourteen normal and two under attack, across 123 sensors and actuators, with fifteen attacks launched during the final two days. The attack model treats the intent space of the system as the attack model, and includes physical effects such as malicious chemical injection and water leakage alongside cyber attacks.

**Why it is in the study.** WADI more than doubles SWaT's dimensionality while dropping the anomaly rate substantially, which makes it the high-dimensional, severely imbalanced stress case. If gradient boosting's advantage is partly about handling many weakly informative features efficiently, WADI is where that should show.

**Known problems.** It is large and contains many missing values, which import as NaN. Imputation strategy must be decided and documented, and must be identical across model families or the comparison is invalid.

### HAI — HIL-based Augmented ICS Security Dataset
**NSR, Affiliated Institute of ETRI, South Korea. Primary dataset. Openly available on GitHub and Kaggle.**

Collected from a realistic ICS testbed augmented with a hardware-in-the-loop simulator emulating steam-turbine power generation and pumped-storage hydropower. The testbed combines three physical control systems, a GE turbine, an Emerson boiler, and a FESTO water treatment system, joined through a dSPACE HIL simulator. Prior to 2018 these three ran independently; the HIL simulator was introduced specifically to couple them, ensuring the variables are highly correlated and the resulting data richer.

Versions: HAI 1.0 released February 2020 with 38 attacks, HAI 20.07 as a debugged release for the HAICon 2020 competition, and HAI 21.03 built on a more tightly coupled simulator with additional attacks and clearer attack effects.

**Why it is in the study.** Three reasons. It is a different physical domain from the iTrust water testbeds, so any finding that holds on both SWaT and HAI is not an artefact of water treatment. It has no access gate, so work could begin immediately rather than waiting on approval. And I have prior working experience with it from XGBoost and dimensionality reduction work at Hanyang University, which reduces the risk of the schedule slipping in week one.

**Note on evaluation.** The HAI authors propose their own metric, TaPR, designed for time-series anomaly detection. Worth reading, but this project reports pointwise and point-adjusted metrics as its primary results to remain comparable with the wider literature.

### Morris SCADA datasets
**Mississippi State University. Primary dataset. Openly available.**

A family of ICS intrusion detection datasets produced by Morris and colleagues from a laboratory control system testbed, most commonly the gas pipeline and power system releases. Each record is largely a snapshot of network and process state with a label.

**Why it is in the study.** This is the essential counterweight. SWaT, WADI, and HAI all sit at the high-temporal-dependency end, and a portfolio consisting only of those can produce a framework with a single branch. The Morris data is closer to independent tabular records, which is precisely the regime where tree-based models are expected to win outright. Without at least one dataset of this type, the claim "use gradient boosting under these conditions" has no empirical support and the framework collapses into a recommendation to always use deep learning.

**Caveat to verify in week 3.** The degree to which these records are genuinely independent needs checking rather than assuming. If strong temporal structure turns out to be present, an alternative low-dependency case is required, and Tennessee Eastman Process data under steady-state conditions is the fallback.

### EPIC — Electric Power and Intelligent Control
**iTrust, SUTD. Held in reserve.**

A small-scale electric power generation and distribution testbed, iTrust's third, integrated in 2017 under guidance from Singapore's Energy Market Authority. It spans a micro-grid stage with generator-coupled motors, 110 rooftop photovoltaic cells and batteries; a transmission stage with an autotransformer; and a smart-home load stage. It can supply power to the SWaT and WADI testbeds concurrently. The dataset covers eight operating scenarios of roughly 30 minutes each, giving over 5,000 sensor and actuator readings with corresponding Modbus network traffic, shipped as both pcap and csv.

**Why it is held back.** A different process domain is genuinely valuable for testing generalisation, but roughly four hours of data in total is thin for training a sequence model, and the value is marginal until the primary results exist. Revisit at week 9 if the schedule allows.

### BATADAL — Battle of the Attack Detection Algorithms
**iTrust, SUTD. Secondary.**

A water distribution dataset originating from a public algorithm competition. Small and clean relative to the others.

**Role.** Useful as a fast smoke test for the pipeline, since a full experiment run completes quickly. Not intended to carry any headline result.

### CISS, Blaq_0, IoT
**iTrust, SUTD. Held but not used.**

- **CISS** (Critical Infrastructure Security Showdown, originally the SUTD Security Showdown) is data collected while live red teams attacked the SWaT testbed in real time. The 2019 run produced 136,805 rows of state information from 28 sensors at one-second intervals across four days. The 2020 release comprises Target files of roughly an hour each under normal operation and CISS_OL files of roughly four hours each under attack, with Target files carrying 82 columns and CISS_OL files 97, the extra columns describing the attack itself: launch and update times, target, type, intent, mode, outcome, anonymised attacker ID, and attack IDs.
- **Blaq_0** is derived from the EPIC testbed, collected during a three-day hackathon in January 2018 where SUTD student teams attacked EPIC and were scored on intent-based success. It is network-level data and is not widely used.
- **IoT** covers Internet of Things device data rather than a process control plant.

**Why not used.** CISS is interesting precisely because its attacks are adversarially designed rather than scripted, and its per-attack metadata would support fine-grained analysis. But it is SWaT again, so it adds no new position on the temporal-dependency axis, and the richer schema is a week of preprocessing work for a marginal gain. Blaq_0 is network-level and lightly used. IoT is outside the ICS scope of the project. All three are documented here so that the decision to exclude them is deliberate and defensible rather than an oversight.

---

## 5. Selection summary

| Dataset | Domain | Approx. sensors | Temporal dependency | Role |
|---|---|---|---|---|
| Morris SCADA | Gas pipeline / power | Low | **Low** (to verify) | Contrast case: where boosting should win |
| SWaT | Water treatment | 51 | High | Primary benchmark, literature comparability |
| HAI | Power generation | High | High | Cross-domain replication |
| WADI | Water distribution | 123 | High | High-dimensional, severely imbalanced stress case |
| BATADAL | Water distribution | Low | Moderate | Fast smoke test |
| EPIC | Electric power | Moderate | High | Reserve, week 9 if ahead |

The framework's credibility rests on the first row. Everything else establishes where deep learning wins; only the contrast case can establish where it does not.

---

## 6. Attribution and terms

iTrust distributes its datasets free of charge to researchers worldwide, on a good faith and as-is basis. Two conditions bear directly on this project:

1. **Credit is required.** iTrust must be explicitly credited as the source in any publication or report. The conventional form is: *"The SWaT, WADI, and BATADAL datasets were provided by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design."*
2. **No support is provided.** Since 1 July 2022 iTrust does not answer queries or provide clarifications about the datasets regardless of when they were downloaded. Bug reports are welcomed but carry no obligation to reply. **Every schema ambiguity must be resolved independently**, which is why week 1 is allocated to reading documentation rather than modelling.

HAI is credited to the National Security Research Institute, an affiliated institute of ETRI, South Korea, citing Shin, Lee, Yun, and Kim, *HAI 1.0: HIL-based Augmented ICS Security Dataset*, USENIX CSET 2020.

Raw data is kept out of version control. `data/raw/` is gitignored.

---

## 7. Open items

- [ ] Confirm iTrust approval email received and archive it, including exact dataset version names for the methodology section
- [ ] Verify disk space before downloading WADI
- [ ] Download HAI 21.03 from GitHub and read the technical details PDF
- [ ] Locate and characterise the Morris SCADA release; verify the low-temporal-dependency assumption empirically rather than by reputation
- [ ] Agree the final dataset list with supervisor at the kickoff meeting
- [ ] Decide and document a single missing-value strategy, applied identically across all model families
