# R-GATE: Radar as Physical Evidence for Camera–Radar 3D Object Detection

R-GATE is a **diagnosis-driven, decision-level arbitration framework** for camera–radar 3D object detection on the [nuScenes](https://www.nuscenes.org/) dataset.

Rather than retraining another large perception network, R-GATE operates on cached predictions from frozen camera and camera–radar experts. It combines cross-expert agreement with physically interpretable radar evidence, including radar support, spatial context, radar cross section (RCS), and ego-motion-compensated planar-velocity agreement.

This repository is a research-code companion to the accompanying dissertation. It contains the scripts, public-normalized learned artifacts and integrity checks used for the R-GATE experiments, including expert-result fusion, radar-sidecar construction, arbiter training, calibration, official nuScenes evaluation, multi-seed replay and paired scene-bootstrap analysis.

It is not a self-contained end-to-end reproducibility package. The nuScenes data, cached expert predictions, upstream detector code and checkpoints, and the original locked run plans are not distributed here.

---

## 🌟 Highlights

* **Diagnosis-first design**
  Recall, score-ranking, and centre-localisation diagnostics are used to identify where the baseline detector still contains recoverable headroom.

* **Multi-expert arbitration**
  Frozen camera and camera–radar detector outputs are combined using source-specific confidence trust and geometry trust.

* **Explicit radar evidence**
  Candidate-level radar features include support count and density, spatial summaries, RCS statistics, and compensated planar-velocity agreement.

* **Lightweight learned arbiter**
  A compact MLP re-scores candidate hypotheses without retraining the underlying 3D perception networks.

* **Auditable evaluation support**
  The repository provides an example configuration template, bundle verification, synthetic smoke tests, official nuScenes evaluation utilities and multi-seed statistical analysis tools.

---

## 📁 Repository Structure

```text
Rgate/
├── configs/        # Example path template; locked run plans are not distributed
├── models/         # Public-normalized learned artifacts across random seeds
├── scripts/        # Fusion, cache construction, arbiter training and analysis
├── tools/          # Evaluation and supporting utilities
├── tests/          # Synthetic smoke tests and structural verification
├── requirements-core.txt
└── verify_bundle.py
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Joyhaotian/Rgate.git
cd Rgate
```

## 2. Create a Python environment

A dedicated Conda environment is recommended:

```bash
conda create -n rgate python=3.8 -y
conda activate rgate
```

## 3. Install the required dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

> Depending on the experiment you want to reproduce, additional dependencies required by the original detector repositories or the official nuScenes devkit may also be needed.

## 4. Verify the installation

Run the repository bundle verifier:

```bash
python verify_bundle.py
```

Then run the synthetic smoke tests:

```bash
python tests/synthetic_smoke.py
```

These tests are intended to verify the repository structure and core execution paths before running experiments that depend on the full nuScenes dataset.

---

# 📦 External Data and Predictions

The published scripts can be exercised only with external resources that are **not distributed directly in this repository**. Reproducing the exact dissertation runs additionally requires the original locked run plans, which are not published here.

You will need:

1. **nuScenes v1.0**
2. The corresponding nuScenes metadata and detection-evaluation environment
3. Cached prediction JSON files from the detector experts used by R-GATE
4. A local experiment configuration derived from `configs/repro.example.json`

The repository includes public-normalized learned artifacts, but `configs/repro.example.json` is a relative-path template rather than a locked dissertation run plan.

The locked expert pool used in the dissertation consists of:

* CRN ResNet-50 — camera + radar
* RepDETR3D EVA02-L — camera
* RepDETR3D VoVNet — camera
* StreamPETR VoVNet — camera

R-GATE operates on their exported nuScenes-format predictions; the underlying detector networks are not retrained by the arbitration pipeline.

---

# 🔬 Running the Published Scripts

The core scripts are located in:

```text
scripts/
```

The supported command sequence, external-input layout and concrete command
templates are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Use `configs/repro.example.json` as a path-safe template. It does not encode the
locked operating points or full inputs used for the dissertation results.

---

## Recommended Replay Workflow

A typical R-GATE replay follows the sequence:

```text
Frozen expert predictions
        ↓
Prediction normalization / caching
        ↓
Candidate grouping
        ↓
Cross-expert evidence construction
        ↓
Radar-sidecar evidence construction
        ↓
Rule-based arbitration
        ↓
Learned arbiter
        ↓
Confidence calibration
        ↓
nuScenes-format prediction export
        ↓
Official nuScenes evaluation
```

The pipeline is designed so that most arbitration experiments can operate on cached detector outputs without repeatedly running the original large perception networks.

---

# 📊 Main Results

The following are locked validation results reported in the dissertation. They are not one-command outputs from a fresh checkout of this repository:

| Configuration                               |     mAP ↑ |     NDS ↑ |    mAVE ↓ |
| ------------------------------------------- | --------: | --------: | --------: |
| CRN-R50 standalone                          |     47.24 |     56.17 |     0.274 |
| Same-pool equal-vote WBF                    |     47.66 |     57.76 |     0.236 |
| E1: trusted anchors + cross-expert re-score |     55.16 |     61.98 |     0.212 |
| E2: + auxiliary-only retrieval              |     55.29 |     62.01 |     0.213 |
| E3: + learned arbiter                       |     55.93 |     62.80 |     0.228 |
| **E4: + calibration**                       | **55.91** | **62.82** | **0.227** |
| E4NR: paired no-explicit-radar model        |     55.87 |     62.75 |     0.236 |

Using the **same four frozen experts**, E4 improves over the equal-vote WBF baseline by:

* **+8.26 mAP**
* **+5.06 NDS points**

The learned-arbiter step contributes a material improvement within the registered staircase, while auxiliary-only retrieval and calibration do not individually produce a material mAP gain.

---

# 📡 What Does Explicit Radar Evidence Add?

The CRN base detector already consumes radar internally. Therefore, the E4 versus E4NR comparison measures only the **residual contribution of the explicit candidate-level radar evidence used by the R-GATE arbiter**.

The observed paired difference is:

* **+0.039 mAP**
* **+0.068 NDS**
* **−0.00919 mAVE**

This should be interpreted as a small observed paired contrast rather than a statistically established standalone radar gain.

The larger improvement of R-GATE over equal-vote WBF should therefore **not** be attributed solely to the explicit radar columns.

---

# 📏 Evaluation

R-GATE uses the official nuScenes detection metrics.

The principal reported metrics are:

* **mAP** — mean Average Precision
* **NDS** — nuScenes Detection Score
* **mAVE** — mean Average Velocity Error

Official evaluation should be performed using the same nuScenes split and evaluation configuration as the corresponding experiment.

The repository also includes tools for:

* deterministic evaluation replay,
* provenance and bundle verification,
* multi-seed experiment comparison,
* paired scene-bootstrap analysis.

---

# ⚠️ Scope

R-GATE is a **multi-expert arbitration system**, not a single-model detector.

The headline result therefore demonstrates the value of diagnosis-driven arbitration over a fixed pool of available expert predictions. It should not be interpreted as a like-for-like single-model state-of-the-art comparison.

Similarly, the explicit-radar ablation measures the marginal contribution of the registered radar-sidecar features after radar has already been consumed by the CRN base detector.

---

# 🧪 Reproducibility

The project is designed around cached predictions and lightweight arbitration so that portions of the decision-level experiments can be inspected or repeated without rerunning all detector inference.

This repository is a research-code companion, not a self-contained end-to-end reproduction package. It does not distribute the nuScenes data, cached expert predictions, upstream checkpoints or the original locked run plans.

Where available, experiment outputs are associated with:

* the example configuration template,
* expert-pool manifests,
* learned artifacts,
* run metadata,
* official nuScenes metric outputs,
* provenance checks,
* random-seed information.

Please verify the repository bundle before scientific replay:

```bash
python -B verify_bundle.py --run-smoke
```

For conditional replay with locally prepared nuScenes data and expert
predictions, follow [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Exact numerical
reproduction additionally requires the locked configurations and exact inputs
provided in the separately submitted project code bundle.

---

# 📖 Citation

If you use R-GATE or this repository in academic work, please cite the accompanying dissertation.

```bibtex
@mastersthesis{chen2026rgate,
  author = {Haotian Chen},
  title  = {R-GATE: Using Radar as Physical Evidence for Camera--Radar 3D Object Detection},
  school = {University of Birmingham},
  year   = {2026}
}
```

---

# 📜 License

The original R-GATE source code, documentation and configuration templates are licensed under [Apache-2.0](LICENSE). The license does not grant rights in nuScenes data, cached expert predictions, upstream detector code or checkpoints, or the data-derived learned JSON artifacts in `models/`. Their applicable boundaries remain described in `NOTICE`.

---

## Acknowledgements

This project builds upon the nuScenes benchmark and publicly released camera and camera–radar 3D detection models. Please also cite the original datasets, detector implementations, and model authors when using their corresponding resources.
