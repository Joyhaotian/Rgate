# R-GATE: Radar-Camera Fusion Pipeline

Welcome to the official repository for **R-GATE**! This project provides a robust, reliable fusion and evaluation pipeline for radar-camera 3D object detection on the nuScenes dataset.

## 🌟 Overview

R-GATE (Reliability-Aware Gating) addresses the challenge of combining dense, high-resolution camera data with sparse, noisy, but geometrically accurate radar data. This repository contains the complete scientific core and evaluation scripts used to reproduce the R-GATE fusion results.

### Key Features
- **Reliability-Aware Fusion**: Dynamically arbitrates between camera and radar proposals based on estimated radar reliability.
- **Score Calibration**: Integrates BGE (Background Generation Error) and AF (Assertion Failure) metrics to calibrate confidence scores.
- **Deterministic Evaluation**: Wraps the official nuScenes detection evaluator for strict, reproducible NDS and mAP calculations.
- **Paired Scene Bootstrap**: Provides statistical significance testing tools for multi-seed performance analysis.

## 📁 Repository Structure

- `configs/` - Example configurations for running the pipeline.
- `models/` - Public-normalized learned artifacts and model checkpoints (across 5 random seeds).
- `scripts/` - Core execution scripts (fusion, cache building, arbiter training, bootstrap analysis).
- `tools/` - Utility scripts including the official nuScenes evaluator wrapper.
- `tests/` - Synthetic smoke tests and structural verification.

## 🚀 Getting Started

### 1. Environment Setup
The project supports Conda for easy environment management. You can create the environment using the provided lock files:
```bash
conda create --name rgate --file requirements-core.txt
```

### 2. Running Verification & Smoke Tests
To ensure the repository is structurally sound and your environment is set up correctly, run the synthetic smoke test and bundle verifier:
```bash
# Verify file closure, hashes, and artifact receipts
python verify_bundle.py

# Run pure-standard-library synthetic tests
python tests/synthetic_smoke.py
```

### 3. Core Scientific Replay
To fully replay the pipeline, you will need:
1. The official nuScenes dataset (v1.0).
2. The initial camera expert result JSONs.

Once downloaded, you can use the scripts in `scripts/` (such as `fuse_nuscenes_expert_results.py`) to run the fusion and evaluation.

## 📊 Evaluation Metrics
This project uses the official nuScenes detection metrics. The primary metrics evaluated are:
* **NDS** (nuScenes Detection Score)
* **mAP** (mean Average Precision)

We strictly separate radar and no-radar baselines to analyze the exact contribution of the R-GATE fusion.

## 📜 License
*Please refer to `LICENSE_PENDING.md` (or the finalized LICENSE file) for usage terms.*