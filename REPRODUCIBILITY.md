# Conditional Replay Guide

This guide explains how to exercise the public R-GATE research code with
locally prepared inputs. It does not turn the repository into a self-contained
end-to-end reproducibility package: nuScenes, cached expert predictions,
upstream detector code and checkpoints, and the locked dissertation run plans
are not distributed here.

The commands below are interfaces supported by the published scripts. Values
such as expert weights, clustering thresholds, training seeds and calibration
settings are experiment choices. They are deliberately shown as local shell
variables rather than claimed to be the locked dissertation settings.

## 1. Verify the checkout

Use Python 3.8 and install the pinned direct dependencies:

```bash
conda env create -f environment.yml
conda activate rgate-repro
python -B verify_bundle.py --run-smoke
```

The verifier authenticates the repository file set and public-normalized model
artifacts. The smoke test uses synthetic data; it does not require nuScenes.

## 2. Prepare external inputs

Copy `configs/repro.example.json` to an untracked local configuration and
populate the paths it names. From the repository root, the example layout is:

```text
data/nuscenes_raw/                         nuScenes v1.0-trainval
artifacts/metadata/sample_info.pkl         sample metadata used by the cache builder
artifacts/expert_results/expert_0.json     nuScenes-format expert prediction
artifacts/expert_results/expert_1.json     nuScenes-format expert prediction
artifacts/expert_results/expert_2.json     nuScenes-format expert prediction
artifacts/expert_results/expert_3.json     nuScenes-format expert prediction
artifacts/cache/                           generated locally
outputs/                                   generated locally
```

Each prediction file must be a nuScenes detection submission with a top-level
`results` mapping keyed by sample token. All experts should cover the same
evaluation split. The four positions in the example config correspond to the
fixed expert order documented in the root README, but the template does not
contain the dissertation's locked weights or operating points.

The scripts accept an expert in either of these forms:

```text
NAME:WEIGHT:RESULT_JSON
NAME:SCORE_WEIGHT:GEOMETRY_WEIGHT:GROUP:MODE:RESULT_JSON
```

`MODE` is `full`, `no_velocity`, or `candidate_only`. Paths containing a colon
are therefore unsuitable for this argument; run under Linux or WSL and use
repository-relative paths as in the example config.

## 3. Build a candidate cache

Choose local expert specifications. The equal weights below demonstrate the
interface only and are not the dissertation's locked settings:

```bash
EXPERT_0='expert_0:1.0:artifacts/expert_results/expert_0.json'
EXPERT_1='expert_1:1.0:artifacts/expert_results/expert_1.json'
EXPERT_2='expert_2:1.0:artifacts/expert_results/expert_2.json'
EXPERT_3='expert_3:1.0:artifacts/expert_results/expert_3.json'

python -B scripts/build_bge_af_arbiter_cache.py \
  --expert "$EXPERT_0" \
  --expert "$EXPERT_1" \
  --expert "$EXPERT_2" \
  --expert "$EXPERT_3" \
  --target-from-info-gt \
  --sample-info-pkl artifacts/metadata/sample_info.pkl \
  --include-radar-evidence \
  --radar-from-nuscenes-tables \
  --nuscenes-root data/nuscenes_raw \
  --stream-build \
  --out artifacts/cache/fullval_cache.jsonl \
  --summary artifacts/cache/fullval_cache_summary.json
```

`--target-from-info-gt` is needed when training from nuScenes ground truth. If
the metadata pickle already embeds radar sweeps, omit
`--radar-from-nuscenes-tables` and `--nuscenes-root`. For an unlabeled inference
cache, omit the target option; training and calibration require labeled rows.

To inspect rule-based weighted fusion independently of the learned arbiter:

```bash
python -B scripts/fuse_nuscenes_expert_results.py \
  --expert "$EXPERT_0" \
  --expert "$EXPERT_1" \
  --expert "$EXPERT_2" \
  --expert "$EXPERT_3" \
  --out outputs/equal_weight_fusion.json \
  --summary outputs/equal_weight_fusion_summary.json
```

## 4. Apply an included learned artifact

The public-normalized artifacts preserve inference parameters; their path-only
metadata was normalized and is audited by `ARTIFACT_MANIFEST.json` and the
receipts under `normalization_receipts/`.

For example, apply the seed-00 explicit-radar model and calibration table:

```bash
python -B scripts/apply_bge_af_arbiter.py \
  --cache-jsonl artifacts/cache/fullval_cache.jsonl \
  --model models/seed_00/radar_model.json \
  --calibration-table models/seed_00/radar_calibration.json \
  --meta-from-result-json artifacts/expert_results/expert_0.json \
  --stream-output \
  --out outputs/seed_00_radar.json \
  --summary outputs/seed_00_radar_summary.json
```

The cache feature schema and experiment choices must match those expected by
the selected artifact. A successful command proves interface compatibility;
matching the dissertation numbers also requires matching its unavailable
locked inputs and settings.

## 5. Train and calibrate a local arbiter

To fit a new local model rather than use the included learned artifacts:

```bash
python -B scripts/train_bge_af_arbiter.py \
  --cache-jsonl artifacts/cache/fullval_cache.jsonl \
  --model-kind mlp \
  --seed local-replay-00 \
  --out-model outputs/local_model.json \
  --summary outputs/local_training_summary.json

python -B scripts/fit_bge_af_score_calibration.py \
  --cache-jsonl artifacts/cache/fullval_cache.jsonl \
  --model outputs/local_model.json \
  --out outputs/local_calibration.json \
  --summary outputs/local_calibration_summary.json

python -B scripts/apply_bge_af_arbiter.py \
  --cache-jsonl artifacts/cache/fullval_cache.jsonl \
  --model outputs/local_model.json \
  --calibration-table outputs/local_calibration.json \
  --meta-from-result-json artifacts/expert_results/expert_0.json \
  --stream-output \
  --out outputs/local_submission.json \
  --summary outputs/local_submission_summary.json
```

Record every non-default option, input hash and random seed for a scientifically
comparable local experiment.

## 6. Run official nuScenes evaluation

First run the evaluation tool without `--execute` to obtain a preflight report:

```bash
python -B tools/nuscenes_official_detection_eval.py \
  --nusc-root data/nuscenes_raw \
  --version v1.0-trainval \
  --eval-set val \
  --candidate local=outputs/local_submission.json \
  --output-dir outputs/official_eval \
  --out outputs/official_eval_preflight.json \
  --md outputs/official_eval_preflight.md
```

If the preflight is ready, repeat the command with `--execute`. Official mAP,
NDS and error metrics come from the resulting nuScenes `metrics_summary.json`,
not from the preflight report.

## 7. Exact dissertation-stage replay

`scripts/materialize_rgate_rq2_stage_results.py` and the paired-bootstrap tools
support the registered dissertation workflow. They require a locked execution
config, its SHA-256 identity, the exact cache and summary, the four expert
predictions, nuScenes tables and sufficient output storage. Those inputs are
part of the separately submitted project code bundle, not this public
repository. Use each script's `--help` output together with that bundle; do not
substitute `configs/repro.example.json` and describe the result as an exact
reproduction.

## Interpretation

There are therefore three distinct claims:

1. A fresh checkout can verify its integrity and exercise core paths on
   synthetic fixtures.
2. With locally prepared nuScenes data and expert predictions, the public code
   can be used for conditional replay and new experiments.
3. Exact numerical reproduction of the dissertation additionally requires its
   locked configurations and exact external inputs.

This repository supports the first two claims. The separately submitted code
bundle is the source for the third.
