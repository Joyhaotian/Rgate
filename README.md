# R-GATE reproducibility candidate

This directory is a double-blind, local-review candidate for reproducing the
R-GATE fusion and evaluation pipeline. It contains the scientific core, hash
manifests, a structural verifier, and a synthetic test. It contains no paper
source, identity metadata, private run records, dataset files, external expert
checkpoints, or measured result tables.

The candidate is **not ready for public release**. Twenty public-normalized learned
artifacts and their receipts are present, but no project license has been selected, and the
double-blind release gate has not closed. Do not create a public repository or
upload this directory in its current state. See `RELEASE_STATUS.md`.

## What is included

- fusion, cache, arbiter training/application, stage materialization, and
  independent checking scripts;
- deterministic result archiving, scene-event construction/checking, paired
  scene bootstrap, and bootstrap-analysis scripts;
- a wrapper around the official nuScenes detection evaluator;
- a relative-path example configuration;
- separate private-source and public-normalized identities for twenty learned
  artifacts across five random seeds, with canonical receipts;
- a pure-standard-library synthetic test, a fail-closed verifier, and audited
  learned-artifact normalizer and all-or-nothing release assembler.

Private full-run orchestration wrappers are not included. A future public
release must replace them with clean, relative-path wrappers before claiming a
one-command full reproduction.

## Reproduction levels

1. **Structure audit (works now).** Checks file closure, hashes, import closure,
   relative paths, anonymous-release rules, Python syntax, and all twenty
   normalized artifact/receipt registrations.
2. **Synthetic smoke (works now).** Exercises score calibration, range bins,
   reliability-aware fusion, streaming nuScenes JSON parsing, and deterministic
   result archiving without third-party packages or real data.
3. **Core scientific replay (external inputs required).** The learned JSONs are
   present; replay additionally requires nuScenes data and four expert result JSONs.
4. **End-to-end expert inference (external).** Requires upstream expert code and
   checkpoints under their own terms. Those materials are not redistributed.

## Quick checks

From this directory:

```bash
python3 -B verify_bundle.py
python3 -B -m unittest \
  tests.synthetic_smoke \
  tests.test_verify_bundle \
  tests.test_normalize_learned_artifact \
  tests.test_assemble_normalized_release
```

The verifier must return zero with no missing-artifact warnings. Every
public-normalized artifact identity, receipt, and parameter/fixture-equivalence
proof is registered and present. The table in `models/README.md` identifies the private
scientific sources, not bytes that may be published unchanged. Any file placed
at a model path without a valid public registration is an error, even with
`--allow-missing-models`.

## Learned-artifact recovery

`tools/normalize_learned_artifact.py` is the only bundled path for converting
one authenticated private learned JSON into a public candidate. It checks the
source bytes against the v2 private size/SHA registry before parsing them, then
changes exactly the registered metadata pointers:

- model: `/training/cache_jsonl`;
- calibration: `/cache_jsonl` and `/model`.

The replacements are fixed anonymous relative paths. After removing those
pointers, the source and public canonical inference-parameter hashes must be
identical. Model artifacts are also compared through the bundled
`model_score` implementation on a deterministic synthetic fixture;
calibration artifacts exercise every registered bin and the global fallback
through the bundled calibration path.

The command accepts one source and creates one entirely new stage outside this
candidate:

```bash
mkdir -p ../public-model-stages
python3 -B tools/normalize_learned_artifact.py \
  --artifact-id seed_00_radar_model \
  --source-json ../private-model-recovery/seed_00_radar_model.json \
  --staging-dir ../public-model-stages/seed_00_radar_model
```

It never writes into `models/`, never edits `ARTIFACT_MANIFEST.json`, and never
changes an availability state. A successful stage contains the intended
relative model path plus a canonical normalization receipt. The receipt has
the source/public byte identities, exact changed pointers, both canonical
parameter hashes, and the fixed-fixture/output hashes; its whole-file SHA-256
is the future `fixture_equivalence_receipt_sha256`. Independent review and
explicit registration are still required before anything is copied into this
candidate. The command will fail now unless the exact private source is
supplied externally. Publication uses Linux `renameat2(RENAME_NOREPLACE)` and
rejects symbolic links in source, manifest, or staging path components. If a
failure occurs after the hidden transaction directory is created, the tool
leaves that external `.rgate-normalize-*` directory for explicit inspection;
it does not publish it and does not recursively delete a path that another
process could have exchanged. The parent is anchored by file descriptor and
checked outside the candidate at entry, immediately before publication, and
again after publication. As with the rest of this local candidate, this assumes
there is no adversarial same-user process relocating user-owned directories
after the final check; such a process can move any user-owned output
independently of this tool.

After all twenty external stages exist, assemble them only into a new sibling
candidate. The output argument has no default and must name a path that does
not exist:

```bash
python3 -B tools/assemble_normalized_release.py \
  --source-candidate . \
  --stages-root ../public-model-stages \
  --output-candidate ../rgate_github_upload
```

The assembler requires exactly one stage for every registered artifact, checks
the canonical receipt and public JSON byte identities, changed pointers,
source/public parameter hashes, fixture hashes, implementation identities, and
anonymous relative paths, then constructs a complete fresh clone. It preserves
the private-source provenance registry, license-pending state, and double-blind
gate. Only after every output byte and the new `SHA256SUMS` are verified does it
publish the directory using `renameat2(RENAME_NOREPLACE)`; unsupported
filesystems and pre-existing or concurrently appearing targets are hard
failures.

## Environment

The structural verifier and synthetic test use only the standard library and
are compatible with the registered Python 3.8.20 environment as well as newer
Python 3 releases. Scientific replay uses these registered direct dependency
pins:

```bash
conda env create -f environment.yml
conda activate rgate-repro
```

The selected MLP artifacts are JSON and do not require LightGBM for inference
or registered MLP training. The copied training script retains an optional
LightGBM code path, but LightGBM is deliberately outside the core environment.
`requirements-core-py38-linux-64.lock` now records a 24-package, hashed pip
wheel resolution for CPython 3.8 on Linux x86_64. It does not claim to lock the
`environment-linux-64.conda.lock` additionally records the 26 exact conda-forge
package URLs and SHA-256 values for the Python 3.8.20 + pip base. Both locks
were dry-run validated without installing packages. They are scoped to Linux
x86_64 and the optional LightGBM branch remains outside them; license,
checkpoint-term and public-release review are still required.

For the reproducibility path on Linux x86_64, use the explicit Conda lock and
then the hashed pip lock (rather than editing the tracked files):

```bash
conda create --name rgate-repro --file environment-linux-64.conda.lock
conda run --name rgate-repro python -m pip install \
  --require-hashes --only-binary=:all: \
  -r requirements-core-py38-linux-64.lock
```

## Data layout

Copy `configs/repro.example.json` outside this integrity-covered directory and
edit the copy for a local run. Runtime inputs and outputs are ignored by the
release verifier; do not add them to `SHA256SUMS` or commit them. The expected
logical layout is:

```text
artifacts/
  expert_results/expert_0.json ... expert_3.json
  cache/fullval_cache.jsonl
  cache/fullval_cache_summary.json
  metadata/sample_info.pkl
data/nuscenes_raw/
models/
outputs/
```

The nuScenes data and expert outputs are user-supplied inputs. Their absence is
not hidden by placeholder files. Avoid symlinks inside the release-covered
source/config/model tree; runtime roots are outside that closure.

## Upstream experts

The four detector checkpoints are external inputs, not R-GATE weights. They are
not redistributed by this candidate. Final public instructions must add an
upstream repository, commit and license decision for each identity.

The checkpoint byte sizes and SHA-256 values below are provenance identifiers
for external inputs used in the registered experiments. They are not download
entitlements or redistribution grants. A source-code repository license is not
treated here as evidence that a separately hosted checkpoint may be
redistributed. This repository neither ships nor mirrors those checkpoint
bytes; reproducers must obtain them from the upstream authors, verify the exact
bytes, and review the terms applicable to each asset. See `NOTICE` for the
registered upstream code revisions and the current license-audit boundary.

`nuscenes-devkit` is software, whereas nuScenes is a separately licensed
dataset. Installing the devkit does not grant dataset access or dataset rights.
Dataset files, annotations, and dataset-derived expert prediction JSONs remain
outside this repository.

| Frozen expert | Bytes | Registered SHA-256 | Redistribution |
|---|---:|---|---|
| CRN R50 | 247343374 | `f725aafc7f033f484f13704134178f2377b3e12bb7e85207b447f8a3ebfa0cb3` | no/unknown |
| StreamPETR VoV | 337742907 | `c9a6b1684cd84a867fc28dfab65571f1536b0eb96af7c18a1c32d363b3823c2d` | no/unknown |
| RepDETR3D VoV | 361967038 | `05482cb475263c62569188e48d7af1b50d769232bc6de85215216e3edad7e240` | no/unknown |
| RepDETR3D EVA02-L | 1337427977 | `e0a7bc16a9e3ab79c7306852e200c9f0f6521eecf9e3b316a85ddb762fd2f85f` | no/unknown |

## Core commands

Each script documents its complete CLI:

```bash
python3 -B scripts/fuse_nuscenes_expert_results.py --help
python3 -B scripts/apply_bge_af_arbiter.py --help
python3 -B scripts/materialize_rgate_rq2_stage_results.py --help
python3 -B scripts/build_nuscenes_scene_metric_events.py --help
python3 -B scripts/run_nuscenes_paired_scene_bootstrap.py --help
```

For a full stage run, explicitly pass a real storage root with
`--storage-mount`; the bundled default is deliberately relative and is suitable
only for inspection or a bounded smoke using the script's test flag. Every
scientific input should be closed by size and SHA-256 before execution.

## Integrity

`SHA256SUMS` covers every regular candidate file except itself.
`SOURCE_MANIFEST.json` records the source identity and bundled identity of every
copied script. Three bundled copies contain documentation/default-path cleanup;
their scientific computations are unchanged. `ARTIFACT_MANIFEST.json` is the
machine-readable learned-artifact gate. Its registered hashes identify private
scientific sources; a future public artifact must have a separate normalized
identity plus parameter and fixed-fixture equivalence proofs.

Do not edit a covered file without regenerating `SHA256SUMS` and rerunning both
checks. Do not interpret a passing structure audit as a reproduced scientific
result.

Citation metadata and a public repository URL will be added only after the
double-blind gate closes. Do not add author-identifying metadata to this private
candidate.
#   R g a t e  
 