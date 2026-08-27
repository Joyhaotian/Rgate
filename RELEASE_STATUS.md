# Release status

Status: **PUBLIC_RESEARCH_CODE_COMPANION**

Scope: publicly readable R-GATE scripts, an example configuration template,
public-normalized learned artifacts and integrity checks accompanying the
dissertation.

This repository is not a self-contained end-to-end reproducibility package.
It does not include nuScenes data or annotations, cached expert predictions,
upstream detector source trees or checkpoints, or the original locked run
plans and full experiment inputs.

The twenty learned JSON artifacts are present only in public-normalized form.
Their byte identities, allowed metadata transformations and fixed-fixture
equivalence receipts are registered in `ARTIFACT_MANIFEST.json` and checked by
`python3 -B verify_bundle.py`.

License status: no project license is granted by this repository. Public
readability does not grant rights to reuse, redistribute or commercially use
the project materials. The applicable boundaries for third-party software,
models and data are recorded in `NOTICE`.
