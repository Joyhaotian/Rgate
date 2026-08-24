# Release status

Status: **NOT_READY_FOR_PUBLIC_RELEASE**

Scope: local, private review only.

Closed artifact gate:

- all twenty learned JSON artifacts are present only in public-normalized form;
- all twenty source/public parameter and fixed-fixture receipts are registered;
- the private scientific-source identities remain preserved separately.

Open gates:

- no project license has been selected;
- the double-blind publication gate is still open;
- third-party dependency and checkpoint terms require final human review;
- hashed pip and Linux-64 Conda locks now cover the declared core replay
  environment; optional LightGBM, license attribution and any platform/solver
  review remain open;
- private full-run orchestration was excluded and needs a clean public wrapper;
- no independent end-to-end replay has been completed from this directory.

The normalized artifacts close only the learned-artifact availability gate.
Public release still requires every open gate above to close, a clean-tree
re-audit, and a fresh human publication decision. Until then, do not push,
mirror, attach, or publish this directory.
