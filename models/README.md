# Learned artifacts

This fresh candidate contains all twenty public-normalized learned JSON
artifacts and their canonical normalization receipts. The original private
scientific-source bytes are not included. Their immutable provenance
identities remain in `ARTIFACT_MANIFEST.json` with
`availability: missing_pending_verified_copy`; that field describes the
non-public source, not the normalized file beside this document.

Each `public_artifacts` registration records the normalized byte identity,
the exact allowed metadata pointers, equal source/public canonical
inference-parameter hashes, and the SHA-256 of its receipt. The receipt files
are under `normalization_receipts/` and are covered by `SHA256SUMS`.

| Public file | Public bytes | Public SHA-256 | Scientific-source SHA-256 | State |
|---|---:|---|---|---|
| `seed_00/no_radar_calibration.json` | 15327 | `82b1dd59bc76583b02cff023c88597fba74d62d816a8b0623676b3192eefc9af` | `e0cf8b5af14ffdbb139b464aba957acfe4e5b4e9ae1ba77cd8cb61904a6591af` | present, normalized, verified |
| `seed_00/no_radar_model.json` | 11942 | `bc193ff303ec0428c4968127da0f22c2e40508ccba36f8b29e7c5eabcb722438` | `5d84c217937be2467c7a70d94197dd9d8c209c2ef3ba58d31227bceccb2f9125` | present, normalized, verified |
| `seed_00/radar_calibration.json` | 15356 | `b01377b76fd0f6d931d945bd62baf13a93d0a85bad6ae78ce2506a0079451feb` | `52edd05f8af51bcd86021f20834654323c6c80432e807213a36ae9803f5e1946` | present, normalized, verified |
| `seed_00/radar_model.json` | 16336 | `ed8c7a1d4c9f124381ae9a80a9af3ed1f74818b63f96983ebf3202d6cabc9534` | `49f0a7bd3cc8cab03f11da4c4ae8997f83d966f54744a32137892c5d2005a172` | present, normalized, verified |
| `seed_01/no_radar_calibration.json` | 15330 | `7d191d38812ecf0d640c8636c12983dbd0f889c2776e0109b3c9134363f5170b` | `0a9c79502426e2ac07ec049567fd3faaf65a4a74532732339c0870e23ec9dd79` | present, normalized, verified |
| `seed_01/no_radar_model.json` | 11969 | `c27f09a1be21e5fdfc6670d30994896971328e92ae608b3c643462fd22e42578` | `d938a3658aff97f2b7b12513e3367b46b81e559b59b51bd503cd1b7bcbd3eed6` | present, normalized, verified |
| `seed_01/radar_calibration.json` | 15329 | `0bf8066c85462ac0536f6c7cb5bef53578e20730ccb4ba5c63eeffa7208b3cff` | `2ef343fe08cbbdc25dd21a4b07279cea48f3e34382284a3b66c4f4bd0b43cb3d` | present, normalized, verified |
| `seed_01/radar_model.json` | 16312 | `c654a2d0a364a4216bce0e41cea92c342956b1cb2400b4957df05e1a8752fac3` | `2edaba2e6380f7adacc821f516e5f627ac14ebb3814fe977dc81394f5c0266a9` | present, normalized, verified |
| `seed_02/no_radar_calibration.json` | 15347 | `11cc30d794e64d3e4384f0e7dff00014cf7c20c622a348b79f243bca05cdb3d0` | `2f74963302957d055669f073999dae51d5c06b68bbdd41eb4785abd87d122a01` | present, normalized, verified |
| `seed_02/no_radar_model.json` | 11924 | `edd85b16b430a6823b805a9e5160cd6cc1711a3fe158f144610e2cd982436626` | `1574dcaef79cdb8361e80afb76ff31cac305969589f72edae23cfa71b05be7d1` | present, normalized, verified |
| `seed_02/radar_calibration.json` | 15322 | `fc487d20e0a5b16f366c245316b8ae132abc61c21f0a5825414bd6ee3d2a9933` | `8505cf7e9ddc3692eef4878cf9eca763b43fd5b0dc7945ea50bf345090c3bb0d` | present, normalized, verified |
| `seed_02/radar_model.json` | 16290 | `af488736aaac51f54fecc269f3e5484b3ccb149fae2593c76e51e6c84df0cee4` | `16c432f39e58bd3d863cee22ba82722ee3eeebd39d4bde0d4ecdc1d786004797` | present, normalized, verified |
| `seed_03/no_radar_calibration.json` | 15347 | `8503ebfc1dc648f7371c47cbf11e4cbac0cb2398072060a483c921a50942ace7` | `249160b064f7a792bcd71003585d12c4bf69226c321b33b4b837ab05f84810aa` | present, normalized, verified |
| `seed_03/no_radar_model.json` | 11993 | `835bd1c049c2e33cf9dd2b807dbc73373343f24acfa6adc0c979b763b97a9611` | `065402dde03f8bc006868cb30de8d440e08e8f59965a03baf902044dc19cc139` | present, normalized, verified |
| `seed_03/radar_calibration.json` | 15334 | `0ed0aa01203b4f23c6d7ac2408210cd9b046d4551e1a33dd6b8d14b49da58ed2` | `1902a27c05264a7e196ab30808b504a2e996fdfc19b1792286f5594b7b53cc95` | present, normalized, verified |
| `seed_03/radar_model.json` | 16362 | `f6761de050d3c18fb5749506a9466e071f48e87fb7a88d5fbbb99ddd63d4eabe` | `5354084452d6a4a3bec9068d0d7e007174361c6a3b3f5d5f54de890623807c5b` | present, normalized, verified |
| `seed_04/no_radar_calibration.json` | 15350 | `d431e5bb9ba907b9b733dfcb6e9e31c226efd70a3d8d33baafe061829b70f888` | `c840b531d6ccacef339a75e6babb2335240f1a3af1678718eea94c89fc4764f9` | present, normalized, verified |
| `seed_04/no_radar_model.json` | 11934 | `d0774a7463c97c70a69955abc5e2432134c07f468931a9595056d5197a0b1422` | `0134f54f449c078e0a7ac887449bc1ed8f600981a060e5c617000a5c13e7ecf0` | present, normalized, verified |
| `seed_04/radar_calibration.json` | 15331 | `ae410bac79ab43a6dd9f2695e432475ff351d5bcfcc3f9c064202092e20cb408` | `a77f9255bf8e15752bc89a57cf56f802fdfa4721c63ea989e57da3c11089596f` | present, normalized, verified |
| `seed_04/radar_model.json` | 16298 | `3ee7171f60b0c3bec4a855512cd9ce41a37b2e7fcb4a82f25950d3828f6f8e3f` | `bcc2f794f5ac2a421e6636c66d55145fcdf09a14affaedf9ff6e9231afa54812` | present, normalized, verified |

Run `python3 -B verify_bundle.py` from the candidate root. Missing, extra,
misregistered, non-canonical, path-bearing, or hash-inconsistent artifacts
and receipts remain hard failures.
