# Generated Artifacts

Local data belongs in these ignored directories:

- `artifacts/source_models/`: imported source weights and tokenizer files;
- `artifacts/work/`: imported generated outputs and vendor Demo inputs;
- `artifacts/packages/`: host-created immutable deployment packages;
- `artifacts/deploy/`: board releases and the active relative `current` link;
- `artifacts/logs/`: conversion and runtime logs.

Imported data is local state and must not be committed. It is not itself a
deployment package. Qwen import records are stored at
`artifacts/work/<model_id>/import-record.json`. For `qwen3_4b`, the verified
vendor Demo is imported at
`artifacts/work/qwen3_4b/install/rknn_Qwen3_demo`, the host package is published
at `artifacts/packages/qwen3_4b/<package_id>`, and the board receives that one
package at `artifacts/deploy/releases/$PACKAGE_ID`. The relative
`artifacts/deploy/current` symlink selects the active release.

Each deployable package must contain a manifest with the source repository and
immutable revision, toolkit version, target platform, quantization and
conversion options, output filename and SHA-256 checksum, and creation
timestamp. Deployment tooling must reject a missing or mismatched checksum.
Neither the imported work tree nor a deploy release is committed to Git.
