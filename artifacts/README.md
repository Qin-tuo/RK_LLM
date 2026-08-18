# Generated Artifacts

Local data belongs in these ignored directories:

- `artifacts/source_models/`: imported source weights and tokenizer files;
- `artifacts/work/`: imported and generated model build inputs;
- `artifacts/packages/`: immutable deployment packages;
- `artifacts/deploy/`: unpacked deployment state;
- `artifacts/logs/`: conversion and runtime logs.

Imported data is local state and must not be committed. It is not itself a
deployment package. The Qwen import record is stored at
`artifacts/work/qwen2_5_0_5b/import-record.json`.

Each deployable package must contain a manifest with the source repository and
immutable revision, toolkit version, target platform, quantization and
conversion options, output filename and SHA-256 checksum, and creation
timestamp. Deployment tooling must reject a missing or mismatched checksum.
