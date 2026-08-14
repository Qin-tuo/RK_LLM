# Generated Artifacts

Generated data belongs below this directory and is ignored by Git:

- `source_models/`: downloaded source weights and tokenizer files;
- `converted_models/`: unpackaged `.rkllm` outputs;
- `packages/`: deployable models and machine-readable manifests;
- `benchmark_runs/`: JSON Lines benchmark records;
- `logs/`: conversion and runtime logs.

Never commit model weights, converted binaries, logs, or benchmark output. Keep only this README in source control.

Each deployable package must contain a manifest with the source repository and immutable revision, RKLLM-Toolkit version, target platform, quantization and conversion options, output filename and SHA-256 checksum, and creation timestamp. Deployment tooling must reject a missing or mismatched checksum.
