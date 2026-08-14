# Model Export

## Pinned inputs

The initial path is intentionally narrow:

- source: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`;
- revision: `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`;
- RKLLM-Toolkit: `1.3.0`;
- target platform: `rk3588`;
- intended example artifact: `deepseek-r1-distill-qwen-1.5b-w8a8-rk3588.rkllm`.

The source of truth is `third_party/versions.yaml`. Do not use a moving branch or an unrecorded model revision.

## Official conversion flow

1. Prepare the dedicated host environment described in [host setup](host-setup.md).
2. Fetch the source model at the exact revision and verify the repository identity.
3. Use the RKLLM-Toolkit Python conversion API and the matching `airockchip/rknn-llm` `1.3.0` examples. Confirm the supported quantization and target arguments in that release; this skeleton does not guess or wrap them.
4. Write temporary output under `artifacts/converted_models/`.
5. Compute the output SHA-256 checksum and package the model with a manifest under `artifacts/packages/`.
6. Review conversion logs for errors before the package is eligible for deployment.

Conversion success means only that the toolkit produced a checked artifact. It is not proof of native-runner integration, model quality, or successful RK3588 inference.

## Required manifest fields

- source repository and immutable revision;
- toolkit version and target platform;
- quantization mode and every conversion option;
- source tokenizer/config identity;
- output filename, size, and SHA-256 checksum;
- conversion host information and UTC creation timestamp.

Weights, `.rkllm` files, packages, and logs must remain outside Git.
