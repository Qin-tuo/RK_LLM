# Model Inputs and Export Status

## Pinned model

The first supported identity is recorded in
`configs/models/qwen2_5_0_5b.yaml`:

- model: `Qwen/Qwen2.5-0.5B-Instruct`;
- revision: `7ae557604adf67be50417f59c2c2f167def9a775`;
- compiler platform: `rk1820` for the RK1828 accelerator;
- target host: aarch64 RK3588.

The model manifest lists every required source and generated file with an exact
size and SHA-256. `manifests/upstream.yaml` separately pins the RKNN3 Toolkit,
Model Zoo, Runtime development files, and target ABI ceilings. These manifests,
not a branch name or a local directory timestamp, are the source of truth.

## What the current project can adopt

The external workspace already contains a source snapshot and generated output
whose file identities are pinned by the model manifest. The manual record
documents the completed source export, GRQ, and RKNN compilation that produced
those outputs outside project automation. Adopt the files into ignored
project-local storage with:

```sh
make host-bootstrap
make host-import MODEL=qwen2_5_0_5b
```

Import verifies all pins before publishing a category, leaves the source
workspace unchanged, and writes an import record beneath
`artifacts/work/qwen2_5_0_5b/`. The imported files provide checked inputs for
the next implementation milestone. They are not an immutable deployment
package and must not be committed.

## What is still missing

This milestone does not provide a wrapper that downloads the model, exports
ONNX/tokenizer/embedding inputs, invokes the RKNN3 compiler, or reproduces the
generated files from scratch. It also does not cross-build the real aarch64
runner, apply ELF compatibility gates, package a deployment, or transfer one to
the board.

Until those commands are implemented and tested, follow the
[manual evidence](rk1828-rknn3-deployment.md) for the completed host-side steps
and preserve all logs and output hashes. Incremental package transfer and the
first RK3588-to-RK1828 board inference have not started and are not verified. A
successful import only proves that the existing files match the recorded
identity; it does not prove model quality, package completeness, or hardware
inference.

## Future export record

The reproducible export milestone must record at least:

- source repository and immutable revision;
- Toolkit and Model Zoo revisions;
- platform, quantization mode, and every conversion option;
- tokenizer/config identities and generated file hashes;
- host OS, Python environment, compiler environment, and complete logs.

Generated data belongs under ignored artifact roots. Git stores only the logic,
configuration, manifests, tests, and documentation needed to reproduce and
verify it.
