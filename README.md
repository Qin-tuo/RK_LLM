# RK_LLM

RK_LLM is the version-controlled foundation for deploying Qwen2.5-0.5B through
RKNN3 on an RK3588 host connected to an RK1828 accelerator. The repository
keeps host preparation and board application logic in one Git project while
leaving large models, vendor repositories, build outputs, and deployment
packages outside Git.

## Current milestone

- The deterministic mock CLI and non-hardware tests remain runnable.
- `manifests/upstream.yaml` and the model manifest pin the verified upstream
  repositories, Runtime development files, source model, and generated files.
- `make host-bootstrap` creates verified local vendor inputs under `.vendor/`.
- `make host-import` adopts the already verified workspace inputs and generated
  Qwen files into ignored project-local artifact directories without changing
  their source files.
- The imported output can be used as evidence for later work, but this project
  does not yet contain the wrapper that rebuilds it.
- The RKNN3 backend remains deliberately unavailable until the Native protocol
  plan implements the project-owned runner contract.

No command in this milestone claims successful hardware inference. The
[manual deployment evidence](docs/rk1828-rknn3-deployment.md) records completed
source export, GRQ, RKNN compilation, aarch64 cross-build, and Ubuntu 22.04 ABI
ceiling checks performed outside project automation. Incremental package
transfer is still pending. The first RK3588-to-RK1828 board inference has not
started and is not verified.

## Mock development

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
rk-llm doctor --backend mock
rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
python3 -m pytest -m "not hardware"
```

Mock output is deterministic functional test data, not RK3588 or RK1828
performance evidence.

## Adopt the verified host workspace

Run these targets from the repository root. Override `WORKSPACE` and
`RKNN3_RUNTIME_DEV_ROOT` when the external workspace uses different paths.

```sh
make host-bootstrap
make host-import
```

Bootstrap creates complete pinned Toolkit and Model Zoo Git checkouts plus a
verified Runtime development-file subset beneath `.vendor/`. Import copies the
pinned source-model files to `artifacts/source_models/` and generated files to
`artifacts/work/`. Both operations reject identity or checksum mismatches.

The next implementation plan must add reproducible export, cross-build,
package, and transfer commands. Those targets are intentionally absent today.

## Repository synchronization boundary

GitHub synchronizes tracked source, board logic, configuration, manifests,
tests, and documentation between the host and board. `.vendor/`, model files,
build workspaces, deployment packages, deployed payloads, and logs are ignored.
The host owns download, pinning, verification, build, packaging, and transfer;
the board owns runtime, benchmark, and application behavior.

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Host setup and verified import](docs/host-setup.md)
- [Board setup and current guard](docs/board-setup.md)
- [Model inputs and export status](docs/model-export.md)
- [Benchmark status and evidence rules](docs/benchmark.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Manual RK3588 + RK1828 deployment evidence](docs/rk1828-rknn3-deployment.md)
