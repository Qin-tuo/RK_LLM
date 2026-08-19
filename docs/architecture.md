# Architecture

## System boundary

RK_LLM uses one Git repository for the host workflow contract and the RK3588
application logic. It does not put vendor repositories or multi-gigabyte model
payloads in Git.

```text
x86 host
  pin -> download -> verify -> export/build -> package -> transfer
                                                    |
                                                    v
RK3588 board
  validate package -> run application/benchmark -> RKNN3 Runtime -> RK1828
```

The current milestone implements immutable manifests, dependency bootstrap,
non-destructive import of existing verified files, the mock CLI, and a guarded
RKNN3 backend. Export/build/package/transfer automation and the Native process
protocol are later milestones. Consequently, the diagram describes ownership,
not a fully automated command chain available today.

## Tracked project content

- `configs/` holds model, runtime, and benchmark configuration.
- `manifests/upstream.yaml` pins all external input identities and hashes.
- `manifests/schemas/` defines immutable deployment-package validation.
- `src/rk_llm/host/` owns verified host bootstrap and import behavior.
- `src/rk_llm/backends/rknn3.py` is a fail-closed boundary pending the Native
  protocol plan; it never falls back to mock.
- `native/rknn3_qwen_runner/` is currently an unavailable stub. It will own
  tokenizer integration, RKNN3 handles, callbacks, and resource shutdown.
- `tools/host/` provides thin entry points only for implemented workflows.
- `tests/` verifies host, package, CLI, and guarded hardware boundaries.

## Ignored local content

`.vendor/` is host-local:

- `rknn3-toolkit/` is a complete pinned upstream Git checkout;
- `rknn3-model-zoo/` is a complete pinned upstream Git checkout;
- `rknn3-runtime/` is only the verified Runtime development-file subset named
  in `manifests/upstream.yaml`, not a complete SDK or repository.

The entire `.vendor/` tree is ignored and is not synchronized to the board.
`artifacts/source_models/`, `artifacts/work/`, `artifacts/packages/`,
`artifacts/deploy/`, and `artifacts/logs/` are also ignored. Generated files are
verified by manifest rather than stored in Git.

## Host and board ownership

The host owns model download, immutable pin verification, conversion, RKNN3
compilation, Ubuntu 22.04-compatible aarch64 runner build, package creation, and
package transfer. This milestone implements only bootstrap and import from that
list.

The board owns package validation, runner lifecycle, generation behavior,
benchmark collection, and higher-level application logic. Board code and
configuration are tracked so GitHub can synchronize them; model and binary
packages are transferred separately.

The manual record in `docs/rk1828-rknn3-deployment.md` is evidence for completed
source export, GRQ, RKNN compilation, aarch64 cross-build, and Ubuntu 22.04 ABI
ceiling checks outside project automation. Incremental package transfer and the
first RK3588-to-RK1828 board inference have not started and are not verified;
the record is not evidence that the guarded project backend can run inference.
