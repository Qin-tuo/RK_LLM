# RK3588 Board Setup

## Current status

The board-side Python package and RKNN3 boundary are tracked, but project-owned
hardware inference is not implemented in this milestone. The native directory
contains an unavailable stub, and `RKNN3Backend` remains guarded until the
Native protocol plan defines and implements runner communication.

The guarded diagnostic is expected to report unavailable even when a candidate
package contains all prerequisite paths. It must never return mock text as a
fallback.

## Synchronization boundary

Clone or update this Git repository on RK3588 to synchronize tracked Python
application logic, configuration, manifests, tests, and documentation. Do not
copy the host `.vendor/` directory to the board. It contains host-side complete
upstream checkouts and a build-time Runtime subset, all of which are ignored.

Models, the aarch64 vendor Demo, Runtime shared libraries, and package manifest
are transferred as one immutable Qwen3 package. They remain outside Git. The
board stores releases under `artifacts/deploy/releases/$PACKAGE_ID`, validates
the incoming package with `rk-llm package-validate`, and atomically activates a
relative `artifacts/deploy/current` symlink. The board owns activation, Demo
lifecycle, application behavior, and real benchmark collection.

## Board baseline

The verified manual baseline uses an aarch64 RK3588 system with Ubuntu 22.04
and glibc 2.35, compatible RKNN3 Runtime/transport components, and RK1828
firmware. Do not upgrade board glibc, drivers, Runtime, transport services, or
firmware as an implicit project step.

The host package builder enforces the recorded ELF symbol ceilings before a
package is eligible for transfer. A successful source build on a newer host is
not proof that its executable can run on this board baseline.

After synchronizing this repository, create the board environment used for
incoming package validation:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/rk-llm package-validate --package artifacts/deploy/current
```

The complete guarded rsync, validation, release activation, and
`bin/rknn_qwen3_demo` invocation are in the
[Qwen3-4B quick deployment workflow](rk1828-qwen3-4b-quick-deployment.md).

## Available checks

The mock backend can be used on a development installation to verify Python
orchestration only:

```sh
python3 -m pip install -e ".[dev]"
rk-llm doctor --backend mock
rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
```

Hardware prerequisite tests remain opt-in and do not perform inference. The
manual [RK3588 + RK1828 evidence record](rk1828-rknn3-deployment.md) covers
completed source export, GRQ, RKNN compilation, aarch64 cross-build, and Ubuntu
22.04 ABI ceiling checks. Incremental package transfer now has a project-owned
validated workflow, but executing it and the first RK3588-to-RK1828 board
inference are not verified. The record does not make the guarded project
backend available.
