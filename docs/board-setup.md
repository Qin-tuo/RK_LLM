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

Models, the aarch64 runner, Runtime shared libraries, package manifest, and
other binary payloads will be transferred as an immutable deployment package by
a later host workflow. They remain outside Git. The board owns validation and
activation of that package, runner lifecycle, application behavior, and real
benchmark collection.

## Board baseline

The verified manual baseline uses an aarch64 RK3588 system with Ubuntu 22.04
and glibc 2.35, compatible RKNN3 Runtime/transport components, and RK1828
firmware. Do not upgrade board glibc, drivers, Runtime, transport services, or
firmware as an implicit project step.

The future host build must enforce the recorded ELF symbol ceilings before a
package is eligible for transfer. A successful source build on a newer host is
not proof that its executable can run on this board baseline.

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
22.04 ABI ceiling checks. Incremental package transfer and the first
RK3588-to-RK1828 board inference have not started and are not verified. The
record does not make the guarded project backend available.
