# RKNN3 Implementation Roadmap

The goal is one Git project that owns the reproducible host-to-board workflow
contract and the RK3588 application logic. GitHub synchronizes tracked logic;
large models, upstream repositories, build outputs, packages, deployments, and
logs stay outside Git.

## Milestone 1: foundation and verified adoption

Status: implemented.

- keep the mock CLI and non-hardware suite runnable;
- pin Toolkit, Model Zoo, Runtime development files, target ABI, source model,
  and existing generated outputs;
- bootstrap complete pinned Toolkit and Model Zoo checkouts plus the verified
  Runtime development subset under ignored `.vendor/`;
- import the verified external source and generated files without modifying the
  source workspace;
- retain a fail-closed RKNN3 backend and unavailable native stub.

Completion of this milestone proves identity and repository boundaries. It does
not prove that the project can rebuild the model or run hardware inference.

## Milestone 2: reproducible host export and RKNN3 build

Status: planned.

- download the immutable Qwen source revision into ignored storage;
- resume or safely restart model export stages;
- invoke the pinned RKNN3 toolchain and verify every generated output;
- preserve commands, environment identity, logs, sizes, and hashes;
- prove reruns either reuse verified output or reject mismatches.

The existing imported output is the comparison baseline for this work, not a
substitute for the new wrapper.

## Milestone 3: real runner, cross-build, and immutable package

Status: planned.

- implement the versioned Native process protocol;
- replace the stub with the project-owned Qwen runner;
- cross-build in an Ubuntu 22.04 environment;
- validate the executable and shared libraries against the recorded aarch64
  architecture and glibc/glibc++ ceilings;
- create a schema-valid immutable package with hashes for every payload.

The RKNN3 backend remains guarded until the protocol and runner pass their
non-hardware contract tests.

## Milestone 4: transfer and board activation

Status: planned.

- transfer packages separately from Git synchronization;
- verify the package before activation on RK3588;
- keep rollback information and never replace board system components as an
  implicit deploy side effect;
- start, stop, and diagnose the runner through tracked board logic.

The host owns build/package/transfer. The board owns runtime and application
behavior. `.vendor/` is never synchronized to the board.

## Milestone 5: hardware inference and benchmark

Status: planned.

- pass an explicit opt-in inference smoke test on the recorded board baseline;
- verify streaming output, errors, cancellation, and repeated requests;
- collect benchmark evidence with package, software, thermal, and power context;
- keep mock and hardware results unambiguously separate.

The manual `rk1828-rknn3-deployment.md` document records completed source
export, GRQ, RKNN compilation, aarch64 cross-build, and Ubuntu 22.04 ABI ceiling
checks outside project automation. Incremental package transfer and the first
RK3588-to-RK1828 board inference have not started and are not verified. Each
later milestone must turn the applicable evidence into tested project
automation before claiming the corresponding capability.
