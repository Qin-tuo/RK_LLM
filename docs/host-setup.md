# Host Setup

## Python development environment

The core package supports Python 3.10 through 3.12. A normal development
environment does not install an RKNN3 wheel or board Runtime from PyPI.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
rk-llm doctor --backend mock
python3 -m pytest -m "not hardware"
```

## Bootstrap pinned external inputs

The stable host target creates `.host-venv/`, installs this project, and invokes
the verified bootstrap module:

```sh
make host-bootstrap \
  WORKSPACE=/home/barry/rk1828-work \
  RKNN3_RUNTIME_DEV_ROOT=/home/barry/rk1828-work/rknn3-model-zoo/3rdparty/rknpu3
```

Bootstrap reads `manifests/upstream.yaml`. It creates complete Git checkouts at
`.vendor/rknn3-toolkit/` and `.vendor/rknn3-model-zoo/`. For each destination it
requires the pinned origin and exact revision; an existing checkout must also
have no changes reported by Git. A repository in `WORKSPACE` may be used only as
an optional clone-object reference and is not adopted as the destination.

For Runtime development files, bootstrap verifies each manifest-declared path
as a regular non-symlink file with the pinned SHA-256. A newly created
`.vendor/rknn3-runtime/` contains only that copied subset. When an existing
Runtime destination has all pinned files, additional files are allowed and are
not inventoried or verified. The directory is therefore not a full upstream SDK
checkout. `.vendor/` remains ignored and host-local.

## Import the verified existing model files

```sh
make host-import \
  WORKSPACE=/home/barry/rk1828-work \
  MODEL=qwen2_5_0_5b
```

The import module reads `configs/models/qwen2_5_0_5b.yaml` and verifies every
declared source/generated pin as a regular non-symlink file with the recorded
size and SHA-256 before publishing a destination. Other files in the source
roots are left alone and are not inventoried. The model manifest records its
repository and revision. Import does not validate the source model Git revision;
it only checks the declared file pins.

Source-model files are adopted under `artifacts/source_models/`; existing RKNN3
outputs are adopted under `artifacts/work/`. A previously imported destination
is reusable only when it contains the exact regular-file set declared for that
category and every file still matches its size and hash. An import record
captures the schema version, model ID, source workspace, copy/link mode,
category statuses, and verified file identities; the import record does not
include either value from the manifest repository/revision pair. The operation
does not alter the source workspace.

## What is not automated yet

The current repository has no supported command that downloads the source
model, regenerates the four Qwen deployment files, cross-builds the real runner,
creates a board package, or transfers it. Use the evidence-backed
[manual record](rk1828-rknn3-deployment.md) for the completed source export,
GRQ, RKNN compilation, aarch64 cross-build, and Ubuntu 22.04 ABI ceiling checks
while those wrappers are implemented in later plans. Incremental transfer and
the first RK3588-to-RK1828 board inference remain unverified.

Do not upgrade the board glibc as a workaround for a host-built binary. The
future runner build must use the Ubuntu 22.04/glibc 2.35 compatibility baseline
recorded by that manual evidence.
