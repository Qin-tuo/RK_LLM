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
  MODEL=qwen3_4b
```

The import module reads `configs/models/qwen3_4b.yaml` and verifies every
declared source/generated/Demo pin as a regular non-symlink file with the
recorded size and SHA-256 before publishing a destination. Other files in the
source roots are left alone and are not inventoried. The model manifest records
its repository and revision. Import does not validate the source model Git
revision; it only checks the declared file pins.

Source-model files are adopted under `artifacts/source_models/`; existing RKNN3
outputs and the vendor Demo are adopted under `artifacts/work/`. The Qwen3 Demo
lands at `artifacts/work/qwen3_4b/install/rknn_Qwen3_demo`. A previously imported
destination is reusable only when it contains the exact regular-file set
declared for that category and every file still matches its size and hash. An import record
captures the schema version, model ID, source workspace, copy/link mode,
category statuses, and verified file identities; the import record does not
include either value from the manifest repository/revision pair. The operation
does not alter the source workspace. Running the same command again must report
all three Qwen3 categories as `reused`.

## Build the immutable Qwen3 vendor package

Commit tracked changes first because the package records a clean project Git
commit. Then run:

```sh
make host-package MODEL=qwen3_4b
.host-venv/bin/rk-llm package-validate \
  --package artifacts/packages/qwen3_4b/<package_id>
```

The builder requires the exact imported Demo inventory, excludes its
`SHA256SUMS` evidence file, and maps the executable, three libraries, and four
model files into the package. It rejects non-AArch64 ELF files, GLIBC above
2.35, GLIBCXX above 3.4.30, a dirty Git tree, changed pins, symlinks, and
conflicting package directories. Publication uses a validated staging tree and
an atomic no-replace rename.

The project does not yet download the model, regenerate the four Qwen
deployment files, or cross-build the vendor Demo. Use the evidence-backed
[manual record](rk1828-rknn3-deployment.md) and the
[Qwen3 workflow](rk1828-qwen3-4b-quick-deployment.md) for those steps. Package
transfer is documented, but the first RK3588-to-RK1828 board inference is not
verified.

Do not upgrade the board glibc as a workaround for a host-built binary. The
vendor Demo build must use the Ubuntu 22.04/glibc 2.35 compatibility baseline
recorded by that manual evidence.
