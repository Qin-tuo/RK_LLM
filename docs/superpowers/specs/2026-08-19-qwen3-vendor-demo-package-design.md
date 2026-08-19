# Qwen3-4B Vendor Demo Package Design

## Goal

Adopt the already-built `Qwen/Qwen3-4B` source files, RKNN3 conversion
outputs, and Model Zoo ARM64 Demo into the ignored local state of `RK_LLM`;
then build a validated immutable package that can be transferred into the
board checkout and run with the official `rknn_qwen3_demo` command.

This milestone packages and deploys the Vendor Demo. It does not implement the
project-owned Native process protocol and does not make
`rk-llm generate --backend rknn3` available.

## Verified Inputs

The source workspace is `/home/barry/rk1828-work`. The input roots are:

- `models/Qwen3-4B` for source-model files;
- `rknn3-model-zoo/examples/Qwen3/model/llm` for conversion outputs;
- `rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo` for the
  installed ARM64 Demo.

The model identity is:

- repository: `Qwen/Qwen3-4B`;
- revision: `1cfa9a7208912126459214e8b04321603b3df60c`;
- project model ID: `qwen3_4b`.

The local directory was populated through a public mirror selection workflow.
Its conversion-relevant model files match the immutable Hugging Face revision.
Mirror-owned metadata that is not part of the official revision, including
`configuration.json`, is not declared or imported. The locally different
`.gitattributes` file is also excluded. Every declared source file is still
pinned by exact size and SHA-256.

Toolkit, Model Zoo, Runtime, target ABI, and firmware identities continue to
come from `manifests/upstream.yaml`. The build target remains RK3588/aarch64
with RK1828 hardware, RKNN compiler platform `rk1820`, and a maximum glibc
version of 2.35.

## Local Artifact Ownership

Qwen3 local state is split by lifecycle rather than by model family:

```text
artifacts/
|-- source_models/
|   `-- qwen3_4b/                         # pinned source-model files
|-- work/
|   `-- qwen3_4b/
|       |-- model/                        # ONNX/config/RKNN/weight/tokenizer/embed
|       |-- install/
|       |   `-- rknn_Qwen3_demo/          # complete Model Zoo install output
|       `-- import-record.json
|-- packages/
|   `-- qwen3_4b/
|       `-- <package_id>/                 # immutable package built from work
`-- deploy/                               # board-only unpacked/activated state
```

All of these payload directories remain ignored by Git. Git tracks only the
model manifest, schemas, code, tests, documentation, and thin tool entry
points.

## Model Manifest Extension

`configs/models/qwen3_4b.yaml` declares three independently verified input
categories:

1. `source_files` beneath `models/Qwen3-4B`;
2. `generated_files` beneath
   `rknn3-model-zoo/examples/Qwen3/model/llm`;
3. `demo_files` beneath the installed `rknn_Qwen3_demo` directory.

The existing source and generated fields remain required. Demo adoption is an
optional extension so the Qwen2.5 manifest and its current destinations remain
unchanged. A manifest that declares demo files must also declare a safe
single-component `demo_name` and a relative `demo_root`.

The importer publishes demo files to
`artifacts/work/<model_id>/install/<demo_name>`. It applies the same guarantees
as the existing source/generated categories: no symlink traversal, regular
files only, exact size and SHA-256, complete preflight before project writes,
sibling staging, atomic no-replace publication, exact-inventory reuse, and an
updated import record. Source workspace files are never modified.

The stable command remains:

```bash
make host-import MODEL=qwen3_4b WORKSPACE=/home/barry/rk1828-work
```

## Vendor Demo Package

The package builder consumes only the verified imported Demo destination. It
does not package directly from the external workspace. It maps the installed
Demo to this canonical layout:

```text
artifacts/packages/qwen3_4b/<package_id>/
|-- manifest.json
|-- bin/
|   `-- rknn_qwen3_demo
|-- lib/
|   |-- librga.so
|   |-- librknn3_api.so
|   `-- librknn3_api_rkcp.so
`-- model/
    |-- Qwen3-4B.rknn
    |-- Qwen3-4B.weight
    |-- Qwen3-4B.tokenizer.gguf
    `-- Qwen3-4B.embed.bin
```

The source `SHA256SUMS` file is not copied because `manifest.json` replaces it
with a complete, schema-validated inventory. No undeclared package file is
allowed.

The package manifest records:

- the exact Qwen3 repository, revision, and source-file pins;
- the project commit and existing pinned RKNN3 toolchain identities;
- the RK3588/RK1828 target and ABI ceilings;
- the recorded Qwen3 export, RKNN, and CMake arguments;
- package profile `vendor_demo` and entry point `bin/rknn_qwen3_demo`;
- exact size, SHA-256, and ELF version data for every package file.

The deployment-package schema retains the existing Qwen2.5/project-runner
contract and adds a separate fixed Qwen3/vendor-demo branch. A Qwen2.5 package
cannot claim the Qwen3 entry point, and a Qwen3 package cannot satisfy the
schema with the project runner. This preserves the provenance hardening rather
than replacing it with loose string patterns.

The stable host command is:

```bash
make host-package MODEL=qwen3_4b
```

The builder creates a mode-0700 sibling staging directory, copies files without
following symlinks, preserves the executable mode for the Demo, records ELF
requirements, writes and fsyncs the manifest, validates the staged package,
and publishes it with no-replace semantics. An identical existing package is
reused. A mismatched existing package or concurrent destination is rejected and
preserved.

## ABI Enforcement

Packaging rejects the Demo executable or any shared library unless all of the
following are true:

- the file is an ELF object for AArch64;
- the maximum required `GLIBC_*` symbol is at most `GLIBC_2.35`;
- the maximum required `GLIBCXX_*` symbol is at most `GLIBCXX_3.4.30`;
- the executable has an execute bit;
- the file size and SHA-256 match the imported Demo pins.

The builder uses `aarch64-linux-gnu-readelf` and reports the failing path and
limit. It never upgrades or modifies board glibc, Runtime, drivers, transport
services, or firmware.

## Validation Command

The CLI gains a read-only package validation command:

```bash
rk-llm package-validate --package /absolute/path/to/package
```

It validates schema, package ID, exact inventory, file types, sizes, hashes,
and symlink safety. On success it prints a compact JSON object containing the
package ID, model ID, package profile, and entry point. It returns exit code 2
with an actionable error on invalid input. It does not activate or execute the
package.

## Transfer And Board Activation

Git synchronizes source and logic. The immutable package is transferred
separately into the board checkout. The documented command uses explicit
variables for the board host and project root, uploads to a unique incoming
directory, validates there, and only then renames it into:

```text
/home/ubuntu/RK_LLM/artifacts/deploy/releases/<package_id>/
```

After validation, an atomic relative `current` symlink selects the release:

```text
artifacts/deploy/current -> releases/<package_id>
```

Transfer and activation never overwrite a different existing release. The
previous `current` target remains available for rollback. `.vendor/`, source
models, work files, and host build state are not transferred.

The Vendor Demo is then run from the activated package with its packaged
libraries and four model arguments. This execution remains a manual command in
this milestone; successful transfer alone is not hardware-inference evidence.

## Failure Handling

- Missing, additional, changed, non-regular, or symlinked inputs fail before
  publication.
- A source revision that cannot be reconciled with declared file identities
  fails manifest adoption.
- Package construction cleans only its own identity-checked staging objects.
- Existing work, package, and board release directories are never recursively
  replaced.
- Transfer interruption leaves only the uniquely named incoming directory and
  cannot change `current`.
- Board validation failure leaves the previous active release unchanged.
- RKNN3 backend availability remains false because the Vendor Demo does not
  implement the project Native protocol.

## Testing

Tests follow red-green TDD and cover:

- Qwen3 manifest parsing, exact pins, and malformed demo declarations;
- Qwen2.5 import behavior remaining unchanged;
- Qwen3 three-category preflight, import, reuse, tamper rejection, symlink
  rejection, race handling, and import-record contents;
- deterministic package IDs and canonical layout;
- Qwen2.5/project-runner and Qwen3/vendor-demo schema separation;
- missing, extra, changed, and unsafe package files;
- wrong architecture and excessive glibc/glibc++ requirements;
- atomic package publication and concurrent-destination preservation;
- CLI package validation success and failure behavior;
- repository documentation and ignored-directory boundaries.

After focused tests, the complete non-hardware suite must pass. Real Qwen3
adoption and package construction are then run against the existing workspace,
followed by package validation and a source-workspace hash-preservation check.
Hardware execution occurs only after the user runs the separately supplied
transfer and board commands.
