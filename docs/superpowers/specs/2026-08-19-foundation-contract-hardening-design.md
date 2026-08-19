# RKNN3 Foundation Contract Hardening Design

## Context

The RKNN3 foundation branch now pins the Qwen source and generated files,
bootstraps verified vendor inputs, imports the existing workspace without
modifying it, and validates deployment-package files. Final cross-module review
found three remaining contract gaps:

1. the native CMake target produces `rknn3_qwen_runner`, while the board probe
   and package layout require `bin/rknn_qwen_runner`;
2. a schema-valid deployment manifest can name an arbitrary 40-character model
   revision without carrying the pinned source-file identities or compiler
   platform;
3. internal metadata and the public exception base still use the retired RKLLM
   backend name.

This change closes those gaps without implementing model rebuild, runner
functionality, package construction, transfer, or hardware inference.

## Decisions

### Runner filename

The project-owned executable name is `rknn_qwen_runner` everywhere. CMake will
produce that filename, the board probe will continue to require it, and the
deployment schema will require a file record whose path is
`bin/rknn_qwen_runner`.

The native source remains an unavailable stub until the Native protocol plan.
This change aligns contracts only; it does not make the runner operational.

### Self-contained model provenance

The deployment manifest remains self-contained. Its `model` object will
require:

- `id: qwen2_5_0_5b`;
- repository `Qwen/Qwen2.5-0.5B-Instruct`;
- revision `7ae557604adf67be50417f59c2c2f167def9a775`;
- `source_files`, fixed to the ten ordered `path`, `size`, and `sha256` records
  in `configs/models/qwen2_5_0_5b.yaml`.

Embedding the exact source pins makes a package independently auditable. A
validator does not need the project checkout or a separately supplied model
manifest to determine which source snapshot the package claims.

### Hardware and compiler identities

The target keeps two separate identities:

- `accelerator: rk1828` names the physical PCIe accelerator;
- `compiler_platform: rk1820` names the platform value used by the verified
  RKNN3 compiler flow.

The compiler platform remains `rk1820` because the current model manifest,
upstream manifest, actual compiler arguments, and board firmware all use that
identifier. Changing it to `rk1828` would invalidate the recorded generated
file provenance and require a separate rebuild and re-pin.

`build.rknn_args` must contain at least one entry. The array still records the
actual command arguments and may grow in later build work; the structured
`compiler_platform` field provides the unambiguous target contract.

### Project naming

The distribution name, import package, command, and repository remain
`rk-llm`, `rk_llm`, `rk-llm`, and `RK_LLM`. These are the project interfaces.
Descriptive metadata will refer to RKNN3 rather than the retired vendor RKLLM
backend. The public exception base becomes `ProjectError`, and internal imports
and exception handling will migrate to it without retaining a legacy alias.

## Validation

Tests will be written before production changes and must demonstrate these
failures on the current branch:

- CMake does not produce the runner filename required by the probe;
- a manifest with a different model revision is accepted;
- a manifest without the ten source pins is accepted, while adding them is
  currently rejected;
- a manifest without `compiler_platform` is accepted;
- empty `rknn_args` are accepted;
- a package without the required runner path is accepted;
- product metadata and source still expose the retired RKLLM identifier.

After implementation, focused artifact, platform, layout, and configuration
tests must pass, followed by the complete non-hardware suite and
`git diff --check`. Existing ignored `.vendor`, `.host-venv`, and imported
artifacts must remain untouched.

## Non-goals

- no model export or RKNN compilation;
- no functional native runner or Native protocol;
- no package builder, transfer, activation, or rollback workflow;
- no board mutation and no hardware-inference claim;
- no change to the verified `rk1820` compiler platform or generated hashes.
