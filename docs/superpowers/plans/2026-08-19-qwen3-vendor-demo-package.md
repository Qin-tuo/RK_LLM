# Qwen3-4B Vendor Demo Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the verified Qwen3-4B source, generated files, and Model Zoo Demo into project-local state, then create and validate an immutable Vendor Demo package ready for transfer to RK3588.

**Architecture:** Extend the existing model manifest and hardened importer with an optional `demo` category, preserving Qwen2.5 behavior. Add a specialized host package builder that consumes only verified imported Qwen3 Demo files, enforces AArch64/glibc ceilings, writes a profile-bound manifest, and atomically publishes a content-addressed package. Expose read-only package validation through the existing CLI and document transfer/activation without implementing the project Native runner protocol.

**Tech Stack:** Python 3.10-3.12, PyYAML, jsonschema Draft 2020-12, pytest, GNU `readelf`, Make, SSH/rsync.

---

## File Map

- `configs/models/qwen3_4b.yaml`: immutable Qwen3 source, generated, and Demo pins.
- `src/rk_llm/manifests/loader.py`: optional Demo manifest fields and validation.
- `src/rk_llm/host/import_existing.py`: variable import categories and Demo destination.
- `src/rk_llm/host/package_vendor_demo.py`: Qwen3 Vendor Demo package builder and module CLI.
- `manifests/schemas/deployment-package.schema.json`: separate fixed Qwen2/project-runner and Qwen3/vendor-demo contracts.
- `src/rk_llm/artifacts/manifest.py`: return validated profile/entrypoint without weakening inventory checks.
- `src/rk_llm/cli.py`: read-only `package-validate` command.
- `Makefile`: `host-package` target.
- `tools/host/package-vendor-demo`: thin executable wrapper for the package module.
- `tests/unit/test_manifests.py`: Qwen3 manifest parsing and malformed Demo fields.
- `tests/unit/test_import_existing.py`: Demo category import, reuse, preflight, and destination tests.
- `tests/integration/test_import_existing.py`: three-category module CLI behavior.
- `tests/unit/test_artifact_manifest.py`: two package profiles and cross-profile rejection.
- `tests/unit/test_package_vendor_demo.py`: builder layout, ABI, atomic publication, and reuse.
- `tests/integration/test_cli.py`: installed package validation command.
- `tests/integration/test_host_package.py`: package module/wrapper integration.
- `tests/unit/test_repository_layout.py`: new tracked boundaries and Make target.
- `README.md`, `artifacts/README.md`, `docs/host-setup.md`, `docs/board-setup.md`, `docs/rk1828-qwen3-4b-quick-deployment.md`: current capability and transfer commands.

### Task 1: Pin Qwen3 And Parse The Demo Category

**Files:**
- Create: `configs/models/qwen3_4b.yaml`
- Modify: `src/rk_llm/manifests/loader.py`
- Modify: `tests/unit/test_manifests.py`

- [ ] **Step 1: Write failing manifest tests**

Add tests that load `configs/models/qwen3_4b.yaml` and assert:

```python
manifest.model_id == "qwen3_4b"
manifest.repository == "Qwen/Qwen3-4B"
manifest.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
manifest.demo_root == Path(
    "rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo"
)
manifest.demo_name == "rknn_Qwen3_demo"
len(manifest.source_files) == 12
len(manifest.generated_files) == 6
len(manifest.demo_files) == 9
```

Add parameterized malformed cases: only one of `demo_root`/`demo_name`/
`demo_files`, unsafe `demo_name` values (`../demo`, `a/b`, `.`, `..`), empty
Demo file list, duplicate file paths within every category, and file paths
that overlap by ancestor (`lib` and `lib/runtime.so`). Each must raise
`ValueError` naming the field before any filesystem access.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_manifests.py -q
```

Expected: new tests fail because `ModelManifest` has no Demo fields and the
Qwen3 manifest does not exist.

- [ ] **Step 3: Add the exact Qwen3 manifest**

Create `configs/models/qwen3_4b.yaml` with platform `rk1820`, the three roots,
`demo_name: rknn_Qwen3_demo`, and these exact pins:

```yaml
schema_version: 1
model_id: qwen3_4b
repository: Qwen/Qwen3-4B
revision: 1cfa9a7208912126459214e8b04321603b3df60c
platform: rk1820
source_root: models/Qwen3-4B
generated_root: rknn3-model-zoo/examples/Qwen3/model/llm
demo_root: rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo
demo_name: rknn_Qwen3_demo
source_files:
  - {path: LICENSE, size: 11343, sha256: 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e}
  - {path: README.md, size: 16857, sha256: 71add1cd091c309b1cb1a6b943f2476eb2b004124b3bd1d32dcd0b24cc1abc47}
  - {path: config.json, size: 726, sha256: 8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a}
  - {path: generation_config.json, size: 239, sha256: 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2}
  - {path: merges.txt, size: 1671853, sha256: 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5}
  - {path: model-00001-of-00003.safetensors, size: 3957900840, sha256: 328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223}
  - {path: model-00002-of-00003.safetensors, size: 3987450520, sha256: 6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5}
  - {path: model-00003-of-00003.safetensors, size: 99630640, sha256: e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1}
  - {path: model.safetensors.index.json, size: 32819, sha256: 6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2}
  - {path: tokenizer.json, size: 11422654, sha256: aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4}
  - {path: tokenizer_config.json, size: 9732, sha256: d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101}
  - {path: vocab.json, size: 2776833, sha256: ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910}
generated_files:
  - {path: Qwen3-4B.onnx, size: 1816180, sha256: edc97465e9f70422548eaf74fe59a680f804ad934c555cc256bee5cd97010b59}
  - {path: Qwen3-4B.config.pkl, size: 6381, sha256: ab4c08ee3070dfc73ea2226b1ba9595debe31911630245f46760b58b0d30cce3}
  - {path: Qwen3-4B.tokenizer.gguf, size: 5933309, sha256: 11752f8d46094557f79a27e242cfb02d8bbf1e09901c9a791114855cf23b090f}
  - {path: Qwen3-4B.embed.bin, size: 777912320, sha256: e9b98cc846d8e5be7a66847fc06a7a5cc14a1c88a9a8b3cc7c09e16134cc2b17}
  - {path: Qwen3-4B.rknn, size: 30681032, sha256: 784a81280d195ae9fbcd8705c26c658283abb9c0be8d51e4e03788cf2ac53e22}
  - {path: Qwen3-4B.weight, size: 2432392704, sha256: 5503eb3c5a39c0367e1f6870a45676387e185302b3eaf4a455bb5c3c6b2e7134}
demo_files:
  - {path: SHA256SUMS, size: 700, sha256: 32abc5c8892a81ee908e76c975a078d2675601bf31fb0197205a7ea333638d69}
  - {path: rknn_qwen3_demo, size: 794824, sha256: 8418947bd24b948c9778fd3f87f439fb046dfa90c19bf24aa09b32118438fb56}
  - {path: lib/librga.so, size: 196736, sha256: d0f5ae5c18c4c54ccee1d697b60f94507c5827389bb8fe9d4b0a4cb39cdc5972}
  - {path: lib/librknn3_api.so, size: 55792, sha256: 113ec97719e04f82e51fcb8badeb18461070ac55ca9a5da87f887f3110b4fcbe}
  - {path: lib/librknn3_api_rkcp.so, size: 8990352, sha256: 5ea77749f44be1f0c2ad0347242d4b431d3907d03eac11d265496ddd80cfd210}
  - {path: model/Qwen3-4B.embed.bin, size: 777912320, sha256: e9b98cc846d8e5be7a66847fc06a7a5cc14a1c88a9a8b3cc7c09e16134cc2b17}
  - {path: model/Qwen3-4B.rknn, size: 30681032, sha256: 784a81280d195ae9fbcd8705c26c658283abb9c0be8d51e4e03788cf2ac53e22}
  - {path: model/Qwen3-4B.tokenizer.gguf, size: 5933309, sha256: 11752f8d46094557f79a27e242cfb02d8bbf1e09901c9a791114855cf23b090f}
  - {path: model/Qwen3-4B.weight, size: 2432392704, sha256: 5503eb3c5a39c0367e1f6870a45676387e185302b3eaf4a455bb5c3c6b2e7134}
```

- [ ] **Step 4: Extend the loader minimally**

Add optional fields to `ModelManifest`:

```python
demo_root: Path | None = None
demo_name: str | None = None
demo_files: tuple[SizedFilePin, ...] = ()
```

Add `_safe_component` and `_unique_pins` helpers. `load_model_manifest` must
require all three Demo fields together, reject an empty declared list, validate
the component, and reject duplicate/ancestor-overlapping file paths. Apply
`_unique_pins` to source and generated lists as well.

- [ ] **Step 5: Verify GREEN and commit**

Run the focused test command from Step 2, then:

```bash
git add configs/models/qwen3_4b.yaml src/rk_llm/manifests/loader.py \
  tests/unit/test_manifests.py
git commit -m "feat: pin Qwen3 vendor demo inputs"
```

### Task 2: Import The Complete Qwen3 Demo Into Work

**Files:**
- Modify: `src/rk_llm/host/import_existing.py`
- Modify: `tests/unit/test_import_existing.py`
- Modify: `tests/integration/test_import_existing.py`

- [ ] **Step 1: Write failing three-category tests**

Extend fixtures with `demo_root`, `demo_name`, and two nested Demo files. Assert
that one call publishes:

```text
artifacts/source_models/demo/model.bin
artifacts/work/demo/model/output.rknn
artifacts/work/demo/install/rknn_Demo/demo
artifacts/work/demo/install/rknn_Demo/lib/runtime.so
```

Assert `statuses` has exactly `source`, `generated`, and `demo`; record entries
carry category `demo`; the second call reports all categories `reused`; a bad
Demo hash prevents every project write; an unexpected existing Demo file is
rejected and preserved; Demo source/destination symlink ancestors are rejected.

- [ ] **Step 2: Run focused import tests and verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_import_existing.py \
  tests/integration/test_import_existing.py -q
```

Expected: Demo assertions fail because only source/generated categories exist.

- [ ] **Step 3: Generalize category routing**

Introduce `_categories(model)` returning the two required categories and the
optional Demo category. Extend `_source_root` and `_destination` explicitly:

```python
if category == "source":
    return workspace / model.source_root
if category == "generated":
    return workspace / model.generated_root
if category == "demo" and model.demo_root is not None:
    return workspace / model.demo_root
raise ConfigurationError(f"unsupported import category: {category}")
```

Demo destination is
`artifacts/work/<model_id>/install/<demo_name>/<pin.path>`. Replace all hard-coded
two-category tuples and root selection branches with `_categories(model)` while
preserving existing descriptor/mapping checks and atomic publication.

- [ ] **Step 4: Verify GREEN, full import regression, and commit**

Run Step 2 plus:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_manifests.py tests/unit/test_import_existing.py \
  tests/integration/test_import_existing.py -q
git add src/rk_llm/host/import_existing.py \
  tests/unit/test_import_existing.py tests/integration/test_import_existing.py
git commit -m "feat: import verified vendor demo files"
```

### Task 3: Support Profile-Bound Qwen3 Packages And CLI Validation

**Files:**
- Modify: `manifests/schemas/deployment-package.schema.json`
- Modify: `src/rk_llm/cli.py`
- Modify: `tests/unit/test_artifact_manifest.py`
- Modify: `tests/integration/test_cli.py`

- [ ] **Step 1: Write failing package-profile tests**

Update Qwen2 fixtures with:

```json
"package_profile": "project_runner",
"entrypoint": "bin/rknn_qwen_runner"
```

Add a Qwen3 fixture with the exact twelve source pins, profile `vendor_demo`,
entry point `bin/rknn_qwen3_demo`, and a declared runner file at that path.
Assert both validate. Assert cross-paired profile, model ID, and entrypoint
combinations fail schema validation. Add CLI tests asserting:

```bash
rk-llm package-validate --package PACKAGE
```

prints only `package_id`, `model_id`, `package_profile`, and `entrypoint` as
JSON, and invalid packages exit through `entrypoint()` with code 2.

- [ ] **Step 2: Run package/CLI tests and verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_artifact_manifest.py tests/integration/test_cli.py -q
```

Expected: Qwen3/profile manifests fail the current Qwen2-only schema and the
CLI parser rejects `package-validate`.

- [ ] **Step 3: Add schema branches without loosening pins**

Require `package_profile` and `entrypoint`. Make `model` a `oneOf` containing
the existing exact Qwen2 object and a second exact Qwen3 object. Add root
conditional branches:

```json
{
  "if": {"properties": {"model": {"properties": {"id": {"const": "qwen3_4b"}}}}},
  "then": {
    "properties": {
      "package_profile": {"const": "vendor_demo"},
      "entrypoint": {"const": "bin/rknn_qwen3_demo"},
      "files": {"contains": {"properties": {"path": {"const": "bin/rknn_qwen3_demo"}}}}
    }
  }
}
```

Keep the equivalent Qwen2 branch requiring `project_runner` and
`bin/rknn_qwen_runner`.

- [ ] **Step 4: Add CLI validation**

Add a `package-validate` parser with required `--package`. Resolve the path
through `_deployment_path`, call `validate_package`, and print:

```python
{
    "package_id": manifest["package_id"],
    "model_id": manifest["model"]["id"],
    "package_profile": manifest["package_profile"],
    "entrypoint": manifest["entrypoint"],
}
```

- [ ] **Step 5: Verify GREEN and commit**

Run Step 2, then commit the four files with:

```bash
git commit -m "feat: validate Qwen3 vendor demo packages"
```

### Task 4: Build An Immutable Vendor Demo Package

**Files:**
- Create: `src/rk_llm/host/package_vendor_demo.py`
- Create: `tools/host/package-vendor-demo`
- Create: `tests/unit/test_package_vendor_demo.py`
- Create: `tests/integration/test_host_package.py`
- Modify: `Makefile`
- Modify: `tests/unit/test_repository_layout.py`

- [ ] **Step 1: Write failing builder tests**

Use tiny Demo files and an injected command runner. Cover:

- package layout mapping excludes `SHA256SUMS`;
- all copied files appear in the manifest with size/hash;
- Demo executable retains execute bits;
- ELF metadata records injected AArch64, glibc, and glibcxx values;
- wrong architecture, glibc > 2.35, glibcxx > 3.4.30, dirty Git worktree,
  changed imported files, symlinks, and undeclared imported files fail;
- staged validation occurs before no-replace publication;
- identical rerun reuses the package;
- conflicting/concurrent final directories are preserved;
- failure cleanup cannot remove a same-name replacement staging directory;
- module CLI and wrapper print one JSON summary;
- Makefile exposes `host-package` and passes project/model/upstream/readelf.

- [ ] **Step 2: Run builder tests and verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_package_vendor_demo.py \
  tests/integration/test_host_package.py \
  tests/unit/test_repository_layout.py -q
```

Expected: collection fails because the builder module and wrapper do not exist.

- [ ] **Step 3: Implement the builder boundary**

Expose `build_vendor_demo_package(project_root: Path, model: ModelManifest,
upstream: UpstreamManifest, *, readelf: str =
"aarch64-linux-gnu-readelf", run: Callable = subprocess.run) ->
dict[str, object]`. Tests pass a recording `run` callable; production uses the
default without shell execution.

Require `model_id == "qwen3_4b"` and complete Demo fields. Verify the imported
Demo has the exact declared inventory and pins. Map the nine inputs to the
eight package payloads (exclude `SHA256SUMS`; move the executable into `bin`).
Call readelf for the executable and three libraries, parse `Machine: AArch64`
and maximum version tuples numerically, and reject values over upstream target
ceilings.

Build the manifest with the recorded commands:

```python
"build": {
    "export_args": ["--quant", "--model_path", "Qwen/Qwen3-4B"],
    "rknn_args": ["--platform", "rk1820", "--dataset_path", "datasets/CMMLU/dataset.txt"],
    "cmake_args": ["-t", "rk3588", "-a", "aarch64", "-d", "Qwen3"],
}
```

Use `git rev-parse HEAD` for `project_commit` only after `git status
--porcelain --untracked-files=all` is empty. Construct a mode-0700 sibling
staging directory, copy without symlink following, fsync files/directories,
write the manifest, calculate its package ID, validate staging, and publish via
Linux `renameat2(RENAME_NOREPLACE)`. Record staging inode identity and clean
only that identity on failure. Validate and reuse an identical final package;
reject every other existing final directory.

- [ ] **Step 4: Add wrapper and Make target**

The executable wrapper follows existing `tools/host/*` conventions and invokes
`python -m rk_llm.host.package_vendor_demo`. Add:

```make
host-package: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.package_vendor_demo \
		--project-root "$(PROJECT_ROOT)" \
		--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml" \
		--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml" \
		--readelf aarch64-linux-gnu-readelf
```

- [ ] **Step 5: Verify GREEN and commit**

Run Step 2 plus package/manifest focused tests. Verify wrapper mode `0755`,
`git diff --check`, then commit with:

```bash
git commit -m "feat: build immutable Qwen3 vendor demo package"
```

### Task 5: Document Transfer, Run Real Adoption, And Verify

**Files:**
- Modify: `README.md`
- Modify: `artifacts/README.md`
- Modify: `docs/host-setup.md`
- Modify: `docs/board-setup.md`
- Modify: `docs/rk1828-qwen3-4b-quick-deployment.md`
- Modify: `tests/unit/test_repository_layout.py`

- [ ] **Step 1: Write failing documentation-contract tests**

Assert docs distinguish work, package, and deploy; name `qwen3_4b`; include
`make host-import`, `make host-package`, `package-validate`; transfer only one
package into `artifacts/deploy/releases`; activate a relative `current` symlink;
and run `bin/rknn_qwen3_demo` with packaged model paths. Remove statements that
package/transfer automation is entirely absent while retaining the explicit
guard that the project Native backend and hardware inference are unverified.

- [ ] **Step 2: Run repository-layout tests and verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  tests/unit/test_repository_layout.py -q
```

- [ ] **Step 3: Update user-facing documentation**

Document host commands and the variable-driven transfer sequence:

```bash
RK3588_HOST=ubuntu@<RK3588_IP>
REMOTE_PROJECT=/home/ubuntu/RK_LLM
PACKAGE_DIR=/absolute/path/from/make-host-package
PACKAGE_ID=$(basename "$PACKAGE_DIR")
REMOTE_INCOMING="$REMOTE_PROJECT/artifacts/deploy/.incoming-$PACKAGE_ID"
REMOTE_RELEASE="$REMOTE_PROJECT/artifacts/deploy/releases/$PACKAGE_ID"

ssh "$RK3588_HOST" "test ! -e '$REMOTE_INCOMING' && test ! -e '$REMOTE_RELEASE' && mkdir -p '$REMOTE_PROJECT/artifacts/deploy/releases'"
rsync -a --protect-args "$PACKAGE_DIR/" "$RK3588_HOST:$REMOTE_INCOMING/"
ssh "$RK3588_HOST" "cd '$REMOTE_PROJECT' && .venv/bin/rk-llm package-validate --package '$REMOTE_INCOMING' && mv '$REMOTE_INCOMING' '$REMOTE_RELEASE' && ln -s 'releases/$PACKAGE_ID' '$REMOTE_PROJECT/artifacts/deploy/.current-$PACKAGE_ID' && mv -Tf '$REMOTE_PROJECT/artifacts/deploy/.current-$PACKAGE_ID' '$REMOTE_PROJECT/artifacts/deploy/current'"
```

Document the activated Vendor Demo invocation with `LD_LIBRARY_PATH`, four
model arguments, device mask `0xff`, and a Chinese smoke prompt.

- [ ] **Step 4: Verify documentation and commit**

Run the layout test, `git diff --check`, and commit:

```bash
git commit -m "docs: add Qwen3 package transfer workflow"
```

- [ ] **Step 5: Run the complete non-hardware suite**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest \
  -m "not hardware" -q
```

Expected: all tests pass with only the explicit hardware test deselected.

- [ ] **Step 6: Import real Qwen3 inputs**

Capture hashes for five representative external files, run:

```bash
make host-import MODEL=qwen3_4b WORKSPACE=/home/barry/rk1828-work
make host-import MODEL=qwen3_4b WORKSPACE=/home/barry/rk1828-work
```

Expected: first result reports three imported categories; second reports three
reused categories; external hashes remain unchanged.

- [ ] **Step 7: Build and validate the real package**

Commit any final code/test fixes first so the tree is clean, then run:

```bash
make host-package MODEL=qwen3_4b
.host-venv/bin/rk-llm package-validate \
  --package "$(find artifacts/packages/qwen3_4b -mindepth 1 -maxdepth 1 -type d -print -quit)"
```

Verify the JSON profile is `vendor_demo`, entrypoint is
`bin/rknn_qwen3_demo`, every package file validates, and `git status --porcelain`
contains no tracked changes. Do not transfer without the user's RK3588 IP.
