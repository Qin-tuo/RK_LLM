# RKNN3 Foundation and Existing Artifact Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing RKLLM/DeepSeek skeleton into an RKNN3/Qwen2.5 project foundation, pin every external input, and non-destructively import the verified `/home/barry/rk1828-work` assets into ignored project-local storage.

**Architecture:** Keep the reusable Python CLI, backend protocol, mock implementation, generation service, metrics, and tests. Replace the hardware-facing contract with an `rknn3` package-root contract, add versioned YAML/JSON manifests, bootstrap pinned vendor checkouts into `.vendor/`, and import verified files into `artifacts/` without modifying the source workspace.

**Tech Stack:** Python 3.10-3.12, PyYAML, jsonschema Draft 2020-12, pytest, Git, Make, SHA-256, existing RKNN3 Toolkit/Model Zoo V1.0.4 assets.

---

## Scope and Plan Sequence

The approved design contains four implementation-sized units. This document executes unit 1 only and leaves
a working, testable repository foundation:

1. **This plan:** backend naming, repository contracts, pinned manifests, bootstrap, and non-destructive import.
2. **Host build/package plan:** resumable export, RKNN3 compile, Ubuntu 22.04 Runner build, ELF gates, immutable package.
3. **Board runtime/deploy plan:** real C++ Runner, JSONL protocol, Python backend, candidate verification, activation, rollback.
4. **Hardware validation plan:** RK3588/RK1828 smoke tests, failure injection, rollback proof, benchmark and final docs.

Do not begin unit 2 until this plan's acceptance commands pass. Execute implementation in
`/home/barry/AI_Infra/RK_LLM/.worktrees/rknn3-foundation`, on a dedicated branch created from `main` after
this plan is committed. Verify that design commit `81b66ad` is an ancestor before making implementation edits.

## Locked File Structure

This plan creates or changes the following boundaries:

- `manifests/upstream.yaml`: immutable vendor repositories, Runtime files, model revision, and hashes.
- `manifests/schemas/deployment-package.schema.json`: deployment-package JSON contract.
- `configs/models/qwen2_5_0_5b.yaml`: exact source and existing output files for the first model.
- `src/rk_llm/manifests/`: typed YAML loading and validation.
- `src/rk_llm/artifacts/manifest.py`: canonical package ID and package-directory validation.
- `src/rk_llm/host/bootstrap.py`: pinned vendor checkout and Runtime development-file adoption.
- `src/rk_llm/host/import_existing.py`: atomic, hash-checked import from the old workspace.
- `src/rk_llm/backends/rknn3.py`: guarded RKNN3 stub until the board-runtime plan implements protocol I/O.
- `src/rk_llm/platform/probe.py`: package-root and architecture prerequisite reporting.
- `native/rknn3_qwen_runner/`: renamed unavailable stub; no vendor API claim in this phase.
- `artifacts/` and `.vendor/`: ignored local state; tracked README files describe their contracts.

## Task 1: Rename the Runtime Configuration Contract

**Files:**
- Modify: `src/rk_llm/config.py`
- Modify: `configs/runtime/rk3588.yaml`
- Modify: `configs/benchmark/smoke.yaml`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_benchmark.py`

- [ ] **Step 1: Write the failing RKNN3 configuration tests**

Replace RKLLM-specific cases in `tests/unit/test_config.py` with these assertions, and rename all
`RuntimeConfig.model_path` expectations to `package_path`:

```python
def test_load_runtime_config_resolves_package_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "backend: rknn3\n"
        "target: rk3588\n"
        "package_path: ../../artifacts/deploy/current\n"
        "max_new_tokens: 64\n",
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.backend == "rknn3"
    assert config.package_path == (tmp_path / "../../artifacts/deploy/current").resolve()
    assert config.max_new_tokens == 64


def test_rknn3_runtime_requires_package_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("backend: rknn3\ntarget: rk3588\n", encoding="utf-8")

    with pytest.raises(ValueError, match="package_path is required for rknn3"):
        load_runtime_config(config_path)
```

Update the malformed-type matrix to contain
`("backend: rknn3\ntarget: rk3588\npackage_path: 12\n", "package_path")`, and update target-pair
parameters from `("rkllm", "host")` to `("rknn3", "host")`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_config.py tests/unit/test_benchmark.py -q
```

Expected: FAIL because `rknn3` is rejected and `RuntimeConfig` has no `package_path`.

- [ ] **Step 3: Implement the renamed immutable configuration field**

Apply these exact semantic changes in `src/rk_llm/config.py`:

```python
@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    target: str
    package_path: Path | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in ("mock", "rknn3"):
            raise ValueError("backend must be mock or rknn3")
        if not isinstance(self.target, str) or self.target not in ("host", "rk3588"):
            raise ValueError("target must be host or rk3588")
        expected_target = "host" if self.backend == "mock" else "rk3588"
        if self.target != expected_target:
            raise ValueError(f"target must be {expected_target} for backend {self.backend}")
```

In `_runtime_from_mapping`, read `package_path`, resolve it relative to the YAML file, and enforce:

```python
backend = _string_value(data, "backend", "mock")
if backend not in {"mock", "rknn3"}:
    raise ValueError("backend must be mock or rknn3")
raw_package = data.get("package_path")
if raw_package is not None and (
    not isinstance(raw_package, str) or not raw_package.strip()
):
    raise ValueError("package_path must be a non-empty string or null")
package_path = (base_dir / raw_package).resolve() if raw_package is not None else None
if backend == "rknn3" and package_path is None:
    raise ValueError("package_path is required for rknn3")
```

Pass `package_path` as the third `RuntimeConfig` constructor argument. Replace the board config with:

```yaml
backend: rknn3
target: rk3588
package_path: ../../artifacts/deploy/current
max_new_tokens: 128
temperature: 0.8
top_p: 0.9
top_k: 1
repeat_penalty: 1.1
```

Keep `configs/benchmark/smoke.yaml` on the mock backend so CI remains hardware-independent.

- [ ] **Step 4: Run configuration and benchmark tests**

Run:

```bash
python3 -m pytest tests/unit/test_config.py tests/unit/test_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add src/rk_llm/config.py configs/runtime/rk3588.yaml \
  configs/benchmark/smoke.yaml tests/unit/test_config.py tests/unit/test_benchmark.py
git commit -m "refactor: adopt RKNN3 package configuration"
```

## Task 2: Replace the Guarded RKLLM Stub with an RKNN3 Stub

**Files:**
- Delete: `src/rk_llm/backends/rkllm.py`
- Create: `src/rk_llm/backends/rknn3.py`
- Modify: `src/rk_llm/platform/probe.py`
- Modify: `src/rk_llm/cli.py`
- Rename: `native/rkllm_runner/` to `native/rknn3_qwen_runner/`
- Rename: `tests/unit/test_rkllm_probe.py` to `tests/unit/test_rknn3_probe.py`
- Modify: `tests/integration/test_cli.py`
- Rename: `tests/hardware/test_rkllm_prerequisites.py` to `tests/hardware/test_rknn3_prerequisites.py`
- Modify: `tests/unit/test_hardware_gate.py`

- [ ] **Step 1: Write failing package-root probe tests**

Create the renamed `tests/unit/test_rknn3_probe.py` around this contract:

```python
def test_rknn3_probe_reports_all_missing_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "x86_64")

    capabilities = probe_rknn3(tmp_path / "missing-package")

    assert capabilities.name == "rknn3"
    assert capabilities.available is False
    assert capabilities.target == "rk3588-rk1828"
    assert capabilities.is_mock is False
    assert "deployment package is missing" in (capabilities.reason or "")
    assert "host architecture is x86_64, expected aarch64" in (
        capabilities.reason or ""
    )


def test_rknn3_backend_never_falls_back_when_unavailable(tmp_path: Path) -> None:
    backend = RKNN3Backend(tmp_path / "missing-package")

    with pytest.raises(BackendUnavailableError, match="deployment package"):
        backend.load()
```

Update CLI tests so choices are `mock|rknn3`, the doctor option is `--package`, and error messages mention
the deployment package rather than `.rkllm`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_rknn3_probe.py tests/integration/test_cli.py \
  tests/unit/test_hardware_gate.py -q
```

Expected: collection/import FAIL because `RKNN3Backend` and `probe_rknn3` do not exist.

- [ ] **Step 3: Implement the guarded RKNN3 boundary**

Create `src/rk_llm/backends/rknn3.py`:

```python
"""Guarded boundary for the RKNN3 Qwen native runner."""

from collections.abc import Iterator
from pathlib import Path

from rk_llm.errors import BackendUnavailableError, NativeRunnerError
from rk_llm.platform.probe import probe_rknn3
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


_UNIMPLEMENTED_MESSAGE = "RKNN3 native protocol is not implemented in this milestone"


class RKNN3Backend:
    def __init__(self, package_path: Path):
        self._package_path = package_path

    def capabilities(self) -> BackendCapabilities:
        prerequisites = probe_rknn3(self._package_path)
        if not prerequisites.available:
            return prerequisites
        return BackendCapabilities(
            name="rknn3",
            available=False,
            streaming=True,
            target="rk3588-rk1828",
            is_mock=False,
            reason=_UNIMPLEMENTED_MESSAGE,
        )

    def load(self) -> None:
        prerequisites = probe_rknn3(self._package_path)
        if not prerequisites.available:
            raise BackendUnavailableError(
                prerequisites.reason or "RKNN3 backend prerequisites unavailable"
            )
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield from ()
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def shutdown(self) -> None:
        return None
```

Replace `probe_rkllm` with this initial `probe_rknn3` in `src/rk_llm/platform/probe.py`:

```python
def probe_rknn3(package_path: Path) -> BackendCapabilities:
    reasons: list[str] = []
    if not package_path.is_dir():
        reasons.append(f"deployment package is missing: {package_path}")
    else:
        runner = package_path / "bin/rknn_qwen_runner"
        manifest = package_path / "manifest.json"
        if not runner.is_file() or not os.access(runner, os.X_OK):
            reasons.append(f"native runner is not executable: {runner}")
        if not manifest.is_file():
            reasons.append(f"deployment manifest is missing: {manifest}")
    machine = platform.machine()
    if machine not in {"aarch64", "arm64"}:
        reasons.append(f"host architecture is {machine}, expected aarch64")
    return BackendCapabilities(
        name="rknn3",
        available=not reasons,
        streaming=True,
        target="rk3588-rk1828",
        is_mock=False,
        reason="; ".join(reasons) if reasons else None,
    )
```

In `src/rk_llm/cli.py`, use `RKNN3Backend`, `artifacts/deploy/current`, `--package`, and backend choices
`("mock", "rknn3")`. `_backend` must pass `config.package_path` and must raise
`ValueError("package_path is required for rknn3")` if absent.

Rename the native directory with `git mv`. Change its project and executable names to
`rknn3_qwen_runner`, and change the stub error to:

```cpp
#include <iostream>

int main() {
  std::cerr << "{\"type\":\"error\",\"code\":\"RKNN3_NATIVE_ADAPTER_NOT_AVAILABLE\"}\n";
  return 78;
}
```

Rename hardware environment variables to `RUN_RK_HARDWARE_TESTS` and `RKNN3_PACKAGE`; keep the hardware
test `xfail` because this phase does not implement model loading.

- [ ] **Step 4: Run all renamed backend and CLI tests**

Run:

```bash
python3 -m pytest tests/unit/test_rknn3_probe.py tests/integration/test_cli.py \
  tests/unit/test_hardware_gate.py -q
```

Expected: PASS, with hardware behavior still explicitly unavailable/xfail.

- [ ] **Step 5: Prove no runtime RKLLM identifier remains**

Run:

```bash
rg -n "RKLLMBackend|probe_rkllm|--backend rkllm|model\.rkllm|RKLLM_RUNNER|RKLLM_MODEL" \
  src tests native configs
```

Expected: no output.

- [ ] **Step 6: Commit the backend boundary rename**

```bash
git add src native configs tests
git commit -m "refactor: replace RKLLM stub with RKNN3 boundary"
```

## Task 3: Add Pinned Upstream and Model Manifests

**Files:**
- Create: `manifests/upstream.yaml`
- Create: `configs/models/qwen2_5_0_5b.yaml`
- Create: `src/rk_llm/manifests/__init__.py`
- Create: `src/rk_llm/manifests/loader.py`
- Create: `tests/unit/test_manifests.py`
- Delete: `third_party/versions.yaml`

- [ ] **Step 1: Write failing typed-manifest tests**

Create `tests/unit/test_manifests.py` with these core cases:

```python
from pathlib import Path

import pytest

from rk_llm.manifests.loader import load_model_manifest, load_upstream_manifest


def test_repository_manifests_pin_verified_versions() -> None:
    upstream = load_upstream_manifest(Path("manifests/upstream.yaml"))
    model = load_model_manifest(Path("configs/models/qwen2_5_0_5b.yaml"))

    assert upstream.toolkit.revision == "cf292045d77c9ad0377b9fb326f216967475071e"
    assert upstream.model_zoo.revision == "f63048265b49bd2c6236790087287bed6c6b76fe"
    assert upstream.runtime.version == "1.0.4"
    assert model.model_id == "qwen2_5_0_5b"
    assert model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert model.platform == "rk1820"
    assert len(model.source_files) == 10
    assert len(model.generated_files) == 6


def test_manifest_rejects_non_sha256_digest(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\n"
        "model_id: demo\nrepository: owner/model\nrevision: abc\nplatform: rk1820\n"
        "source_files: [{path: model.bin, size: 1, sha256: short}]\n"
        "generated_files: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha256"):
        load_model_manifest(path)
```

- [ ] **Step 2: Run the test and verify import failure**

Run:

```bash
python3 -m pytest tests/unit/test_manifests.py -q
```

Expected: collection FAIL because `rk_llm.manifests` does not exist.

- [ ] **Step 3: Create exact version-controlled YAML data**

`manifests/upstream.yaml` must contain:

```yaml
schema_version: 1
rknn3_toolkit:
  repository: https://github.com/airockchip/rknn3-toolkit.git
  release: V1.0.4
  revision: cf292045d77c9ad0377b9fb326f216967475071e
rknn3_model_zoo:
  repository: https://github.com/airockchip/rknn3-model-zoo.git
  release: V1.0.4
  revision: f63048265b49bd2c6236790087287bed6c6b76fe
runtime:
  version: 1.0.4
  files:
    - path: include/float16.h
      sha256: 6e230c07bbcfd0ea75c64d44d1b07ed3e549a88d4bb6908b4b9941d4a04fb424
    - path: include/rknn3_api.h
      sha256: 64202b613bb87c6445499cb871e7227f02e9af4720e30750344477ae6e87c16d
    - path: Linux/aarch64/librknn3_api.so
      sha256: 113ec97719e04f82e51fcb8badeb18461070ac55ca9a5da87f887f3110b4fcbe
    - path: Linux/aarch64/librknn3_api_rkcp.so
      sha256: 5ea77749f44be1f0c2ad0347242d4b431d3907d03eac11d265496ddd80cfd210
    - path: Linux/aarch64/librknn3_api_native.so
      sha256: 8ec78e9d294e6ecf2be6ad9e16004ae5c50bcb9a8567d8bac7310ab27b66dd11
target:
  host_soc: rk3588
  accelerator: rk1828
  compiler_platform: rk1820
  architecture: aarch64
  glibc_max: "2.35"
  glibcxx_max: "3.4.30"
```

`configs/models/qwen2_5_0_5b.yaml` must record all 10 source files and six generated files. Each entry has
`path`, `size`, and `sha256`; use the exact values captured in the approved design session:

```yaml
schema_version: 1
model_id: qwen2_5_0_5b
repository: Qwen/Qwen2.5-0.5B-Instruct
revision: 7ae557604adf67be50417f59c2c2f167def9a775
platform: rk1820
source_root: models/Qwen2.5-0.5B-Instruct
generated_root: rknn3-model-zoo/examples/Qwen2_5/model/llm
source_files:
  - {path: .gitattributes, size: 1519, sha256: 11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361}
  - {path: LICENSE, size: 11343, sha256: 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e}
  - {path: README.md, size: 4917, sha256: b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15}
  - {path: config.json, size: 659, sha256: 18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45}
  - {path: generation_config.json, size: 242, sha256: e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6}
  - {path: merges.txt, size: 1671839, sha256: 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3}
  - {path: model.safetensors, size: 988097824, sha256: fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe}
  - {path: tokenizer.json, size: 7031645, sha256: c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539}
  - {path: tokenizer_config.json, size: 7305, sha256: 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583}
  - {path: vocab.json, size: 2776833, sha256: ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910}
generated_files:
  - {path: Qwen2.5-0.5B-Instruct.onnx, size: 754920840, sha256: e31d74a2d5f4dbd52bf9d733eee292b2fd8b162a920a8ea92ea17463d2af7586}
  - {path: Qwen2.5-0.5B-Instruct.config.pkl, size: 5067, sha256: af4b89296afeea4cfe8072c55b5dba5319bdd4c9ff9733a4c51cb8fa00105e9b}
  - {path: Qwen2.5-0.5B-Instruct.tokenizer.gguf, size: 5931031, sha256: f2c2188ff62a9eae426fe1902405a99745cde3144443bd5298435a541560c4ee}
  - {path: Qwen2.5-0.5B-Instruct.embed.bin, size: 272269312, sha256: d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0}
  - {path: Qwen2.5-0.5B-Instruct.rknn, size: 17939072, sha256: 013dd8c92fa7c08feaac9b3fd9c6dc8370b5913589bb5ba8d2d7c61a8552ee6a}
  - {path: Qwen2.5-0.5B-Instruct.weight, size: 333308416, sha256: 94bbef9ec8eb5eee08473105af3d88bcce062283db763adba15804d03b7e40f8}
```

- [ ] **Step 4: Implement strict typed YAML loaders**

Create these immutable dataclasses in `src/rk_llm/manifests/loader.py`:

```python
@dataclass(frozen=True)
class GitPin:
    repository: str
    release: str
    revision: str


@dataclass(frozen=True)
class DigestPin:
    path: Path
    sha256: str


@dataclass(frozen=True)
class RuntimePin:
    version: str
    files: tuple[DigestPin, ...]


@dataclass(frozen=True)
class TargetPin:
    host_soc: str
    accelerator: str
    compiler_platform: str
    architecture: str
    glibc_max: str
    glibcxx_max: str


@dataclass(frozen=True)
class UpstreamManifest:
    toolkit: GitPin
    model_zoo: GitPin
    runtime: RuntimePin
    target: TargetPin


@dataclass(frozen=True)
class SizedFilePin:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    repository: str
    revision: str
    platform: str
    source_root: Path
    generated_root: Path
    source_files: tuple[SizedFilePin, ...]
    generated_files: tuple[SizedFilePin, ...]
```

Use `yaml.safe_load`, reject non-mappings, require `schema_version == 1`,
validate revisions/digests with `re.fullmatch(r"[0-9a-f]{40}", value)` and
`re.fullmatch(r"[0-9a-f]{64}", value)`, require positive sizes, and reject absolute or parent-traversing paths:

```python
def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value
```

Export only the two loader functions from `src/rk_llm/manifests/__init__.py`. Delete the obsolete
`third_party/versions.yaml` after all tests consume the new source of truth.

- [ ] **Step 5: Run manifest tests**

Run:

```bash
python3 -m pytest tests/unit/test_manifests.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit pinned inputs**

```bash
git add manifests configs/models src/rk_llm/manifests tests/unit/test_manifests.py \
  third_party/versions.yaml
git commit -m "feat: pin RKNN3 and Qwen inputs"
```

## Task 4: Implement the Deployment Manifest Contract

**Files:**
- Create: `manifests/schemas/deployment-package.schema.json`
- Create: `src/rk_llm/artifacts/__init__.py`
- Create: `src/rk_llm/artifacts/manifest.py`
- Create: `tests/unit/test_artifact_manifest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing package-ID and path-safety tests**

Create tests using a minimal package tree:

```python
def _file_record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "elf": None,
    }


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_id": "0" * 16,
        "created_at": "2026-08-18T00:00:00Z",
        "model": {
            "id": "qwen2_5_0_5b",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        "toolchain": {
            "project_commit": "a" * 40,
            "toolkit": {"release": "V1.0.4", "revision": "cf292045d77c9ad0377b9fb326f216967475071e"},
            "model_zoo": {"release": "V1.0.4", "revision": "f63048265b49bd2c6236790087287bed6c6b76fe"},
            "runtime_version": "1.0.4",
            "firmware_version": "1.0.4",
            "builder": {"image": "ubuntu:22.04", "compiler": "aarch64-linux-gnu-g++ 11.4.0"},
        },
        "target": {
            "host_soc": "rk3588",
            "accelerator": "rk1828",
            "architecture": "aarch64",
            "glibc_max": "2.35",
            "glibcxx_max": "3.4.30",
        },
        "build": {"export_args": [], "rknn_args": ["--platform", "rk1820"], "cmake_args": []},
        "files": [],
    }


def _write_valid_package(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    package = tmp_path / "package"
    runner = package / "bin/rknn_qwen_runner"
    runner.parent.mkdir(parents=True)
    payload = b"runner"
    runner.write_bytes(payload)
    manifest = _valid_manifest()
    manifest["files"] = [_file_record("bin/rknn_qwen_runner", payload)]
    manifest["package_id"] = compute_package_id(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package, manifest


def test_package_id_ignores_created_at_and_package_id() -> None:
    first = _valid_manifest()
    second = {**first, "package_id": "f" * 16, "created_at": "2030-01-01T00:00:00Z"}

    assert compute_package_id(first) == compute_package_id(second)


def test_validate_package_rejects_parent_traversal(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = _valid_manifest()
    manifest["files"] = [_file_record("../outside", b"x")]
    manifest["package_id"] = compute_package_id(manifest)
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="safe relative path"):
        validate_package(package)


def test_validate_package_rejects_undeclared_file(tmp_path: Path) -> None:
    package, manifest = _write_valid_package(tmp_path)
    (package / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ArtifactError, match="undeclared files"):
        validate_package(package)
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```bash
python3 -m pytest tests/unit/test_artifact_manifest.py -q
```

Expected: collection FAIL because `rk_llm.artifacts.manifest` does not exist.

- [ ] **Step 3: Add the exact package schema and runtime dependency**

Add `jsonschema>=4.23,<5` to `[project].dependencies`. The Draft 2020-12 schema must require exactly these
top-level keys and set `additionalProperties` to `false`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["schema_version", "package_id", "created_at", "model", "toolchain", "target", "build", "files"],
  "properties": {
    "schema_version": {"const": 1},
    "package_id": {"type": "string", "pattern": "^[0-9a-f]{16}$"},
    "created_at": {"type": "string", "format": "date-time"},
    "model": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "repository", "revision"],
      "properties": {
        "id": {"const": "qwen2_5_0_5b"},
        "repository": {"const": "Qwen/Qwen2.5-0.5B-Instruct"},
        "revision": {"pattern": "^[0-9a-f]{40}$"}
      }
    },
    "toolchain": {
      "type": "object",
      "additionalProperties": false,
      "required": ["project_commit", "toolkit", "model_zoo", "runtime_version", "firmware_version", "builder"],
      "properties": {
        "project_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "toolkit": {
          "type": "object",
          "additionalProperties": false,
          "required": ["release", "revision"],
          "properties": {
            "release": {"const": "V1.0.4"},
            "revision": {"const": "cf292045d77c9ad0377b9fb326f216967475071e"}
          }
        },
        "model_zoo": {
          "type": "object",
          "additionalProperties": false,
          "required": ["release", "revision"],
          "properties": {
            "release": {"const": "V1.0.4"},
            "revision": {"const": "f63048265b49bd2c6236790087287bed6c6b76fe"}
          }
        },
        "runtime_version": {"const": "1.0.4"},
        "firmware_version": {"const": "1.0.4"},
        "builder": {
          "type": "object",
          "additionalProperties": false,
          "required": ["image", "compiler"],
          "properties": {
            "image": {"const": "ubuntu:22.04"},
            "compiler": {"type": "string", "minLength": 1}
          }
        }
      }
    },
    "target": {
      "type": "object",
      "additionalProperties": false,
      "required": ["host_soc", "accelerator", "architecture", "glibc_max", "glibcxx_max"],
      "properties": {
        "host_soc": {"const": "rk3588"},
        "accelerator": {"const": "rk1828"},
        "architecture": {"const": "aarch64"},
        "glibc_max": {"const": "2.35"},
        "glibcxx_max": {"const": "3.4.30"}
      }
    },
    "build": {
      "type": "object",
      "additionalProperties": false,
      "required": ["export_args", "rknn_args", "cmake_args"],
      "properties": {
        "export_args": {"type": "array", "items": {"type": "string"}},
        "rknn_args": {"type": "array", "items": {"type": "string"}},
        "cmake_args": {"type": "array", "items": {"type": "string"}}
      }
    },
    "files": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["path", "size", "sha256", "elf"],
        "properties": {
          "path": {"type": "string", "minLength": 1},
          "size": {"type": "integer", "minimum": 1},
          "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
          "elf": {
            "oneOf": [
              {"type": "null"},
              {
                "type": "object",
                "additionalProperties": false,
                "required": ["glibc", "glibcxx"],
                "properties": {
                  "glibc": {"type": ["string", "null"]},
                  "glibcxx": {"type": ["string", "null"]}
                }
              }
            ]
          }
        }
      }
    }
  },
  "additionalProperties": false
}
```

Refresh the editable development environment before implementing the validator:

```bash
python3 -m pip install -e ".[dev]"
```

Expected: installation succeeds and `python3 -c 'import jsonschema'` exits 0.

- [ ] **Step 4: Implement canonical IDs and directory validation**

In `src/rk_llm/artifacts/manifest.py`, implement:

```python
def _schema_path() -> Path:
    configured = os.environ.get("RK_LLM_ROOT")
    root = Path(configured) if configured is not None else Path(__file__).resolve().parents[3]
    if not root.is_absolute():
        raise ArtifactError("RK_LLM_ROOT must be an absolute path")
    return root / "manifests/schemas/deployment-package.schema.json"


def _load_schema() -> dict[str, object]:
    path = _schema_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"failed to load deployment schema {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"deployment schema root must be an object: {path}")
    return value


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError("deployment file path must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"bin", "lib", "model"}:
        raise ArtifactError(f"deployment file path must be a safe relative path: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload(manifest: Mapping[str, object]) -> bytes:
    payload = {key: value for key, value in manifest.items() if key not in {"package_id", "created_at"}}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_package_id(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_payload(manifest)).hexdigest()[:16]


def validate_package(package_root: Path) -> dict[str, object]:
    manifest_path = package_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        Draft202012Validator(_load_schema()).validate(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise ArtifactError(f"invalid deployment manifest {manifest_path}: {error}") from error
    if manifest["package_id"] != compute_package_id(manifest):
        raise ArtifactError("deployment package_id does not match canonical manifest content")
    declared: set[Path] = set()
    for record in manifest["files"]:
        relative = _safe_relative_path(record["path"])
        if relative in declared:
            raise ArtifactError(f"duplicate deployment file: {relative}")
        declared.add(relative)
        actual = package_root / relative
        if not actual.is_file() or actual.is_symlink():
            raise ArtifactError(f"deployment file is missing or is a symlink: {relative}")
        if actual.stat().st_size != record["size"]:
            raise ArtifactError(f"deployment file size mismatch: {relative}")
        if _sha256(actual) != record["sha256"]:
            raise ArtifactError(f"deployment file SHA-256 mismatch: {relative}")
    present = {
        path.relative_to(package_root)
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if present != declared:
        raise ArtifactError(f"undeclared files: {sorted(str(path) for path in present - declared)}")
    return manifest
```

Import `Draft202012Validator`, `FormatChecker`, and `ValidationError`; construct the validator with
`format_checker=FormatChecker()` so `created_at` is checked as an RFC 3339 date-time.

- [ ] **Step 5: Run artifact tests and the full non-hardware suite**

Run:

```bash
python3 -m pytest tests/unit/test_artifact_manifest.py -q
python3 -m pytest -m "not hardware" -q
```

Expected: both PASS.

- [ ] **Step 6: Commit the deployment contract**

```bash
git add pyproject.toml manifests/schemas src/rk_llm/artifacts \
  tests/unit/test_artifact_manifest.py
git commit -m "feat: validate immutable deployment packages"
```

## Task 5: Bootstrap Pinned Vendor Checkouts and Runtime Files

**Files:**
- Create: `src/rk_llm/host/__init__.py`
- Create: `src/rk_llm/host/bootstrap.py`
- Create: `tools/host/bootstrap`
- Create: `tests/unit/test_host_bootstrap.py`
- Create: `tests/integration/test_host_bootstrap.py`

- [ ] **Step 1: Write failing dirty-checkout and Runtime-hash tests**

Create unit tests around injected command execution and temporary files:

```python
def test_existing_dirty_checkout_is_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / ".vendor/rknn3-model-zoo"
    checkout.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=" M tracked-file\n", stderr="")

    pin = GitPin("https://example.invalid/model-zoo.git", "V1.0.4", "a" * 40)

    with pytest.raises(ConfigurationError, match="uncommitted changes"):
        ensure_checkout(pin, checkout, fake_run)

    assert checkout.is_dir()
    assert len(calls) == 1
    assert calls[0][-2:] == ["status", "--porcelain"]


def test_runtime_file_hash_mismatch_does_not_populate_vendor_directory(
    tmp_path: Path,
) -> None:
    runtime_source = tmp_path / "runtime"
    (runtime_source / "include").mkdir(parents=True)
    (runtime_source / "include/rknn3_api.h").write_bytes(b"wrong")

    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        adopt_runtime_files(
            runtime_source,
            tmp_path / ".vendor/rknn3-runtime",
            (DigestPin(Path("include/rknn3_api.h"), "0" * 64),),
        )

    assert not (tmp_path / ".vendor/rknn3-runtime").exists()
```

The integration test uses a real local Git repository:

```python
def test_checkout_is_detached_at_exact_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    (source / "tracked.txt").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-m", "fixture",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    ensure_checkout(GitPin(source.as_uri(), "fixture", revision), destination)

    actual = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    symbolic = subprocess.run(
        ["git", "-C", str(destination), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
    )
    assert actual == revision
    assert symbolic.returncode == 1
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
python3 -m pytest tests/unit/test_host_bootstrap.py \
  tests/integration/test_host_bootstrap.py -q
```

Expected: collection FAIL because `rk_llm.host.bootstrap` does not exist.

- [ ] **Step 3: Implement non-destructive Git checkout management**

Use an injected callable compatible with `subprocess.run` and argument lists only. The core sequence is:

```python
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _checked(run: CommandRunner, args: list[str]) -> None:
    try:
        run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "command failed").strip()
        raise ConfigurationError(f"command failed: {args!r}: {detail}") from error


def _capture(run: CommandRunner, args: list[str]) -> str:
    try:
        result = run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "command failed").strip()
        raise ConfigurationError(f"command failed: {args!r}: {detail}") from error
    return result.stdout


def ensure_checkout(
    pin: GitPin,
    destination: Path,
    run: CommandRunner = subprocess.run,
    reference: Path | None = None,
) -> None:
    if destination.exists():
        status = _capture(run, ["git", "-C", str(destination), "status", "--porcelain"])
        if status.strip():
            raise ConfigurationError(f"vendor checkout has uncommitted changes: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--no-checkout"]
        if reference is not None:
            command.extend(["--reference-if-able", str(reference)])
        command.extend([pin.repository, str(destination)])
        _checked(run, command)
    _checked(run, ["git", "-C", str(destination), "fetch", "origin", pin.revision])
    _checked(run, ["git", "-C", str(destination), "checkout", "--detach", pin.revision])
    actual = _capture(run, ["git", "-C", str(destination), "rev-parse", "HEAD"]).strip()
    if actual != pin.revision:
        raise ConfigurationError(
            f"vendor checkout revision mismatch: expected {pin.revision}, got {actual}"
        )
```

Do not run `git clean`, `git reset --hard`, recursive deletion, or force checkout.

- [ ] **Step 4: Implement atomic Runtime development-file adoption**

Verify every source file first. Copy into a sibling temporary directory, fsync files, then rename that complete
directory to `.vendor/rknn3-runtime`. If the destination exists, verify it and reuse it; a mismatch is an error.
Require an absolute `RKNN3_RUNTIME_DEV_ROOT` in the command entry point.

Use this atomic shape in `bootstrap.py`:

```python
def _verify_digest(root: Path, pin: DigestPin) -> None:
    path = root / pin.path
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"Runtime development file is missing: {path}")
    actual = _sha256(path)
    if actual != pin.sha256:
        raise ArtifactError(
            f"Runtime development file SHA-256 mismatch: {path}: "
            f"expected {pin.sha256}, got {actual}"
        )


def adopt_runtime_files(
    source_root: Path,
    destination: Path,
    files: tuple[DigestPin, ...],
) -> None:
    if not source_root.is_absolute():
        raise ConfigurationError("RKNN3_RUNTIME_DEV_ROOT must be an absolute path")
    for pin in files:
        _verify_digest(source_root, pin)
    if destination.exists():
        for pin in files:
            _verify_digest(destination, pin)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for pin in files:
            target = temporary / pin.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / pin.path, target)
            with target.open("rb") as copied:
                os.fsync(copied.fileno())
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
```

Create executable `tools/host/bootstrap` as the stable direct entry point:

```sh
#!/usr/bin/env sh
set -eu
exec "${RK_LLM_PYTHON:-python3}" -m rk_llm.host.bootstrap "$@"
```

Mark it executable with `chmod +x tools/host/bootstrap`.

The exact module CLI for the implementation worktree is:

```text
python -m rk_llm.host.bootstrap \
  --project-root /home/barry/AI_Infra/RK_LLM/.worktrees/rknn3-foundation \
  --upstream-manifest /home/barry/AI_Infra/RK_LLM/.worktrees/rknn3-foundation/manifests/upstream.yaml \
  --runtime-dev-root /home/barry/rk1828-work/rknn3-model-zoo/3rdparty/rknpu3 \
  --seed-workspace /home/barry/rk1828-work
```

When `--seed-workspace` is present, use its clean `rknn3-toolkit` and `rknn3-model-zoo` repositories only as
Git object references. Keep the official GitHub URLs as the destination remotes.

- [ ] **Step 5: Run bootstrap tests**

Run:

```bash
python3 -m pytest tests/unit/test_host_bootstrap.py \
  tests/integration/test_host_bootstrap.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit bootstrap support**

```bash
git add src/rk_llm/host tools/host/bootstrap tests/unit/test_host_bootstrap.py \
  tests/integration/test_host_bootstrap.py
git commit -m "feat: bootstrap pinned RKNN3 dependencies"
```

## Task 6: Import the Verified Existing Workspace

**Files:**
- Create: `src/rk_llm/host/import_existing.py`
- Create: `tools/host/import-existing`
- Create: `tests/unit/test_import_existing.py`
- Create: `tests/integration/test_import_existing.py`
- Modify: `artifacts/README.md`

- [ ] **Step 1: Write failing atomic-import tests**

Build tiny source fixtures from an in-test `ModelManifest` rather than copying real models:

```python
def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _workspace_fixture(tmp_path: Path) -> tuple[Path, Path, ModelManifest]:
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    source = workspace / "models/demo/model.bin"
    generated = workspace / "model-zoo/demo/output.rknn"
    source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    generated.write_bytes(b"output")
    model = ModelManifest(
        model_id="demo",
        repository="owner/demo",
        revision="a" * 40,
        platform="rk1820",
        source_root=Path("models/demo"),
        generated_root=Path("model-zoo/demo"),
        source_files=(SizedFilePin(Path("model.bin"), 6, _digest(b"source")),),
        generated_files=(SizedFilePin(Path("output.rknn"), 6, _digest(b"output")),),
    )
    return workspace, project, model


def _tree_hashes(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): _digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_import_copies_verified_files_and_writes_record(tmp_path: Path) -> None:
    workspace, project, model = _workspace_fixture(tmp_path)

    record = import_existing(workspace, project, model, mode="copy")

    imported = project / "artifacts/source_models/demo/model.bin"
    assert imported.read_bytes() == b"source"
    assert imported.stat().st_ino != (workspace / "models/demo/model.bin").stat().st_ino
    assert record["model_id"] == "demo"
    assert record["mode"] == "copy"
    assert len(record["files"]) == 2


def test_import_reuses_matching_target_and_rejects_mismatch(tmp_path: Path) -> None:
    workspace, project, model = _workspace_fixture(tmp_path)
    import_existing(workspace, project, model, mode="copy")
    target = project / "artifacts/work/demo/model/output.rknn"
    target.write_bytes(b"changed")

    with pytest.raises(ArtifactError, match="existing destination does not match"):
        import_existing(workspace, project, model, mode="copy")


def test_failed_import_keeps_source_workspace_unchanged(tmp_path: Path) -> None:
    workspace, project, model = _workspace_fixture(tmp_path)
    before = _tree_hashes(workspace)
    bad_model = replace(
        model,
        source_files=(SizedFilePin(Path("model.bin"), 6, "0" * 64),),
    )

    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        import_existing(workspace, project, bad_model, mode="copy")

    assert _tree_hashes(workspace) == before
    assert not (project / "artifacts/source_models/demo").exists()
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
python3 -m pytest tests/unit/test_import_existing.py \
  tests/integration/test_import_existing.py -q
```

Expected: collection FAIL because `rk_llm.host.import_existing` does not exist.

- [ ] **Step 3: Implement verified import planning and atomic copy**

The implementation maps files from the manifest as follows:

```python
def _destination(project_root: Path, model_id: str, category: str, path: Path) -> Path:
    if category == "source":
        return project_root / "artifacts/source_models" / model_id / path
    return project_root / "artifacts/work" / model_id / "model" / path


def _source_root(workspace: Path, model: ModelManifest, category: str) -> Path:
    relative = model.source_root if category == "source" else model.generated_root
    return workspace / relative
```

For each file, reject source symlinks, verify exact size and SHA-256 before any write, then populate a sibling
temporary import root. `mode="copy"` uses `shutil.copy2`; `mode="hardlink"` uses `os.link` and records that
choice. The default is `copy`. Rename complete category directories only after all files verify. If an existing
destination file matches, reuse it; if it differs, stop without replacing it.

Use this category-level transaction so a failed copy cannot leave a partially populated category:

```python
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_file(path: Path, pin: SizedFilePin, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"{label} file is missing or is a symlink: {path}")
    if path.stat().st_size != pin.size:
        raise ArtifactError(f"{label} size mismatch: {path}")
    actual = _sha256(path)
    if actual != pin.sha256:
        raise ArtifactError(
            f"{label} SHA-256 mismatch: {path}: expected {pin.sha256}, got {actual}"
        )


def _populate_category(
    source_root: Path,
    destination_root: Path,
    files: tuple[SizedFilePin, ...],
    mode: str,
) -> str:
    for pin in files:
        _verify_pinned_file(source_root / pin.path, pin, "source")
    if destination_root.exists():
        for pin in files:
            _verify_pinned_file(destination_root / pin.path, pin, "existing destination")
        present = {
            path.relative_to(destination_root)
            for path in destination_root.rglob("*")
            if path.is_file()
        }
        expected = {pin.path for pin in files}
        if present != expected:
            raise ArtifactError(f"existing destination has unexpected files: {destination_root}")
        return "reused"
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_root.name}.", dir=destination_root.parent)
    )
    try:
        for pin in files:
            target = temporary / pin.path
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "copy":
                shutil.copy2(source_root / pin.path, target)
            else:
                os.link(source_root / pin.path, target)
            _verify_pinned_file(target, pin, "imported destination")
        os.replace(temporary, destination_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "imported"


def _write_record(path: Path, record: dict[str, object]) -> None:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def import_existing(
    workspace: Path,
    project_root: Path,
    model: ModelManifest,
    mode: str = "copy",
) -> dict[str, object]:
    if not workspace.is_absolute() or not project_root.is_absolute():
        raise ConfigurationError("workspace and project_root must be absolute paths")
    if mode not in {"copy", "hardlink"}:
        raise ConfigurationError("mode must be copy or hardlink")
    for source_root, pins in (
        (workspace / model.source_root, model.source_files),
        (workspace / model.generated_root, model.generated_files),
    ):
        for pin in pins:
            _verify_pinned_file(source_root / pin.path, pin, "source")
    statuses = {
        "source": _populate_category(
            workspace / model.source_root,
            project_root / "artifacts/source_models" / model.model_id,
            model.source_files,
            mode,
        ),
        "generated": _populate_category(
            workspace / model.generated_root,
            project_root / "artifacts/work" / model.model_id / "model",
            model.generated_files,
            mode,
        ),
    }
    files = [
        {"category": category, "path": str(pin.path), "size": pin.size, "sha256": pin.sha256}
        for category, pins in (
            ("source", model.source_files),
            ("generated", model.generated_files),
        )
        for pin in pins
    ]
    record: dict[str, object] = {
        "schema_version": 1,
        "model_id": model.model_id,
        "source_workspace": str(workspace),
        "mode": mode,
        "statuses": statuses,
        "files": files,
    }
    record_path = project_root / "artifacts/work" / model.model_id / "import-record.json"
    _write_record(record_path, record)
    return record
```

Write `artifacts/work/qwen2_5_0_5b/import-record.json` atomically with:

```json
{
  "schema_version": 1,
  "model_id": "qwen2_5_0_5b",
  "source_workspace": "/home/barry/rk1828-work",
  "mode": "copy",
  "files": [
    {"category": "source", "path": "model.safetensors", "size": 988097824, "sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"}
  ]
}
```

The CLI requires absolute `--workspace` and `--project-root`, accepts only `--mode copy|hardlink`, and prints
one JSON summary to stdout. It never deletes or chmods the source workspace.

Create executable `tools/host/import-existing`:

```sh
#!/usr/bin/env sh
set -eu
exec "${RK_LLM_PYTHON:-python3}" -m rk_llm.host.import_existing "$@"
```

Run `chmod +x tools/host/import-existing`.

- [ ] **Step 4: Run import tests**

Run:

```bash
python3 -m pytest tests/unit/test_import_existing.py \
  tests/integration/test_import_existing.py -q
```

Expected: PASS.

- [ ] **Step 5: Document ignored artifact ownership**

Update `artifacts/README.md` with the exact five ignored subdirectories, the import record location, and the
statement that imported data is local state and must not be committed or treated as a deployment package.

- [ ] **Step 6: Commit the importer**

```bash
git add src/rk_llm/host/import_existing.py tools/host/import-existing tests/unit/test_import_existing.py \
  tests/integration/test_import_existing.py artifacts/README.md
git commit -m "feat: import verified RKNN3 workspace assets"
```

## Task 7: Add Stable Make Targets, Ignore Rules, and Foundation Documentation

**Files:**
- Modify: `.gitignore`
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/host-setup.md`
- Modify: `docs/board-setup.md`
- Modify: `docs/model-export.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `requirements/toolkit.txt`
- Modify: `requirements/board.txt`
- Modify: `tests/unit/test_repository_layout.py`

- [ ] **Step 1: Write failing repository-contract tests**

Replace the old RKLLM layout assertions with:

```python
def test_rknn3_foundation_has_required_boundaries() -> None:
    required = (
        "configs/models/qwen2_5_0_5b.yaml",
        "configs/runtime/rk3588.yaml",
        "manifests/upstream.yaml",
        "manifests/schemas/deployment-package.schema.json",
        "native/rknn3_qwen_runner/CMakeLists.txt",
        "src/rk_llm/backends/rknn3.py",
        "src/rk_llm/host/bootstrap.py",
        "src/rk_llm/host/import_existing.py",
        "tools/host/bootstrap",
        "tools/host/import-existing",
        "artifacts/README.md",
    )
    assert [path for path in required if not Path(path).is_file()] == []


def test_generated_and_vendor_roots_are_ignored() -> None:
    patterns = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())
    assert {
        ".vendor/",
        ".host-venv/",
        "artifacts/source_models/",
        "artifacts/work/",
        "artifacts/packages/",
        "artifacts/deploy/",
        "artifacts/logs/",
    } <= patterns


def test_product_docs_do_not_advertise_old_rkllm_flow() -> None:
    product_docs = [Path("README.md"), *Path("docs").glob("*.md")]
    stale = [
        str(path)
        for path in product_docs
        if path.name != "rk1828-rknn3-deployment.md"
        and any(token in path.read_text(encoding="utf-8") for token in ("DeepSeek-R1", "--backend rkllm", ".rkllm"))
    ]
    assert stale == []
```

- [ ] **Step 2: Run layout tests and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_repository_layout.py -q
```

Expected: FAIL on old ignore rules and RKLLM product documentation.

- [ ] **Step 3: Add stable Make targets**

Keep `install`, `test`, and `smoke`; add:

```make
PROJECT_ROOT := $(abspath .)
HOST_VENV ?= $(PROJECT_ROOT)/.host-venv
HOST_PYTHON := $(HOST_VENV)/bin/python
MODEL ?= qwen2_5_0_5b
WORKSPACE ?= /home/barry/rk1828-work
RKNN3_RUNTIME_DEV_ROOT ?= $(WORKSPACE)/rknn3-model-zoo/3rdparty/rknpu3

.PHONY: host-env host-bootstrap host-import

host-env:
	python3 -m venv "$(HOST_VENV)"
	"$(HOST_PYTHON)" -m pip install -e ".[dev]"

host-bootstrap: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.bootstrap \
		--project-root "$(PROJECT_ROOT)" \
		--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml" \
		--runtime-dev-root "$(RKNN3_RUNTIME_DEV_ROOT)" \
		--seed-workspace "$(WORKSPACE)"

host-import: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.import_existing \
		--project-root "$(PROJECT_ROOT)" \
		--workspace "$(WORKSPACE)" \
		--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml" \
		--mode copy
```

Do not add `host-build`, `host-runner`, `host-package`, or `deploy` commands until their implementation plans
provide working targets. Documentation must not advertise commands that only print placeholders.

- [ ] **Step 4: Rewrite product docs around the approved RKNN3 direction**

Update the listed documents to state the current milestone precisely:

- mock CLI remains runnable;
- pinned manifests, bootstrap, and import are implemented;
- the existing verified outputs can be adopted but not yet rebuilt through the new wrapper;
- the RKNN3 backend remains guarded until the Native protocol plan;
- `docs/rk1828-rknn3-deployment.md` remains the evidence-backed manual reference;
- no document claims the new unified commands have completed hardware inference.

Replace `requirements/toolkit.txt` and `requirements/board.txt` RKLLM pins with comments pointing to
`manifests/upstream.yaml`; vendor wheels and Runtime libraries remain external and verified by bootstrap.

- [ ] **Step 5: Run layout and full non-hardware tests**

Run:

```bash
python3 -m pytest tests/unit/test_repository_layout.py -q
python3 -m pytest -m "not hardware" -q
git diff --check
```

Expected: all tests PASS and `git diff --check` has no output.

- [ ] **Step 6: Commit the repository entry points and docs**

```bash
git add .gitignore Makefile README.md \
  docs/architecture.md docs/host-setup.md docs/board-setup.md \
  docs/model-export.md docs/implementation-roadmap.md \
  requirements/toolkit.txt requirements/board.txt tests/unit/test_repository_layout.py
git commit -m "docs: make RKNN3 foundation the project entry point"
```

## Task 8: Verify Bootstrap and Perform the Real Non-Destructive Import

**Files:**
- No tracked files expected; only ignored `.host-venv/`, `.vendor/`, and `artifacts/` state.

- [ ] **Step 1: Record the external workspace before import**

Run:

```bash
cd /home/barry/rk1828-work
sha256sum \
  models/Qwen2.5-0.5B-Instruct/model.safetensors \
  rknn3-model-zoo/examples/Qwen2_5/model/llm/Qwen2.5-0.5B-Instruct.rknn \
  rknn3-model-zoo/examples/Qwen2_5/model/llm/Qwen2.5-0.5B-Instruct.weight \
  rknn3-model-zoo/examples/Qwen2_5/model/llm/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
  rknn3-model-zoo/examples/Qwen2_5/model/llm/Qwen2.5-0.5B-Instruct.embed.bin \
  > /tmp/rk-llm-foundation-import.before.sha256
```

Expected: five digest lines and exit code 0.

- [ ] **Step 2: Run bootstrap using the existing clean repositories as Git references**

Run from the implementation worktree root:

```bash
make host-bootstrap \
  WORKSPACE=/home/barry/rk1828-work \
  RKNN3_RUNTIME_DEV_ROOT=/home/barry/rk1828-work/rknn3-model-zoo/3rdparty/rknpu3
```

Expected: both `.vendor` repositories resolve to their pinned commits, five Runtime development files verify,
and the command exits 0 without modifying `/home/barry/rk1828-work`.

- [ ] **Step 3: Import source and generated model files**

Run:

```bash
make host-import MODEL=qwen2_5_0_5b WORKSPACE=/home/barry/rk1828-work
```

Expected: JSON summary reports `mode: copy`, 16 verified files, and an import record under
`artifacts/work/qwen2_5_0_5b/import-record.json`.

- [ ] **Step 4: Re-run import to prove idempotence**

Run the same `make host-import` command again.

Expected: exit 0; all 16 destinations are reported as reused; no duplicate or temporary directory remains.

- [ ] **Step 5: Prove source hashes and Git boundaries remain intact**

Run:

```bash
cd /home/barry/rk1828-work
sha256sum -c /tmp/rk-llm-foundation-import.before.sha256

cd /home/barry/AI_Infra/RK_LLM/.worktrees/rknn3-foundation
git status --short
git check-ignore -v .vendor artifacts/source_models artifacts/work .host-venv
python3 -m pytest -m "not hardware" -q
```

Expected: five `OK` lines; no tracked changes; every generated root is ignored; all non-hardware tests PASS.

- [ ] **Step 6: Record final review evidence**

Run:

```bash
git log --oneline --decorate -9
git status --short --branch
```

Expected: the plan commit followed by seven focused implementation commits, design commit `81b66ad` in the
visible ancestry, a clean worktree, and no push performed.

## Plan 1 Acceptance Gate

Do not begin the host build/package plan until all of these are true:

- `rk-llm` exposes only `mock` and `rknn3` backends;
- old RKLLM/DeepSeek product identifiers remain only in historical design documents;
- upstream, Runtime, source model, and existing generated files have exact version/hash manifests;
- deployment package IDs and file validation have passing path-safety tests;
- bootstrap preserves dirty vendor checkouts and verifies Runtime files before adoption;
- the old workspace hashes are unchanged after import;
- imported files are project-local, ignored, hash-verified, and idempotently reusable;
- the complete non-hardware test suite passes without vendor SDKs or hardware.
