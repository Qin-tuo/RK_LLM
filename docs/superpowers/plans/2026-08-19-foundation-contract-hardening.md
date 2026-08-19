# RKNN3 Foundation Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the runner filename, bind deployment packages to the verified Qwen/RKNN3 inputs, and remove retired RKLLM backend naming from product code.

**Architecture:** Keep package validation schema-driven and self-contained. The JSON manifest carries the exact ten source pins, fixed model revision, physical accelerator, compiler platform, and required runner path; the native output and board probe use that same path. Rename only the internal exception base and descriptions while preserving the `rk-llm` distribution, `rk_llm` package, CLI, and repository interfaces.

**Tech Stack:** Python 3.12, pytest, JSON Schema Draft 2020-12, CMake, YAML manifests, Git.

---

### Task 1: Align the Native Runner Filename

**Files:**
- Modify: `native/rknn3_qwen_runner/CMakeLists.txt`
- Modify: `tests/unit/test_repository_layout.py`

- [ ] **Step 1: Write the failing contract test**

Add:

```python
def test_native_runner_output_matches_board_probe_contract() -> None:
    cmake = Path("native/rknn3_qwen_runner/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    probe = Path("src/rk_llm/platform/probe.py").read_text(encoding="utf-8")
    assert "add_executable(rknn_qwen_runner " in cmake
    assert 'package_path / "bin/rknn_qwen_runner"' in probe
    assert "rknn3_qwen_runner" not in cmake
```

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_repository_layout.py::test_native_runner_output_matches_board_probe_contract -q
```

Expected: FAIL because CMake declares `rknn3_qwen_runner`.

- [ ] **Step 3: Make the minimal CMake change**

```cmake
add_executable(rknn_qwen_runner src/main.cpp)
```

Do not add runner functionality, installation, or packaging rules.

- [ ] **Step 4: Verify GREEN and commit**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_repository_layout.py tests/unit/test_rknn3_probe.py -q
git add native/rknn3_qwen_runner/CMakeLists.txt tests/unit/test_repository_layout.py
git commit -m "fix: align RKNN3 runner package contract"
```

### Task 2: Require Self-Contained Package Provenance

**Files:**
- Modify: `manifests/schemas/deployment-package.schema.json`
- Modify: `tests/unit/test_artifact_manifest.py`

- [ ] **Step 1: Define the exact source-pin fixture**

Add this constant to `tests/unit/test_artifact_manifest.py` and use
`copy.deepcopy(SOURCE_FILES)` as `_valid_manifest()["model"]["source_files"]`:

```python
SOURCE_FILES = [
    {"path": ".gitattributes", "size": 1519, "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"},
    {"path": "LICENSE", "size": 11343, "sha256": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"},
    {"path": "README.md", "size": 4917, "sha256": "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15"},
    {"path": "config.json", "size": 659, "sha256": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"},
    {"path": "generation_config.json", "size": 242, "sha256": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6"},
    {"path": "merges.txt", "size": 1671839, "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"},
    {"path": "model.safetensors", "size": 988097824, "sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"},
    {"path": "tokenizer.json", "size": 7031645, "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"},
    {"path": "tokenizer_config.json", "size": 7305, "sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"},
    {"path": "vocab.json", "size": 2776833, "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"},
]
```

Add `"compiler_platform": "rk1820"` to the valid target. Add a test that
loads `configs/models/qwen2_5_0_5b.yaml` with `load_model_manifest()` and asserts
the resulting ten `{path, size, sha256}` records equal `SOURCE_FILES`.

- [ ] **Step 2: Write rejection tests**

For each case below, mutate a valid package, call `_write_manifest()` to
recompute its package ID, and assert `validate_package()` raises
`ArtifactError` containing `invalid deployment manifest`:

```python
manifest["model"]["revision"] = "f" * 40
manifest["model"]["source_files"][0]["sha256"] = "f" * 64
del manifest["target"]["compiler_platform"]
manifest["target"]["compiler_platform"] = "rk1828"
manifest["build"]["rknn_args"] = []
```

Add a separate runner test that replaces `bin/rknn_qwen_runner` and its file
record with `bin/other_runner`; validation must reject it.

- [ ] **Step 3: Verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_artifact_manifest.py -q
```

Expected: the valid fixture fails because the current schema forbids
`source_files` and `compiler_platform`, while the old schema accepts the other
invalid provenance cases.

- [ ] **Step 4: Strengthen the JSON schema**

Make `model.revision` a `const` equal to
`7ae557604adf67be50417f59c2c2f167def9a775`. Require `model.source_files` and
define it as `{"const": [...]}` containing the exact ten JSON objects from
`SOURCE_FILES` above in the same order.

Require this target property:

```json
"compiler_platform": {"const": "rk1820"}
```

Require non-empty compiler arguments:

```json
"rknn_args": {
  "type": "array",
  "minItems": 1,
  "items": {"type": "string"}
}
```

Add these keywords to the top-level `files` array schema:

```json
"contains": {
  "type": "object",
  "required": ["path"],
  "properties": {
    "path": {"const": "bin/rknn_qwen_runner"}
  }
},
"minContains": 1
```

- [ ] **Step 5: Verify GREEN and commit**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_artifact_manifest.py -q
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest -m "not hardware" -q
git diff --check
git add manifests/schemas/deployment-package.schema.json tests/unit/test_artifact_manifest.py
git commit -m "fix: bind packages to pinned RKNN3 inputs"
```

### Task 3: Remove Retired RKLLM Product Naming

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/rk_llm/__init__.py`
- Modify: `src/rk_llm/errors.py`
- Modify: `src/rk_llm/cli.py`
- Modify: `src/rk_llm/host/bootstrap.py`
- Modify: `src/rk_llm/host/import_existing.py`
- Modify: `tests/unit/test_rknn3_probe.py`
- Modify: `tests/unit/test_repository_layout.py`

- [ ] **Step 1: Write neutral-naming tests**

Change `tests/unit/test_rknn3_probe.py` to import `ProjectError` and assert all
four specialized errors inherit from it. Add:

```python
def test_product_metadata_and_source_do_not_name_retired_rkllm_backend() -> None:
    paths = (Path("pyproject.toml"), *Path("src/rk_llm").rglob("*.py"))
    findings = [
        str(path)
        for path in paths
        if "RKLLM" in path.read_text(encoding="utf-8")
    ]
    assert findings == []
```

- [ ] **Step 2: Verify RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_rknn3_probe.py tests/unit/test_repository_layout.py::test_product_metadata_and_source_do_not_name_retired_rkllm_backend -q
```

Expected: collection fails because `ProjectError` is absent, and the scan finds
legacy descriptions and `RKLLMProjectError`.

- [ ] **Step 3: Rename only internal product naming**

Rename `RKLLMProjectError` to `ProjectError` in `src/rk_llm/errors.py` and make
`ConfigurationError`, `ArtifactError`, `BackendUnavailableError`, and
`NativeRunnerError` inherit from it. Update imports, `except` tuples, and
`type[...]` annotations in `cli.py`, `host/bootstrap.py`, and
`host/import_existing.py`.

Use these descriptions:

```toml
description = "Incremental pure-text RKNN3 deployment project"
```

```python
"""Incremental pure-text RKNN3 deployment package."""
```

Do not rename `rk-llm`, `rk_llm`, `RK_LLM`, CLI entry points, or environment
variables. Do not retain a legacy exception alias.

- [ ] **Step 4: Verify GREEN and commit**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest tests/unit/test_rknn3_probe.py tests/unit/test_repository_layout.py -q
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest -m "not hardware" -q
rg -n "RKLLMProjectError|Incremental pure-text RKLLM" pyproject.toml src tests
git diff --check
git add pyproject.toml src/rk_llm tests/unit/test_rknn3_probe.py tests/unit/test_repository_layout.py
git commit -m "refactor: remove retired RKLLM product naming"
```

Expected: tests pass; `rg` and `git diff --check` have no output.

### Final Verification

- [ ] Run the complete non-hardware suite and static checks:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .host-venv/bin/python -m pytest -m "not hardware" -q
git diff --check e39692d..HEAD
git status --short --branch
```

- [ ] Re-run final cross-module review against the updated HEAD.

- [ ] Confirm ignored `.vendor`, `.host-venv`, and imported artifact counts and
hashes are unchanged. Do not rerun bootstrap or import for this contract-only
change.
