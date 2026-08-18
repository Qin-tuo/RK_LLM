from pathlib import Path


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


def test_local_vendor_and_artifact_roots_are_ignored() -> None:
    ignore_patterns = set(
        Path(".gitignore").read_text(encoding="utf-8").splitlines()
    )
    required = {
        ".vendor/",
        ".host-venv/",
        "artifacts/source_models/",
        "artifacts/work/",
        "artifacts/packages/",
        "artifacts/deploy/",
        "artifacts/logs/",
    }

    assert required <= ignore_patterns


def test_product_docs_do_not_advertise_the_legacy_rkllm_path() -> None:
    documentation = (
        Path("README.md"),
        *(
            path
            for path in Path("docs").glob("*.md")
            if path.name != "rk1828-rknn3-deployment.md"
        ),
        *Path("tools").rglob("*.md"),
    )
    stale_tokens = ("DeepSeek-R1", "RKLLM", "--backend rkllm", ".rkllm")
    findings = [
        f"{path}:{token}"
        for path in documentation
        for token in stale_tokens
        if token in path.read_text(encoding="utf-8")
    ]

    assert findings == []


def test_manual_evidence_status_is_scoped_to_completed_steps() -> None:
    readme = " ".join(
        Path("README.md").read_text(encoding="utf-8").lower().split()
    )
    required_evidence = (
        "source export",
        "grq",
        "rknn compilation",
        "aarch64 cross-build",
        "ubuntu 22.04 abi ceiling",
        "incremental package transfer",
        "first rk3588-to-rk1828 board inference",
        "has not started and is not verified",
    )
    product_docs = (
        Path("README.md"),
        Path("docs/architecture.md"),
        Path("docs/board-setup.md"),
        Path("docs/implementation-roadmap.md"),
    )
    overclaims = (
        "flow already exercised",
        "already exercised external workflow",
        "previously exercised external",
    )

    assert all(term in readme for term in required_evidence)
    assert [
        f"{path}:{claim}"
        for path in product_docs
        for claim in overclaims
        if claim in path.read_text(encoding="utf-8").lower()
    ] == []


def test_makefile_exposes_only_implemented_host_foundation_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    required_lines = (
        "PROJECT_ROOT := $(abspath .)",
        "HOST_VENV ?= $(PROJECT_ROOT)/.host-venv",
        "HOST_PYTHON := $(HOST_VENV)/bin/python",
        "MODEL ?= qwen2_5_0_5b",
        "WORKSPACE ?= /home/barry/rk1828-work",
        "RKNN3_RUNTIME_DEV_ROOT ?= $(WORKSPACE)/rknn3-model-zoo/3rdparty/rknpu3",
        'python3 -m venv "$(HOST_VENV)"',
        '"$(HOST_PYTHON)" -m pip install -e ".[dev]"',
        '"$(HOST_PYTHON)" -m rk_llm.host.bootstrap',
        '--project-root "$(PROJECT_ROOT)"',
        '--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml"',
        '--runtime-dev-root "$(RKNN3_RUNTIME_DEV_ROOT)"',
        '--seed-workspace "$(WORKSPACE)"',
        '"$(HOST_PYTHON)" -m rk_llm.host.import_existing',
        '--workspace "$(WORKSPACE)"',
        '--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml"',
        "--mode copy",
    )

    assert all(line in makefile for line in required_lines)
    assert all(f"{target}:" in makefile for target in ("install", "test", "smoke"))
    assert ".PHONY:" in makefile
    assert all(
        target in makefile
        for target in ("host-env", "host-bootstrap", "host-import")
    )
    assert all(
        f"{target}:" not in makefile
        for target in ("host-build", "host-runner", "host-package", "deploy")
    )


def test_vendor_requirements_point_to_the_pinned_manifest() -> None:
    for path in (Path("requirements/toolkit.txt"), Path("requirements/board.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        active_requirements = [
            line for line in lines if line.strip() and not line.lstrip().startswith("#")
        ]

        assert active_requirements == []
        assert "manifests/upstream.yaml" in "\n".join(lines)
        assert "bootstrap" in "\n".join(lines).lower()

    board_requirements = Path("requirements/board.txt").read_text(encoding="utf-8")
    assert "sizes" not in board_requirements


def test_host_docs_distinguish_bootstrap_and_import_validation() -> None:
    host_setup = Path("docs/host-setup.md").read_text(encoding="utf-8")
    normalized_host_setup = " ".join(host_setup.split())

    assert "additional files are allowed" in normalized_host_setup
    assert "exact regular-file set" in normalized_host_setup
    assert (
        "does not validate the source model Git revision" in normalized_host_setup
    )
    assert "Import records the model revision" not in normalized_host_setup
    assert (
        "model manifest records its repository and revision"
        in normalized_host_setup
    )
    assert "import record does not include either value" in normalized_host_setup
