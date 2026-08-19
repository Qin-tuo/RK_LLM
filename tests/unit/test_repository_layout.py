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
        "src/rk_llm/host/package_vendor_demo.py",
        "tools/host/bootstrap",
        "tools/host/import-existing",
        "tools/host/package-vendor-demo",
        "artifacts/README.md",
    )

    assert [path for path in required if not Path(path).is_file()] == []


def test_project_metadata_and_python_sources_do_not_use_retired_product_name() -> None:
    sources = (Path("pyproject.toml"), *Path("src/rk_llm").rglob("*.py"))
    findings = [
        str(path)
        for path in sources
        if "RKLLM" in path.read_text(encoding="utf-8")
    ]

    assert findings == []


def test_rknn3_runner_name_matches_the_deployment_package_contract() -> None:
    cmake = Path("native/rknn3_qwen_runner/CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    probe = Path("src/rk_llm/platform/probe.py").read_text(encoding="utf-8")

    assert "add_executable(rknn_qwen_runner src/main.cpp)" in cmake
    assert 'package_path / "bin/rknn_qwen_runner"' in probe
    assert "add_executable(rknn3_qwen_runner" not in cmake


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
        "is not verified",
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


def test_qwen3_package_transfer_workflow_uses_project_artifact_boundaries() -> None:
    paths = (
        Path("README.md"),
        Path("artifacts/README.md"),
        Path("docs/host-setup.md"),
        Path("docs/board-setup.md"),
        Path("docs/rk1828-qwen3-4b-quick-deployment.md"),
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    required = (
        "qwen3_4b",
        "make host-import",
        "make host-package",
        "package-validate",
        "artifacts/work/qwen3_4b/install/rknn_Qwen3_demo",
        "artifacts/packages/qwen3_4b/<package_id>",
        "artifacts/deploy/releases/$PACKAGE_ID",
        "artifacts/deploy/current",
        "rsync -a --protect-args",
        "ln -s 'releases/$PACKAGE_ID'",
        "bin/rknn_qwen3_demo",
        "model/Qwen3-4B.rknn",
        "model/Qwen3-4B.weight",
        "model/Qwen3-4B.tokenizer.gguf",
        "model/Qwen3-4B.embed.bin",
        "0xff",
        "你好，请用三句话介绍你自己。",
    )
    stale = (
        "Those targets are intentionally absent today.",
        "creates a board package, or transfers it",
        "will be transferred as an immutable deployment package by a later",
    )

    assert all(token in combined for token in required)
    assert all(token not in combined for token in stale)


def test_qwen3_quick_deployment_transfers_one_validated_package() -> None:
    quickstart = Path(
        "docs/rk1828-qwen3-4b-quick-deployment.md"
    ).read_text(encoding="utf-8")

    assert 'make host-import MODEL=qwen3_4b' in quickstart
    assert 'make host-package MODEL=qwen3_4b' in quickstart
    assert 'REMOTE_INCOMING="$REMOTE_PROJECT/artifacts/deploy/.incoming-$PACKAGE_ID"' in quickstart
    assert 'REMOTE_RELEASE="$REMOTE_PROJECT/artifacts/deploy/releases/$PACKAGE_ID"' in quickstart
    assert '"$PACKAGE_DIR/" "$RK3588_HOST:$REMOTE_INCOMING/"' in quickstart
    assert 'mv \'$REMOTE_INCOMING\' \'$REMOTE_RELEASE\'' in quickstart
    assert 'scp -r "$DEMO_DIR"' not in quickstart


def test_makefile_exposes_implemented_host_targets() -> None:
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
        '"$(HOST_PYTHON)" -m rk_llm.host.package_vendor_demo',
        '--workspace "$(WORKSPACE)"',
        '--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml"',
        "--mode copy",
        '--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml"',
        "--readelf aarch64-linux-gnu-readelf",
    )

    assert all(line in makefile for line in required_lines)
    assert all(f"{target}:" in makefile for target in ("install", "test", "smoke"))
    assert ".PHONY:" in makefile
    assert all(
        target in makefile
        for target in ("host-env", "host-bootstrap", "host-import", "host-package")
    )
    assert all(
        f"{target}:" not in makefile
        for target in ("host-build", "host-runner", "deploy")
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
