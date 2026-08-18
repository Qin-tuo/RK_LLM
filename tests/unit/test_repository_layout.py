import re
from pathlib import Path


def test_operational_skeleton_has_required_boundaries() -> None:
    required = (
        "configs/runtime/mock.yaml",
        "configs/runtime/rk3588.yaml",
        "configs/benchmark/smoke.yaml",
        "configs/models/qwen2_5_0_5b.yaml",
        "manifests/upstream.yaml",
        "native/rknn3_qwen_runner/CMakeLists.txt",
        "native/rknn3_qwen_runner/src/main.cpp",
        "tools/export/README.md",
        "tools/deploy/README.md",
        "tools/benchmark/README.md",
        "artifacts/README.md",
        "docs/architecture.md",
        "docs/host-setup.md",
        "docs/board-setup.md",
        "docs/model-export.md",
        "docs/benchmark.md",
    )
    assert [path for path in required if not Path(path).is_file()] == []
    assert not Path("third_party/versions.yaml").exists()


def test_documented_generation_commands_select_a_backend() -> None:
    documentation = (
        Path("README.md"),
        Path("Makefile"),
        *Path("docs").rglob("*.md"),
        *Path("tools").rglob("*.md"),
    )
    command_pattern = re.compile(r"rk-llm (?:generate|benchmark)[^&`]*")
    missing_backend = [
        f"{path}:{line_number}:{command.group(0).strip()}"
        for path in documentation
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        for command in command_pattern.finditer(line)
        if "--backend" not in command.group(0)
    ]
    assert missing_backend == []


def test_toolkit_install_uses_the_cloned_official_wheel_directory() -> None:
    requirements = Path("requirements/toolkit.txt").read_text(encoding="utf-8")
    host_setup = Path("docs/host-setup.md").read_text(encoding="utf-8")

    assert "--no-index" in requirements
    assert (
        "--find-links ./third_party/rknn-llm/rkllm-toolkit/packages"
        in requirements
    )
    assert "rkllm-toolkit==1.3.0" in requirements
    assert "https://github.com/airockchip/rknn-llm.git" in host_setup
    assert "--branch release-v1.3.0" in host_setup
    assert "third_party/rknn-llm" in host_setup


def test_cloned_upstream_toolkit_repository_is_ignored() -> None:
    ignore_patterns = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert "third_party/rknn-llm/" in ignore_patterns


def test_documented_toolkit_virtual_environment_is_ignored() -> None:
    ignore_patterns = Path(".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".toolkit-venv/" in ignore_patterns
