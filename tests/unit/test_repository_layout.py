import re
from pathlib import Path


def test_operational_skeleton_has_required_boundaries() -> None:
    required = (
        "configs/runtime/mock.yaml",
        "configs/runtime/rk3588.yaml",
        "configs/benchmark/smoke.yaml",
        "third_party/versions.yaml",
        "native/rkllm_runner/CMakeLists.txt",
        "native/rkllm_runner/src/main.cpp",
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
