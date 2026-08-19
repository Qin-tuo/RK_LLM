import os
import subprocess
import sys
from pathlib import Path


def test_fake_prerequisites_cannot_report_hardware_inference_passed(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "package"
    runner = package_path / "bin/rknn_qwen_runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    (package_path / "manifest.json").write_text("{}", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        RUN_RK_HARDWARE_TESTS="1",
        RKNN3_PACKAGE=str(package_path),
    )
    command = (
        "import platform, pytest; "
        'platform.machine = lambda: "aarch64"; '
        "raise SystemExit(pytest.main("
        '["tests/hardware/test_rknn3_prerequisites.py", "-q"]'
        "))"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 xfailed" in result.stdout
    assert "passed" not in result.stdout
