"""RKNN3 runtime prerequisite detection."""

import os
import platform
from pathlib import Path

from rk_llm.types import BackendCapabilities


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
