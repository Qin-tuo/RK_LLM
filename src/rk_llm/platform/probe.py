"""RKLLM runtime prerequisite detection."""

import os
import platform
from pathlib import Path

from rk_llm.types import BackendCapabilities


def probe_rkllm(runner_path: Path, model_path: Path) -> BackendCapabilities:
    reasons: list[str] = []
    if not runner_path.is_file() or not os.access(runner_path, os.X_OK):
        reasons.append(f"native runner is not executable: {runner_path}")
    if not model_path.is_file():
        reasons.append(f"RKLLM model is missing: {model_path}")
    machine = platform.machine()
    if machine not in {"aarch64", "arm64"}:
        reasons.append(f"host architecture is {machine}, expected aarch64")
    return BackendCapabilities(
        name="rkllm",
        available=not reasons,
        streaming=True,
        target="rk",
        is_mock=False,
        reason="; ".join(reasons) if reasons else None,
    )
