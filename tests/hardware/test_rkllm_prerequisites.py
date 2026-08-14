"""Opt-in RKLLM prerequisite probe; this is not an inference smoke test."""

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.hardware


@pytest.mark.skipif(os.environ.get("RUN_RK_HARDWARE_TESTS") != "1", reason="RK hardware disabled")
def test_rkllm_prerequisite_paths_and_architecture_are_visible() -> None:
    from rk_llm.platform.probe import probe_rkllm

    capabilities = probe_rkllm(
        Path(os.environ["RKLLM_RUNNER"]), Path(os.environ["RKLLM_MODEL"])
    )
    assert capabilities.available, capabilities.reason
    pytest.xfail(
        "prerequisites are visible, but the native protocol, RKLLM Runtime/model "
        "loading, and inference are not implemented"
    )
