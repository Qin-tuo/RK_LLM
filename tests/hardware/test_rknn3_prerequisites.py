"""Opt-in RKNN3 prerequisite probe; this is not an inference smoke test."""

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.hardware


@pytest.mark.skipif(os.environ.get("RUN_RK_HARDWARE_TESTS") != "1", reason="RK hardware disabled")
def test_rknn3_prerequisite_paths_and_architecture_are_visible() -> None:
    from rk_llm.platform.probe import probe_rknn3

    capabilities = probe_rknn3(Path(os.environ["RKNN3_PACKAGE"]))
    assert capabilities.available, capabilities.reason
    pytest.xfail(
        "prerequisites are visible, but the native protocol, RKNN3 Runtime/model "
        "loading, and inference are not implemented"
    )
