import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.hardware


@pytest.mark.skipif(os.environ.get("RUN_RK_HARDWARE_TESTS") != "1", reason="RK hardware disabled")
def test_rkllm_runtime_and_model_are_discoverable() -> None:
    from rk_llm.platform.probe import probe_rkllm

    capabilities = probe_rkllm(
        Path(os.environ["RKLLM_RUNNER"]), Path(os.environ["RKLLM_MODEL"])
    )
    assert capabilities.available, capabilities.reason
