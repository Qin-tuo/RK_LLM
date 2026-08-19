from pathlib import Path

import pytest

from rk_llm.config import RuntimeConfig, load_benchmark_config, load_runtime_config
from rk_llm.errors import ConfigurationError


def test_load_runtime_config_resolves_package_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "backend: rknn3\n"
        "target: rk3588\n"
        "package_path: ../../artifacts/deploy/current\n"
        "max_new_tokens: 64\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.backend == "rknn3"
    assert config.package_path == (tmp_path / "../../artifacts/deploy/current").resolve()
    assert config.max_new_tokens == 64


def test_rknn3_runtime_requires_package_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("backend: rknn3\ntarget: rk3588\n", encoding="utf-8")
    with pytest.raises(ValueError, match="package_path is required for rknn3"):
        load_runtime_config(config_path)


def test_runtime_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backend: gpu\ntarget: host\n", encoding="utf-8")
    with pytest.raises(ValueError, match="backend"):
        load_runtime_config(path)


def test_malformed_yaml_is_reported_as_configuration_error_with_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("backend: [mock\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=str(path)):
        load_runtime_config(path)


def test_benchmark_config_requires_prompts(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text("iterations: 1\nprompts: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts"):
        load_benchmark_config(path)


@pytest.mark.parametrize(
    "yaml_prompts",
    ["hello", "{one: two}", "[hello, null]", "[hello, true]", "[hello, 7]"],
)
def test_benchmark_prompts_must_be_a_list_of_strings(
    tmp_path: Path, yaml_prompts: str
) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text(f"iterations: 1\nprompts: {yaml_prompts}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompts"):
        load_benchmark_config(path)


@pytest.mark.parametrize("yaml_iterations", ["1.9", "true"])
def test_benchmark_iterations_must_be_an_integer(
    tmp_path: Path, yaml_iterations: str
) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        f"iterations: {yaml_iterations}\nprompts: [hello]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="iterations"):
        load_benchmark_config(path)


@pytest.mark.parametrize(
    ("runtime_yaml", "field"),
    [
        ("backend: true\ntarget: host\n", "backend"),
        ("backend: mock\ntarget: []\n", "target"),
        ("backend: rknn3\ntarget: rk3588\npackage_path: 12\n", "package_path"),
        ("backend: mock\ntarget: host\npackage_path: '   '\n", "package_path"),
        ("backend: mock\ntarget: host\nmax_new_tokens: true\n", "max_new_tokens"),
        ("backend: mock\ntarget: host\ntop_k: 1.9\n", "top_k"),
        ("backend: mock\ntarget: host\ntemperature: true\n", "temperature"),
        ("backend: mock\ntarget: host\ntop_p: true\n", "top_p"),
        ("backend: mock\ntarget: host\nrepeat_penalty: true\n", "repeat_penalty"),
    ],
)
def test_runtime_config_rejects_malformed_field_types(
    tmp_path: Path, runtime_yaml: str, field: str
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(runtime_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_runtime_config(path)


def test_runtime_config_accepts_integer_and_float_sampling_values(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        "backend: mock\n"
        "target: host\n"
        "package_path: null\n"
        "max_new_tokens: 64\n"
        "temperature: 1\n"
        "top_p: 0.75\n"
        "top_k: 4\n"
        "repeat_penalty: 2\n",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.package_path is None
    assert config.temperature == 1.0
    assert config.top_p == 0.75
    assert config.repeat_penalty == 2.0


@pytest.mark.parametrize(
    ("backend", "target"),
    [
        ("mock", ""),
        ("mock", "s100"),
        ("mock", "arbitrary"),
        ("mock", "rk3588"),
        ("rknn3", "host"),
    ],
)
def test_runtime_config_enforces_rk_backend_target_pairs(
    tmp_path: Path, backend: str, target: str
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        f"backend: {backend}\ntarget: '{target}'\npackage_path: package\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("backend", "target"),
    [
        ("mock", ""),
        ("mock", "s100"),
        ("mock", "arbitrary"),
        ("mock", "rk3588"),
        ("rknn3", "host"),
    ],
)
def test_runtime_config_type_enforces_backend_target_pairs(
    backend: str, target: str
) -> None:
    with pytest.raises(ValueError, match="target"):
        RuntimeConfig(backend, target, Path("package"), 128, 0.8, 0.9, 1, 1.1)


def test_benchmark_nested_runtime_enforces_rk_target(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        "prompts: [hello]\n"
        "iterations: 1\n"
        "runtime:\n"
        "  backend: mock\n"
        "  target: s100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target"):
        load_benchmark_config(path)


@pytest.mark.parametrize("field", ["temperature", "top_p", "repeat_penalty"])
@pytest.mark.parametrize("yaml_value", [".nan", ".inf", "-.inf"])
def test_runtime_config_rejects_non_finite_sampling_values(
    tmp_path: Path, field: str, yaml_value: str
) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        f"backend: mock\ntarget: host\n{field}: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_runtime_config(path)
