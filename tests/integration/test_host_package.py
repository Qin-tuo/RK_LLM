import json
import os
import subprocess
import sys
from pathlib import Path


def test_package_vendor_demo_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rk_llm.host.package_vendor_demo", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--project-root" in result.stdout
    assert "--model-manifest" in result.stdout
    assert "--upstream-manifest" in result.stdout
    assert "--readelf" in result.stdout
    assert result.stderr == ""


def test_package_vendor_demo_wrapper_invokes_module_help(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RK_LLM_PYTHON"] = sys.executable

    result = subprocess.run(
        [str(Path("tools/host/package-vendor-demo").resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--project-root" in result.stdout
    assert result.stderr == ""


def test_module_cli_prints_one_json_summary(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    import rk_llm.host.package_vendor_demo as module

    summary = {
        "package_id": "0123456789abcdef",
        "model_id": "qwen3_4b",
        "status": "created",
        "package_path": str(tmp_path / "package"),
    }
    monkeypatch.setattr(module, "load_model_manifest", lambda path: object())
    monkeypatch.setattr(module, "load_upstream_manifest", lambda path: object())
    monkeypatch.setattr(
        module, "build_vendor_demo_package", lambda *args, **kwargs: summary
    )

    result = module.main(
        [
            "--project-root",
            str(tmp_path),
            "--model-manifest",
            str(tmp_path / "model.yaml"),
            "--upstream-manifest",
            str(tmp_path / "upstream.yaml"),
            "--readelf",
            "readelf",
        ]
    )

    assert result == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.count("\n") == 1
    assert json.loads(output.out) == summary
