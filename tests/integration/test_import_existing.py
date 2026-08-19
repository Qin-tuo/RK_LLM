import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_fixture(tmp_path: Path, sha256: str | None = None) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    manifest = project / "configs/models/demo.yaml"
    source = workspace / "models/demo/model.bin"
    generated = workspace / "model-zoo/demo/output.rknn"
    demo = workspace / "model-zoo/install/rknn_Demo/demo"
    runtime = workspace / "model-zoo/install/rknn_Demo/lib/runtime.so"
    source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    generated.write_bytes(b"output")
    demo.write_bytes(b"executable")
    runtime.write_bytes(b"runtime")
    source_sha = sha256 or hashlib.sha256(b"source").hexdigest()
    generated_sha = hashlib.sha256(b"output").hexdigest()
    demo_sha = hashlib.sha256(b"executable").hexdigest()
    runtime_sha = hashlib.sha256(b"runtime").hexdigest()
    manifest.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model_id: demo",
                "repository: example/demo",
                f"revision: {'a' * 40}",
                "platform: rk1820",
                "source_root: models/demo",
                "generated_root: model-zoo/demo",
                "demo_root: model-zoo/install/rknn_Demo",
                "demo_name: rknn_Demo",
                "source_files:",
                f"  - {{path: model.bin, size: 6, sha256: {source_sha}}}",
                "generated_files:",
                f"  - {{path: output.rknn, size: 6, sha256: {generated_sha}}}",
                "demo_files:",
                f"  - {{path: demo, size: 10, sha256: {demo_sha}}}",
                f"  - {{path: lib/runtime.so, size: 7, sha256: {runtime_sha}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workspace, project, manifest


def _command(workspace: Path, project: Path, manifest: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rk_llm.host.import_existing",
        "--project-root",
        str(project),
        "--workspace",
        str(workspace),
        "--model-manifest",
        str(manifest),
        "--mode",
        "copy",
    ]


def test_module_cli_imports_tiny_workspace_and_prints_one_json_summary(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _write_fixture(tmp_path)

    result = subprocess.run(
        _command(workspace, project, manifest),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    record = json.loads(result.stdout)
    assert record["model_id"] == "demo"
    assert record["mode"] == "copy"
    assert record["statuses"] == {
        "source": "imported",
        "generated": "imported",
        "demo": "imported",
    }
    assert len(record["files"]) == 4
    assert (project / "artifacts/source_models/demo/model.bin").read_bytes() == b"source"
    assert (project / "artifacts/work/demo/model/output.rknn").read_bytes() == b"output"
    assert (
        project / "artifacts/work/demo/install/rknn_Demo/demo"
    ).read_bytes() == b"executable"
    assert (
        project / "artifacts/work/demo/install/rknn_Demo/lib/runtime.so"
    ).read_bytes() == b"runtime"


def test_module_cli_reports_artifact_errors_on_stderr_with_exit_two(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _write_fixture(tmp_path, sha256="f" * 64)

    result = subprocess.run(
        _command(workspace, project, manifest),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "expected sha256" in result.stderr
    assert not (project / "artifacts/source_models/demo").exists()


def test_executable_wrapper_invokes_import_module_help(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RK_LLM_PYTHON"] = sys.executable

    result = subprocess.run(
        [str(Path("tools/host/import-existing").resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--workspace" in result.stdout
    assert "--model-manifest" in result.stdout
    assert result.stderr == ""
