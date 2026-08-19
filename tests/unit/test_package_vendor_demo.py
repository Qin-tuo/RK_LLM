from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import rk_llm.host.package_vendor_demo as package_vendor_demo
from rk_llm.artifacts.manifest import compute_package_id, validate_package
from rk_llm.errors import ArtifactError, ConfigurationError
from rk_llm.host.package_vendor_demo import build_vendor_demo_package
from rk_llm.manifests.loader import (
    SizedFilePin,
    load_model_manifest,
    load_upstream_manifest,
)


DEMO_CONTENTS = {
    "SHA256SUMS": b"checksums\n",
    "rknn_qwen3_demo": b"demo-elf\n",
    "lib/librga.so": b"rga-elf\n",
    "lib/librknn3_api.so": b"api-elf\n",
    "lib/librknn3_api_rkcp.so": b"rkcp-elf\n",
    "model/Qwen3-4B.embed.bin": b"embed\n",
    "model/Qwen3-4B.rknn": b"rknn\n",
    "model/Qwen3-4B.tokenizer.gguf": b"tokenizer\n",
    "model/Qwen3-4B.weight": b"weight\n",
}


def _pin(path: str, content: bytes) -> SizedFilePin:
    return SizedFilePin(
        path=Path(path),
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _fixture(tmp_path: Path) -> tuple[Path, object, object, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    model = dataclasses.replace(
        load_model_manifest(Path("configs/models/qwen3_4b.yaml")),
        demo_files=tuple(_pin(path, content) for path, content in DEMO_CONTENTS.items()),
    )
    upstream = load_upstream_manifest(Path("manifests/upstream.yaml"))
    demo_root = (
        project_root
        / "artifacts/work/qwen3_4b/install/rknn_Qwen3_demo"
    )
    for relative, content in DEMO_CONTENTS.items():
        destination = demo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (demo_root / "rknn_qwen3_demo").chmod(0o755)
    return project_root, model, upstream, demo_root


class CommandRunner:
    def __init__(
        self,
        *,
        dirty: bool = False,
        machine: str = "AArch64",
        glibc: str = "2.35",
        glibcxx: str = "3.4.30",
    ) -> None:
        self.dirty = dirty
        self.machine = machine
        self.glibc = glibc
        self.glibcxx = glibcxx
        self.calls: list[list[str]] = []

    def __call__(
        self, args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[-3:] == ["status", "--porcelain", "--untracked-files=all"]:
            stdout = "?? dirty\n" if self.dirty else ""
        elif args[-2:] == ["rev-parse", "HEAD"]:
            stdout = f"{'a' * 40}\n"
        elif args[-1:] == ["--version"]:
            stdout = "aarch64-linux-gnu-g++ 11.4.0\n"
        elif "-h" in args:
            stdout = f"ELF Header:\n  Machine: {self.machine}\n"
        elif "--version-info" in args:
            stdout = (
                "Version needs section:\n"
                f"  Name: GLIBC_{self.glibc}\n"
                f"  Name: GLIBCXX_{self.glibcxx}\n"
                "  Name: GLIBC_2.17\n"
            )
        else:
            raise AssertionError(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def _build(tmp_path: Path, **runner_options: object) -> tuple[dict[str, object], Path]:
    project_root, model, upstream, _ = _fixture(tmp_path)
    summary = build_vendor_demo_package(
        project_root,
        model,
        upstream,
        run=CommandRunner(**runner_options),
    )
    return summary, Path(summary["package_path"])


def test_builder_maps_vendor_demo_and_records_verified_manifest(
    tmp_path: Path,
) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)
    runner = CommandRunner()

    summary = build_vendor_demo_package(project_root, model, upstream, run=runner)

    package_root = Path(summary["package_path"])
    manifest = validate_package(package_root)
    expected = {
        "bin/rknn_qwen3_demo",
        "lib/librga.so",
        "lib/librknn3_api.so",
        "lib/librknn3_api_rkcp.so",
        "model/Qwen3-4B.embed.bin",
        "model/Qwen3-4B.rknn",
        "model/Qwen3-4B.tokenizer.gguf",
        "model/Qwen3-4B.weight",
    }
    assert {record["path"] for record in manifest["files"]} == expected
    assert not (package_root / "SHA256SUMS").exists()
    assert os.access(package_root / "bin/rknn_qwen3_demo", os.X_OK)
    assert manifest["package_profile"] == "vendor_demo"
    assert manifest["entrypoint"] == "bin/rknn_qwen3_demo"
    assert manifest["target"]["compiler_platform"] == "rk1820"
    elf_records = [record for record in manifest["files"] if record["elf"]]
    assert len(elf_records) == 4
    assert all(
        record["elf"] == {"glibc": "2.35", "glibcxx": "3.4.30"}
        for record in elf_records
    )
    assert summary == {
        "package_id": manifest["package_id"],
        "model_id": "qwen3_4b",
        "status": "created",
        "package_path": str(package_root),
    }
    assert any(call[-1:] == ["--version"] for call in runner.calls)


@pytest.mark.parametrize(
    ("runner_options", "message"),
    [
        ({"machine": "Advanced Micro Devices X86-64"}, "AArch64"),
        ({"glibc": "2.38"}, "GLIBC"),
        ({"glibcxx": "3.4.31"}, "GLIBCXX"),
    ],
)
def test_builder_rejects_incompatible_elf(
    tmp_path: Path, runner_options: dict[str, str], message: str
) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)

    with pytest.raises(ArtifactError, match=message):
        build_vendor_demo_package(
            project_root,
            model,
            upstream,
            run=CommandRunner(**runner_options),
        )


def test_builder_rejects_dirty_project_before_copy(tmp_path: Path) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)

    with pytest.raises(ConfigurationError, match="uncommitted"):
        build_vendor_demo_package(
            project_root, model, upstream, run=CommandRunner(dirty=True)
        )

    assert not (project_root / "artifacts/packages").exists()


@pytest.mark.parametrize("mutation", ["changed", "symlink", "undeclared"])
def test_builder_rejects_untrusted_imported_demo(
    tmp_path: Path, mutation: str
) -> None:
    project_root, model, upstream, demo_root = _fixture(tmp_path)
    target = demo_root / "model/Qwen3-4B.rknn"
    if mutation == "changed":
        target.write_bytes(b"changed\n")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to("Qwen3-4B.weight")
    else:
        (demo_root / "undeclared.bin").write_bytes(b"unexpected\n")

    with pytest.raises(ArtifactError):
        build_vendor_demo_package(
            project_root, model, upstream, run=CommandRunner()
        )


def test_builder_reuses_identical_immutable_package(tmp_path: Path) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)

    first = build_vendor_demo_package(
        project_root, model, upstream, run=CommandRunner()
    )
    second = build_vendor_demo_package(
        project_root, model, upstream, run=CommandRunner()
    )

    assert first["status"] == "created"
    assert second == {**first, "status": "reused"}


def test_builder_validates_staging_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)
    events: list[str] = []
    real_validate = package_vendor_demo.validate_package
    real_publish = package_vendor_demo._publish_noreplace

    def validate(path: Path) -> dict[str, object]:
        events.append("validate")
        return real_validate(path)

    def publish(source: Path, destination: Path) -> None:
        events.append("publish")
        real_publish(source, destination)

    monkeypatch.setattr(package_vendor_demo, "validate_package", validate)
    monkeypatch.setattr(package_vendor_demo, "_publish_noreplace", publish)

    build_vendor_demo_package(
        project_root, model, upstream, run=CommandRunner()
    )

    assert events == ["validate", "publish"]


def test_builder_preserves_conflicting_existing_package(tmp_path: Path) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)
    first = build_vendor_demo_package(
        project_root, model, upstream, run=CommandRunner()
    )
    package_root = Path(first["package_path"])
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build"]["cmake_args"].append("--conflict")
    manifest["package_id"] = compute_package_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(ArtifactError, match="conflict"):
        build_vendor_demo_package(
            project_root, model, upstream, run=CommandRunner()
        )

    assert manifest_path.read_bytes() == before


def test_builder_preserves_concurrent_final_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)
    marker = b"concurrent\n"

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_bytes(marker)
        raise FileExistsError(destination)

    monkeypatch.setattr(package_vendor_demo, "_publish_noreplace", race)

    with pytest.raises(ArtifactError, match="appeared"):
        build_vendor_demo_package(
            project_root, model, upstream, run=CommandRunner()
        )

    package_parent = project_root / "artifacts/packages/qwen3_4b"
    final = next(path for path in package_parent.iterdir() if not path.name.startswith("."))
    assert (final / "owner.txt").read_bytes() == marker


def test_failure_cleanup_preserves_same_name_replacement_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, model, upstream, _ = _fixture(tmp_path)
    replacement = b"replacement\n"
    detached: list[Path] = []

    def fail_after_replacement(source: Path, destination: Path) -> None:
        moved = source.with_name(f"{source.name}.moved")
        source.rename(moved)
        detached.append(moved)
        source.mkdir()
        (source / "owner.txt").write_bytes(replacement)
        raise OSError("publish failed")

    monkeypatch.setattr(
        package_vendor_demo, "_publish_noreplace", fail_after_replacement
    )

    with pytest.raises(ArtifactError, match="publish"):
        build_vendor_demo_package(
            project_root, model, upstream, run=CommandRunner()
        )

    assert detached and not detached[0].exists()
    replacement_roots = list(
        (project_root / "artifacts/packages/qwen3_4b").glob(".*.staging.*")
    )
    assert len(replacement_roots) == 1
    assert (replacement_roots[0] / "owner.txt").read_bytes() == replacement
