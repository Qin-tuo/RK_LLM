from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import pytest

import rk_llm.artifacts.manifest as artifact_manifest
from rk_llm.artifacts.manifest import (
    _safe_relative_path,
    _schema_path,
    canonical_payload,
    compute_package_id,
    validate_package,
)
from rk_llm.errors import ArtifactError


RUNNER = b"rknn-qwen-runner\n"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _valid_manifest(file_path: str = "bin/rknn_qwen_runner") -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_id": "0" * 16,
        "created_at": "2026-08-18T00:00:00Z",
        "model": {
            "id": "qwen2_5_0_5b",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": MODEL_REVISION,
        },
        "toolchain": {
            "project_commit": "a" * 40,
            "toolkit": {
                "release": "V1.0.4",
                "revision": "cf292045d77c9ad0377b9fb326f216967475071e",
            },
            "model_zoo": {
                "release": "V1.0.4",
                "revision": "f63048265b49bd2c6236790087287bed6c6b76fe",
            },
            "runtime_version": "1.0.4",
            "firmware_version": "1.0.4",
            "builder": {
                "image": "ubuntu:22.04",
                "compiler": "aarch64-linux-gnu-g++ 11.4.0",
            },
        },
        "target": {
            "host_soc": "rk3588",
            "accelerator": "rk1828",
            "architecture": "aarch64",
            "glibc_max": "2.35",
            "glibcxx_max": "3.4.30",
        },
        "build": {
            "export_args": [],
            "rknn_args": ["--platform", "rk1820"],
            "cmake_args": [],
        },
        "files": [
            {
                "path": file_path,
                "size": len(RUNNER),
                "sha256": hashlib.sha256(RUNNER).hexdigest(),
                "elf": None,
            }
        ],
    }


def _write_manifest(package_root: Path, manifest: dict[str, object]) -> None:
    manifest["package_id"] = compute_package_id(manifest)
    (package_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _make_package(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    package_root = tmp_path / "package"
    runner = package_root / "bin" / "rknn_qwen_runner"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(RUNNER)
    manifest = _valid_manifest()
    _write_manifest(package_root, manifest)
    return package_root, manifest


def test_package_id_ignores_package_id_and_created_at() -> None:
    manifest = _valid_manifest()
    original_payload = canonical_payload(manifest)
    original_id = compute_package_id(manifest)
    manifest["package_id"] = "f" * 16
    manifest["created_at"] = "2030-01-01T12:34:56Z"

    assert canonical_payload(manifest) == original_payload
    assert compute_package_id(manifest) == original_id
    assert original_id == hashlib.sha256(original_payload).hexdigest()[:16]


def test_validate_package_returns_valid_manifest(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)

    assert validate_package(package_root) == manifest


def test_validate_package_rejects_out_of_range_rfc3339_offsets(
    tmp_path: Path,
) -> None:
    package_root, manifest = _make_package(tmp_path)

    for created_at in (
        "2026-08-18T00:00:00+00:60",
        "2026-08-18T00:00:00+24:00",
    ):
        manifest["created_at"] = created_at
        _write_manifest(package_root, manifest)

        with pytest.raises(ArtifactError, match="invalid deployment manifest"):
            validate_package(package_root)


def test_validate_package_rejects_non_ascii_rfc3339_digits(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    manifest["created_at"] = "２０２６-08-18T00:00:00Z"
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="invalid deployment manifest"):
        validate_package(package_root)


def test_validate_package_enforces_rfc3339_leap_second_position(
    tmp_path: Path,
) -> None:
    package_root, manifest = _make_package(tmp_path)

    for created_at in (
        "2016-12-31t23:59:60z",
        "2017-01-01T00:59:60+01:00",
    ):
        manifest["created_at"] = created_at
        _write_manifest(package_root, manifest)

        assert validate_package(package_root) == manifest

    manifest["created_at"] = "2016-12-31T23:59:60+01:00"
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="invalid deployment manifest"):
        validate_package(package_root)


def test_validate_package_rejects_changed_package_id(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    manifest["package_id"] = "f" * 16
    (package_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="package_id"):
        validate_package(package_root)


def test_validate_package_rejects_parent_traversal(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    file_record = manifest["files"][0]
    assert isinstance(file_record, dict)
    file_record["path"] = "../outside"
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="safe relative path"):
        validate_package(package_root)


@pytest.mark.parametrize("path", ["etc/config", "share/data", "runner"])
def test_safe_relative_path_allows_only_package_directories(path: str) -> None:
    with pytest.raises(ArtifactError, match="safe relative path"):
        _safe_relative_path(path)


@pytest.mark.parametrize("path", ["bin/tool", "lib/runtime.so", "model/qwen.rknn"])
def test_safe_relative_path_accepts_allowed_package_directories(path: str) -> None:
    assert _safe_relative_path(path) == Path(path)


@pytest.mark.parametrize(
    "path",
    ["", "/bin/tool", "bin/../outside", "bin", "bin/", "lib", "model", 42],
)
def test_safe_relative_path_rejects_unsafe_values(path: object) -> None:
    with pytest.raises(ArtifactError, match="safe relative path"):
        _safe_relative_path(path)


def test_schema_path_rejects_relative_project_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RK_LLM_ROOT", "relative/root")

    with pytest.raises(ArtifactError, match="absolute"):
        _schema_path()


def test_validate_package_rejects_undeclared_file(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    (package_root / "lib").mkdir()
    (package_root / "lib" / "undeclared.so").write_bytes(b"runtime")

    with pytest.raises(ArtifactError, match="undeclared"):
        validate_package(package_root)


def test_validate_package_rejects_undeclared_file_symlink(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    (package_root / "lib").mkdir()
    target = tmp_path / "undeclared.so"
    target.write_bytes(b"runtime")
    (package_root / "lib" / "undeclared.so").symlink_to(target)

    with pytest.raises(ArtifactError, match="undeclared"):
        validate_package(package_root)


def test_validate_package_rejects_manifest_symlink(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    manifest_path = package_root / "manifest.json"
    outside_manifest = tmp_path / "manifest.json"
    manifest_path.replace(outside_manifest)
    manifest_path.symlink_to(outside_manifest)

    with pytest.raises(ArtifactError, match="symlink"):
        validate_package(package_root)


def test_validate_package_rejects_broken_file_symlink(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    (package_root / "lib").mkdir()
    (package_root / "lib" / "broken.so").symlink_to(tmp_path / "missing.so")

    with pytest.raises(ArtifactError, match="symlink"):
        validate_package(package_root)


def test_validate_package_rejects_directory_symlink(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    target = tmp_path / "runtime"
    target.mkdir()
    (package_root / "lib").mkdir()
    (package_root / "lib" / "runtime").symlink_to(
        target, target_is_directory=True
    )

    with pytest.raises(ArtifactError, match="symlink"):
        validate_package(package_root)


def test_validate_package_rejects_declared_file_through_symlink_parent(
    tmp_path: Path,
) -> None:
    package_root, manifest = _make_package(tmp_path)
    runner = package_root / "bin" / "rknn_qwen_runner"
    model_runner = package_root / "model" / "rknn_qwen_runner"
    model_runner.parent.mkdir()
    runner.replace(model_runner)
    runner.parent.rmdir()
    runner.parent.symlink_to(model_runner.parent, target_is_directory=True)
    file_record = manifest["files"][0]
    assert isinstance(file_record, dict)
    model_record = copy.deepcopy(file_record)
    model_record["path"] = "model/rknn_qwen_runner"
    manifest["files"] = [file_record, model_record]
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="symlink"):
        validate_package(package_root)


def test_validate_package_allows_package_root_symlink(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    active_package = tmp_path / "current"
    active_package.symlink_to(package_root, target_is_directory=True)

    assert validate_package(active_package) == manifest


def test_validate_package_rejects_missing_file(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    (package_root / "bin" / "rknn_qwen_runner").unlink()

    with pytest.raises(ArtifactError, match="regular file"):
        validate_package(package_root)


def test_validate_package_rejects_symlink(tmp_path: Path) -> None:
    package_root, _ = _make_package(tmp_path)
    runner = package_root / "bin" / "rknn_qwen_runner"
    outside = tmp_path / "runner"
    outside.write_bytes(RUNNER)
    runner.unlink()
    runner.symlink_to(outside)

    with pytest.raises(ArtifactError, match="symlink"):
        validate_package(package_root)


def test_validate_package_rejects_size_mismatch(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    file_record = manifest["files"][0]
    assert isinstance(file_record, dict)
    file_record["size"] = len(RUNNER) + 1
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="size"):
        validate_package(package_root)


def test_validate_package_rejects_hash_mismatch(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    file_record = manifest["files"][0]
    assert isinstance(file_record, dict)
    file_record["sha256"] = "b" * 64
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="sha256"):
        validate_package(package_root)


def test_validate_package_wraps_descendant_lstat_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _make_package(tmp_path)
    runner = package_root / "bin" / "rknn_qwen_runner"
    original_lstat = Path.lstat

    def failing_lstat(path: Path, *args: object, **kwargs: object) -> object:
        if path == runner:
            raise OSError("lstat failed")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", failing_lstat)

    with pytest.raises(ArtifactError, match=r"rknn_qwen_runner.*lstat failed"):
        validate_package(package_root)


def test_validate_package_wraps_descendant_scandir_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _make_package(tmp_path)
    hidden_directory = package_root / "model" / "hidden"
    hidden_directory.mkdir(parents=True)
    (hidden_directory / "undeclared.rknn").write_bytes(b"hidden")
    original_scandir = os.scandir

    def failing_scandir(path: Path) -> os.ScandirIterator[str]:
        if Path(path) == hidden_directory:
            raise PermissionError("scan denied")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", failing_scandir)

    with pytest.raises(ArtifactError, match=r"model/hidden.*scan denied"):
        validate_package(package_root)


def test_validate_package_wraps_hash_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _ = _make_package(tmp_path)

    def failing_sha256(path: Path) -> str:
        raise OSError("hash read failed")

    monkeypatch.setattr(artifact_manifest, "_sha256", failing_sha256)

    with pytest.raises(ArtifactError, match=r"rknn_qwen_runner.*hash read failed"):
        validate_package(package_root)


def test_validate_package_rejects_duplicate_paths(tmp_path: Path) -> None:
    package_root, manifest = _make_package(tmp_path)
    file_record = manifest["files"][0]
    assert isinstance(file_record, dict)
    manifest["files"] = [file_record, copy.deepcopy(file_record)]
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="duplicate"):
        validate_package(package_root)


def _set_schema_version(manifest: dict[str, object]) -> None:
    manifest["schema_version"] = 2


def _set_invalid_created_at(manifest: dict[str, object]) -> None:
    manifest["created_at"] = "2026-08-18"


def _add_top_level_property(manifest: dict[str, object]) -> None:
    manifest["unexpected"] = True


def _set_wrong_model_id(manifest: dict[str, object]) -> None:
    manifest["model"]["id"] = "other"


def _set_wrong_model_repository(manifest: dict[str, object]) -> None:
    manifest["model"]["repository"] = "other/model"


def _set_wrong_toolkit_release(manifest: dict[str, object]) -> None:
    manifest["toolchain"]["toolkit"]["release"] = "V2.0.0"


def _set_wrong_model_zoo_revision(manifest: dict[str, object]) -> None:
    manifest["toolchain"]["model_zoo"]["revision"] = "b" * 40


def _set_wrong_runtime_version(manifest: dict[str, object]) -> None:
    manifest["toolchain"]["runtime_version"] = "2.0.0"


def _set_wrong_firmware_version(manifest: dict[str, object]) -> None:
    manifest["toolchain"]["firmware_version"] = "2.0.0"


def _set_wrong_builder_image(manifest: dict[str, object]) -> None:
    manifest["toolchain"]["builder"]["image"] = "ubuntu:24.04"


def _set_wrong_target(manifest: dict[str, object]) -> None:
    manifest["target"]["accelerator"] = "rk9999"


@pytest.mark.parametrize(
    "mutate",
    [
        _set_schema_version,
        _set_invalid_created_at,
        _add_top_level_property,
        _set_wrong_model_id,
        _set_wrong_model_repository,
        _set_wrong_toolkit_release,
        _set_wrong_model_zoo_revision,
        _set_wrong_runtime_version,
        _set_wrong_firmware_version,
        _set_wrong_builder_image,
        _set_wrong_target,
    ],
    ids=lambda mutate: mutate.__name__,
)
def test_validate_package_rejects_schema_violations(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    package_root, manifest = _make_package(tmp_path)
    mutate(manifest)
    _write_manifest(package_root, manifest)

    with pytest.raises(ArtifactError, match="invalid deployment manifest"):
        validate_package(package_root)
