import errno
import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import rk_llm.host.import_existing as importer
from rk_llm.errors import ArtifactError, ConfigurationError
from rk_llm.host.import_existing import import_existing
from rk_llm.manifests.loader import ModelManifest, SizedFilePin


def _pin(path: str, content: bytes) -> SizedFilePin:
    return SizedFilePin(
        path=Path(path),
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, ModelManifest]:
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    source = workspace / "models/demo/model.bin"
    generated = workspace / "model-zoo/demo/output.rknn"
    source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    project.mkdir()
    source.write_bytes(b"source")
    generated.write_bytes(b"output")
    manifest = ModelManifest(
        model_id="demo",
        repository="example/demo",
        revision="a" * 40,
        platform="rk1820",
        source_root=Path("models/demo"),
        generated_root=Path("model-zoo/demo"),
        source_files=(_pin("model.bin", b"source"),),
        generated_files=(_pin("output.rknn", b"output"),),
    )
    return workspace, project, manifest


def _fixture_with_demo(tmp_path: Path) -> tuple[Path, Path, ModelManifest]:
    workspace, project, manifest = _fixture(tmp_path)
    demo_root = workspace / "model-zoo/install/rknn_Demo"
    executable = demo_root / "demo"
    runtime = demo_root / "lib/runtime.so"
    runtime.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    runtime.write_bytes(b"runtime")
    return workspace, project, replace(
        manifest,
        demo_root=Path("model-zoo/install/rknn_Demo"),
        demo_name="rknn_Demo",
        demo_files=(
            _pin("demo", b"executable"),
            _pin("lib/runtime.so", b"runtime"),
        ),
    )


def _source_hashes(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


def test_copy_imports_both_categories_and_writes_deterministic_record(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    before = _source_hashes(workspace)

    record = import_existing(workspace, project, manifest)

    source = workspace / "models/demo/model.bin"
    generated = workspace / "model-zoo/demo/output.rknn"
    imported_source = project / "artifacts/source_models/demo/model.bin"
    imported_generated = project / "artifacts/work/demo/model/output.rknn"
    assert imported_source.read_bytes() == b"source"
    assert imported_generated.read_bytes() == b"output"
    assert imported_source.stat().st_ino != source.stat().st_ino
    assert imported_generated.stat().st_ino != generated.stat().st_ino
    assert _source_hashes(workspace) == before
    assert record == {
        "schema_version": 1,
        "model_id": "demo",
        "source_workspace": str(workspace),
        "mode": "copy",
        "statuses": {"source": "imported", "generated": "imported"},
        "files": [
            {
                "category": "source",
                "path": "model.bin",
                "size": 6,
                "sha256": hashlib.sha256(b"source").hexdigest(),
            },
            {
                "category": "generated",
                "path": "output.rknn",
                "size": 6,
                "sha256": hashlib.sha256(b"output").hexdigest(),
            },
        ],
    }
    record_path = project / "artifacts/work/demo/import-record.json"
    assert json.loads(record_path.read_text(encoding="utf-8")) == record
    assert record_path.read_text(encoding="utf-8") == (
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def test_copy_imports_and_reuses_complete_demo_category(tmp_path: Path) -> None:
    workspace, project, manifest = _fixture_with_demo(tmp_path)
    before = _source_hashes(workspace)

    record = import_existing(workspace, project, manifest)

    demo_root = project / "artifacts/work/demo/install/rknn_Demo"
    executable = demo_root / "demo"
    runtime = demo_root / "lib/runtime.so"
    assert executable.read_bytes() == b"executable"
    assert runtime.read_bytes() == b"runtime"
    assert executable.stat().st_ino != (
        workspace / "model-zoo/install/rknn_Demo/demo"
    ).stat().st_ino
    assert record["statuses"] == {
        "source": "imported",
        "generated": "imported",
        "demo": "imported",
    }
    assert [entry["category"] for entry in record["files"]] == [
        "source",
        "generated",
        "demo",
        "demo",
    ]
    assert _source_hashes(workspace) == before

    original_inodes = (executable.stat().st_ino, runtime.stat().st_ino)
    reused = import_existing(workspace, project, manifest)

    assert reused["statuses"] == {
        "source": "reused",
        "generated": "reused",
        "demo": "reused",
    }
    assert (executable.stat().st_ino, runtime.stat().st_ino) == original_inodes


def test_bad_demo_hash_fails_before_any_destination_write(tmp_path: Path) -> None:
    workspace, project, manifest = _fixture_with_demo(tmp_path)
    bad_demo = replace(manifest.demo_files[1], sha256="0" * 64)
    bad_manifest = replace(
        manifest,
        demo_files=(manifest.demo_files[0], bad_demo),
    )

    with pytest.raises(ArtifactError, match="demo source file") as error:
        import_existing(workspace, project, bad_manifest)

    assert "expected sha256" in str(error.value)
    assert not (project / "artifacts").exists()


def test_unexpected_existing_demo_file_is_rejected_and_preserved(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture_with_demo(tmp_path)
    import_existing(workspace, project, manifest)
    unexpected = (
        project / "artifacts/work/demo/install/rknn_Demo/unexpected.bin"
    )
    unexpected.write_bytes(b"keep")

    with pytest.raises(ArtifactError, match="unexpected.bin"):
        import_existing(workspace, project, manifest)

    assert unexpected.read_bytes() == b"keep"


def test_demo_source_root_symlink_is_rejected_before_any_write(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture_with_demo(tmp_path)
    demo_root = workspace / "model-zoo/install/rknn_Demo"
    external = tmp_path / "external-demo"
    demo_root.rename(external)
    demo_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(ArtifactError, match="symlink"):
        import_existing(workspace, project, manifest)

    assert not (project / "artifacts").exists()


def test_demo_destination_symlink_is_rejected_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture_with_demo(tmp_path)
    destination = project / "artifacts/work/demo/install/rknn_Demo"
    destination.parent.mkdir(parents=True)
    external = tmp_path / "external-destination"
    external.mkdir()
    destination.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        importer,
        "_sha256",
        lambda path: pytest.fail(f"hashed before destination preflight: {path}"),
    )

    with pytest.raises(ConfigurationError, match="symlink"):
        import_existing(workspace, project, manifest)

    assert destination.is_symlink()


def test_rerun_reuses_matching_targets_and_preserves_mismatched_target(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    import_existing(workspace, project, manifest)
    source_target = project / "artifacts/source_models/demo/model.bin"
    generated_target = project / "artifacts/work/demo/model/output.rknn"
    original_inodes = (source_target.stat().st_ino, generated_target.stat().st_ino)

    reused = import_existing(workspace, project, manifest)

    assert reused["statuses"] == {"source": "reused", "generated": "reused"}
    assert (source_target.stat().st_ino, generated_target.stat().st_ino) == (
        original_inodes
    )
    record_path = project / "artifacts/work/demo/import-record.json"
    record_before_failure = record_path.read_bytes()
    source_target.write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="existing destination") as error:
        import_existing(workspace, project, manifest)

    assert str(source_target) in str(error.value)
    assert source_target.read_bytes() == b"tampered"
    assert generated_target.read_bytes() == b"output"
    assert record_path.read_bytes() == record_before_failure


def test_bad_manifest_hash_fails_before_any_destination_write(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    before = _source_hashes(workspace)
    bad_generated = replace(
        manifest.generated_files[0], sha256="0" * 64
    )
    bad_manifest = replace(manifest, generated_files=(bad_generated,))

    with pytest.raises(ArtifactError, match="generated source file") as error:
        import_existing(workspace, project, bad_manifest)

    assert "expected sha256" in str(error.value)
    assert _source_hashes(workspace) == before
    assert not (project / "artifacts/source_models/demo").exists()
    assert not (project / "artifacts/work/demo/model").exists()


def test_hardlink_mode_shares_inodes_and_is_recorded(tmp_path: Path) -> None:
    workspace, project, manifest = _fixture(tmp_path)

    record = import_existing(workspace, project, manifest, mode="hardlink")

    assert record["mode"] == "hardlink"
    assert (
        project / "artifacts/source_models/demo/model.bin"
    ).stat().st_ino == (workspace / "models/demo/model.bin").stat().st_ino
    assert (
        project / "artifacts/work/demo/model/output.rknn"
    ).stat().st_ino == (workspace / "model-zoo/demo/output.rknn").stat().st_ino


@pytest.mark.parametrize("mode", ["move", "", "COPY"])
def test_invalid_mode_is_rejected(mode: str, tmp_path: Path) -> None:
    workspace, project, manifest = _fixture(tmp_path)

    with pytest.raises(ConfigurationError, match="mode"):
        import_existing(workspace, project, manifest, mode=mode)

    assert not (project / "artifacts").exists()


@pytest.mark.parametrize("relative_argument", ["workspace", "project_root"])
def test_relative_roots_are_rejected(
    relative_argument: str, tmp_path: Path
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    if relative_argument == "workspace":
        workspace = Path("relative-workspace")
    else:
        project = Path("relative-project")

    with pytest.raises(ConfigurationError, match="absolute"):
        import_existing(workspace, project, manifest)


@pytest.mark.parametrize(
    "model_id",
    ["", ".", "..", "../outside", "nested/id", r"nested\id"],
)
def test_unsafe_model_id_is_rejected_before_any_write(
    model_id: str, tmp_path: Path
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    bad_manifest = replace(manifest, model_id=model_id)

    with pytest.raises(ConfigurationError, match="model_id"):
        import_existing(workspace, project, bad_manifest)

    assert not (project / "artifacts").exists()


def test_absolute_model_id_cannot_write_outside_project(tmp_path: Path) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    outside = tmp_path / "outside-model"
    bad_manifest = replace(manifest, model_id=str(outside))

    with pytest.raises(ConfigurationError, match="model_id"):
        import_existing(workspace, project, bad_manifest)

    assert not outside.exists()
    assert not (project / "artifacts").exists()


def test_symlinked_source_file_is_rejected(tmp_path: Path) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    source = workspace / "models/demo/model.bin"
    external = tmp_path / "external-model.bin"
    external.write_bytes(b"source")
    source.unlink()
    source.symlink_to(external)

    with pytest.raises(ArtifactError, match="symlink") as error:
        import_existing(workspace, project, manifest)

    assert str(source) in str(error.value)
    assert external.read_bytes() == b"source"
    assert not (project / "artifacts/source_models/demo").exists()


@pytest.mark.parametrize("linked_component", ["source_root", "generated_ancestor"])
def test_source_ancestor_symlink_is_rejected_before_any_hash_or_write(
    linked_component: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    if linked_component == "source_root":
        linked = workspace / "models/demo"
        external = tmp_path / "external-source"
    else:
        linked = workspace / "model-zoo"
        external = tmp_path / "external-generated"
    linked.rename(external)
    linked.symlink_to(external, target_is_directory=True)
    external_before = _source_hashes(external)
    monkeypatch.setattr(
        importer,
        "_sha256",
        lambda path: pytest.fail(f"hashed before path preflight: {path}"),
    )

    with pytest.raises(ArtifactError, match="symlink") as error:
        import_existing(workspace, project, manifest)

    assert str(linked) in str(error.value)
    assert _source_hashes(external) == external_before
    assert not (project / "artifacts").exists()


def test_source_pin_resolving_outside_declared_root_is_rejected(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    outside = workspace / "models/outside.bin"
    outside.write_bytes(b"source")
    escaped_pin = _pin("../outside.bin", b"source")
    bad_manifest = replace(manifest, source_files=(escaped_pin,))
    before = _source_hashes(workspace)

    with pytest.raises(ArtifactError, match="outside declared source root"):
        import_existing(workspace, project, bad_manifest)

    assert _source_hashes(workspace) == before
    assert not (project / "artifacts").exists()


@pytest.mark.parametrize(
    "linked_component", ["artifacts", "source_models", "destination_root"]
)
def test_destination_ancestor_symlink_is_rejected_before_any_hash_or_write(
    linked_component: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    external = tmp_path / f"external-{linked_component}"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_bytes(b"keep")
    if linked_component == "artifacts":
        linked = project / "artifacts"
    elif linked_component == "source_models":
        (project / "artifacts").mkdir()
        linked = project / "artifacts/source_models"
    else:
        (project / "artifacts/source_models").mkdir(parents=True)
        linked = project / "artifacts/source_models/demo"
    linked.symlink_to(external, target_is_directory=True)
    external_before = _source_hashes(external)
    monkeypatch.setattr(
        importer,
        "_sha256",
        lambda path: pytest.fail(f"hashed before path preflight: {path}"),
    )

    with pytest.raises(ConfigurationError, match="symlink") as error:
        import_existing(workspace, project, manifest)

    assert str(linked) in str(error.value)
    assert _source_hashes(external) == external_before
    assert linked.is_symlink()
    if linked_component != "artifacts":
        assert not (project / "artifacts/work").exists()


def test_destination_symlink_inserted_after_preflight_is_rejected_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    external = tmp_path / "external-after-preflight"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_bytes(b"keep")
    original_preflight = importer._preflight_paths

    def preflight_then_swap(*args: object, **kwargs: object) -> None:
        original_preflight(*args, **kwargs)
        (project / "artifacts").symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(importer, "_preflight_paths", preflight_then_swap)
    monkeypatch.setattr(
        importer,
        "_sha256",
        lambda path: pytest.fail(f"hashed after destination swap: {path}"),
    )

    with pytest.raises(
        (ArtifactError, ConfigurationError), match="symlink"
    ):
        import_existing(workspace, project, manifest)

    assert _source_hashes(external) == {"keep.txt": hashlib.sha256(b"keep").hexdigest()}
    assert (project / "artifacts").is_symlink()


def test_destination_symlink_inserted_before_record_write_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    external = tmp_path / "external-before-record"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_bytes(b"keep")
    preserved_artifacts = tmp_path / "preserved-artifacts"
    original_write_record = importer._write_record

    def swap_then_write(
        path: Path,
        record: dict[str, object],
        **kwargs: object,
    ) -> tuple[int, int]:
        (project / "artifacts").rename(preserved_artifacts)
        (project / "artifacts").symlink_to(external, target_is_directory=True)
        return original_write_record(path, record, **kwargs)

    monkeypatch.setattr(importer, "_write_record", swap_then_write)

    with pytest.raises(
        (ArtifactError, ConfigurationError), match="symlink"
    ):
        import_existing(workspace, project, manifest)

    assert _source_hashes(external) == {"keep.txt": hashlib.sha256(b"keep").hexdigest()}
    assert (project / "artifacts").is_symlink()
    assert (preserved_artifacts / "source_models/demo/model.bin").is_file()
    assert (preserved_artifacts / "work/demo/model/output.rknn").is_file()


@pytest.mark.parametrize(
    ("pin", "message"),
    [
        (
            SizedFilePin(
                Path("model.bin"),
                7,
                hashlib.sha256(b"source").hexdigest(),
            ),
            "expected size 7, actual 6",
        ),
        (SizedFilePin(Path("model.bin"), 6, "f" * 64), "expected sha256"),
    ],
)
def test_source_size_and_hash_are_checked_exactly(
    pin: SizedFilePin, message: str, tmp_path: Path
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    bad_manifest = replace(manifest, source_files=(pin,))

    with pytest.raises(ArtifactError, match=message):
        import_existing(workspace, project, bad_manifest)

    assert not (project / "artifacts/source_models/demo").exists()


def test_source_hash_read_failure_is_reported_as_artifact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    failure = OSError("hash read denied")

    def fail_hash(path: Path) -> str:
        raise failure

    monkeypatch.setattr(importer, "_sha256", fail_hash)

    with pytest.raises(ArtifactError, match="cannot read") as error:
        import_existing(workspace, project, manifest)

    assert error.value.__cause__ is failure
    assert manifest.source_files[0].sha256 in str(error.value)
    assert not (project / "artifacts").exists()


def test_unexpected_existing_destination_file_is_rejected_and_preserved(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    import_existing(workspace, project, manifest)
    unexpected = project / "artifacts/source_models/demo/unexpected.bin"
    unexpected.write_bytes(b"keep")

    with pytest.raises(ArtifactError, match="unexpected.bin"):
        import_existing(workspace, project, manifest)

    assert unexpected.read_bytes() == b"keep"


def test_existing_destination_missing_declared_file_is_rejected_unchanged(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    import_existing(workspace, project, manifest)
    missing = project / "artifacts/source_models/demo/model.bin"
    generated = project / "artifacts/work/demo/model/output.rknn"
    record_path = project / "artifacts/work/demo/import-record.json"
    missing.unlink()
    generated_before = generated.read_bytes()
    record_before = record_path.read_bytes()

    with pytest.raises(ArtifactError, match="existing destination file") as error:
        import_existing(workspace, project, manifest)

    assert str(missing) in str(error.value)
    assert not missing.exists()
    assert generated.read_bytes() == generated_before
    assert record_path.read_bytes() == record_before
    assert list(missing.parent.iterdir()) == []


def test_existing_destination_root_file_is_rejected_and_preserved(
    tmp_path: Path,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination = project / "artifacts/source_models/demo"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"keep")
    before = _source_hashes(workspace)

    with pytest.raises(ArtifactError, match="real directory"):
        import_existing(workspace, project, manifest)

    assert destination.read_bytes() == b"keep"
    assert _source_hashes(workspace) == before
    assert not (project / "artifacts/work/demo/model").exists()


@pytest.mark.parametrize("mode", ["copy", "hardlink"])
def test_category_failure_removes_only_its_temporary_directory(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    second = workspace / "models/demo/tokenizer.json"
    second.write_bytes(b"tokenizer")
    manifest = replace(
        manifest,
        source_files=(
            manifest.source_files[0],
            _pin("tokenizer.json", b"tokenizer"),
        ),
    )
    before = _source_hashes(workspace)
    failure = OSError(f"{mode} failed")
    calls = 0
    operation = importer.shutil.copy2 if mode == "copy" else importer.os.link

    def fail_second(
        source: Path | str,
        destination: Path | str,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return operation(source, destination, **kwargs)

    if mode == "copy":
        monkeypatch.setattr(importer.shutil, "copy2", fail_second)
    else:
        monkeypatch.setattr(importer.os, "link", fail_second)

    with pytest.raises(OSError) as error:
        import_existing(workspace, project, manifest, mode=mode)

    assert error.value is failure
    destination_parent = project / "artifacts/source_models"
    assert not (destination_parent / "demo").exists()
    assert list(destination_parent.iterdir()) == []
    assert not (project / "artifacts/work/demo/model").exists()
    assert _source_hashes(workspace) == before


def test_category_cleanup_failure_does_not_mask_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    import_failure = RuntimeError("import failed")
    cleanup_calls = 0

    def fail_import(source: Path, destination: Path) -> None:
        raise import_failure

    def fail_cleanup(*args: object, **kwargs: object) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise OSError("cleanup failed")

    monkeypatch.setattr(importer.shutil, "copy2", fail_import)
    monkeypatch.setattr(importer, "_remove_owned_directory", fail_cleanup)

    with pytest.raises(RuntimeError) as error:
        import_existing(workspace, project, manifest)

    assert error.value is import_failure
    assert cleanup_calls == 1


def test_category_cleanup_preserves_replacement_and_finds_renamed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination_parent = project / "artifacts/source_models"
    moved_staging = destination_parent / "moved-staging"
    replacement: Path | None = None
    failure = OSError("copy failed after staging replacement")

    def replace_staging_then_fail(source: Path, destination: Path) -> None:
        nonlocal replacement
        staging = next(destination_parent.glob(".demo.*"))
        replacement = destination_parent / staging.name
        staging.rename(moved_staging)
        replacement.mkdir()
        (replacement / "competitor.bin").write_bytes(b"keep")
        raise failure

    monkeypatch.setattr(importer.shutil, "copy2", replace_staging_then_fail)

    with pytest.raises(OSError) as error:
        import_existing(workspace, project, manifest)

    assert error.value is failure
    assert replacement is not None
    assert (replacement / "competitor.bin").read_bytes() == b"keep"
    assert not moved_staging.exists()


def test_category_publish_rejects_and_preserves_destination_created_in_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination = project / "artifacts/source_models/demo"
    raced_inode: int | None = None

    def destination_appears_before_publish(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal raced_inode
        assert source_dir_fd == destination_dir_fd
        assert source_name.startswith(".demo.")
        assert destination_name == "demo"
        os.mkdir(destination_name, dir_fd=destination_dir_fd)
        raced_inode = os.stat(
            destination_name,
            dir_fd=destination_dir_fd,
            follow_symlinks=False,
        ).st_ino
        raise FileExistsError(
            errno.EEXIST,
            os.strerror(errno.EEXIST),
            destination_name,
        )

    monkeypatch.setattr(
        importer, "_rename_noreplace_at", destination_appears_before_publish
    )

    with pytest.raises(ArtifactError, match="appeared during import"):
        import_existing(workspace, project, manifest)

    assert destination.is_dir()
    assert destination.stat().st_ino == raced_inode
    assert list(destination.iterdir()) == []
    assert sorted(path.name for path in destination.parent.iterdir()) == ["demo"]
    assert not (project / "artifacts/work/demo/model").exists()


def test_reuse_rejects_destination_parent_replaced_during_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    import_existing(workspace, project, manifest)
    parent = project / "artifacts/source_models"
    detached = tmp_path / "detached-source-models"
    replacement = parent / "demo"
    original_present_files = importer._present_files_at
    swapped = False

    def scan_then_replace(root_fd: int, root: Path) -> set[Path]:
        nonlocal swapped
        present = original_present_files(root_fd, root)
        if not swapped and root == replacement:
            swapped = True
            parent.rename(detached)
            replacement.mkdir(parents=True)
            (replacement / "unexpected.bin").write_bytes(b"keep")
        return present

    monkeypatch.setattr(importer, "_present_files_at", scan_then_replace)

    with pytest.raises(ArtifactError, match="changed during import"):
        import_existing(workspace, project, manifest)

    assert (replacement / "unexpected.bin").read_bytes() == b"keep"
    assert (detached / "demo/model.bin").read_bytes() == b"source"


def test_publish_rejects_parent_replaced_during_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    parent = project / "artifacts/source_models"
    detached = tmp_path / "detached-publish-parent"
    original_rename = importer._rename_noreplace_at
    swapped = False

    def replace_parent_then_rename(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        if not swapped and destination_name == "demo":
            swapped = True
            parent.rename(detached)
            parent.mkdir()
        original_rename(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )

    monkeypatch.setattr(importer, "_rename_noreplace_at", replace_parent_then_rename)

    with pytest.raises(ArtifactError, match="changed during import"):
        import_existing(workspace, project, manifest)

    assert list(parent.iterdir()) == []
    assert list(detached.iterdir()) == []
    assert not (project / "artifacts/work/demo/model").exists()


def test_publish_rejects_replacement_after_rename_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination = project / "artifacts/source_models/demo"
    detached_name = "detached-staged-demo"
    original_rename = importer._rename_noreplace_at
    swapped = False

    def replace_after_rename(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        original_rename(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )
        if not swapped and destination_name == "demo":
            swapped = True
            os.rename(
                destination_name,
                detached_name,
                src_dir_fd=destination_dir_fd,
                dst_dir_fd=destination_dir_fd,
            )
            os.mkdir(destination_name, dir_fd=destination_dir_fd)
            replacement_fd = os.open(
                destination_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=destination_dir_fd,
            )
            try:
                unexpected_fd = os.open(
                    "unexpected.bin",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement_fd,
                )
                os.close(unexpected_fd)
            finally:
                os.close(replacement_fd)

    monkeypatch.setattr(importer, "_rename_noreplace_at", replace_after_rename)

    with pytest.raises(ArtifactError, match="changed during import"):
        import_existing(workspace, project, manifest)

    assert (destination / "unexpected.bin").is_file()
    assert not (destination.parent / detached_name).exists()
    assert not (project / "artifacts/work/demo/model").exists()


def test_temporary_parent_open_failure_does_not_leak_source_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination = project / "artifacts/source_models/demo"
    original_open_parent = importer._open_relative_parent
    failure = OSError("temporary parent open failed")

    def fail_temporary_parent(*args: object, **kwargs: object) -> tuple[int, str]:
        if kwargs.get("label") == "temporary import parent":
            raise failure
        return original_open_parent(*args, **kwargs)

    before = len(list(Path("/proc/self/fd").iterdir()))
    monkeypatch.setattr(importer, "_open_relative_parent", fail_temporary_parent)

    with pytest.raises(OSError) as error:
        importer._populate_category(
            workspace / manifest.source_root,
            destination,
            manifest.source_files,
            "copy",
        )

    assert error.value is failure
    assert len(list(Path("/proc/self/fd").iterdir())) == before
    assert not destination.exists()


def test_staging_root_open_failure_removes_new_empty_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    destination = project / "artifacts/source_models/demo"
    destination_parent = destination.parent
    source_before = _source_hashes(workspace)
    original_open = importer.os.open
    failure = OSError("staging root open failed")
    failed = False

    def fail_first_staging_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal failed
        if not failed and Path(path).name.startswith(".demo."):
            failed = True
            raise failure
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(importer.os, "open", fail_first_staging_open)

    with pytest.raises(OSError) as error:
        import_existing(workspace, project, manifest)

    assert error.value is failure
    assert _source_hashes(workspace) == source_before
    assert not destination.exists()
    assert list(destination_parent.iterdir()) == []


def test_atomic_no_replace_rename_preserves_existing_empty_directory(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "temporary"
    destination = tmp_path / "destination"
    temporary.mkdir()
    destination.mkdir()
    staged = temporary / "model.bin"
    staged.write_bytes(b"staged")
    destination_inode = destination.stat().st_ino

    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(FileExistsError):
            importer._rename_noreplace_at(
                parent_fd,
                temporary.name,
                parent_fd,
                destination.name,
            )
    finally:
        os.close(parent_fd)

    assert destination.stat().st_ino == destination_inode
    assert list(destination.iterdir()) == []
    assert staged.read_bytes() == b"staged"


def test_record_replacement_is_atomic_and_rerun_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    first = import_existing(workspace, project, manifest)
    record_path = project / "artifacts/work/demo/import-record.json"
    first_bytes = record_path.read_bytes()
    original_replace = importer.os.replace
    failure = OSError("record replace failed")

    def fail_record_replace(
        source: Path | str,
        destination: Path | str,
        **kwargs: object,
    ) -> None:
        if str(destination) == record_path.name:
            raise failure
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(importer.os, "replace", fail_record_replace)

    with pytest.raises(OSError) as error:
        import_existing(workspace, project, manifest)

    assert error.value is failure
    assert record_path.read_bytes() == first_bytes
    assert first["statuses"] == {"source": "imported", "generated": "imported"}
    assert sorted(path.name for path in record_path.parent.iterdir()) == [
        "import-record.json",
        "model",
    ]

    monkeypatch.setattr(importer.os, "replace", original_replace)
    second = import_existing(workspace, project, manifest)
    third = import_existing(workspace, project, manifest)
    assert second == third
    assert second["statuses"] == {"source": "reused", "generated": "reused"}
    assert json.loads(record_path.read_text(encoding="utf-8")) == second


def test_record_cleanup_preserves_replacement_and_finds_renamed_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    original_replace = importer.os.replace
    replacement_name: str | None = None
    moved_name = "moved-import-record.tmp"
    failure = OSError("record publication failed")

    def replace_record_temp_then_fail(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal replacement_name
        if destination != "import-record.json":
            original_replace(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            return
        replacement_name = source
        os.rename(
            source,
            moved_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        replacement_fd = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        os.write(replacement_fd, b"competitor")
        os.close(replacement_fd)
        raise failure

    monkeypatch.setattr(importer.os, "replace", replace_record_temp_then_fail)

    with pytest.raises(OSError) as error:
        import_existing(workspace, project, manifest)

    assert error.value is failure
    assert replacement_name is not None
    record_parent = project / "artifacts/work/demo"
    assert (record_parent / replacement_name).read_bytes() == b"competitor"
    assert not (record_parent / moved_name).exists()


def test_project_artifacts_replacement_between_categories_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    original_populate = importer._populate_category
    detached = tmp_path / "detached-artifacts"
    replacement = project / "artifacts"
    calls = 0

    def populate_then_replace(*args: object, **kwargs: object) -> str:
        nonlocal calls
        status = original_populate(*args, **kwargs)
        calls += 1
        if calls == 1:
            replacement.rename(detached)
            replacement.mkdir()
            (replacement / "competitor.bin").write_bytes(b"keep")
        return status

    monkeypatch.setattr(importer, "_populate_category", populate_then_replace)

    with pytest.raises(ArtifactError, match="changed during import"):
        import_existing(workspace, project, manifest)

    assert (replacement / "competitor.bin").read_bytes() == b"keep"
    assert list(replacement.iterdir()) == [replacement / "competitor.bin"]
    assert (detached / "source_models/demo/model.bin").read_bytes() == b"source"


@pytest.mark.parametrize("replaced", ["source_root", "workspace"])
def test_source_tree_replacement_after_category_is_rejected(
    replaced: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, project, manifest = _fixture(tmp_path)
    original_populate = importer._populate_category
    source_root = workspace / manifest.source_root
    replaced_path = source_root if replaced == "source_root" else workspace
    detached = tmp_path / f"detached-{replaced}"
    calls = 0

    def populate_then_replace(*args: object, **kwargs: object) -> str:
        nonlocal calls
        status = original_populate(*args, **kwargs)
        calls += 1
        if calls == 1:
            replaced_path.rename(detached)
            replaced_path.mkdir(parents=True)
            (replaced_path / "competitor.bin").write_bytes(b"keep")
        return status

    monkeypatch.setattr(importer, "_populate_category", populate_then_replace)

    with pytest.raises(ArtifactError, match="changed during import"):
        import_existing(workspace, project, manifest)

    assert (replaced_path / "competitor.bin").read_bytes() == b"keep"


def test_import_existing_wrapper_has_exact_content_and_executable_mode() -> None:
    wrapper = Path("tools/host/import-existing")

    assert wrapper.read_text(encoding="utf-8") == (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'exec "${RK_LLM_PYTHON:-python3}" -m rk_llm.host.import_existing "$@"\n'
    )
    assert wrapper.stat().st_mode & stat.S_IXUSR
