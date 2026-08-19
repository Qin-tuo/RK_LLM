import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

import rk_llm.host.bootstrap as bootstrap
from rk_llm.errors import ArtifactError, ConfigurationError
from rk_llm.host.bootstrap import adopt_runtime_files, ensure_checkout
from rk_llm.manifests.loader import (
    DigestPin,
    GitPin,
    RuntimePin,
    TargetPin,
    UpstreamManifest,
)


REVISION = "a" * 40
OTHER_REVISION = "b" * 40
PIN = GitPin(
    repository="https://example.com/official.git",
    release="V1.0.4",
    revision=REVISION,
)


def _completed(
    args: list[str], stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=stderr)


def _digest_pin(path: str, content: bytes) -> DigestPin:
    return DigestPin(Path(path), hashlib.sha256(content).hexdigest())


def test_dirty_existing_checkout_is_preserved_and_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()
    tracked = destination / "tracked-file"
    tracked.write_text("keep me", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return _completed(args, stdout=" M tracked-file\n")

    with pytest.raises(ConfigurationError, match="uncommitted changes"):
        ensure_checkout(PIN, destination, run=run)

    assert calls == [
        (
            [
                "git",
                "-C",
                str(destination),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            {"check": True, "capture_output": True, "text": True},
        )
    ]
    assert destination.is_dir()
    assert tracked.read_text(encoding="utf-8") == "keep me"


def test_absent_checkout_uses_reference_and_official_remote(tmp_path: Path) -> None:
    destination = tmp_path / "vendor" / "toolkit"
    reference = tmp_path / "seed"
    reference.mkdir()
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        calls.append(args)
        if args[-3:] == ["remote", "get-url", "origin"]:
            stdout = f"{PIN.repository}\n"
        else:
            stdout = f"{REVISION}\n" if args[-2:] == ["rev-parse", "HEAD"] else ""
        return _completed(args, stdout=stdout)

    ensure_checkout(PIN, destination, run=run, reference=reference)

    assert destination.parent.is_dir()
    assert calls == [
        [
            "git",
            "clone",
            "--no-checkout",
            "--reference-if-able",
            str(reference),
            "--dissociate",
            PIN.repository,
            str(destination),
        ],
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
        ["git", "-C", str(destination), "fetch", "origin", REVISION],
        ["git", "-C", str(destination), "checkout", "--detach", REVISION],
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
    ]


def test_clean_checkout_reuse_is_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[-3:] == ["remote", "get-url", "origin"]:
            stdout = f"{PIN.repository}\n"
        else:
            stdout = f"{REVISION}\n" if args[-2:] == ["rev-parse", "HEAD"] else ""
        return _completed(args, stdout=stdout)

    ensure_checkout(PIN, destination, run=run)
    ensure_checkout(PIN, destination, run=run)

    expected = [
        [
            "git",
            "-C",
            str(destination),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
        ["git", "-C", str(destination), "fetch", "origin", REVISION],
        ["git", "-C", str(destination), "checkout", "--detach", REVISION],
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
    ]
    assert calls == expected * 2


def test_git_command_failure_is_wrapped_with_command_and_output(tmp_path: Path) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()
    calls = 0

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(args)
        if calls == 2:
            return _completed(args, stdout=f"{PIN.repository}\n")
        raise subprocess.CalledProcessError(
            128, args, output="fetch output", stderr="fetch failed"
        )

    with pytest.raises(ConfigurationError) as error:
        ensure_checkout(PIN, destination, run=run)

    message = str(error.value)
    assert "git" in message
    assert "fetch" in message
    assert "fetch failed" in message
    assert "fetch output" in message


def test_checkout_rejects_exact_revision_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[-3:] == ["remote", "get-url", "origin"]:
            stdout = f"{PIN.repository}\n"
        else:
            stdout = (
                f"{OTHER_REVISION}\n"
                if args[-2:] == ["rev-parse", "HEAD"]
                else ""
            )
        return _completed(args, stdout=stdout)

    with pytest.raises(ConfigurationError) as error:
        ensure_checkout(PIN, destination, run=run)

    assert REVISION in str(error.value)
    assert OTHER_REVISION in str(error.value)


def test_existing_checkout_symlink_is_rejected_without_git(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    destination = tmp_path / "checkout"
    destination.symlink_to(seed, target_is_directory=True)
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed(args)

    with pytest.raises(ConfigurationError, match="real directory"):
        ensure_checkout(PIN, destination, run=run)

    assert calls == []
    assert destination.is_symlink()
    assert seed.is_dir()


@pytest.mark.parametrize("kind", ["file", "fifo"])
def test_existing_checkout_non_directory_is_rejected_without_git(
    tmp_path: Path, kind: str
) -> None:
    destination = tmp_path / "checkout"
    if kind == "file":
        destination.write_text("preserve", encoding="utf-8")
    else:
        os.mkfifo(destination)
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed(args)

    with pytest.raises(ConfigurationError, match="real directory"):
        ensure_checkout(PIN, destination, run=run)

    assert calls == []
    assert os.path.lexists(destination)


def test_existing_checkout_origin_mismatch_stops_before_git_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkout"
    destination.mkdir()
    other_repository = "https://example.com/not-official.git"
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[-3:] == ["remote", "get-url", "origin"]:
            return _completed(args, stdout=f"{other_repository}\n")
        return _completed(args)

    with pytest.raises(ConfigurationError) as error:
        ensure_checkout(PIN, destination, run=run)

    assert PIN.repository in str(error.value)
    assert other_repository in str(error.value)
    assert calls == [
        [
            "git",
            "-C",
            str(destination),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        ["git", "-C", str(destination), "remote", "get-url", "origin"],
    ]


def test_runtime_source_root_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="RKNN3_RUNTIME_DEV_ROOT must be an absolute path",
    ):
        adopt_runtime_files(Path("relative/runtime"), tmp_path / "runtime", ())


def test_runtime_hash_mismatch_is_checked_before_destination_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "include/api.h"
    source_file.parent.mkdir()
    source_file.write_bytes(b"actual")
    pin = _digest_pin("include/api.h", b"expected")
    destination = tmp_path / ".vendor/rknn3-runtime"

    with pytest.raises(ArtifactError) as error:
        adopt_runtime_files(source, destination, (pin,))

    message = str(error.value)
    assert str(source_file) in message
    assert pin.sha256 in message
    assert hashlib.sha256(b"actual").hexdigest() in message
    assert not destination.exists()
    assert source_file.read_bytes() == b"actual"


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_runtime_source_requires_regular_non_symlink_files(
    tmp_path: Path, kind: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "include/api.h"
    if kind == "symlink":
        target = source / "target"
        target.write_bytes(b"api")
        path.parent.mkdir()
        path.symlink_to(target)
    pin = _digest_pin("include/api.h", b"api")

    with pytest.raises(ArtifactError) as error:
        adopt_runtime_files(source, tmp_path / "runtime", (pin,))

    assert str(path) in str(error.value)
    assert pin.sha256 in str(error.value)
    assert "actual" in str(error.value)


@pytest.mark.parametrize("link_position", ["root", "ancestor"])
def test_runtime_rejects_symlinked_source_root_or_ancestor(
    tmp_path: Path, link_position: str
) -> None:
    external = tmp_path / "external"
    if link_position == "root":
        external_source = external
        external_source.mkdir()
        source = tmp_path / "source"
        symlink = source
        source.symlink_to(external_source, target_is_directory=True)
    else:
        external_source = external / "source"
        external_source.mkdir(parents=True)
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(external, target_is_directory=True)
        source = linked_parent / "source"
        symlink = linked_parent
    pin = _digest_pin("include/api.h", b"external-api")
    external_file = external_source / pin.path
    external_file.parent.mkdir()
    external_file.write_bytes(b"external-api")
    destination = tmp_path / "runtime"

    with pytest.raises(ArtifactError, match="symlink") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(symlink) in str(error.value)
    assert external_file.read_bytes() == b"external-api"
    assert not destination.exists()


def test_runtime_rejects_symlinked_source_path_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "api.h"
    external_file.write_bytes(b"external-api")
    (source / "include").symlink_to(external, target_is_directory=True)
    pin = _digest_pin("include/api.h", b"external-api")
    destination = tmp_path / "runtime"

    with pytest.raises(ArtifactError, match="symlink") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(source / "include") in str(error.value)
    assert external_file.read_bytes() == b"external-api"
    assert not destination.exists()


def test_runtime_rejects_existing_destination_root_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pin = _digest_pin("include/api.h", b"api")
    source_file = source / pin.path
    source_file.parent.mkdir()
    source_file.write_bytes(b"api")
    external = tmp_path / "external-destination"
    external_file = external / pin.path
    external_file.parent.mkdir(parents=True)
    external_file.write_bytes(b"api")
    destination = tmp_path / "runtime"
    destination.symlink_to(external, target_is_directory=True)

    with pytest.raises(ArtifactError, match="symlink") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(destination) in str(error.value)
    assert destination.is_symlink()
    assert external_file.read_bytes() == b"api"


def test_runtime_rejects_existing_destination_component_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "runtime"
    source.mkdir()
    destination.mkdir()
    pin = _digest_pin("include/api.h", b"api")
    source_file = source / pin.path
    source_file.parent.mkdir()
    source_file.write_bytes(b"api")
    external = tmp_path / "external-destination"
    external.mkdir()
    external_file = external / "api.h"
    external_file.write_bytes(b"api")
    (destination / "include").symlink_to(external, target_is_directory=True)

    with pytest.raises(ArtifactError, match="symlink") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(destination / "include") in str(error.value)
    assert external_file.read_bytes() == b"api"


def test_runtime_rejects_symlinked_new_destination_ancestor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pin = _digest_pin("include/api.h", b"api")
    source_file = source / pin.path
    source_file.parent.mkdir()
    source_file.write_bytes(b"api")
    external = tmp_path / "external-project"
    external.mkdir()
    linked_project = tmp_path / "linked-project"
    linked_project.symlink_to(external, target_is_directory=True)
    destination = linked_project / ".vendor/runtime"

    with pytest.raises(ConfigurationError, match="symlink") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(linked_project) in str(error.value)
    assert list(external.iterdir()) == []


def test_runtime_rejects_pinned_path_outside_declared_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    external_file = external / "api.h"
    external_file.write_bytes(b"external-api")
    pin = _digest_pin("../outside/api.h", b"external-api")
    destination = tmp_path / "vendor/runtime"

    with pytest.raises(ArtifactError, match="outside") as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(external_file) in str(error.value)
    assert external_file.read_bytes() == b"external-api"
    assert not destination.exists()


def test_runtime_adoption_is_atomic_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    contents = {
        "include/api.h": b"header",
        "Linux/aarch64/librknn3.so": b"library",
    }
    pins = tuple(_digest_pin(path, content) for path, content in contents.items())
    for relative_path, content in contents.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source_snapshot = {
        path: ((source / path).read_bytes(), (source / path).stat().st_mtime_ns)
        for path in contents
    }
    destination = tmp_path / ".vendor/rknn3-runtime"

    adopt_runtime_files(source, destination, pins)

    assert {
        path: (destination / path).read_bytes() for path in contents
    } == contents
    assert {
        path: ((source / path).read_bytes(), (source / path).stat().st_mtime_ns)
        for path in contents
    } == source_snapshot
    assert [path for path in destination.parent.iterdir() if path != destination] == []


def test_matching_runtime_destination_is_reused_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "runtime"
    source.mkdir()
    destination.mkdir()
    pin = _digest_pin("include/api.h", b"api")
    for root in (source, destination):
        path = root / pin.path
        path.parent.mkdir()
        path.write_bytes(b"api")
    original_inode = (destination / pin.path).stat().st_ino
    monkeypatch.setattr(
        bootstrap.shutil,
        "copy2",
        lambda *args, **kwargs: pytest.fail("matching destination must be reused"),
    )

    adopt_runtime_files(source, destination, (pin,))

    assert (destination / pin.path).stat().st_ino == original_inode


def test_existing_runtime_mismatch_is_preserved_and_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "runtime"
    source.mkdir()
    destination.mkdir()
    pin = _digest_pin("include/api.h", b"expected")
    for root, content in ((source, b"expected"), (destination, b"existing")):
        path = root / pin.path
        path.parent.mkdir()
        path.write_bytes(content)

    with pytest.raises(ArtifactError) as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(destination / pin.path) in str(error.value)
    assert (destination / pin.path).read_bytes() == b"existing"


def test_copy_failure_does_not_expose_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pins = (
        _digest_pin("include/first.h", b"first"),
        _digest_pin("include/second.h", b"second"),
    )
    for pin, content in zip(pins, (b"first", b"second"), strict=True):
        path = source / pin.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    destination = tmp_path / "vendor/runtime"
    original_copy2 = bootstrap.shutil.copy2
    calls = 0

    def fail_second_copy(source_path: Path, destination_path: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("copy failed")
        return Path(original_copy2(source_path, destination_path))

    monkeypatch.setattr(bootstrap.shutil, "copy2", fail_second_copy)

    with pytest.raises(OSError, match="copy failed"):
        adopt_runtime_files(source, destination, pins)

    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
    assert [(source / pin.path).is_file() for pin in pins] == [True, True]


def test_cleanup_failure_does_not_mask_adoption_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pin = _digest_pin("include/api.h", b"api")
    source_path = source / pin.path
    source_path.parent.mkdir()
    source_path.write_bytes(b"api")
    destination = tmp_path / "vendor/runtime"
    adoption_error = RuntimeError("adoption failed")
    cleanup_calls: list[tuple[Path, bool]] = []

    def fail_adoption(source_file: Path, destination_file: Path) -> Path:
        raise adoption_error

    def fail_cleanup(path: Path, *, ignore_errors: bool = False) -> None:
        cleanup_calls.append((path, ignore_errors))
        if not ignore_errors:
            raise OSError("cleanup failed")

    monkeypatch.setattr(bootstrap.shutil, "copy2", fail_adoption)
    monkeypatch.setattr(bootstrap.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError) as error:
        adopt_runtime_files(source, destination, (pin,))

    assert error.value is adoption_error
    assert len(cleanup_calls) == 1
    temporary, ignore_errors = cleanup_calls[0]
    assert temporary.parent == destination.parent
    assert ignore_errors is True


def test_runtime_reverifies_staged_copy_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pin = _digest_pin("include/api.h", b"verified")
    source_path = source / pin.path
    source_path.parent.mkdir()
    source_path.write_bytes(b"verified")
    destination = tmp_path / "vendor/runtime"
    original_copy2 = bootstrap.shutil.copy2

    def mutate_then_copy(source_file: Path, destination_file: Path) -> Path:
        source_file.write_bytes(b"changed-after-verification")
        return Path(original_copy2(source_file, destination_file))

    monkeypatch.setattr(bootstrap.shutil, "copy2", mutate_then_copy)

    with pytest.raises(ArtifactError) as error:
        adopt_runtime_files(source, destination, (pin,))

    assert str(destination) not in str(error.value)
    assert pin.sha256 in str(error.value)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def _manifest() -> UpstreamManifest:
    toolkit = GitPin("https://example.com/toolkit.git", "V1", REVISION)
    model_zoo = GitPin("https://example.com/model-zoo.git", "V1", OTHER_REVISION)
    runtime = RuntimePin("1.0.4", (_digest_pin("include/api.h", b"api"),))
    target = TargetPin("rk3588", "rk1828", "rk1820", "aarch64", "2.35", "3.4.30")
    return UpstreamManifest(toolkit, model_zoo, runtime, target)


def test_cli_populates_vendor_paths_and_uses_matching_seed_references(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    manifest_path = tmp_path / "upstream.yaml"
    manifest_path.write_text("manifest", encoding="utf-8")
    runtime_root = tmp_path / "runtime-source"
    runtime_root.mkdir()
    seed_workspace = tmp_path / "seeds"
    toolkit_seed = seed_workspace / "rknn3-toolkit"
    model_zoo_seed = seed_workspace / "rknn3-model-zoo"
    toolkit_seed.mkdir(parents=True)
    model_zoo_seed.mkdir()
    manifest = _manifest()
    checkout_calls: list[tuple[GitPin, Path, Path | None]] = []
    runtime_calls: list[tuple[Path, Path, tuple[DigestPin, ...]]] = []
    monkeypatch.setattr(bootstrap, "load_upstream_manifest", lambda path: manifest)
    monkeypatch.setattr(
        bootstrap,
        "ensure_checkout",
        lambda pin, destination, reference=None: checkout_calls.append(
            (pin, destination, reference)
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "adopt_runtime_files",
        lambda source, destination, files: runtime_calls.append(
            (source, destination, files)
        ),
    )

    assert (
        bootstrap.main(
            [
                "--project-root",
                str(project_root),
                "--upstream-manifest",
                str(manifest_path),
                "--runtime-dev-root",
                str(runtime_root),
                "--seed-workspace",
                str(seed_workspace),
            ]
        )
        == 0
    )

    vendor = project_root / ".vendor"
    assert checkout_calls == [
        (manifest.toolkit, vendor / "rknn3-toolkit", toolkit_seed),
        (manifest.model_zoo, vendor / "rknn3-model-zoo", model_zoo_seed),
    ]
    assert runtime_calls == [
        (runtime_root, vendor / "rknn3-runtime", manifest.runtime.files)
    ]
    assert capsys.readouterr().out == f"Bootstrapped RKNN3 dependencies in {vendor}\n"


@pytest.mark.parametrize(
    "option",
    ["--project-root", "--upstream-manifest", "--runtime-dev-root", "--seed-workspace"],
)
def test_cli_requires_absolute_paths(option: str, tmp_path: Path) -> None:
    arguments = [
        "--project-root",
        str(tmp_path / "project"),
        "--upstream-manifest",
        str(tmp_path / "upstream.yaml"),
        "--runtime-dev-root",
        str(tmp_path / "runtime"),
        "--seed-workspace",
        str(tmp_path / "seeds"),
    ]
    arguments[arguments.index(option) + 1] = "relative/path"

    with pytest.raises(SystemExit) as error:
        bootstrap.main(arguments)

    assert error.value.code == 2


def test_bootstrap_wrapper_has_exact_content_and_executable_mode() -> None:
    wrapper = Path("tools/host/bootstrap")

    assert wrapper.read_text(encoding="utf-8") == (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'exec "${RK_LLM_PYTHON:-python3}" -m rk_llm.host.bootstrap "$@"\n'
    )
    assert wrapper.stat().st_mode & stat.S_IXUSR
