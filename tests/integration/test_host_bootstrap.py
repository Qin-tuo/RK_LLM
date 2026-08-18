import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rk_llm.errors import ConfigurationError
from rk_llm.host.bootstrap import ensure_checkout
from rk_llm.manifests.loader import GitPin


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_real_local_repository_is_checked_out_at_detached_pinned_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "init", str(source)], check=True, capture_output=True, text=True
    )
    _git(source, "config", "user.name", "Bootstrap Test")
    _git(source, "config", "user.email", "bootstrap@example.com")
    tracked = source / "tracked.txt"
    tracked.write_text("pinned\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "pinned")
    revision = _git(source, "rev-parse", "HEAD").stdout.strip()
    tracked.write_text("later\n", encoding="utf-8")
    _git(source, "commit", "-am", "later")
    destination = tmp_path / "vendor/checkout"
    pin = GitPin(str(source), "local", revision)

    ensure_checkout(pin, destination)

    assert _git(destination, "rev-parse", "HEAD").stdout.strip() == revision
    detached = subprocess.run(
        ["git", "-C", str(destination), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert detached.returncode == 1
    assert (destination / "tracked.txt").read_text(encoding="utf-8") == "pinned\n"
    assert (
        _git(destination, "remote", "get-url", "origin").stdout.strip()
        == str(source)
    )


def test_existing_checkout_detects_untracked_files_hidden_by_git_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "init", str(source)], check=True, capture_output=True, text=True
    )
    _git(source, "config", "user.name", "Bootstrap Test")
    _git(source, "config", "user.email", "bootstrap@example.com")
    tracked = source / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "pinned")
    revision = _git(source, "rev-parse", "HEAD").stdout.strip()
    destination = tmp_path / "vendor/checkout"
    pin = GitPin(str(source), "local", revision)
    ensure_checkout(pin, destination)
    _git(destination, "config", "status.showUntrackedFiles", "no")
    untracked = destination / "generated.bin"
    untracked.write_bytes(b"preserve")

    with pytest.raises(ConfigurationError, match="uncommitted changes"):
        ensure_checkout(pin, destination)

    assert untracked.read_bytes() == b"preserve"
    assert _git(destination, "rev-parse", "HEAD").stdout.strip() == revision


def test_reference_clone_is_self_contained_after_temporary_seed_removal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(
        ["git", "init", str(source)], check=True, capture_output=True, text=True
    )
    _git(source, "config", "user.name", "Bootstrap Test")
    _git(source, "config", "user.email", "bootstrap@example.com")
    tracked = source / "tracked.txt"
    tracked.write_text("pinned\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "pinned")
    revision = _git(source, "rev-parse", "HEAD").stdout.strip()
    seed = tmp_path / "temporary-seed.git"
    subprocess.run(
        ["git", "clone", "--mirror", str(source), str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    destination = tmp_path / "vendor/checkout"
    pin = GitPin(str(source), "local", revision)

    ensure_checkout(pin, destination, reference=seed)

    alternates = destination / ".git/objects/info/alternates"
    assert not alternates.exists()
    shutil.rmtree(seed)
    _git(destination, "cat-file", "-e", f"{revision}^{{commit}}")
    assert _git(destination, "rev-parse", "HEAD").stdout.strip() == revision
    _git(destination, "checkout", "--detach", revision)
    assert (destination / "tracked.txt").read_text(encoding="utf-8") == "pinned\n"


def test_checkout_rejects_symlinked_destination_ancestor_before_git(
    tmp_path: Path,
) -> None:
    external_parent = tmp_path / "external-parent"
    external_repository = external_parent / "checkout"
    external_repository.mkdir(parents=True)
    subprocess.run(
        ["git", "init", str(external_repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(external_repository, "config", "user.name", "Bootstrap Test")
    _git(external_repository, "config", "user.email", "bootstrap@example.com")
    tracked = external_repository / "tracked.txt"
    tracked.write_text("external\n", encoding="utf-8")
    _git(external_repository, "add", "tracked.txt")
    _git(external_repository, "commit", "-m", "external")
    original_revision = _git(external_repository, "rev-parse", "HEAD").stdout.strip()
    project = tmp_path / "project"
    project.mkdir()
    linked_vendor = project / ".vendor"
    linked_vendor.symlink_to(external_parent, target_is_directory=True)
    destination = linked_vendor / "checkout"
    pin = GitPin(str(external_repository), "local", original_revision)
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[-3:] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{pin.repository}\n", stderr=""
            )
        if args[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{original_revision}\n", stderr=""
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with pytest.raises(ConfigurationError, match="symlink") as error:
        ensure_checkout(pin, destination, run=run)

    assert str(linked_vendor) in str(error.value)
    assert calls == []
    assert (
        _git(external_repository, "rev-parse", "HEAD").stdout.strip()
        == original_revision
    )
    assert tracked.read_text(encoding="utf-8") == "external\n"


def test_executable_wrapper_invokes_module_help(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RK_LLM_PYTHON"] = sys.executable

    result = subprocess.run(
        [str(Path("tools/host/bootstrap").resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--project-root" in result.stdout
    assert result.stderr == ""
