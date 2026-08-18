"""Bootstrap immutable RKNN3 source checkouts and runtime files."""

import argparse
import hashlib
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from rk_llm.errors import ArtifactError, ConfigurationError, RKLLMProjectError
from rk_llm.manifests.loader import DigestPin, GitPin, load_upstream_manifest


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _command_error(
    args: Sequence[str], error: subprocess.CalledProcessError
) -> ConfigurationError:
    details = [
        f"command failed with exit code {error.returncode}: {shlex.join(args)}"
    ]
    if error.stderr:
        details.append(f"stderr: {error.stderr.strip()}")
    if error.stdout:
        details.append(f"stdout: {error.stdout.strip()}")
    return ConfigurationError("; ".join(details))


def _checked(
    args: list[str], run: CommandRunner = subprocess.run
) -> subprocess.CompletedProcess[str]:
    try:
        return run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise _command_error(args, error) from error


def _capture(args: list[str], run: CommandRunner = subprocess.run) -> str:
    return _checked(args, run).stdout


def ensure_checkout(
    pin: GitPin,
    destination: Path,
    run: CommandRunner = subprocess.run,
    reference: Path | None = None,
) -> None:
    """Create or advance a clean checkout to an exact detached revision."""
    _reject_symlink_components(
        destination,
        ConfigurationError,
        "checkout destination must be a real directory;",
    )
    destination_text = str(destination)
    if os.path.lexists(destination):
        if not stat.S_ISDIR(destination.lstat().st_mode):
            raise ConfigurationError(
                f"checkout destination must be a real directory: {destination}"
            )
        status = _capture(
            [
                "git",
                "-C",
                destination_text,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            run,
        )
        if status:
            raise ConfigurationError(
                f"checkout {destination} has uncommitted changes"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone = ["git", "clone", "--no-checkout"]
        if reference is not None:
            clone.extend(
                ["--reference-if-able", str(reference), "--dissociate"]
            )
        clone.extend([pin.repository, destination_text])
        _checked(clone, run)

    actual_repository = _capture(
        ["git", "-C", destination_text, "remote", "get-url", "origin"], run
    ).strip()
    if actual_repository != pin.repository:
        raise ConfigurationError(
            f"checkout {destination} origin mismatch: "
            f"expected {pin.repository}, actual {actual_repository}"
        )

    _checked(
        ["git", "-C", destination_text, "fetch", "origin", pin.revision], run
    )
    _checked(
        [
            "git",
            "-C",
            destination_text,
            "checkout",
            "--detach",
            pin.revision,
        ],
        run,
    )
    actual_revision = _capture(
        ["git", "-C", destination_text, "rev-parse", "HEAD"], run
    ).strip()
    if actual_revision != pin.revision:
        raise ConfigurationError(
            f"checkout {destination} revision mismatch: "
            f"expected {pin.revision}, actual {actual_revision}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_components(
    path: Path,
    error_type: type[RKLLMProjectError],
    label: str,
) -> None:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute_path.anchor)
    for part in absolute_path.parts[1:]:
        current = current.parent if part == ".." else current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise error_type(f"cannot inspect {label} {current}: {error}") from error
        if stat.S_ISLNK(mode):
            raise error_type(
                f"{label} {path} contains symlink component {current}"
            )


def _real_directory(
    path: Path,
    error_type: type[RKLLMProjectError],
    label: str,
) -> Path:
    _reject_symlink_components(path, error_type, label)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise error_type(
            f"{label} must be a real directory: {path}: {error}"
        ) from error
    if not stat.S_ISDIR(mode):
        raise error_type(f"{label} must be a real directory: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise error_type(f"cannot resolve {label} {path}: {error}") from error


def _pinned_path(root: Path, resolved_root: Path, pin: DigestPin) -> Path:
    path = root / pin.path
    try:
        _reject_symlink_components(path, ArtifactError, "runtime file path")
    except ArtifactError as error:
        raise ArtifactError(
            f"runtime file {path}: expected {pin.sha256}, "
            f"actual <symlink component>: {error}"
        ) from error
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ArtifactError(f"cannot resolve runtime file {path}: {error}") from error
    if not resolved_path.is_relative_to(resolved_root):
        raise ArtifactError(
            f"runtime file {path} resolves outside declared root "
            f"{root}: {resolved_path}"
        )
    return path


def _verify_digest(root: Path, pin: DigestPin) -> None:
    resolved_root = _real_directory(root, ArtifactError, "runtime root")
    path = _pinned_path(root, resolved_root, pin)
    if not os.path.lexists(path):
        raise ArtifactError(
            f"runtime file {path}: expected {pin.sha256}, actual <missing>"
        )
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(
            f"runtime file {path}: expected {pin.sha256}, "
            "actual <not a regular non-symlink file>"
        )
    try:
        actual = _sha256(path)
    except OSError as error:
        raise ArtifactError(
            f"runtime file {path}: expected {pin.sha256}, "
            f"actual <unreadable: {error}>"
        ) from error
    if actual != pin.sha256:
        raise ArtifactError(
            f"runtime file {path}: expected {pin.sha256}, actual {actual}"
        )


def adopt_runtime_files(
    source_root: Path, destination: Path, files: tuple[DigestPin, ...]
) -> None:
    """Verify and atomically adopt only the pinned runtime files."""
    if not source_root.is_absolute():
        raise ConfigurationError(
            "RKNN3_RUNTIME_DEV_ROOT must be an absolute path"
        )

    _real_directory(source_root, ArtifactError, "runtime source root")
    for pin in files:
        _verify_digest(source_root, pin)

    if os.path.lexists(destination):
        _real_directory(destination, ArtifactError, "runtime destination root")
        for pin in files:
            _verify_digest(destination, pin)
        return

    _reject_symlink_components(
        destination.parent,
        ConfigurationError,
        "runtime destination parent",
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ConfigurationError(
            f"cannot create runtime destination parent {destination.parent}: {error}"
        ) from error
    _real_directory(
        destination.parent,
        ConfigurationError,
        "runtime destination parent",
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=str(destination.parent)
        )
    )
    try:
        for pin in files:
            source_path = source_root / pin.path
            temporary_path = temporary / pin.path
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, temporary_path)
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
        for pin in files:
            _verify_digest(temporary, pin)
        if os.path.lexists(destination):
            raise ArtifactError(
                f"runtime destination appeared during adoption: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if os.path.lexists(temporary):
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-llm-host-bootstrap")
    parser.add_argument("--project-root", required=True, type=_absolute_path)
    parser.add_argument("--upstream-manifest", required=True, type=_absolute_path)
    parser.add_argument("--runtime-dev-root", required=True, type=_absolute_path)
    parser.add_argument("--seed-workspace", type=_absolute_path)
    return parser


def _reference(seed_workspace: Path | None, name: str) -> Path | None:
    if seed_workspace is None:
        return None
    candidate = seed_workspace / name
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_upstream_manifest(args.upstream_manifest)
    vendor = args.project_root / ".vendor"
    ensure_checkout(
        manifest.toolkit,
        vendor / "rknn3-toolkit",
        reference=_reference(args.seed_workspace, "rknn3-toolkit"),
    )
    ensure_checkout(
        manifest.model_zoo,
        vendor / "rknn3-model-zoo",
        reference=_reference(args.seed_workspace, "rknn3-model-zoo"),
    )
    adopt_runtime_files(
        args.runtime_dev_root,
        vendor / "rknn3-runtime",
        manifest.runtime.files,
    )
    print(f"Bootstrapped RKNN3 dependencies in {vendor}")
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RKLLMProjectError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    entrypoint()
