"""Build an immutable deployment package from an imported vendor Demo."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from rk_llm.artifacts.manifest import (
    canonical_payload,
    compute_package_id,
    validate_package,
)
from rk_llm.errors import ArtifactError, ConfigurationError, ProjectError
from rk_llm.manifests.loader import (
    ModelManifest,
    SizedFilePin,
    UpstreamManifest,
    load_model_manifest,
    load_upstream_manifest,
)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_ELF_INPUTS = frozenset(
    {
        "rknn_qwen3_demo",
        "lib/librga.so",
        "lib/librknn3_api.so",
        "lib/librknn3_api_rkcp.so",
    }
)
_PACKAGE_PATHS = {
    "rknn_qwen3_demo": Path("bin/rknn_qwen3_demo"),
    "lib/librga.so": Path("lib/librga.so"),
    "lib/librknn3_api.so": Path("lib/librknn3_api.so"),
    "lib/librknn3_api_rkcp.so": Path("lib/librknn3_api_rkcp.so"),
    "model/Qwen3-4B.embed.bin": Path("model/Qwen3-4B.embed.bin"),
    "model/Qwen3-4B.rknn": Path("model/Qwen3-4B.rknn"),
    "model/Qwen3-4B.tokenizer.gguf": Path("model/Qwen3-4B.tokenizer.gguf"),
    "model/Qwen3-4B.weight": Path("model/Qwen3-4B.weight"),
}


def _checked(
    args: list[str], run: CommandRunner
) -> subprocess.CompletedProcess[str]:
    try:
        return run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = error.stderr or error.stdout or str(error)
        raise ConfigurationError(
            f"command failed ({' '.join(args)}): {details.strip()}"
        ) from error


def _version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError as error:
        raise ConfigurationError(f"invalid ABI version: {value}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_directory(path: Path, label: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ConfigurationError(f"cannot inspect {label} {current}: {error}") from error
        if stat.S_ISLNK(details.st_mode):
            raise ConfigurationError(
                f"{label} contains symlink component: {current}"
            )
    try:
        details = path.lstat()
    except OSError as error:
        raise ConfigurationError(f"{label} must be a real directory: {path}") from error
    if not stat.S_ISDIR(details.st_mode):
        raise ConfigurationError(f"{label} must be a real directory: {path}")


def _inventory(root: Path) -> set[Path]:
    present: set[Path] = set()

    def scan(directory: Path) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root)
                    details = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(details.st_mode):
                        scan(path)
                    elif stat.S_ISREG(details.st_mode):
                        present.add(relative)
                    elif stat.S_ISLNK(details.st_mode):
                        raise ArtifactError(
                            f"imported Demo contains symlink: {relative}"
                        )
                    else:
                        raise ArtifactError(
                            f"imported Demo contains unsupported entry: {relative}"
                        )
        except ArtifactError:
            raise
        except OSError as error:
            raise ArtifactError(f"cannot inspect imported Demo {root}: {error}") from error

    scan(root)
    return present


def _verify_demo(root: Path, pins: tuple[SizedFilePin, ...]) -> None:
    expected = {pin.path for pin in pins}
    actual = _inventory(root)
    if actual != expected:
        missing = sorted(path.as_posix() for path in expected - actual)
        unexpected = sorted(path.as_posix() for path in actual - expected)
        raise ArtifactError(
            "imported Demo inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for pin in pins:
        path = root / pin.path
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode):
            raise ArtifactError(f"imported Demo file is not regular: {pin.path}")
        if details.st_size != pin.size:
            raise ArtifactError(f"imported Demo size mismatch: {pin.path}")
        if _sha256(path) != pin.sha256:
            raise ArtifactError(f"imported Demo sha256 mismatch: {pin.path}")


def _elf_metadata(
    path: Path,
    readelf: str,
    target_glibc: str,
    target_glibcxx: str,
    run: CommandRunner,
) -> dict[str, str | None]:
    header = _checked([readelf, "-h", str(path)], run).stdout
    match = re.search(r"^\s*Machine:\s*(.+?)\s*$", header, re.MULTILINE)
    if match is None or match.group(1) != "AArch64":
        actual = match.group(1) if match is not None else "<missing>"
        raise ArtifactError(
            f"ELF {path.name} must target AArch64, actual {actual}"
        )

    versions = _checked([readelf, "--version-info", str(path)], run).stdout
    glibc_values = re.findall(r"\bGLIBC_([0-9]+(?:\.[0-9]+)*)", versions)
    glibcxx_values = re.findall(
        r"\bGLIBCXX_([0-9]+(?:\.[0-9]+)*)", versions
    )
    glibc = max(glibc_values, key=_version) if glibc_values else None
    glibcxx = max(glibcxx_values, key=_version) if glibcxx_values else None
    if glibc is not None and _version(glibc) > _version(target_glibc):
        raise ArtifactError(
            f"ELF {path.name} requires GLIBC_{glibc}, above GLIBC_{target_glibc}"
        )
    if glibcxx is not None and _version(glibcxx) > _version(target_glibcxx):
        raise ArtifactError(
            f"ELF {path.name} requires GLIBCXX_{glibcxx}, "
            f"above GLIBCXX_{target_glibcxx}"
        )
    return {"glibc": glibc, "glibcxx": glibcxx}


def _project_state(project_root: Path, run: CommandRunner) -> tuple[str, str]:
    status = _checked(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        run,
    ).stdout
    if status:
        raise ConfigurationError(
            f"project {project_root} has uncommitted changes"
        )
    commit = _checked(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], run
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ConfigurationError(f"invalid project Git commit: {commit!r}")
    compiler = _checked(["aarch64-linux-gnu-g++", "--version"], run).stdout
    compiler = compiler.splitlines()[0].strip() if compiler else ""
    if not compiler:
        raise ConfigurationError("cannot determine cross-compiler version")
    return commit, compiler


def _file_record(
    source: Path,
    destination: Path,
    elf: dict[str, str | None] | None,
) -> dict[str, object]:
    details = source.stat()
    return {
        "path": destination.as_posix(),
        "size": details.st_size,
        "sha256": _sha256(source),
        "elf": elf,
    }


def _manifest(
    model: ModelManifest,
    upstream: UpstreamManifest,
    project_commit: str,
    compiler: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "package_id": "0" * 16,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "package_profile": "vendor_demo",
        "entrypoint": "bin/rknn_qwen3_demo",
        "model": {
            "id": model.model_id,
            "repository": model.repository,
            "revision": model.revision,
            "source_files": [
                {
                    "path": pin.path.as_posix(),
                    "size": pin.size,
                    "sha256": pin.sha256,
                }
                for pin in model.source_files
            ],
        },
        "toolchain": {
            "project_commit": project_commit,
            "toolkit": {
                "release": upstream.toolkit.release,
                "revision": upstream.toolkit.revision,
            },
            "model_zoo": {
                "release": upstream.model_zoo.release,
                "revision": upstream.model_zoo.revision,
            },
            "runtime_version": upstream.runtime.version,
            "firmware_version": upstream.runtime.version,
            "builder": {"image": "ubuntu:22.04", "compiler": compiler},
        },
        "target": {
            "host_soc": upstream.target.host_soc,
            "accelerator": upstream.target.accelerator,
            "compiler_platform": upstream.target.compiler_platform,
            "architecture": upstream.target.architecture,
            "glibc_max": upstream.target.glibc_max,
            "glibcxx_max": upstream.target.glibcxx_max,
        },
        "build": {
            "export_args": ["--quant", "--model_path", "Qwen/Qwen3-4B"],
            "rknn_args": [
                "--platform",
                "rk1820",
                "--dataset_path",
                "datasets/CMMLU/dataset.txt",
            ],
            "cmake_args": ["-t", "rk3588", "-a", "aarch64", "-d", "Qwen3"],
        },
        "files": records,
    }
    manifest["package_id"] = compute_package_id(manifest)
    return manifest


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_noreplace(source: Path, destination: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "renameat2(RENAME_NOREPLACE) is unavailable on this host",
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    return details.st_dev, details.st_ino


def _remove_owned_staging(
    parent: Path, preferred: Path, expected: tuple[int, int]
) -> None:
    try:
        entries = list(parent.iterdir())
    except OSError:
        return
    entries.sort(key=lambda path: path != preferred)
    for candidate in entries:
        try:
            details = candidate.stat(follow_symlinks=False)
        except OSError:
            continue
        if (details.st_dev, details.st_ino) != expected:
            continue
        if not stat.S_ISDIR(details.st_mode):
            return
        shutil.rmtree(candidate, ignore_errors=True)
        return


def _same_package(package_root: Path, manifest: dict[str, object]) -> bool:
    try:
        current = validate_package(package_root)
    except ArtifactError:
        return False
    return canonical_payload(current) == canonical_payload(manifest)


def build_vendor_demo_package(
    project_root: Path,
    model: ModelManifest,
    upstream: UpstreamManifest,
    *,
    readelf: str = "aarch64-linux-gnu-readelf",
    run: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Validate an imported Qwen3 Demo and publish one immutable package."""
    project_root = Path(project_root)
    if not project_root.is_absolute():
        raise ConfigurationError("project_root must be an absolute path")
    _real_directory(project_root, "project root")
    if model.model_id != "qwen3_4b":
        raise ConfigurationError("vendor Demo packaging only supports qwen3_4b")
    if (
        model.demo_root is None
        or model.demo_name != "rknn_Qwen3_demo"
        or not model.demo_files
    ):
        raise ConfigurationError("qwen3_4b manifest has incomplete Demo fields")
    if model.platform != "rk1820" or upstream.target.compiler_platform != "rk1820":
        raise ConfigurationError("Qwen3 compiler_platform must be rk1820")
    if {pin.path.as_posix() for pin in model.demo_files} != {
        "SHA256SUMS",
        *_PACKAGE_PATHS,
    }:
        raise ConfigurationError("qwen3_4b Demo file mapping is incomplete")

    project_commit, compiler = _project_state(project_root, run)
    demo_root = (
        project_root
        / "artifacts/work"
        / model.model_id
        / "install"
        / model.demo_name
    )
    _real_directory(demo_root, "imported Demo root")
    _verify_demo(demo_root, model.demo_files)

    records: list[dict[str, object]] = []
    for source_name, destination in _PACKAGE_PATHS.items():
        source = demo_root / source_name
        elf = None
        if source_name in _ELF_INPUTS:
            elf = _elf_metadata(
                source,
                readelf,
                upstream.target.glibc_max,
                upstream.target.glibcxx_max,
                run,
            )
        records.append(_file_record(source, destination, elf))
    records.sort(key=lambda record: str(record["path"]))
    manifest = _manifest(model, upstream, project_commit, compiler, records)
    package_id = str(manifest["package_id"])
    package_parent = project_root / "artifacts/packages" / model.model_id
    package_root = package_parent / package_id

    if os.path.lexists(package_root):
        if _same_package(package_root, manifest):
            return {
                "package_id": package_id,
                "model_id": model.model_id,
                "status": "reused",
                "package_path": str(package_root),
            }
        raise ArtifactError(f"immutable package conflict: {package_root}")

    package_parent.mkdir(parents=True, exist_ok=True)
    _real_directory(package_parent, "package parent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{package_id}.staging.", dir=str(package_parent)
        )
    )
    staging.chmod(0o700)
    staging_identity = _identity(staging)
    owns_staging = True
    try:
        for source_name, destination in _PACKAGE_PATHS.items():
            source = demo_root / source_name
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            source_pin = next(
                pin for pin in model.demo_files if pin.path.as_posix() == source_name
            )
            if target.stat().st_size != source_pin.size or _sha256(target) != source_pin.sha256:
                raise ArtifactError(f"staged artifact changed during copy: {destination}")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_tree(staging)
        validate_package(staging)
        try:
            _publish_noreplace(staging, package_root)
        except FileExistsError as error:
            if _same_package(package_root, manifest):
                return {
                    "package_id": package_id,
                    "model_id": model.model_id,
                    "status": "reused",
                    "package_path": str(package_root),
                }
            raise ArtifactError(
                f"package destination appeared during publication: {package_root}"
            ) from error
        except OSError as error:
            raise ArtifactError(f"cannot publish package {package_root}: {error}") from error
        owns_staging = False
        parent_fd = os.open(package_parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if owns_staging:
            _remove_owned_staging(package_parent, staging, staging_identity)

    return {
        "package_id": package_id,
        "model_id": model.model_id,
        "status": "created",
        "package_path": str(package_root),
    }


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-llm-host-package-vendor-demo")
    parser.add_argument("--project-root", required=True, type=_absolute_path)
    parser.add_argument("--model-manifest", required=True, type=_absolute_path)
    parser.add_argument("--upstream-manifest", required=True, type=_absolute_path)
    parser.add_argument("--readelf", default="aarch64-linux-gnu-readelf")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = load_model_manifest(args.model_manifest)
    upstream = load_upstream_manifest(args.upstream_manifest)
    summary = build_vendor_demo_package(
        args.project_root,
        model,
        upstream,
        readelf=args.readelf,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ProjectError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    entrypoint()

