from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from rk_llm.errors import ArtifactError


_PACKAGE_DIRECTORIES = frozenset({"bin", "lib", "model"})
_DATE_TIME = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"[Tt](?P<hour>[01][0-9]|2[0-3]):(?P<minute>[0-5][0-9]):"
    r"(?P<second>[0-5][0-9]|60)(?:\.[0-9]+)?"
    r"(?:(?P<zulu>[Zz])|(?P<sign>[+-])"
    r"(?P<offset_hour>[01][0-9]|2[0-3]):"
    r"(?P<offset_minute>[0-5][0-9]))$"
)


def _is_date_time(value: object) -> bool:
    if not isinstance(value, str):
        return True
    match = _DATE_TIME.fullmatch(value)
    if match is None:
        return False

    second = int(match["second"])
    try:
        local_time = datetime(
            int(match["year"]),
            int(match["month"]),
            int(match["day"]),
            int(match["hour"]),
            int(match["minute"]),
            min(second, 59),
        )
    except ValueError:
        return False
    if second != 60:
        return True

    offset = timedelta()
    if match["zulu"] is None:
        offset = timedelta(
            hours=int(match["offset_hour"]),
            minutes=int(match["offset_minute"]),
        )
        if match["sign"] == "-":
            offset = -offset
    try:
        utc_time = local_time.replace(tzinfo=timezone(offset)).astimezone(
            timezone.utc
        )
    except (OverflowError, ValueError):
        return False
    return (
        utc_time.month,
        utc_time.day,
        utc_time.hour,
        utc_time.minute,
        utc_time.second,
    ) in {
        (6, 30, 23, 59, 59),
        (12, 31, 23, 59, 59),
    }


_FORMAT_CHECKER = FormatChecker()
_FORMAT_CHECKER.checks("date-time")(_is_date_time)


def _schema_path() -> Path:
    configured_root = os.environ.get("RK_LLM_ROOT")
    if configured_root is not None:
        project_root = Path(configured_root)
        if not project_root.is_absolute():
            raise ArtifactError("RK_LLM_ROOT must be an absolute path")
    else:
        project_root = Path(__file__).resolve().parents[3]
    return project_root / "manifests" / "schemas" / "deployment-package.schema.json"


def _load_schema() -> dict[str, object]:
    path = _schema_path()
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"invalid deployment schema {path}: {error}") from error
    if not isinstance(schema, dict):
        raise ArtifactError(f"invalid deployment schema {path}: root must be an object")
    return schema


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactError("artifact path must be a safe relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] not in _PACKAGE_DIRECTORIES
    ):
        raise ArtifactError(f"artifact path {value!r} must be a safe relative path")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload(manifest: Mapping[str, object]) -> bytes:
    payload = dict(manifest)
    payload.pop("package_id", None)
    payload.pop("created_at", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_package_id(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_payload(manifest)).hexdigest()[:16]


def _manifest(package_root: Path) -> dict[str, object]:
    path = package_root / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator(
            _load_schema(), format_checker=_FORMAT_CHECKER
        ).validate(manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise ArtifactError(f"invalid deployment manifest {path}: {error}") from error
    return manifest


def _assert_within_package(package_root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(package_root.resolve())
    except (OSError, ValueError) as error:
        raise ArtifactError(f"artifact file {path} escapes package root") from error


def _scan_directory(
    package_root: Path, directory: Path, regular_files: set[Path]
) -> None:
    relative_directory = directory.relative_to(package_root)
    context = relative_directory if relative_directory.parts else Path(".")
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative_path = path.relative_to(package_root)
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as error:
                    raise ArtifactError(
                        "failed to inspect package descendant "
                        f"{relative_path}: {error}"
                    ) from error
                if stat.S_ISLNK(mode):
                    raise ArtifactError(
                        "package contains an undeclared or unsafe symlink: "
                        f"{relative_path}"
                    )
                if stat.S_ISDIR(mode):
                    _scan_directory(package_root, path, regular_files)
                elif stat.S_ISREG(mode):
                    if relative_path != Path("manifest.json"):
                        regular_files.add(relative_path)
                else:
                    raise ArtifactError(
                        f"unsupported package descendant type: {relative_path}"
                    )
    except ArtifactError:
        raise
    except OSError as error:
        raise ArtifactError(
            f"failed to scan package directory {context}: {error}"
        ) from error


def _inventory(package_root: Path) -> set[Path]:
    regular_files: set[Path] = set()
    _scan_directory(package_root, package_root, regular_files)
    return regular_files


def _artifact_stat(path: Path, relative_path: Path) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ArtifactError(
            f"artifact file must be a regular file: {relative_path}"
        ) from error
    except OSError as error:
        raise ArtifactError(
            f"failed to stat artifact file {relative_path}: {error}"
        ) from error
    if stat.S_ISLNK(file_stat.st_mode):
        raise ArtifactError(f"artifact file must not be a symlink: {relative_path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ArtifactError(f"artifact file must be a regular file: {relative_path}")
    return file_stat


def validate_package(package_root: Path) -> dict[str, object]:
    """Validate a package.

    The caller must keep exclusive or stable ownership of the package tree for
    the duration of validation.
    """

    package_root = Path(package_root)
    actual = _inventory(package_root)
    manifest = _manifest(package_root)

    expected_package_id = compute_package_id(manifest)
    if manifest["package_id"] != expected_package_id:
        raise ArtifactError(
            "deployment manifest package_id does not match canonical payload"
        )

    declared: set[Path] = set()
    for record in manifest["files"]:
        relative_path = _safe_relative_path(record["path"])
        if relative_path in declared:
            raise ArtifactError(f"duplicate artifact path: {relative_path}")
        declared.add(relative_path)

        artifact_path = package_root / relative_path
        _assert_within_package(package_root, artifact_path)
        actual_size = _artifact_stat(artifact_path, relative_path).st_size
        if actual_size != record["size"]:
            raise ArtifactError(
                f"artifact size mismatch for {relative_path}: "
                f"expected {record['size']}, got {actual_size}"
            )
        try:
            actual_sha256 = _sha256(artifact_path)
        except OSError as error:
            raise ArtifactError(
                f"failed to hash artifact file {relative_path}: {error}"
            ) from error
        if actual_sha256 != record["sha256"]:
            raise ArtifactError(f"artifact sha256 mismatch for {relative_path}")

    undeclared = actual - declared
    if undeclared:
        paths = ", ".join(sorted(path.as_posix() for path in undeclared))
        raise ArtifactError(f"undeclared artifact files: {paths}")
    return manifest
