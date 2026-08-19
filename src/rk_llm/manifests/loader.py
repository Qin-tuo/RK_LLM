from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True)
class GitPin:
    repository: str
    release: str
    revision: str


@dataclass(frozen=True)
class DigestPin:
    path: Path
    sha256: str


@dataclass(frozen=True)
class RuntimePin:
    version: str
    files: tuple[DigestPin, ...]


@dataclass(frozen=True)
class TargetPin:
    host_soc: str
    accelerator: str
    compiler_platform: str
    architecture: str
    glibc_max: str
    glibcxx_max: str


@dataclass(frozen=True)
class UpstreamManifest:
    toolkit: GitPin
    model_zoo: GitPin
    runtime: RuntimePin
    target: TargetPin


@dataclass(frozen=True)
class SizedFilePin:
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    repository: str
    revision: str
    platform: str
    source_root: Path
    generated_root: Path
    source_files: tuple[SizedFilePin, ...]
    generated_files: tuple[SizedFilePin, ...]
    demo_root: Path | None = None
    demo_name: str | None = None
    demo_files: tuple[SizedFilePin, ...] = ()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"failed to parse manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    return data


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _required_string(data: dict[str, Any], key: str, field: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _schema_version(data: dict[str, Any]) -> None:
    value = data.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("schema_version must be 1")


def _revision(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git revision")
    return value


def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _safe_component(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
        or Path(value).parts != (value,)
    ):
        raise ValueError(f"{field} must be a safe path component")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _git_pin(value: object, field: str) -> GitPin:
    data = _mapping(value, field)
    return GitPin(
        repository=_required_string(data, "repository", f"{field}.repository"),
        release=_required_string(data, "release", f"{field}.release"),
        revision=_revision(data.get("revision"), f"{field}.revision"),
    )


def _digest_pin(value: object, field: str) -> DigestPin:
    data = _mapping(value, field)
    return DigestPin(
        path=_relative_path(data.get("path"), f"{field}.path"),
        sha256=_digest(data.get("sha256"), f"{field}.sha256"),
    )


def _sized_file_pin(value: object, field: str) -> SizedFilePin:
    data = _mapping(value, field)
    size = data.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{field}.size must be a positive integer")
    return SizedFilePin(
        path=_relative_path(data.get("path"), f"{field}.path"),
        size=size,
        sha256=_digest(data.get("sha256"), f"{field}.sha256"),
    )


def _sized_file_pins(values: list[object], field: str) -> tuple[SizedFilePin, ...]:
    pins = tuple(
        _sized_file_pin(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    paths: list[Path] = []
    for pin in pins:
        if pin.path in paths:
            raise ValueError(f"{field} contains duplicate path: {pin.path}")
        if any(pin.path in path.parents or path in pin.path.parents for path in paths):
            raise ValueError(f"{field} contains overlapping paths: {pin.path}")
        paths.append(pin.path)
    return pins


def load_upstream_manifest(path: Path) -> UpstreamManifest:
    data = _read_yaml(path)
    _schema_version(data)

    runtime_data = _mapping(data.get("runtime"), "runtime")
    runtime_files = _sequence(runtime_data.get("files"), "runtime.files")
    target_data = _mapping(data.get("target"), "target")
    return UpstreamManifest(
        toolkit=_git_pin(data.get("rknn3_toolkit"), "rknn3_toolkit"),
        model_zoo=_git_pin(data.get("rknn3_model_zoo"), "rknn3_model_zoo"),
        runtime=RuntimePin(
            version=_required_string(runtime_data, "version", "runtime.version"),
            files=tuple(
                _digest_pin(value, f"runtime.files[{index}]")
                for index, value in enumerate(runtime_files)
            ),
        ),
        target=TargetPin(
            host_soc=_required_string(target_data, "host_soc", "target.host_soc"),
            accelerator=_required_string(
                target_data, "accelerator", "target.accelerator"
            ),
            compiler_platform=_required_string(
                target_data, "compiler_platform", "target.compiler_platform"
            ),
            architecture=_required_string(
                target_data, "architecture", "target.architecture"
            ),
            glibc_max=_required_string(target_data, "glibc_max", "target.glibc_max"),
            glibcxx_max=_required_string(
                target_data, "glibcxx_max", "target.glibcxx_max"
            ),
        ),
    )


def load_model_manifest(path: Path) -> ModelManifest:
    data = _read_yaml(path)
    _schema_version(data)

    source_files = _sequence(data.get("source_files"), "source_files")
    generated_files = _sequence(data.get("generated_files"), "generated_files")
    demo_fields = ("demo_root", "demo_name", "demo_files")
    declared_demo_fields = tuple(field in data for field in demo_fields)
    if any(declared_demo_fields) and not all(declared_demo_fields):
        raise ValueError("demo_root, demo_name, and demo_files must be declared together")

    demo_root: Path | None = None
    demo_name: str | None = None
    demo_files: tuple[SizedFilePin, ...] = ()
    if all(declared_demo_fields):
        demo_root = _relative_path(data.get("demo_root"), "demo_root")
        demo_name = _safe_component(data.get("demo_name"), "demo_name")
        raw_demo_files = _sequence(data.get("demo_files"), "demo_files")
        if not raw_demo_files:
            raise ValueError("demo_files must not be empty")
        demo_files = _sized_file_pins(raw_demo_files, "demo_files")

    return ModelManifest(
        model_id=_required_string(data, "model_id", "model_id"),
        repository=_required_string(data, "repository", "repository"),
        revision=_revision(data.get("revision"), "revision"),
        platform=_required_string(data, "platform", "platform"),
        source_root=_relative_path(data.get("source_root"), "source_root"),
        generated_root=_relative_path(data.get("generated_root"), "generated_root"),
        source_files=_sized_file_pins(source_files, "source_files"),
        generated_files=_sized_file_pins(generated_files, "generated_files"),
        demo_root=demo_root,
        demo_name=demo_name,
        demo_files=demo_files,
    )
