"""Import pinned model assets from an existing RKNN3 workspace."""

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from rk_llm.errors import ArtifactError, ConfigurationError, ProjectError
from rk_llm.manifests.loader import (
    ModelManifest,
    SizedFilePin,
    load_model_manifest,
)


_MODES = ("copy", "hardlink")
_RENAME_NOREPLACE = 1
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW


def _identity(details: os.stat_result) -> tuple[int, int]:
    return details.st_dev, details.st_ino


def _reject_symlink_components(
    path: Path,
    error_type: type[ProjectError],
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
            raise error_type(
                f"cannot inspect {label} {current}: {error}"
            ) from error
        if stat.S_ISLNK(mode):
            raise error_type(
                f"{label} {path} contains symlink component {current}"
            )


def _resolved_path(
    path: Path,
    error_type: type[ProjectError],
    label: str,
    *,
    directory: bool = False,
) -> Path:
    _reject_symlink_components(path, error_type, label)
    if directory:
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise error_type(
                f"{label} must be a real directory: {path}: {error}"
            ) from error
        if not stat.S_ISDIR(mode):
            raise error_type(f"{label} must be a real directory: {path}")
    try:
        return path.resolve(strict=directory)
    except (OSError, RuntimeError) as error:
        raise error_type(f"cannot resolve {label} {path}: {error}") from error


def _open_child_directory(
    parent_fd: int,
    name: str,
    current: Path,
    create: bool,
    error_type: type[ProjectError],
    label: str,
) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise error_type(f"{label} must be a real directory: {current}")
        try:
            os.mkdir(name, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as error:
            raise error_type(f"cannot create {label} {current}: {error}") from error
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as error:
            raise error_type(f"cannot open {label} {current}: {error}") from error
    except OSError as error:
        try:
            mode = os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode
        except OSError:
            mode = 0
        if stat.S_ISLNK(mode):
            raise error_type(f"{label} contains symlink component {current}") from error
        raise error_type(f"cannot open {label} {current}: {error}") from error


def _open_directory(
    path: Path,
    *,
    create: bool,
    error_type: type[ProjectError],
    label: str,
) -> int:
    normalized = Path(os.path.normpath(path))
    if not normalized.is_absolute():
        raise error_type(f"{label} must be absolute: {path}")
    descriptor = os.open(normalized.anchor, _DIRECTORY_FLAGS)
    current = Path(normalized.anchor)
    try:
        for part in normalized.parts[1:]:
            current /= part
            child = _open_child_directory(
                descriptor,
                part,
                current,
                create,
                error_type,
                label,
            )
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _assert_directory_mapping(path: Path, expected_fd: int, label: str) -> None:
    try:
        current_fd = _open_directory(
            path,
            create=False,
            error_type=ArtifactError,
            label=label,
        )
    except ArtifactError as error:
        raise ArtifactError(
            f"{label} changed during import: {path}: {error}"
        ) from error
    try:
        expected = os.fstat(expected_fd)
        current = os.fstat(current_fd)
    finally:
        os.close(current_fd)
    if _identity(expected) != _identity(current):
        raise ArtifactError(f"{label} changed during import: {path}")


def _matching_entry_name(
    parent_fd: int,
    preferred_name: str,
    expected_identity: tuple[int, int],
    *,
    directory: bool,
) -> str | None:
    try:
        with os.scandir(parent_fd) as entries:
            names = [entry.name for entry in entries]
    except OSError:
        return None
    if preferred_name in names:
        names.remove(preferred_name)
        names.insert(0, preferred_name)
    for name in names:
        try:
            details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            continue
        if _identity(details) != expected_identity:
            continue
        if directory != stat.S_ISDIR(details.st_mode):
            continue
        if not directory and not stat.S_ISREG(details.st_mode):
            continue
        return name
    return None


def _unlink_owned_file(
    parent_fd: int,
    preferred_name: str,
    expected_identity: tuple[int, int],
) -> bool:
    name = _matching_entry_name(
        parent_fd,
        preferred_name,
        expected_identity,
        directory=False,
    )
    if name is None:
        return False
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(current) != expected_identity or not stat.S_ISREG(
            current.st_mode
        ):
            return False
        os.unlink(name, dir_fd=parent_fd)
    except OSError:
        return False
    return True


def _remove_owned_directory(
    parent_fd: int,
    preferred_name: str,
    expected_identity: tuple[int, int],
) -> bool:
    name = _matching_entry_name(
        parent_fd,
        preferred_name,
        expected_identity,
        directory=True,
    )
    if name is None:
        return False
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        if _identity(os.fstat(directory_fd)) != expected_identity:
            return False
    finally:
        os.close(directory_fd)
    name = _matching_entry_name(
        parent_fd,
        name,
        expected_identity,
        directory=True,
    )
    if name is None:
        return False
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _identity(current) != expected_identity
        ):
            return False
        shutil.rmtree(Path(f"/proc/self/fd/{parent_fd}/{name}"))
    except OSError:
        return False
    return True


def _validate_directory_components(
    path: Path,
    error_type: type[ProjectError],
    label: str,
) -> None:
    normalized = Path(os.path.normpath(path))
    descriptor = os.open(normalized.anchor, _DIRECTORY_FLAGS)
    current = Path(normalized.anchor)
    try:
        for part in normalized.parts[1:]:
            current /= part
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                return
            except OSError as error:
                try:
                    mode = os.stat(
                        part, dir_fd=descriptor, follow_symlinks=False
                    ).st_mode
                except OSError:
                    mode = 0
                if stat.S_ISLNK(mode):
                    raise error_type(
                        f"{label} contains symlink component {current}"
                    ) from error
                raise error_type(f"cannot open {label} {current}: {error}") from error
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _open_relative_parent(
    root_fd: int,
    relative: Path,
    *,
    create: bool,
    error_type: type[ProjectError],
    label: str,
) -> tuple[int, str]:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise error_type(f"{label} must be a safe relative path: {relative}")
    descriptor = os.dup(root_fd)
    current = Path()
    try:
        for part in relative.parts[:-1]:
            current /= part
            child = _open_child_directory(
                descriptor,
                part,
                current,
                create,
                error_type,
                label,
            )
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, relative.parts[-1]


def _open_relative_directory(
    root_fd: int,
    relative: Path,
    *,
    create: bool,
    error_type: type[ProjectError],
    label: str,
) -> int:
    if not relative.parts:
        return os.dup(root_fd)
    parent_fd, name = _open_relative_parent(
        root_fd,
        relative,
        create=create,
        error_type=error_type,
        label=label,
    )
    try:
        return _open_child_directory(
            parent_fd,
            name,
            relative,
            create,
            error_type,
            label,
        )
    finally:
        os.close(parent_fd)


def _assert_entry_identity(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    path: Path,
    label: str,
    *,
    directory: bool,
) -> None:
    try:
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ArtifactError(f"{label} changed during import: {path}: {error}") from error
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(details.st_mode) or _identity(details) != expected_identity:
        raise ArtifactError(f"{label} changed during import: {path}")


def _verify_pinned_file_at(
    root_fd: int,
    root: Path,
    pin: SizedFilePin,
    label: str,
) -> None:
    path = root / pin.path
    expected = f"expected size {pin.size}, sha256 {pin.sha256}"
    parent_fd, name = _open_relative_parent(
        root_fd,
        pin.path,
        create=False,
        error_type=ArtifactError,
        label=label,
    )
    try:
        try:
            file_fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError as error:
            raise ArtifactError(
                f"{label} {path}: {expected}, actual <missing>"
            ) from error
        except OSError as error:
            try:
                mode = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                ).st_mode
            except OSError:
                mode = 0
            actual = "<symlink>" if stat.S_ISLNK(mode) else f"<cannot open: {error}>"
            raise ArtifactError(
                f"{label} {path}: {expected}, actual {actual}"
            ) from error
        try:
            details = os.fstat(file_fd)
            if not stat.S_ISREG(details.st_mode):
                raise ArtifactError(
                    f"{label} {path}: {expected}, actual <not a regular file>"
                )
            if details.st_size != pin.size:
                raise ArtifactError(
                    f"{label} {path}: expected size {pin.size}, "
                    f"actual {details.st_size}"
                )
            try:
                actual_sha256 = _sha256(Path(f"/proc/self/fd/{file_fd}"))
            except OSError as error:
                raise ArtifactError(
                    f"{label} {path}: expected sha256 {pin.sha256}, "
                    f"actual <cannot read: {error}>"
                ) from error
            if actual_sha256 != pin.sha256:
                raise ArtifactError(
                    f"{label} {path}: expected sha256 {pin.sha256}, "
                    f"actual {actual_sha256}"
                )
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rename_noreplace_at(
    source_dir_fd: int,
    source_name: str,
    destination_dir_fd: int,
    destination_name: str,
) -> None:
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
        source_dir_fd,
        os.fsencode(source_name),
        destination_dir_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _present_files_at(root_fd: int, root: Path) -> set[Path]:
    present: set[Path] = set()

    def scan(directory_fd: int, relative_root: Path) -> None:
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    relative = relative_root / entry.name
                    details = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(details.st_mode):
                        present.add(relative)
                        continue
                    if stat.S_ISDIR(details.st_mode):
                        child_fd = os.open(
                            entry.name,
                            _DIRECTORY_FLAGS,
                            dir_fd=directory_fd,
                        )
                        try:
                            scan(child_fd, relative)
                        finally:
                            os.close(child_fd)
                        continue
                    raise ArtifactError(
                        f"existing destination {root} contains unexpected "
                        f"non-regular file: {relative.as_posix()}"
                    )
        except ArtifactError:
            raise
        except OSError as error:
            raise ArtifactError(
                f"cannot inspect existing destination {root}: {error}"
            ) from error

    scan(root_fd, Path())
    return present


def _populate_category(
    source_root: Path,
    destination_root: Path,
    files: tuple[SizedFilePin, ...],
    mode: str,
    *,
    trusted_source_fd: int | None = None,
    trusted_destination_parent_fd: int | None = None,
    identity_out: list[tuple[int, int]] | None = None,
) -> str:
    source_fd = (
        os.dup(trusted_source_fd)
        if trusted_source_fd is not None
        else _open_directory(
            source_root,
            create=False,
            error_type=ArtifactError,
            label="source root",
        )
    )
    try:
        for pin in files:
            _verify_pinned_file_at(source_fd, source_root, pin, "source file")

        destination_parent_fd = (
            os.dup(trusted_destination_parent_fd)
            if trusted_destination_parent_fd is not None
            else _open_directory(
                destination_root.parent,
                create=True,
                error_type=ArtifactError,
                label="destination parent",
            )
        )
        temporary_name: str | None = None
        temporary_identity: tuple[int, int] | None = None
        try:
            try:
                destination_details = os.stat(
                    destination_root.name,
                    dir_fd=destination_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                destination_details = None
            except OSError as error:
                raise ArtifactError(
                    f"cannot inspect existing destination "
                    f"{destination_root}: {error}"
                ) from error

            if destination_details is not None:
                if not stat.S_ISDIR(destination_details.st_mode):
                    raise ArtifactError(
                        f"existing destination {destination_root} "
                        "must be a real directory"
                    )
                try:
                    destination_fd = os.open(
                        destination_root.name,
                        _DIRECTORY_FLAGS,
                        dir_fd=destination_parent_fd,
                    )
                except OSError as error:
                    raise ArtifactError(
                        f"cannot open existing destination "
                        f"{destination_root}: {error}"
                    ) from error
                try:
                    for pin in files:
                        _verify_pinned_file_at(
                            destination_fd,
                            destination_root,
                            pin,
                            "existing destination file",
                        )
                    expected = {pin.path for pin in files}
                    actual = _present_files_at(destination_fd, destination_root)
                    if actual != expected:
                        unexpected = sorted(
                            path.as_posix()
                            for path in actual.difference(expected)
                        )
                        missing = sorted(
                            path.as_posix()
                            for path in expected.difference(actual)
                        )
                        raise ArtifactError(
                            f"existing destination {destination_root} "
                            f"file set mismatch: unexpected {unexpected}, "
                            f"missing {missing}"
                        )
                    _assert_directory_mapping(
                        destination_root,
                        destination_fd,
                        "existing destination",
                    )
                    destination_identity = _identity(os.fstat(destination_fd))
                finally:
                    os.close(destination_fd)
                if identity_out is not None:
                    identity_out.append(destination_identity)
                return "reused"

            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination_root.name}.",
                    dir=f"/proc/self/fd/{destination_parent_fd}",
                )
            )
            temporary_name = temporary.name
            try:
                temporary_fd = os.open(
                    temporary_name,
                    _DIRECTORY_FLAGS,
                    dir_fd=destination_parent_fd,
                )
            except BaseException:
                try:
                    os.rmdir(temporary_name, dir_fd=destination_parent_fd)
                except OSError:
                    pass
                temporary_name = None
                raise
            try:
                temporary_identity = _identity(os.fstat(temporary_fd))
                for pin in files:
                    source_parent_fd, source_name = _open_relative_parent(
                        source_fd,
                        pin.path,
                        create=False,
                        error_type=ArtifactError,
                        label="source file parent",
                    )
                    try:
                        temporary_parent_fd, temporary_file_name = (
                            _open_relative_parent(
                                temporary_fd,
                                pin.path,
                                create=True,
                                error_type=ArtifactError,
                                label="temporary import parent",
                            )
                        )
                        try:
                            if mode == "copy":
                                source_file_fd = os.open(
                                    source_name,
                                    _FILE_FLAGS,
                                    dir_fd=source_parent_fd,
                                )
                                try:
                                    shutil.copy2(
                                        Path(f"/proc/self/fd/{source_file_fd}"),
                                        Path(
                                            f"/proc/self/fd/"
                                            f"{temporary_parent_fd}/"
                                            f"{temporary_file_name}"
                                        ),
                                    )
                                finally:
                                    os.close(source_file_fd)
                            else:
                                os.link(
                                    source_name,
                                    temporary_file_name,
                                    src_dir_fd=source_parent_fd,
                                    dst_dir_fd=temporary_parent_fd,
                                    follow_symlinks=False,
                                )
                        finally:
                            os.close(temporary_parent_fd)
                    finally:
                        os.close(source_parent_fd)
                    _verify_pinned_file_at(
                        temporary_fd,
                        temporary,
                        pin,
                        "imported file",
                    )
            finally:
                os.close(temporary_fd)

            _assert_directory_mapping(
                destination_root.parent,
                destination_parent_fd,
                "destination parent",
            )
            try:
                _rename_noreplace_at(
                    destination_parent_fd,
                    temporary_name,
                    destination_parent_fd,
                    destination_root.name,
                )
            except FileExistsError as error:
                raise ArtifactError(
                    f"existing destination appeared during import: "
                    f"{destination_root}"
                ) from error
            except OSError as error:
                raise ArtifactError(
                    f"cannot atomically publish imported destination "
                    f"{destination_root}: {error}"
                ) from error
            try:
                _assert_directory_mapping(
                    destination_root.parent,
                    destination_parent_fd,
                    "destination parent",
                )
                published_fd = os.open(
                    destination_root.name,
                    _DIRECTORY_FLAGS,
                    dir_fd=destination_parent_fd,
                )
                try:
                    if _identity(os.fstat(published_fd)) != temporary_identity:
                        raise ArtifactError(
                            f"imported destination changed during import: "
                            f"{destination_root}"
                        )
                    _assert_directory_mapping(
                        destination_root,
                        published_fd,
                        "imported destination",
                    )
                finally:
                    os.close(published_fd)
            except BaseException:
                try:
                    if temporary_identity is not None:
                        _remove_owned_directory(
                            destination_parent_fd,
                            destination_root.name,
                            temporary_identity,
                        )
                except BaseException:
                    pass
                temporary_name = None
                raise
            temporary_name = None
            if identity_out is not None and temporary_identity is not None:
                identity_out.append(temporary_identity)
            return "imported"
        except BaseException:
            if temporary_name is not None:
                try:
                    if temporary_identity is not None:
                        _remove_owned_directory(
                            destination_parent_fd,
                            temporary_name,
                            temporary_identity,
                        )
                except BaseException:
                    pass
            raise
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_fd)


def _categories(
    model: ModelManifest,
) -> tuple[tuple[str, tuple[SizedFilePin, ...]], ...]:
    categories = (
        ("source", model.source_files),
        ("generated", model.generated_files),
    )
    demo_declared = (
        model.demo_root is not None,
        model.demo_name is not None,
        bool(model.demo_files),
    )
    if any(demo_declared) and not all(demo_declared):
        raise ConfigurationError(
            "demo_root, demo_name, and nonempty demo_files must be declared together"
        )
    if all(demo_declared):
        return (*categories, ("demo", model.demo_files))
    return categories


def _source_relative_root(model: ModelManifest, category: str) -> Path:
    if category == "source":
        return model.source_root
    if category == "generated":
        return model.generated_root
    if category == "demo" and model.demo_root is not None:
        return model.demo_root
    raise ConfigurationError(f"unsupported import category: {category}")


def _source_root(
    workspace: Path, model: ModelManifest, category: str
) -> Path:
    return workspace / _source_relative_root(model, category)


def _destination(
    project_root: Path,
    model: ModelManifest,
    category: str,
    path: Path,
) -> Path:
    if category == "source":
        return project_root / "artifacts/source_models" / model.model_id / path
    if category == "generated":
        return project_root / "artifacts/work" / model.model_id / "model" / path
    if category == "demo" and model.demo_name is not None:
        return (
            project_root
            / "artifacts/work"
            / model.model_id
            / "install"
            / model.demo_name
            / path
        )
    raise ConfigurationError(f"unsupported import category: {category}")


def _destination_root(
    project_root: Path, model: ModelManifest, category: str
) -> Path:
    return _destination(project_root, model, category, Path())


def _preflight_paths(
    workspace: Path,
    project_root: Path,
    model: ModelManifest,
    categories: tuple[tuple[str, tuple[SizedFilePin, ...]], ...],
) -> None:
    resolved_workspace = _resolved_path(
        workspace,
        ArtifactError,
        "source workspace",
        directory=True,
    )
    for category, files in categories:
        source_root = _source_root(workspace, model, category)
        resolved_source_root = _resolved_path(
            source_root,
            ArtifactError,
            f"{category} source root",
            directory=True,
        )
        if not resolved_source_root.is_relative_to(resolved_workspace):
            raise ArtifactError(
                f"{category} source root {source_root} resolves outside "
                f"workspace {workspace}: {resolved_source_root}"
            )
        for pin in files:
            source_path = source_root / pin.path
            resolved_source_path = _resolved_path(
                source_path,
                ArtifactError,
                f"{category} source file",
            )
            if not resolved_source_path.is_relative_to(resolved_source_root):
                raise ArtifactError(
                    f"{category} source file {source_path} resolves outside "
                    f"declared source root {source_root}: {resolved_source_path}"
                )

    resolved_project_root = _resolved_path(
        project_root,
        ConfigurationError,
        "project root",
    )
    if os.path.lexists(project_root):
        try:
            project_mode = project_root.lstat().st_mode
        except OSError as error:
            raise ConfigurationError(
                f"cannot inspect project root {project_root}: {error}"
            ) from error
        if not stat.S_ISDIR(project_mode):
            raise ConfigurationError(
                f"project root must be a real directory: {project_root}"
            )

    destination_roots = [
        _destination_root(project_root, model, category)
        for category, _ in categories
    ]
    for destination_root in destination_roots:
        resolved_destination = _resolved_path(
            destination_root,
            ConfigurationError,
            "destination path",
        )
        if not resolved_destination.is_relative_to(resolved_project_root):
            raise ConfigurationError(
                f"destination path {destination_root} resolves outside "
                f"project root {project_root}: {resolved_destination}"
            )
        if os.path.lexists(destination_root):
            try:
                destination_mode = destination_root.lstat().st_mode
            except OSError as error:
                raise ArtifactError(
                    f"cannot inspect existing destination "
                    f"{destination_root}: {error}"
                ) from error
            if not stat.S_ISDIR(destination_mode):
                raise ArtifactError(
                    f"existing destination {destination_root} "
                    "must be a real directory"
                )

    destination_paths = [
        _destination(project_root, model, category, pin.path)
        for category, files in categories
        for pin in files
    ]
    destination_paths.append(
        project_root
        / "artifacts/work"
        / model.model_id
        / "import-record.json"
    )
    for destination_path in destination_paths:
        resolved_destination = _resolved_path(
            destination_path,
            ConfigurationError,
            "destination path",
        )
        if not resolved_destination.is_relative_to(resolved_project_root):
            raise ConfigurationError(
                f"destination path {destination_path} resolves outside "
                f"project root {project_root}: {resolved_destination}"
            )


def _write_record(
    path: Path,
    record: dict[str, object],
    *,
    trusted_parent_fd: int | None = None,
) -> tuple[int, int]:
    parent_fd = (
        os.dup(trusted_parent_fd)
        if trusted_parent_fd is not None
        else _open_directory(
            path.parent,
            create=True,
            error_type=ConfigurationError,
            label="import record parent",
        )
    )
    try:
        _assert_directory_mapping(
            path.parent,
            parent_fd,
            "import record parent",
        )
        try:
            existing = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as error:
            raise ArtifactError(
                f"cannot inspect import record {path}: {error}"
            ) from error
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ConfigurationError(
                f"import record path must not be a symlink: {path}"
            )

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=f"/proc/self/fd/{parent_fd}",
        )
        temporary_name = Path(temporary_path).name
        try:
            temporary_identity = _identity(os.fstat(descriptor))
        except BaseException:
            os.close(descriptor)
            raise
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    record,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                existing = os.stat(
                    path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and stat.S_ISLNK(existing.st_mode):
                raise ConfigurationError(
                    f"import record path must not be a symlink: {path}"
                )
            _assert_directory_mapping(
                path.parent,
                parent_fd,
                "import record parent",
            )
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            installed = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(installed.st_mode)
                or _identity(installed) != temporary_identity
            ):
                raise ArtifactError(
                    f"import record changed during import: {path}"
                )
            _assert_directory_mapping(
                path.parent,
                parent_fd,
                "import record parent",
            )
            return temporary_identity
        except BaseException:
            try:
                _unlink_owned_file(
                    parent_fd,
                    temporary_name,
                    temporary_identity,
                )
            except BaseException:
                pass
            raise
    finally:
        os.close(parent_fd)


def import_existing(
    workspace: Path,
    project_root: Path,
    model: ModelManifest,
    mode: str = "copy",
) -> dict[str, object]:
    """Verify and atomically import one model's pinned local artifacts."""
    if not workspace.is_absolute():
        raise ConfigurationError("workspace must be an absolute path")
    if not project_root.is_absolute():
        raise ConfigurationError("project_root must be an absolute path")
    if mode not in _MODES:
        raise ConfigurationError("import mode must be copy or hardlink")
    model_id = model.model_id
    if (
        not isinstance(model_id, str)
        or not model_id
        or model_id in {".", ".."}
        or "/" in model_id
        or "\\" in model_id
        or "\0" in model_id
        or Path(model_id).parts != (model_id,)
    ):
        raise ConfigurationError(
            "model_id must be a nonempty safe single path component: "
            f"{model_id!r}"
        )

    categories = _categories(model)
    if model.demo_name is not None and (
        model.demo_name in {".", ".."}
        or "/" in model.demo_name
        or "\\" in model.demo_name
        or "\0" in model.demo_name
        or Path(model.demo_name).parts != (model.demo_name,)
    ):
        raise ConfigurationError(
            "demo_name must be a nonempty safe single path component: "
            f"{model.demo_name!r}"
        )
    _preflight_paths(workspace, project_root, model, categories)
    for category, _ in categories:
        _validate_directory_components(
            _destination_root(project_root, model, category),
            ConfigurationError,
            "destination path",
        )
    _validate_directory_components(
        project_root / "artifacts/work" / model.model_id,
        ConfigurationError,
        "import record parent",
    )
    records = [
        {
            "category": category,
            "path": pin.path.as_posix(),
            "size": pin.size,
            "sha256": pin.sha256,
        }
        for category, files in categories
        for pin in files
    ]
    workspace_fd = _open_directory(
        workspace,
        create=False,
        error_type=ArtifactError,
        label="source workspace",
    )
    source_fds: dict[str, int] = {}
    project_fd: int | None = None
    artifacts_fd: int | None = None
    try:
        for category, _ in categories:
            relative_root = _source_relative_root(model, category)
            source_fds[category] = _open_relative_directory(
                workspace_fd,
                relative_root,
                create=False,
                error_type=ArtifactError,
                label=f"{category} source root",
            )

        def assert_source_roots() -> None:
            _assert_directory_mapping(
                workspace,
                workspace_fd,
                "source workspace",
            )
            for category, _ in categories:
                _assert_directory_mapping(
                    _source_root(workspace, model, category),
                    source_fds[category],
                    f"{category} source root",
                )

        assert_source_roots()
        for category, files in categories:
            root = _source_root(workspace, model, category)
            for pin in files:
                _verify_pinned_file_at(
                    source_fds[category],
                    root,
                    pin,
                    f"{category} source file",
                )
        assert_source_roots()

        project_fd = _open_directory(
            project_root,
            create=True,
            error_type=ConfigurationError,
            label="project root",
        )
        artifacts_fd = _open_child_directory(
            project_fd,
            "artifacts",
            project_root / "artifacts",
            True,
            ConfigurationError,
            "artifacts root",
        )

        def assert_import_roots() -> None:
            assert_source_roots()
            if project_fd is None or artifacts_fd is None:
                raise AssertionError("import root descriptors are unavailable")
            _assert_directory_mapping(project_root, project_fd, "project root")
            _assert_directory_mapping(
                project_root / "artifacts",
                artifacts_fd,
                "artifacts root",
            )

        assert_import_roots()
        statuses: dict[str, str] = {}
        destination_identities: dict[str, tuple[int, int]] = {}
        for category, files in categories:
            assert_import_roots()
            destination_root = _destination_root(
                project_root,
                model,
                category,
            )
            relative_destination = destination_root.relative_to(
                project_root / "artifacts"
            )
            destination_parent_fd, destination_name = _open_relative_parent(
                artifacts_fd,
                relative_destination,
                create=True,
                error_type=ArtifactError,
                label="destination parent",
            )
            identity: list[tuple[int, int]] = []
            try:
                statuses[category] = _populate_category(
                    _source_root(workspace, model, category),
                    destination_root,
                    files,
                    mode,
                    trusted_source_fd=source_fds[category],
                    trusted_destination_parent_fd=destination_parent_fd,
                    identity_out=identity,
                )
            finally:
                os.close(destination_parent_fd)
            if len(identity) != 1:
                raise ArtifactError(
                    f"cannot establish imported destination identity: "
                    f"{destination_root}"
                )
            destination_identities[category] = identity[0]
            assert_import_roots()

        record: dict[str, object] = {
            "schema_version": 1,
            "model_id": model.model_id,
            "source_workspace": str(workspace),
            "mode": mode,
            "statuses": statuses,
            "files": records,
        }
        record_path = (
            project_root
            / "artifacts/work"
            / model.model_id
            / "import-record.json"
        )
        relative_record = record_path.relative_to(project_root / "artifacts")
        record_parent_fd, record_name = _open_relative_parent(
            artifacts_fd,
            relative_record,
            create=True,
            error_type=ArtifactError,
            label="import record parent",
        )
        try:
            record_identity = _write_record(
                record_path,
                record,
                trusted_parent_fd=record_parent_fd,
            )
            assert_import_roots()
            for category, _ in categories:
                destination_root = _destination_root(
                    project_root,
                    model,
                    category,
                )
                relative_destination = destination_root.relative_to(
                    project_root / "artifacts"
                )
                parent_fd, name = _open_relative_parent(
                    artifacts_fd,
                    relative_destination,
                    create=False,
                    error_type=ArtifactError,
                    label="imported destination",
                )
                try:
                    _assert_entry_identity(
                        parent_fd,
                        name,
                        destination_identities[category],
                        destination_root,
                        "imported destination",
                        directory=True,
                    )
                finally:
                    os.close(parent_fd)
            _assert_entry_identity(
                record_parent_fd,
                record_name,
                record_identity,
                record_path,
                "import record",
                directory=False,
            )
            _assert_directory_mapping(
                record_path.parent,
                record_parent_fd,
                "import record parent",
            )
            assert_import_roots()
        finally:
            os.close(record_parent_fd)
        return record
    finally:
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        if project_fd is not None:
            os.close(project_fd)
        for descriptor in source_fds.values():
            os.close(descriptor)
        os.close(workspace_fd)


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-llm-host-import-existing")
    parser.add_argument("--project-root", required=True, type=_absolute_path)
    parser.add_argument("--workspace", required=True, type=_absolute_path)
    parser.add_argument("--model-manifest", required=True, type=_absolute_path)
    parser.add_argument("--mode", choices=_MODES, default="copy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = load_model_manifest(args.model_manifest)
    record = import_existing(args.workspace, args.project_root, model, args.mode)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ProjectError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    entrypoint()
