from pathlib import Path

import pytest
import yaml

from rk_llm.manifests.loader import load_model_manifest, load_upstream_manifest


VALID_REVISION = "a" * 40
VALID_DIGEST = "b" * 64


def _model_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "demo",
        "repository": "owner/model",
        "revision": VALID_REVISION,
        "platform": "rk1820",
        "source_root": "models/demo",
        "generated_root": "generated/demo",
        "source_files": [
            {"path": "model.bin", "size": 1, "sha256": VALID_DIGEST}
        ],
        "generated_files": [],
    }


def _with_demo(data: dict[str, object]) -> dict[str, object]:
    data.update(
        {
            "demo_root": "model-zoo/install/rknn_Demo",
            "demo_name": "rknn_Demo",
            "demo_files": [
                {"path": "demo", "size": 1, "sha256": VALID_DIGEST},
                {"path": "lib/runtime.so", "size": 1, "sha256": VALID_DIGEST},
            ],
        }
    )
    return data


def _upstream_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "rknn3_toolkit": {
            "repository": "https://example.com/toolkit.git",
            "release": "V1.0.4",
            "revision": VALID_REVISION,
        },
        "rknn3_model_zoo": {
            "repository": "https://example.com/model-zoo.git",
            "release": "V1.0.4",
            "revision": VALID_REVISION,
        },
        "runtime": {
            "version": "1.0.4",
            "files": [{"path": "include/api.h", "sha256": VALID_DIGEST}],
        },
        "target": {
            "host_soc": "rk3588",
            "accelerator": "rk1828",
            "compiler_platform": "rk1820",
            "architecture": "aarch64",
            "glibc_max": "2.35",
            "glibcxx_max": "3.4.30",
        },
    }


def _write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_repository_manifests_pin_verified_versions() -> None:
    upstream = load_upstream_manifest(Path("manifests/upstream.yaml"))
    model = load_model_manifest(Path("configs/models/qwen2_5_0_5b.yaml"))
    assert upstream.toolkit.repository == (
        "https://github.com/airockchip/rknn3-toolkit.git"
    )
    assert upstream.toolkit.release == "V1.0.4"
    assert upstream.toolkit.revision == "cf292045d77c9ad0377b9fb326f216967475071e"
    assert upstream.model_zoo.repository == (
        "https://github.com/airockchip/rknn3-model-zoo.git"
    )
    assert upstream.model_zoo.release == "V1.0.4"
    assert upstream.model_zoo.revision == "f63048265b49bd2c6236790087287bed6c6b76fe"
    assert upstream.runtime.version == "1.0.4"
    assert [(str(pin.path), pin.sha256) for pin in upstream.runtime.files] == [
        (
            "include/float16.h",
            "6e230c07bbcfd0ea75c64d44d1b07ed3e549a88d4bb6908b4b9941d4a04fb424",
        ),
        (
            "include/rknn3_api.h",
            "64202b613bb87c6445499cb871e7227f02e9af4720e30750344477ae6e87c16d",
        ),
        (
            "Linux/aarch64/librknn3_api.so",
            "113ec97719e04f82e51fcb8badeb18461070ac55ca9a5da87f887f3110b4fcbe",
        ),
        (
            "Linux/aarch64/librknn3_api_rkcp.so",
            "5ea77749f44be1f0c2ad0347242d4b431d3907d03eac11d265496ddd80cfd210",
        ),
        (
            "Linux/aarch64/librknn3_api_native.so",
            "8ec78e9d294e6ecf2be6ad9e16004ae5c50bcb9a8567d8bac7310ab27b66dd11",
        ),
    ]
    assert upstream.target.host_soc == "rk3588"
    assert upstream.target.accelerator == "rk1828"
    assert upstream.target.compiler_platform == "rk1820"
    assert upstream.target.architecture == "aarch64"
    assert upstream.target.glibc_max == "2.35"
    assert upstream.target.glibcxx_max == "3.4.30"
    assert model.model_id == "qwen2_5_0_5b"
    assert model.repository == "Qwen/Qwen2.5-0.5B-Instruct"
    assert model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert model.platform == "rk1820"
    assert model.source_root == Path("models/Qwen2.5-0.5B-Instruct")
    assert model.generated_root == Path(
        "rknn3-model-zoo/examples/Qwen2_5/model/llm"
    )
    assert [
        (str(pin.path), pin.size, pin.sha256) for pin in model.source_files
    ] == [
        (
            ".gitattributes",
            1519,
            "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        ),
        (
            "LICENSE",
            11343,
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        ),
        (
            "README.md",
            4917,
            "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15",
        ),
        (
            "config.json",
            659,
            "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        ),
        (
            "generation_config.json",
            242,
            "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
        ),
        (
            "merges.txt",
            1671839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        (
            "model.safetensors",
            988097824,
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        ),
        (
            "tokenizer.json",
            7031645,
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        (
            "tokenizer_config.json",
            7305,
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        ),
        (
            "vocab.json",
            2776833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ]
    assert [
        (str(pin.path), pin.size, pin.sha256) for pin in model.generated_files
    ] == [
        (
            "Qwen2.5-0.5B-Instruct.onnx",
            754920840,
            "e31d74a2d5f4dbd52bf9d733eee292b2fd8b162a920a8ea92ea17463d2af7586",
        ),
        (
            "Qwen2.5-0.5B-Instruct.config.pkl",
            5067,
            "af4b89296afeea4cfe8072c55b5dba5319bdd4c9ff9733a4c51cb8fa00105e9b",
        ),
        (
            "Qwen2.5-0.5B-Instruct.tokenizer.gguf",
            5931031,
            "f2c2188ff62a9eae426fe1902405a99745cde3144443bd5298435a541560c4ee",
        ),
        (
            "Qwen2.5-0.5B-Instruct.embed.bin",
            272269312,
            "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0",
        ),
        (
            "Qwen2.5-0.5B-Instruct.rknn",
            17939072,
            "013dd8c92fa7c08feaac9b3fd9c6dc8370b5913589bb5ba8d2d7c61a8552ee6a",
        ),
        (
            "Qwen2.5-0.5B-Instruct.weight",
            333308416,
            "94bbef9ec8eb5eee08473105af3d88bcce062283db763adba15804d03b7e40f8",
        ),
    ]


def test_repository_qwen3_manifest_pins_vendor_demo_inputs() -> None:
    model = load_model_manifest(Path("configs/models/qwen3_4b.yaml"))

    assert model.model_id == "qwen3_4b"
    assert model.repository == "Qwen/Qwen3-4B"
    assert model.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert model.platform == "rk1820"
    assert model.source_root == Path("models/Qwen3-4B")
    assert model.generated_root == Path(
        "rknn3-model-zoo/examples/Qwen3/model/llm"
    )
    assert model.demo_root == Path(
        "rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo"
    )
    assert model.demo_name == "rknn_Qwen3_demo"
    assert len(model.source_files) == 12
    assert len(model.generated_files) == 6
    assert len(model.demo_files) == 9
    assert model.source_files[5].path == Path(
        "model-00001-of-00003.safetensors"
    )
    assert (
        model.source_files[5].sha256
        == "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"
    )
    assert model.demo_files[1].path == Path("rknn_qwen3_demo")
    assert (
        model.demo_files[1].sha256
        == "8418947bd24b948c9778fd3f87f439fb046dfa90c19bf24aa09b32118438fb56"
    )


@pytest.mark.parametrize("missing", ["demo_root", "demo_name", "demo_files"])
def test_model_manifest_requires_complete_demo_declaration(
    tmp_path: Path, missing: str
) -> None:
    data = _with_demo(_model_data())
    data.pop(missing)
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="demo_root, demo_name, and demo_files"):
        load_model_manifest(path)


@pytest.mark.parametrize("demo_name", ["../demo", "a/b", ".", "..", ""])
def test_model_manifest_rejects_unsafe_demo_name(
    tmp_path: Path, demo_name: str
) -> None:
    data = _with_demo(_model_data())
    data["demo_name"] = demo_name
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="demo_name must be a safe path component"):
        load_model_manifest(path)


def test_model_manifest_rejects_empty_demo_file_list(tmp_path: Path) -> None:
    data = _with_demo(_model_data())
    data["demo_files"] = []
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="demo_files must not be empty"):
        load_model_manifest(path)


@pytest.mark.parametrize("field", ["source_files", "generated_files", "demo_files"])
def test_model_manifest_rejects_duplicate_file_paths(
    tmp_path: Path, field: str
) -> None:
    data = _with_demo(_model_data())
    data[field] = [
        {"path": "same.bin", "size": 1, "sha256": VALID_DIGEST},
        {"path": "same.bin", "size": 1, "sha256": "c" * 64},
    ]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match=f"{field} contains duplicate path"):
        load_model_manifest(path)


@pytest.mark.parametrize("field", ["source_files", "generated_files", "demo_files"])
def test_model_manifest_rejects_ancestor_overlapping_file_paths(
    tmp_path: Path, field: str
) -> None:
    data = _with_demo(_model_data())
    data[field] = [
        {"path": "lib", "size": 1, "sha256": VALID_DIGEST},
        {"path": "lib/runtime.so", "size": 1, "sha256": "c" * 64},
    ]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match=f"{field} contains overlapping paths"):
        load_model_manifest(path)


def test_manifest_rejects_non_sha256_digest(tmp_path: Path) -> None:
    data = _model_data()
    data["source_files"] = [{"path": "model.bin", "size": 1, "sha256": "short"}]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="sha256"):
        load_model_manifest(path)


@pytest.mark.parametrize("schema_version", [True, 2, "1", None])
def test_model_manifest_requires_schema_version_one(
    tmp_path: Path, schema_version: object
) -> None:
    data = _model_data()
    data["schema_version"] = schema_version
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="schema_version"):
        load_model_manifest(path)


def test_model_manifest_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "bad.yaml", ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_model_manifest(path)


def test_model_manifest_wraps_malformed_yaml_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed-model-manifest.yaml"
    path.write_text("schema_version: [1\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_model_manifest(path)

    assert f"failed to parse manifest {path}" in str(error.value)


def test_upstream_manifest_rejects_malformed_nested_mapping(tmp_path: Path) -> None:
    data = _upstream_data()
    data["rknn3_toolkit"] = []
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="rknn3_toolkit must be a mapping"):
        load_upstream_manifest(path)


@pytest.mark.parametrize("revision", ["abc", "A" * 40, 40, None])
def test_model_manifest_rejects_invalid_git_revision(
    tmp_path: Path, revision: object
) -> None:
    data = _model_data()
    data["revision"] = revision
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="revision"):
        load_model_manifest(path)


@pytest.mark.parametrize("revision", ["abc", "A" * 40, 40, None])
def test_upstream_manifest_rejects_invalid_git_revision(
    tmp_path: Path, revision: object
) -> None:
    data = _upstream_data()
    toolkit = data["rknn3_toolkit"]
    assert isinstance(toolkit, dict)
    toolkit["revision"] = revision
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="rknn3_toolkit.revision"):
        load_upstream_manifest(path)


@pytest.mark.parametrize("value", ["", 7, None])
def test_model_manifest_requires_non_empty_strings(
    tmp_path: Path, value: object
) -> None:
    data = _model_data()
    data["repository"] = value
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="repository must be a non-empty string"):
        load_model_manifest(path)


def test_model_manifest_requires_file_lists(tmp_path: Path) -> None:
    data = _model_data()
    data["source_files"] = {"path": "model.bin"}
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="source_files must be a list"):
        load_model_manifest(path)


def test_model_manifest_requires_mapping_file_entries(tmp_path: Path) -> None:
    data = _model_data()
    data["source_files"] = ["model.bin"]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match=r"source_files\[0\] must be a mapping"):
        load_model_manifest(path)


@pytest.mark.parametrize(
    ("field", "unsafe_path"),
    [
        ("source_root", "/models/demo"),
        ("generated_root", "generated/../demo"),
    ],
)
def test_model_manifest_rejects_unsafe_root_paths(
    tmp_path: Path, field: str, unsafe_path: str
) -> None:
    data = _model_data()
    data[field] = unsafe_path
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="safe relative path"):
        load_model_manifest(path)


def test_model_manifest_rejects_parent_traversing_file_path(tmp_path: Path) -> None:
    data = _model_data()
    data["source_files"] = [
        {"path": "../model.bin", "size": 1, "sha256": VALID_DIGEST}
    ]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="safe relative path"):
        load_model_manifest(path)


def test_upstream_manifest_rejects_absolute_runtime_file_path(
    tmp_path: Path,
) -> None:
    data = _upstream_data()
    runtime = data["runtime"]
    assert isinstance(runtime, dict)
    runtime["files"] = [{"path": "/include/api.h", "sha256": VALID_DIGEST}]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="safe relative path"):
        load_upstream_manifest(path)


@pytest.mark.parametrize("size", [True, 0, -1, 1.5, "1"])
def test_model_manifest_requires_positive_integer_sizes(
    tmp_path: Path, size: object
) -> None:
    data = _model_data()
    data["source_files"] = [
        {"path": "model.bin", "size": size, "sha256": VALID_DIGEST}
    ]
    path = _write_yaml(tmp_path / "bad.yaml", data)

    with pytest.raises(ValueError, match="size must be a positive integer"):
        load_model_manifest(path)
