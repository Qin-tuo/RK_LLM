# RKNN3 Unified Host-to-Board Workflow Design

## Purpose

将已经在 `/home/barry/rk1828-work` 跑通的 Qwen2.5-0.5B RKNN3 模型流程纳入
`RK_LLM`，使同一个 GitHub 仓库同时管理：

- x86 宿主机上的模型下载、量化、RKNN3 编译、ARM64 Runner 构建、打包和部署；
- RK3588 上的 Python 业务逻辑、C++ RKNN3 Runner、运行诊断和硬件测试；
- 两端共用的模型配置、第三方版本、产物清单、哈希规则、文档和命令入口。

“统一项目”指所有操作都从 `RK_LLM` 仓库根目录通过项目命令执行。它不意味着把
第三方仓库、虚拟环境、源模型和约 610 MB 的部署产物提交到普通 Git 历史。

## Product Direction

本设计取代 `2026-08-14-rk-llm-skeleton-design.md` 中以 RKLLM 1.3.0、DeepSeek 和
`.rkllm` 为目标的产品方向。现有骨架中与厂商后端无关的部分继续保留：

- Python `src` 布局、CLI 和配置加载；
- backend protocol、mock backend 和 generation service；
- benchmark、metrics、错误类型和非硬件测试；
- Python 进程与 Native Runner 之间的逐行 JSON 协议边界。

以下内容被替换：

- `rkllm` backend 改为 `rknn3` backend；
- DeepSeek/RKLLM 配置和文档改为 Qwen2.5/RKNN3；
- `native/rkllm_runner` 改为 `native/rknn3_qwen_runner`；
- RKLLM Toolkit/Runtime 版本清单改为 RKNN3 Toolkit、Model Zoo、Runtime 和固件清单。

## First Milestone Scope

首个里程碑只支持：

- 源模型：`Qwen/Qwen2.5-0.5B-Instruct`；
- 源模型 revision：`7ae557604adf67be50417f59c2c2f167def9a775`；
- RKNN3 Toolkit：`V1.0.4`，commit
  `cf292045d77c9ad0377b9fb326f216967475071e`；
- RKNN3 Model Zoo：`V1.0.4`，commit
  `f63048265b49bd2c6236790087287bed6c6b76fe`；
- 编译器目标：`rk1820`；
- 运行拓扑：RK3588 主机通过 PCIe 调用 RK1828；
- RKNN3 API、传输服务和 RK1828 固件：`1.0.4`；
- RK3588 系统基线：Ubuntu 22.04、glibc 2.35；
- 输入输出：纯文本 prompt 和流式文本生成。

目录和配置为后续模型扩展保留边界，但第一版不实现通用多模型框架。

## Non-Goals

第一阶段不包含：

- Git LFS、GitHub Release 或普通 Git 中的模型和部署包分发；
- 自动升级端侧 glibc、驱动、RKNN3 Runtime、传输服务或 RK1828 固件；
- 多模态输入、多模型选择、HTTP 服务、ROS 2 或任务规划；
- GitHub CI 中执行模型下载、量化、RKNN3 编译或真实硬件测试；
- 把完整 Toolkit、Model Zoo 或厂商二进制复制进本项目源码；
- 在未通过 RK3588 硬件测试时宣称端侧推理完成。

## System Architecture

```text
                         GitHub: Qin-tuo/RK_LLM
              source, config, manifests, tests, documentation
                            /                 \
                     clone/pull/push      clone/pull/push
                          /                     \
                x86 development host          RK3588 + RK1828
                --------------------          ---------------
                tools/host                    Python application
                   |                              |
                .vendor (ignored)              rknn3 backend
                   |                              |
                source model                  C++ Qwen runner
                   |                              |
                export + GRQ                  RKNN3 Runtime
                   |                              |
                RKNN3 compile                 PCIe -> RK1828
                   |
                Ubuntu 22.04 ARM64 build
                   |
                versioned package
                   |
                   +---------- SSH/rsync ---------->
                             artifacts/deploy (ignored)
```

GitHub 和 SSH/rsync 是两条不同的数据通道：

- GitHub 只同步项目源码和小型元数据；
- SSH/rsync 从宿主机传输经过验证的大型部署包；
- 两条通道都由 `RK_LLM` 中的命令管理；
- Git 更新不得删除、覆盖或提交端侧 `artifacts/deploy/` 中的部署包。

## Repository Layout

```text
RK_LLM/
|-- Makefile
|-- pyproject.toml
|-- configs/
|   |-- models/
|   |   `-- qwen2_5_0_5b.yaml
|   |-- runtime/
|   |   |-- mock.yaml
|   |   `-- rk3588.yaml
|   `-- benchmark/
|-- manifests/
|   |-- upstream.yaml
|   `-- schemas/
|       `-- deployment-package.schema.json
|-- src/rk_llm/
|   |-- cli.py
|   |-- config.py
|   |-- generation/
|   |-- backends/
|   |   |-- base.py
|   |   |-- mock.py
|   |   `-- rknn3.py
|   |-- platform/
|   |   `-- probe.py
|   `-- metrics/
|-- native/rknn3_qwen_runner/
|   |-- CMakeLists.txt
|   |-- include/
|   `-- src/
|-- tools/
|   |-- host/
|   |   |-- bootstrap
|   |   |-- build_model
|   |   |-- build_runner
|   |   |-- package
|   |   `-- deploy
|   |-- board/
|   |   |-- setup
|   |   |-- verify
|   |   |-- activate
|   |   `-- rollback
|   `-- benchmark/
|-- docker/
|   `-- arm64-builder/
|       `-- Dockerfile
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- hardware/
|-- docs/
|-- artifacts/                 # ignored except README
|   |-- source_models/
|   |-- work/
|   |-- packages/
|   |-- deploy/
|   `-- logs/
`-- .vendor/                   # ignored
    |-- rknn3-toolkit/
    |-- rknn3-model-zoo/
    `-- rknn3-runtime/
```

Each file or directory has one owner:

| Area | Primary responsibility | Normal editing environment |
| --- | --- | --- |
| `tools/host/`, `docker/` | Download, export, compile, package, deploy | x86 host |
| `src/rk_llm/` | CLI, business orchestration, process lifecycle | RK3588 or x86 tests |
| `native/rknn3_qwen_runner/` | Tokenizer and RKNN3 inference integration | RK3588 logic; cross-built on host |
| `tools/board/` | Verification, activation, rollback | RK3588 |
| `configs/`, `manifests/` | Shared versioned contracts | Both |
| `artifacts/`, `.vendor/` | Local/generated state | Never committed |

Both machines may create commits because their normal ownership areas differ. Before pushing, each machine
must incorporate the current remote branch with a fast-forward pull or rebase. A non-fast-forward push or
merge conflict is an explicit stop condition; automation must not force-push or auto-resolve it.

## External Dependency Strategy

The repository uses pinned bootstrap downloads rather than Git submodules or copied upstream source.

`manifests/upstream.yaml` records repository URLs, exact commits, release labels, source model revision,
expected Runtime/API version and verified checksums. `tools/host/bootstrap`:

1. clones Toolkit and Model Zoo into `.vendor/` when absent;
2. fetches and checks out the exact commit when present;
3. fails if a vendor checkout has uncommitted changes instead of deleting them;
4. creates project-local Python environments outside Git;
5. locates the separately supplied RKNN3 Runtime development files through the absolute
   `RKNN3_RUNTIME_DEV_ROOT` path;
6. verifies Runtime headers and libraries against recorded hashes before exposing them to the build.

The bootstrap process never modifies system drivers or endpoint Runtime installations. Missing licensed or
vendor-supplied files produce an actionable error naming the required release and expected location.

## Host Workflow

All commands run from the `RK_LLM` root. `Makefile` is a stable user-facing facade; focused tools beneath
`tools/host/` implement the stages and can be tested independently.

### Bootstrap

```text
make host-bootstrap
```

Bootstrap validates host architecture, disk space, Docker availability, Python version, CUDA visibility,
pinned upstream checkouts and Runtime development files. It does not download the source model or start a
long conversion.

### Model Build

```text
make host-build MODEL=qwen2_5_0_5b
```

The build command performs the validated sequence:

1. download the exact Hugging Face revision into `artifacts/source_models/`;
2. verify source file sizes and hashes;
3. export GRQ model, ONNX, config, tokenizer and embedding data through the pinned Model Zoo scripts;
4. compile the RKNN graph and weight with `--platform rk1820`;
5. reload the produced RKNN model with Toolkit 1.0.4;
6. save logs and intermediate state under `artifacts/work/` and `artifacts/logs/`.

Stages are resumable only when their recorded inputs, arguments and outputs match. A mere existing filename
is not sufficient for reuse.

### ARM64 Runner Build

```text
make host-runner MODEL=qwen2_5_0_5b
```

The runner is cross-compiled in the pinned Ubuntu 22.04 container. The build consumes project-owned C++
source plus verified external RKNN3 headers and libraries. It must scan the executable and every packaged
shared library, rejecting any GLIBC requirement newer than 2.35 or GLIBCXX requirement newer than 3.4.30.
Board verification also compares these requirements with the symbols exported by the board's actual libc and
libstdc++ rather than trusting the declared OS version alone.

### Package

```text
make host-package MODEL=qwen2_5_0_5b
```

Packaging is read-only with respect to successful build outputs. It creates a new immutable directory under
`artifacts/packages/` and never edits an existing package in place.

### Deploy

```text
make deploy MODEL=qwen2_5_0_5b BOARD=ubuntu@<RK3588_IP>
```

Deploy transfers the package by SSH/rsync and invokes tracked board-side verification and activation tools.
It does not perform `git pull` for the board, does not install a driver and does not upgrade system packages.

## Deployment Package Contract

```text
qwen2.5-0.5b-rk1828-<package-id>/
|-- manifest.json
|-- bin/
|   `-- rknn_qwen_runner
|-- lib/
|   |-- librga.so
|   |-- librknn3_api.so
|   `-- librknn3_api_rkcp.so
`-- model/
    |-- Qwen2.5-0.5B-Instruct.rknn
    |-- Qwen2.5-0.5B-Instruct.weight
    |-- Qwen2.5-0.5B-Instruct.tokenizer.gguf
    `-- Qwen2.5-0.5B-Instruct.embed.bin
```

`package-id` is the first 16 hexadecimal characters of the SHA-256 over canonical UTF-8 JSON containing all
manifest fields except `package_id` and `created_at`. Canonical JSON sorts object keys and uses no insignificant
whitespace. `created_at` is retained for audit but does not change an otherwise identical package ID.
`manifest.json` contains:

- manifest schema version and package ID;
- source model repository, revision and source hashes;
- Toolkit, Model Zoo, Runtime/API and firmware versions;
- export, quantization, RKNN compilation and CMake arguments;
- target topology and required board architecture;
- builder OS, toolchain and container identity;
- relative path, byte size and SHA-256 for every packaged file;
- highest required GLIBC and GLIBCXX symbol version for every ELF;
- Git commit of the project-owned Runner source.

The manifest uses only relative paths. Validators reject absolute paths, `..` traversal, duplicate entries,
missing files, extra undeclared deployment files and hashes that do not match.

## Board Workflow

The default board clone lives at `/home/ubuntu/RK_LLM`. An alternative absolute root may be provided through
`RK_LLM_ROOT`; relative or empty roots are rejected.

### Source Synchronization

```text
git pull --ff-only
```

Git synchronizes board business logic, C++ source, configuration, tests and board tools. It does not
synchronize models or change the active package.

### Setup and Diagnostics

```text
make board-setup
make board-doctor MODEL=qwen2_5_0_5b
```

Setup installs only project Python dependencies. Doctor checks aarch64 architecture, Ubuntu/glibc baseline,
package manifest, file hashes, ELF dependencies, RKNN3 Runtime visibility, transport service and RK1828
status. Every unavailable prerequisite is reported separately.

### Inference

```text
rk-llm generate --backend rknn3 \
  --config configs/runtime/rk3588.yaml \
  --prompt "hello"
```

The Python layer owns request validation, application flow, process lifecycle, streaming output, errors and
metrics. The C++ runner owns tokenizer loading, model handles, RKNN3 calls and orderly resource release.

Changing Python business logic can be tested directly on RK3588 and committed. Changing C++ Runner logic
requires the host to rebuild and deploy a package whose manifest records the new project commit.

## Python-to-Native Protocol

Python launches the active package's Runner with the active manifest path. Communication is newline-delimited
JSON with protocol version `1`:

- Runner to Python: `ready`, `chunk`, `complete`, `error`;
- Python to Runner: `generate`, `shutdown`;
- every generation event carries a request ID;
- generated text is contained in JSON values, never inferred from logs;
- Runner diagnostics go to stderr;
- malformed JSON, unknown event types, request ID mismatch, timeout or unexpected process exit are fatal;
- the `rknn3` backend never falls back to `mock`.

The first implementation processes one generation request at a time. Concurrency and long-running service
pooling are deferred until the single-request hardware path passes.

## Safe Deployment and Rollback

On the board, ignored deployment state is organized as:

```text
artifacts/deploy/
|-- .incoming/
|-- releases/
|   |-- <package-id-a>/
|   `-- <package-id-b>/
|-- current -> releases/<package-id-b>
`-- previous -> releases/<package-id-a>
```

A deployment transaction is:

1. upload to a fresh `.incoming/<package-id>` directory;
2. validate manifest schema, declared files, hashes, architecture and symbol versions;
3. check Runtime, transport service and RK1828 status;
4. start the candidate Runner, load the model and run a fixed smoke prompt;
5. move the verified candidate into `releases/<package-id>`;
6. atomically update `previous` and `current` symlinks;
7. report the active package ID and smoke-test result.

Any failure before activation returns non-zero and leaves `current` unchanged. Interrupted uploads remain
outside `releases/`. `tools/board/rollback` atomically activates `previous` after revalidating it. Cleanup is
explicit and never removes `current` or `previous`.

## Error Handling

- Configuration and manifest errors name the field and rejected value without exposing credentials.
- Missing vendor dependencies name the expected release, hash and explicit configuration variable.
- Host subprocess failures preserve the command stage, exit code and log path.
- A GLIBC requirement above 2.35 blocks packaging, not merely deployment.
- SSH failure, checksum failure and board health failure are distinct exit conditions.
- Runner startup, model load, protocol and inference failures remain distinct project exceptions.
- Signals and Python exceptions trigger Runner shutdown; forced termination is a bounded fallback.
- No command deletes an existing external workspace or overwrites the installed board Runtime.

## Testing Strategy

### GitHub CI

CI runs without vendor SDKs, source models or hardware:

- Python unit tests for configuration, manifest validation, package IDs, path safety and error mapping;
- mock backend generation and benchmark tests;
- fake Runner protocol tests for ready, streaming, completion, error, timeout and malformed records;
- repository layout and `.gitignore` tests;
- CLI integration tests that prove `rknn3` never falls back to mock.

### Host Integration

Opt-in host tests cover pinned checkout resolution, resumable stage decisions, source hash validation,
packaging, all-ELF GLIBC checks and deploy dry-run behavior. External commands are represented by recorded
fixtures in normal tests; full model conversion remains explicit.

### Ubuntu 22.04 Container

An explicit heavy test rebuilds the ARM64 Runner, verifies the AArch64 ELF interpreter, scans all GLIBC and
GLIBCXX requirements and constructs a package from existing verified model outputs.

### RK3588 Hardware

Opt-in hardware tests require explicit environment variables and verify:

- board architecture and Runtime/firmware prerequisites;
- active manifest and all file hashes;
- RK1828 visibility;
- candidate Runner startup and model loading;
- non-empty generation for a fixed prompt;
- clean shutdown and real timing data;
- rollback to the prior verified package.

Hardware tests must not run automatically merely because a device exists.

## Migration of the Existing Workspace

The first implementation is non-destructive:

1. do not move or delete `/home/barry/rk1828-work`;
2. add the new ignored `.vendor/` and `artifacts/` layout inside `RK_LLM`;
3. import or reuse verified existing inputs through an explicit migration command;
4. record source and generated artifact hashes before and after import;
5. execute the new package validator and Ubuntu 22.04 Runner build;
6. keep the old workspace until the new project performs a successful RK3588 smoke test and rollback test.

The migration command must not make an external directory writable through hidden symlinks. Any reuse by
copy, reflink or hardlink is explicit in its output, and package validation treats the result as project-local
state.

## Acceptance Criteria

The unified milestone is complete when all of the following are true:

1. A fresh host clone can run project-root commands to bootstrap the exact upstream versions.
2. The host can build or safely adopt the verified Qwen2.5-0.5B artifacts without manual `cd` into Model Zoo.
3. The Runner builds in Ubuntu 22.04 and no deployment ELF requires GLIBC newer than 2.35 or GLIBCXX newer
   than 3.4.30; board verification also checks the actual installed libraries.
4. Packaging produces a schema-valid immutable bundle whose package ID and hashes are reproducible.
5. Git tracks no source model, converted model, vendor checkout, virtual environment or deployment package.
6. RK3588 can clone/pull the same repository without affecting its active ignored deployment package.
7. Host deployment uploads a candidate, verifies it on RK3588, runs a hardware smoke prompt and activates it
   only on success.
8. `rk-llm doctor --backend rknn3` reports real board prerequisites and the active package ID.
9. `rk-llm generate --backend rknn3 ...` returns non-empty streamed text through the Python/C++ protocol.
10. A failed candidate leaves the current package running, and rollback restores the previous verified package.
11. Non-hardware tests pass on x86 without vendor dependencies; hardware tests remain explicitly opt-in.
12. Host-owned and board-owned source changes can be committed and synchronized through the same GitHub
    repository without committing generated artifacts.

## Delivery Sequence

Implementation is divided into testable milestones:

1. repository contracts, version manifest, ignore rules and artifact schema;
2. host bootstrap and non-destructive import of the verified existing workspace;
3. host model build orchestration and resumable stage records;
4. project-owned C++ RKNN3 Qwen Runner and Ubuntu 22.04 cross-build;
5. immutable package creation and all-ELF compatibility validation;
6. board verification, candidate activation and rollback;
7. Python `rknn3` backend and versioned Native protocol;
8. actual RK3588 + RK1828 smoke test and benchmark documentation.

Each milestone leaves the x86 mock and non-hardware tests passing. Hardware completion is claimed only after
the corresponding tracked command and opt-in hardware test pass on the intended device.
