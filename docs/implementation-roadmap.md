# RK_LLM 实现路线图

这份文档只描述 RK_LLM 从当前骨架走到 RK3588 真机推理的大致推进流程。它不是逐行开发手册；每进入一个新阶段，再针对该阶段补充具体命令和实现细节。

## 最终目标

```text
原始大模型
  -> RKLLM-Toolkit 转换和量化
  -> .rkllm 模型
  -> RK3588 上的 C++ runner
  -> RKLLM Runtime
  -> RK3588 NPU
  -> Python RKLLMBackend
  -> rk-llm CLI 输出纯文本
```

第一版固定使用：

- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 模型 revision：`ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- 目标板：RK3588
- RKLLM-Toolkit：`1.3.0`
- RKLLM Runtime：`1.3.0`
- 量化方式：W8A8
- 功能范围：单轮、纯文本、流式输出

版本的唯一记录位置是 [`third_party/versions.yaml`](../third_party/versions.yaml)。在完成第一条真机链路之前，不同时更换模型、Toolkit、Runtime 或量化方式。

## 阶段 1：跑通现有 Mock

### 要做什么

在开发电脑上安装项目，运行 mock doctor、文本生成、benchmark 和非硬件测试。

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements/dev.txt
rk-llm doctor --backend mock
rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
rk-llm benchmark --backend mock --config configs/benchmark/smoke.yaml \
  --output artifacts/benchmark_runs/mock.jsonl
python3 -m pytest -m "not hardware"
```

### 为什么

这一阶段先证明 CLI、配置、Backend 接口、流式输出和 benchmark 框架正常。后面接入 RKLLM 时，如果出现问题，可以判断是新接入的问题，而不是上层骨架本身的问题。

### 完成标志

- `doctor` 报告 `available: true` 和 `is_mock: true`。
- `generate` 输出 `mock: hello`。
- benchmark 生成合法 JSONL。
- 非硬件测试全部通过。

## 阶段 2：在普通电脑运行原始模型

### 要做什么

在具备足够内存的 CPU/GPU 主机上，用原始 Hugging Face 模型和它自己的 tokenizer 跑通参考推理。确认 chat template、特殊 token、停止 token 和采样参数，并保存几组固定 prompt 与输出。

这一阶段使用原始模型，不使用 `.rkllm`，也不涉及 RK3588。

### 为什么

需要先证明模型文件、tokenizer 和输入格式正确。否则模型转换后出现乱码、无法停止或回答异常时，无法判断问题来自原始模型还是 RKLLM 转换。

### 完成标志

- 固定 revision 的模型可以重复加载。
- 中文和英文 prompt 都能生成合理文本。
- 已记录实际使用的 chat template、EOS token 和采样参数。
- 已保存一小组后续用于对比的参考输入和输出。

## 阶段 3：转换为 RKLLM 模型

### 要做什么

在独立的模型转换主机环境中安装 RKLLM-Toolkit `1.3.0`，按照官方 `airockchip/rknn-llm` 对应版本的示例，将固定 revision 的模型转换成面向 RK3588 的 W8A8 `.rkllm` 文件。

转换完成后同时保存：

- `.rkllm` 模型文件。
- 完整转换日志。
- 模型来源和 revision。
- Toolkit 版本、目标平台和量化参数。
- 文件大小和 SHA-256。
- 转换主机环境信息。

产物放在 `artifacts/converted_models/` 或外部模型仓库，不提交到 Git。详细约束见 [`model-export.md`](model-export.md)。

### 为什么

RK3588 的 NPU 不能直接运行 Hugging Face checkpoint。RKLLM-Toolkit 需要完成模型图转换、算子处理和量化，生成 RKLLM Runtime 能加载的格式。

### 完成标志

- 转换进程正常结束，日志中没有未处理错误。
- `.rkllm` 文件非空且 SHA-256 已记录。
- 任何人能根据记录确认它来自哪个模型、哪个版本和哪些转换参数。

## 阶段 4：使用官方 Demo 在 RK3588 推理

### 要做什么

准备 RK3588 的系统镜像、内核、NPU 驱动、固件和 RKLLM Runtime `1.3.0`。先编译并运行官方 C++ 文本模型 demo，加载阶段 3 生成的 `.rkllm`，执行阶段 2 保存的参考 prompt。

此时不要先调试本项目的 Python Backend，也不要先修改本项目 C++ runner。

### 为什么

官方 demo 是板卡环境门禁。它可以单独验证：

- NPU 驱动是否工作。
- Runtime 动态库是否匹配。
- `.rkllm` 是否能加载。
- RKLLM 回调是否返回文本。

如果官方 demo 都不能运行，问题通常不在本项目的 Python 或进程协议中。

### 完成标志

- 官方 demo 能加载模型并正常退出。
- 中文和英文 prompt 都能返回可读文本。
- 多次重复运行没有崩溃或持续增长的内存占用。
- 已记录板卡镜像、内核、驱动、固件和 Runtime 版本。

## 阶段 5：接入当前 RK_LLM 项目

### 要做什么

首先把 [`native/rkllm_runner`](../native/rkllm_runner/) 的不可用 stub 替换成真正的 C++ RKLLM 适配器，集中管理模型初始化、推理回调、终止和销毁。

然后建立 Python 与 runner 之间的逐行 JSON 通信：

```text
Python 发送请求
  -> C++ runner 调用 RKLLM Runtime
  -> 回调产生文本片段
  -> runner 输出 chunk
  -> Python RKLLMBackend 转换成 TextChunk
  -> GenerationService 流式输出
```

最后让 [`RKLLMBackend`](../src/rk_llm/backends/rkllm.py) 启动并管理 runner。真实后端失败时必须返回明确错误，不能退回 MockBackend。

### 为什么

C++ 层负责厂商 C API、句柄和回调；Python 层负责配置、CLI、结果组织和 benchmark。这样厂商接口变化被限制在 native 边界内。

### 完成标志

- `rk-llm doctor --backend rkllm` 在正确环境中报告可用。
- `rk-llm generate --backend rkllm ...` 输出真实模型文本。
- 文本片段可以流式返回。
- runner 异常、模型错误和 Runtime 错误会明确传回 Python。
- mock 与非硬件测试仍然通过。

## 阶段 6：测试、稳定和优化

### 要做什么

在功能正确后，再处理模型常驻、KV Cache、超时、abort 和重复请求。使用真实 RK3588 benchmark 记录：

- 模型加载时间。
- 首 token 延迟 TTFT。
- Decode tokens/s。
- 输入和输出 token 数。
- 内存峰值。
- 温度、散热和功耗模式。
- 短 prompt、长 prompt 和重复运行稳定性。

Mock benchmark 与 RK3588 benchmark 必须分开，真实结果必须携带完整板卡和软件环境说明。详细要求见 [`benchmark.md`](benchmark.md)。

### 为什么

先保证正确，再优化性能。否则模型加载、文本质量、Runtime 兼容和性能问题会混在一起，难以定位。

### 完成标志

- 模型可以常驻并处理重复请求。
- 超时或中止后仍可安全释放或重建 runner。
- 参考 prompt 的结果没有明显转换异常。
- benchmark 可以重复执行并生成带环境说明的结果。
- 已形成可以重新部署的 runner、Runtime、模型、manifest 和配置组合。

## 推进原则

始终按以下顺序排查：

```text
原始模型是否正确
  -> 转换是否正确
  -> 官方 Demo 是否正确
  -> 自定义 C++ runner 是否正确
  -> Python Backend 是否正确
  -> 性能是否达标
```

不要跳过官方 demo，不要同时升级多个版本，不要把 Mock 结果当成真机结果，也不要在真实后端失败时自动回退到 Mock。

当前应从阶段 1 开始。阶段 1 完成并理解现有调用链后，再为阶段 2 编写具体操作说明。
