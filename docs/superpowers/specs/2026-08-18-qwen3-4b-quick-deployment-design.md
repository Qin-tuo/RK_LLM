# Qwen3-4B RK1828 快速部署文档设计

## 目标

新增 `docs/rk1828-qwen3-4b-quick-deployment.md`，为已经按
`docs/rk1828-rknn3-deployment.md` 跑通 Qwen2.5-0.5B 的环境提供一条最短、可验证的
Qwen3-4B 部署路径。原完整手册保持不变。

## 读者与前置条件

读者已经具备以下环境：

- x86 Ubuntu 转换机、Python 3.12 虚拟环境和 CUDA 可用；
- RKNN3 Toolkit 1.0.4 与配套 Model Zoo 依赖已安装；
- `rknn3-model-zoo/3rdparty/rknpu3` 中的 RK3588 ARM64 开发文件已补齐；
- RK3588 + RK1828 Runtime、固件和 PCIe 链路已经通过 Qwen2.5-0.5B 验证；
- Ubuntu 22.04 Docker 交叉编译方式可用。

文档不重复解释上述环境的完整安装过程，而是链接原手册作为前置参考。

## 文档结构

快速文档按执行顺序包含以下内容：

1. 固定模型、目录和版本变量，并检查磁盘、内存、Swap、CUDA 与 RK1828 状态；
2. 对公开模型源执行小流量测速，自动从当前最快的源下载准确的
   `Qwen/Qwen3-4B` 模型，并检查下载完整性；
3. 使用 Qwen3 官方示例执行 GRQ 量化和 ONNX、Config、Tokenizer、Embedding 导出；
4. 使用 W4A16、GRQ、group32 和 `--platform rk1820` 编译 RKNN3 模型；
5. 重新加载 `.rknn` 与 `.weight`，确认返回码为 `0`；
6. 在 Ubuntu 22.04 Docker 中交叉编译 `Qwen3` RK3588 ARM64 Demo；
7. 检查程序架构和部署包内全部 ELF 的 glibc 版本；
8. 传输部署包，在 RK3588 上运行并使用 `rknn-smi` 监控 RK1828；
9. 给出成功标准和高频阻塞问题的处理入口。

## 固定参数

- 模型仓库：`Qwen/Qwen3-4B`；
- 本地模型目录：`~/rk1828-work/models/Qwen3-4B`；
- Model Zoo 示例：`examples/Qwen3`；
- 输出前缀：`Qwen3-4B`；
- RKNN3 编译目标：`--platform rk1820`；
- RK3588 Demo 构建参数：`-t rk3588 -a aarch64 -d Qwen3`；
- RK1828 核掩码：`0xff`；
- 首次运行上下文：保持 Qwen3 Demo 的 `MAX_CONTEXT_LEN=1024`。

不使用 `Qwen3-4B-Base`、`Qwen3-4B-FP8`、GGUF、VL 或 2507 变体。

## 下载源自动选择

下载阶段使用一段可直接执行的 Bash 脚本。脚本预填
`MODEL_REPO=Qwen/Qwen3-4B` 和测速文件
`model-00001-of-00003.safetensors`；以后下载其他公开模型时，只需修改模型仓库和一个
大文件名。

脚本依次对以下三个无需登录的公开地址请求同一文件的前 1 MiB：

- Hugging Face 官方站；
- HF Mirror；
- ModelScope。

每次请求设置连接和总耗时上限，跟随重定向，并同时检查 `curl` 返回码、HTTP 状态码和
实际接收字节数。只有 HTTP 206 且实际收到完整 1 MiB 样本的响应才有效；服务器忽略
Range 并返回 HTTP 200 时立即判定该源本次测速无效，避免继续传输整个模型分片。脚本以
`curl` 的实际平均下载速度为依据，输出每个可用源的 MiB/s，并选择当前测速最快的源；
三个源全部失败时立即退出，不开始正式下载。

选中源后按以下方式下载：

- Hugging Face 官方站：执行 `hf download`；
- HF Mirror：临时设置 `HF_ENDPOINT=https://hf-mirror.com` 后执行 `hf download`；
- ModelScope：执行 `modelscope download --max-workers 4`。

脚本开始前检查 `curl`、`hf` 和 `modelscope` 命令，并检测是否已有针对同一模型仓库的
`hf` 或 `modelscope` 下载进程。缺少任一命令或发现已有下载进程时先报错退出。测速只
代表执行时的当前网络状况，不会永久缓存结果。若正式下载期间速度明显变化，可先停止当前
下载进程，再重新运行测速和下载脚本；不允许多个下载器同时写入同一模型目录。

切换来源时不删除已有文件。若从 Hugging Face 下载中断后改用 ModelScope，脚本将模型
目录内的 `.cache/huggingface` 移到模型目录外的时间戳备份目录，避免两种下载器的临时
状态相互影响，同时保留恢复所需数据。

正式下载完成后必须检查 `config.json`、`model.safetensors.index.json`、
`tokenizer.json`、`tokenizer_config.json` 和 Qwen3-4B 的三个明确分片均存在且非空，
并确认当前模型目录内没有 `.incomplete` 文件。缺少任一文件都停止后续转换。

## 可靠性与错误处理

文档只保留会直接影响部署结果的检查：

- 转换前确认主机 RAM、Swap、磁盘和 CUDA；Qwen3-4B 转换资源不足时先停止并扩容；
- 模型下载前确认三个公开源的当前速度，只允许恰好收到 1 MiB 的完整测速样本参与选择；
- 下载源均不可用、下载命令缺失或模型文件不完整时，不进入 GRQ 导出；
- 切换下载源时保留已有文件和 Hugging Face 临时状态，不执行破坏性清理；
- GRQ 遇到 CUDA OOM 时不把不完整输出当成成功产物；
- C++ 构建缺头文件或动态库时回到原手册的 Runtime 开发文件恢复步骤；
- Ubuntu 24.04 产生高于 glibc 2.35 的依赖时，统一改用 Ubuntu 22.04 Docker 重编译；
- 不覆盖端侧已匹配的 Runtime、传输服务或 RK1828 固件。

## 验收标准

完成部署必须同时满足：

- Qwen3-4B 的三个 Safetensors 分片、配置和 Tokenizer 文件均存在且非空，模型目录内
  没有 `.incomplete` 文件；
- `.rknn`、`.weight`、`.tokenizer.gguf`、`.embed.bin` 四个文件均存在且非空；
- RKNN3 模型重新加载返回 `0`；
- Demo 是 ARM64 ELF，部署包没有高于 glibc 2.35 的依赖；
- 板端 Demo 能持续输出有意义的 token；
- `rknn-smi` 能看到 RK1828 利用率和显存占用变化；
- 日志中没有 Runtime、固件、PCIe 或版本不匹配错误。

Qwen3-4B 的实际转换耗时、产物大小、端侧显存和实测性能在首次运行后记录，文档不提前
填入未经当前设备验证的数值。
