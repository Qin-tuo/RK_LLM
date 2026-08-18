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
2. 下载准确的 `Qwen/Qwen3-4B` 模型并检查下载完整性；
3. 使用 Qwen3 官方示例执行 GRQ 量化和 ONNX、Config、Tokenizer、Embedding 导出；
4. 使用 W4A16、GRQ、group32 和 `--platform rk1820` 编译 RKNN3 模型；
5. 重新加载 `.rknn` 与 `.weight`，确认返回码为 `0`；
6. 在 Ubuntu 22.04 Docker 中交叉编译 `Qwen3` RK3588 ARM64 Demo；
7. 检查程序架构和部署包内全部 ELF 的 glibc 版本；
8. 传输部署包，在 RK3588 上运行并使用 `rknn-smi` 监控 RK1828；
9. 给出成功标准和四类高频阻塞问题的处理入口。

## 固定参数

- Hugging Face 模型：`Qwen/Qwen3-4B`；
- 本地模型目录：`~/rk1828-work/models/Qwen3-4B`；
- Model Zoo 示例：`examples/Qwen3`；
- 输出前缀：`Qwen3-4B`；
- RKNN3 编译目标：`--platform rk1820`；
- RK3588 Demo 构建参数：`-t rk3588 -a aarch64 -d Qwen3`；
- RK1828 核掩码：`0xff`；
- 首次运行上下文：保持 Qwen3 Demo 的 `MAX_CONTEXT_LEN=1024`。

不使用 `Qwen3-4B-Base`、`Qwen3-4B-FP8`、GGUF、VL 或 2507 变体。

## 可靠性与错误处理

文档只保留会直接影响部署结果的检查：

- 转换前确认主机 RAM、Swap、磁盘和 CUDA；Qwen3-4B 转换资源不足时先停止并扩容；
- GRQ 遇到 CUDA OOM 时不把不完整输出当成成功产物；
- C++ 构建缺头文件或动态库时回到原手册的 Runtime 开发文件恢复步骤；
- Ubuntu 24.04 产生高于 glibc 2.35 的依赖时，统一改用 Ubuntu 22.04 Docker 重编译；
- 不覆盖端侧已匹配的 Runtime、传输服务或 RK1828 固件。

## 验收标准

完成部署必须同时满足：

- `.rknn`、`.weight`、`.tokenizer.gguf`、`.embed.bin` 四个文件均存在且非空；
- RKNN3 模型重新加载返回 `0`；
- Demo 是 ARM64 ELF，部署包没有高于 glibc 2.35 的依赖；
- 板端 Demo 能持续输出有意义的 token；
- `rknn-smi` 能看到 RK1828 利用率和显存占用变化；
- 日志中没有 Runtime、固件、PCIe 或版本不匹配错误。

Qwen3-4B 的实际转换耗时、产物大小、端侧显存和实测性能在首次运行后记录，文档不提前
填入未经当前设备验证的数值。
