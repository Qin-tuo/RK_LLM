# RK3588 + RK1828 的 RKNN3 模型部署流程

本文记录当前设备检查结果，以及在 x86 Ubuntu 宿主机上将
`Qwen/Qwen2.5-0.5B-Instruct` 转换并部署到 RK3588 + RK1828 的完整流程。

> 状态日期：2026-08-18
>
> 当前检查点：RKNN3 Toolkit 1.0.4、Model Zoo 依赖和 RTX 5070 CUDA 环境均已验证；
> Qwen2.5-0.5B 的 GRQ 量化、ONNX/配置/Tokenizer/Embedding 导出和 RK3588 ARM64 C++
> Demo 交叉编译均已完成。Demo 已验证兼容端侧 glibc 2.35。Toolkit 1.0.4 接受
> `--platform rk1820` 和 `--platform rk1828`，但两种参数实测生成的 `.rknn/.weight` 哈希
> 完全一致，模型内均包含 `RK1820` 标识，说明该版本将两者归一到同一套 RK182X 模型格式。
> 当前模型已在 RK1828 上完整生成文本并正常释放资源。实测稳定配置为
> `Work Mode=EFFICIENT`、`Prefill Mode=PERFORMANCE`、`core_mask=0xff`；把 Work Mode 改为
> `NORMAL` 后，同一模型和同一提示词会在首次 `rknn3_session_wait` 时使设备端掉线。

## 1. 整体链路

```text
Hugging Face 模型
        |
        v
model-zoo/export_llm.py
        |
        v
ONNX + Config + Tokenizer + Embedding
        |
        v
model-zoo/export_rknn.py 调用 RKNN3 Toolkit
        |
        v
model.rknn + model.weight
        |
        v
x86 宿主机交叉编译 ARM64 C++ Demo
        |
        v
复制到 RK3588
        |
        v
C++ Demo -> RKNN3 Runtime -> PCIe -> RK1828 固件/NPU
```

各组件的职责：

| 组件 | 作用 | 运行位置 |
| --- | --- | --- |
| `rknn3-toolkit` | 模型转换、量化和 RKNN 编译 | x86 宿主机的 Python 虚拟环境 |
| `rknn3-model-zoo` | 官方导出脚本、C++ Demo 和构建脚本 | 转换在 x86，Demo 在 RK3588 |
| AutoGPTQ | Model Zoo 的通用 GPTQ 辅助依赖，不是本流程的主要量化器 | x86 宿主机的 Python 虚拟环境 |
| RKNN3 Runtime | 向应用提供 C/C++ API，通过 PCIe 调用协处理器 | RK3588 |
| RK1828 固件 | 管理 RK1828 上的实际 NPU 执行 | RK1828 |
| `RK_LLM` | 后续接入已验证推理链路的上层项目 | RK3588 |

### 1.1 流程为什么分为宿主机和端侧

Hugging Face 提供的 `model.safetensors` 是通用 PyTorch 权重，RK1828 不能直接加载。
模型必须先在资源更充足的 x86 宿主机上完成导出、量化和编译，再把生成的 RKNN3
专用模型复制到端侧。

RK3588 在运行阶段负责应用逻辑、Tokenizer 和 RKNN3 Runtime；计算任务经 PCIe 发送到
RK1828。RK1828 负责实际的 NPU 推理，但不运行宿主机上的 Python Toolkit。

完整过程按输入、处理和输出划分如下：

| 阶段 | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| 环境准备 | x86 Ubuntu | 创建 venv，安装 CUDA 版 PyTorch、Toolkit 和 Zoo | 隔离的模型转换环境 |
| 模型下载 | Hugging Face 仓库 | 下载 Qwen2.5-0.5B | `model.safetensors`、配置和 Tokenizer 文件 |
| 模型导出 | Qwen 原始模型 | `export_llm.py --quant` 导出并执行 GRQ 预量化 | ONNX、Config、Tokenizer、Embed |
| RKNN 编译 | ONNX 和 Config | RKNN3 Toolkit 按 W4A16 编译 | `.rknn` 和 `.weight` |
| Demo 编译 | Model Zoo C++ 源码 | 为 RK3588 交叉编译 ARM64 程序 | `rknn_qwen2_5_demo` |
| 端侧部署 | Demo 和四个模型文件 | 复制到 RK3588 的 `/home/ubuntu/userdata` | 可运行的完整部署包 |
| 硬件推理 | Prompt | Runtime 经 PCIe 调用 RK1828 | 持续生成的文本 token |
| 项目接入 | 已验证的官方 Demo | 将真实推理链路接入 `RK_LLM` | RK1828 硬件后端 |

量化相关组件的关系：

| 名称 | 本流程中的作用 |
| --- | --- |
| AutoGPTQ | Model Zoo 的公共依赖；关闭其可选 CUDA 扩展不会关闭 PyTorch CUDA |
| GRQ | `export_llm.py --quant` 通过 `rknn.utils.grq` 调用的主要量化算法 |
| W4A16 | 最终 RKNN3 模型的数据格式：4-bit 权重、16-bit 激活 |

模型文件在各阶段的变化为：

```text
model.safetensors
        -> Qwen2.5-0.5B-Instruct.onnx
        -> Qwen2.5-0.5B-Instruct.rknn
           Qwen2.5-0.5B-Instruct.weight
           Qwen2.5-0.5B-Instruct.tokenizer.gguf
           Qwen2.5-0.5B-Instruct.embed.bin
```

## 2. 当前硬件与软件状态

RK3588 主机：

- 板卡：CM3588 V2；
- 系统：Ubuntu 22.04；
- 内核：Linux 6.1.118；
- CPU：4 x Cortex-A76 + 4 x Cortex-A55；
- 内存：16 GB，无 Swap；
- RK3588 NPU 驱动：RKNPU `0.9.8`。

RK1828 PCIe 协处理器：

- PCIe 地址：`0000:01:00.0`；
- 产品：`RM1828MC0-F`；
- 芯片：RK1828，1 颗；
- 板载内存：5120 MB；
- 连接：PCIe Gen2 x1；
- `rknn-smi`：`1.3.0`；
- PCIe 驱动：`3.3.0`；
- RK1828 固件：`1.0.4`；
- RKNN3 API：`1.0.4`；
- 实测状态：`Online`、`Health OK`；
- 已验证运行模式：Work Mode `EFFICIENT`、Prefill Mode `PERFORMANCE`；
- Qwen2.5-0.5B Core Mask：`0xff`，必须使用全部 8 个 NPU 核。

端侧已有以下运行组件：

```text
/usr/lib/librknn3_api.so
/usr/lib/librknn3_api_rkcp.so
/usr/bin/rknn3_transfer_proxy
/usr/bin/rknn3_startup
/usr/lib/modules/pcie-rkep.ko
/lib/firmware/rknn3_rk1820.img
/usr/bin/rknn-smi
/usr/bin/pcie_upgrade_tool
/etc/systemd/system/rk182x.service
```

这些文件不属于任何 Deb 包，且时间戳一致，判断为板卡厂商镜像或厂商部署包批量安装。
`rk182x.service` 已启用并正常运行。当前 Runtime、API、固件和宿主机 Toolkit 都是
`1.0.4`，无需升级或覆盖端侧 Runtime。

## 3. x86 宿主机基础环境

宿主机为 Ubuntu 24.04，默认 Python 为 3.12。安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs python3.12 python3.12-venv python3.12-dev \
  cmake build-essential \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

git lfs install
```

Ubuntu 24.04 默认仓库没有 `python3.10-venv`，应使用 `python3.12-venv`。
如果一条 `apt install` 命令中有任意包无法定位，该次命令中的其他包也可能没有安装，
需要修正包名后重新执行整条命令。

## 4. 获取 Toolkit 和 Model Zoo

```bash
mkdir -p ~/rk1828-work
cd ~/rk1828-work

git clone https://github.com/airockchip/rknn3-toolkit.git
git clone https://github.com/airockchip/rknn3-model-zoo.git
```

如果 Toolkit 克隆因连接重置而失败，可保留失败目录后进行浅克隆：

```bash
mv rknn3-toolkit rknn3-toolkit.incomplete

GIT_LFS_SKIP_SMUDGE=1 git -c http.version=HTTP/1.1 clone \
  --depth 1 --single-branch \
  https://github.com/airockchip/rknn3-toolkit.git

git -C rknn3-toolkit lfs pull \
  --include="rknn3-toolkit/packages/**"
```

## 5. 创建 Python 虚拟环境

`venv` 用于隔离此项目的 Python、pip 和依赖包，不是端侧 Runtime，也不会改变系统
Python。

```bash
cd ~/rk1828-work
python3.12 -m venv .venv
source .venv/bin/activate

unset PYTHONPATH
export PYTHONNOUSERSITE=1

python --version
which python
python -m pip --version
```

预期结果：

```text
Python 3.12.x
/home/<USER>/rk1828-work/.venv/bin/python
```

每次打开新终端后都要重新执行：

```bash
source ~/rk1828-work/.venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

退出环境使用：

```bash
deactivate
```

## 6. 安装 RKNN3 Toolkit 1.0.4

先安装官方 Python 3.12 依赖：

```bash
cd ~/rk1828-work
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r \
  rknn3-toolkit/rknn3-toolkit/packages/requirements_cp312-1.0.4.txt
```

安装 Toolkit wheel：

```bash
RKNN3_WHEEL=~/rk1828-work/rknn3-toolkit/rknn3-toolkit/packages/rknn3_toolkit-1.0.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

ls -lh "$RKNN3_WHEEL"
file "$RKNN3_WHEEL"
python -m pip install "$RKNN3_WHEEL"
```

注意安装包名称是 `rknn3-toolkit`，但 Python 导入名仍然是 `rknn`，主要类名是
`RKNN`：

```bash
python -c "from rknn.api import RKNN; print('RKNN3 Toolkit 1.0.4 import OK')"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -m pip check
```

当前实测结果：

```text
RKNN3 Toolkit 1.0.4 import OK
PyTorch: 2.7.0+cu128
TorchVision: 0.22.0+cu128
CUDA available: True
No broken requirements found.
```

PyTorch 安装的 `nvidia-*` 包是 CUDA wheel 的依赖。当前 `CUDA available: True` 表示宿主机
确实可以使用 NVIDIA GPU，有利于后续 GRQ 量化。

宿主机使用 NVIDIA GeForce RTX 5070 Laptop GPU，计算能力为 `sm_120`。CUDA 12.6 版
PyTorch 2.7.0 不包含 `sm_120` 内核，会出现 `no kernel image is available`；已改用
PyTorch 2.7.0+cu128，并通过实际 CUDA 张量运算验证：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); print(torch.ones(4, device='cuda') * 2)"
```

关键结果应包含：

```text
NVIDIA GeForce RTX 5070 Laptop GPU
(12, 0)
sm_120
tensor([2., 2., 2., 2.], device='cuda:0')
```

## 7. 下载首个验证模型

首个模型使用：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

0.5B 用于先验证完整工具链；验证通过后再尝试 1.5B 或更大的模型。

```bash
source ~/rk1828-work/.venv/bin/activate
python -m pip install huggingface_hub
mkdir -p ~/rk1828-work/models

hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ~/rk1828-work/models/Qwen2.5-0.5B-Instruct
```

`hf-cli skill is not installed` 是面向 AI Agent 的提示，不是模型下载错误，不需要执行
`hf skills add -g --claude`。

下载完成后检查：

```bash
MODEL_DIR=~/rk1828-work/models/Qwen2.5-0.5B-Instruct

test -f "$MODEL_DIR/config.json"
find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -ls
find "$MODEL_DIR" -name '*.incomplete' -print
```

只有确认没有 `.incomplete` 文件后才开始模型转换。

## 8. 安装 Model Zoo 额外依赖

```bash
cd ~/rk1828-work/rknn3-model-zoo
source ~/rk1828-work/.venv/bin/activate

unset PYTHONPATH
export PYTHONNOUSERSITE=1

# AutoGPTQ 的可选 CUDA 扩展与本流程无关，关闭扩展可避免隔离构建阶段找不到 torch。
BUILD_CUDA_EXT=0 python -m pip install \
  --no-build-isolation \
  "auto_gptq==0.7.1"

python -m pip install -r requirements.txt

# requirements.txt 解析过程中可能把 torch 升级到 2.8.0，需恢复 Toolkit 1.0.4
# 要求的 torch 2.7.0；cu128 版本同时支持 RTX 5070 的 sm_120。
python -m pip install \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "from rknn.api import RKNN; print('RKNN3 Toolkit OK')"
python -c "import auto_gptq, accelerate, timm, optimum; print('Model Zoo dependencies OK')"

# 只保留 Model Zoo 根目录，避免 ROS 的 PYTHONPATH 污染虚拟环境。
export PYTHONPATH="$PWD"
```

AutoGPTQ 打印 `CUDA extension not installed` 不代表 PyTorch CUDA 不可用。这里关闭的是
AutoGPTQ 自己的可选扩展；`export_llm.py` 使用的 GRQ 量化仍通过 PyTorch CUDA 执行。
`torch.cuda.amp.custom_fwd/custom_bwd` 的 `FutureWarning` 也只是弃用提醒，不影响当前转换。

## 9. 导出 Qwen2.5-0.5B

官方脚本默认使用 Qwen2.5-3B。这里必须显式传入本地 0.5B 模型路径和输出名，避免
脚本重新下载或转换默认的 3B 模型。

```bash
cd ~/rk1828-work/rknn3-model-zoo
export PYTHONPATH="$PWD"
cd examples/Qwen2_5/python

python export_llm.py --quant \
  --model_path ~/rk1828-work/models/Qwen2.5-0.5B-Instruct \
  --export_llm_path ../model/llm/Qwen2.5-0.5B-Instruct.onnx \
  2>&1 | tee qwen2.5-0.5b-export.log
```

此步骤负责：

1. 加载 Hugging Face 模型；
2. 使用 CUDA 执行 GRQ 量化；
3. 导出 ONNX；
4. 导出 RKNN3 LLM 配置；
5. 导出 `tokenizer.gguf`；
6. 导出 `embed.bin`。

预期中间文件：

```text
../model/llm/Qwen2.5-0.5B-Instruct.onnx
../model/llm/Qwen2.5-0.5B-Instruct.config.pkl
../model/llm/Qwen2.5-0.5B-Instruct.tokenizer.gguf
../model/llm/Qwen2.5-0.5B-Instruct.embed.bin
```

本次实测耗时约 2 分 37 秒，最终输出包括：

```text
GRQ quantization success!
Exported to .../Qwen2.5-0.5B-Instruct.onnx
Model configuration exported to ../model/llm/Qwen2.5-0.5B-Instruct.config.pkl
Tokenizer exported to ../model/llm/Qwen2.5-0.5B-Instruct.tokenizer.gguf
Embedding weight exported to ../model/llm/Qwen2.5-0.5B-Instruct.embed.bin
```

执行期间的 `Can't initialize NVML` 只表示 PyTorch 无法通过 NVML 读取 GPU 管理指标。
在 `torch.cuda.is_available()` 为 `True` 且上述 CUDA 张量测试成功的情况下，它不影响模型
计算或导出。GPU 状态可另开终端使用 `nvidia-smi` 监控；若 `nvidia-smi` 本身也失败，
再检查 NVIDIA 内核驱动和 `libnvidia-ml.so.1`。

## 10. 编译为 RKNN3 模型

```bash
cd ~/rk1828-work/rknn3-model-zoo/examples/Qwen2_5/python
export PYTHONPATH=~/rk1828-work/rknn3-model-zoo
set -o pipefail

# 无论内部字符串是什么，都先按时间戳隔离现有产物，避免覆盖后无法比较。
BACKUP_DIR="../model/llm/previous-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
for file in \
  ../model/llm/Qwen2.5-0.5B-Instruct.rknn \
  ../model/llm/Qwen2.5-0.5B-Instruct.weight; do
  [ ! -f "$file" ] || mv "$file" "$BACKUP_DIR/"
done

# 先单独验证平台参数。返回值不是 0 时不要开始耗时编译。
python -c "from rknn.api import RKNN; r=RKNN(verbose=True); ret=r.config(target_platform='rk1820', quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32'); print('rknn.config return:', ret); r.release(); raise SystemExit(ret)"

python export_rknn.py \
  --onnx_path ../model/llm/Qwen2.5-0.5B-Instruct.onnx \
  --config ../model/llm/Qwen2.5-0.5B-Instruct.config.pkl \
  --rknn_path ../model/llm/Qwen2.5-0.5B-Instruct.rknn \
  --dataset_path ../../../datasets/CMMLU/dataset.txt \
  --platform rk1820 \
  2>&1 | tee qwen2.5-0.5b-rknn.log
```

`--platform` 是必填参数；省略时脚本只会打印
`the following arguments are required: --platform` 并退出，不会开始编译。

### 10.1 Toolkit 1.0.4 的 RK182X 平台映射

实际 PCIe 协处理器是 **RK1828**，`rknn-smi` 显示的芯片型号无需也不应该改变。
在当前安装的 RKNN3 Toolkit 1.0.4 中，`rknn.config(target_platform='rk1828')` 返回 `0`，
但它与 `target_platform='rk1820'` 生成的 `.rknn/.weight` 字节完全相同，内部可打印标识均为
`RK1820`。因此这个版本实际上把两个输入映射到同一套 RK182X 模型格式。

官方资料本身存在表述差异：《用户指南》把 RK1820、RK1828、RK3572 列为可选目标，并在 FAQ
中称三者模型互不兼容；同版本 API 参考和 wheel 内 `rknn.py` 只列出 `rk1820`，Model Zoo
面向 RK1820/RK1828 的示例也使用 `target_platform='rk1820'`。本流程以当前安装包的实际输出
和官方示例为准，固定使用 `rk1820`，不再通过 `strings` 区分 RK1820 与 RK1828。

因此：

- RK1828 是真实硬件型号；
- 当前 Toolkit 1.0.4 使用 `--platform rk1820` 生成 RK182X 模型；
- `--platform rk1828` 虽然被接受，但不会生成不同的二进制；
- 端侧固件文件名仍为 `rknn3_rk1820.img`，不代表模型目标也应选择 RK1820；
- `build-linux.sh -t rk3588` 仍保持不变，因为 C++ Demo 运行在 RK3588 主机。

这里的两个平台参数属于不同对象：

- `export_rknn.py --platform rk1820`：当前 Toolkit 1.0.4 的 RK182X 编译目标；
- `build-linux.sh -t rk3588`：运行 C++ Demo 的 ARM64 主机是 RK3588。

RKNN3 使用权重分离模式，检查最终四个部署文件：

```bash
ls -lh ../model/llm/Qwen2.5-0.5B-Instruct.{rknn,weight,tokenizer.gguf,embed.bin}
```

文件用途：

| 文件 | 用途 |
| --- | --- |
| `.rknn` | 图结构和执行信息 |
| `.weight` | 量化后的模型权重 |
| `.tokenizer.gguf` | 文本与 token 之间的转换规则 |
| `.embed.bin` | Token embedding 权重 |

### 10.2 当前模型与运行模式验证

首次编译于 2026-08-18 10:50 完成，编译日志本身没有报错：

```text
I rknn building done.
I RKNN: Stage code generation completed successfully
I RKNN: === RKNN Compiler All stages completed successfully ===
```

模型可打印字符串中包含 `RK1820`：

```bash
strings ../model/llm/Qwen2.5-0.5B-Instruct.rknn | grep -m1 RK1820
```

当前产物如下：

| 文件 | 字节数 | 约合大小 |
| --- | ---: | ---: |
| `Qwen2.5-0.5B-Instruct.rknn` | 17,939,072 | 17.1 MiB |
| `Qwen2.5-0.5B-Instruct.weight` | 333,308,416 | 317.9 MiB |
| `Qwen2.5-0.5B-Instruct.tokenizer.gguf` | 5,931,031 | 5.7 MiB |
| `Qwen2.5-0.5B-Instruct.embed.bin` | 272,269,312 | 259.7 MiB |

2026-08-18 的成功/失败对照使用完全相同的模型文件、`Hi` 提示词和 `0xff` Core Mask，唯一已知
关键差异是 Work Mode：

| Work Mode | Prefill Mode | 结果 |
| --- | --- | --- |
| `EFFICIENT` | `PERFORMANCE` | 完整生成 `Hello! How can I assist you today?`，状态查询与资源释放正常 |
| `NORMAL` | `PERFORMANCE` | 首次 `rknn3_session_wait` 断链，返回 `ERROR_PIPE`、`ERROR_NO_DEVICE` |

成功日志确认 API、`rknn3server`、`rknn3rt_srv` 均为 `1.0.4`，8 个 NPU 核的命令、权重、内部
缓冲和 KV Cache 均成功分配与传输。实测性能为 Prefill `617.44 token/s`、Generate
`129.10 token/s`。模型已经包含 Prefill `[1, 128, 896]` 和 Decode `[1, 1, 896]` 两种输入
Shape，无需为了该问题重新导出或重新编译。

`open /dev/dma_heap/system-dma32 fail` 不是本次故障：Runtime 随后已从普通 DMA Heap 成功分配。
首次连接先访问通用 Proxy、取得哈希后重连到带哈希的 Socket 也属于正常流程，只要随后出现
`Transfer interface successfully opened via retry` 即可。

切换 Work Mode 为 `NORMAL` 后，故障发生于全部模型数据传输成功、发送首次推理任务之后，且没有
返回任何 token。per-device transfer proxy 子进程随即退出；`rknn-smi` 读取到 PCIe magic、
RC/EP version 均为 `0xffffffff`，并判定 `PCIe device 0 is not alive`。这说明 RK1828 固件或
执行服务在 `NORMAL` 模式的 8 核 LLM 路径中崩溃/复位，不是 Tokenizer、模型哈希、平台字符串、
内存分配或提示词导致。部署目录中的 Runtime 动态库与 `/usr/lib` 中的版本也逐字节一致。

设备恢复后应再次确认 `Health OK`、全部 PCIe 错误计数为 `0`，并同时存在父 transfer proxy 和
per-device transfer proxy 子进程。`systemctl` 只跟踪父进程，因此显示 `active (running)` 不能
单独证明 RK1828 仍然在线。`Reset Reason` 为 `NPOR` 时只反映最近一次上电复位，无法还原此前
的崩溃原因。

`rknn-smi log -t collect` 默认写入 `/data`，但当前系统没有该目录，因此应显式指定日志目录：

```bash
sudo mkdir -p /home/ubuntu/userdata/rknn-logs
sudo rknn-smi log -t collect -s /home/ubuntu/userdata/rknn-logs
sudo find /home/ubuntu/userdata/rknn-logs -maxdepth 2 -type f -ls
```

本次重启后的设备同时报告 current/last log 均无数据，因此即使修正输出目录，也可能无法取回上次
崩溃的设备端日志。

### 10.3 验证重新编译产物

`strings model.rknn | grep RK1820` 只能说明文件中存在该文本，不能证明当前构建实际采用了
RK1820。wheel 的公共代码和二进制中同时存在 RK1820、RK1828 及 RK182X 相关标识，模型还
可能携带共享架构或后端名称。因此 `strings` 只作辅助观察，不能作为部署门槛。

先检查新产物及日志：

```bash
cd ~/rk1828-work/rknn3-model-zoo/examples/Qwen2_5/python

sha256sum \
  ../model/llm/Qwen2.5-0.5B-Instruct.rknn \
  ../model/llm/Qwen2.5-0.5B-Instruct.weight

stat -c '%y %s %n' \
  ../model/llm/Qwen2.5-0.5B-Instruct.rknn \
  ../model/llm/Qwen2.5-0.5B-Instruct.weight

grep -Ein 'rk1828|rk1820|target.*platform|platform|error|fail' \
  qwen2.5-0.5b-rknn.log | head -100

strings ../model/llm/Qwen2.5-0.5B-Instruct.rknn \
  | grep -m5 -E 'RK1820|RK1828|RK182X'
```

当前已在稳定模式下跑通的模型 SHA-256 为：

```text
013dd8c92fa7c08feaac9b3fd9c6dc8370b5913589bb5ba8d2d7c61a8552ee6a  Qwen2.5-0.5B-Instruct.rknn
94bbef9ec8eb5eee08473105af3d88bcce062283db763adba15804d03b7e40f8  Qwen2.5-0.5B-Instruct.weight
```

判定规则：

- `rknn.config return` 为 `0`：参数被 Toolkit 接受；
- `rk1820` 与 `rk1828` 两次构建哈希完全相同：确认当前版本执行了 RK182X 平台归一化；
- 端侧文件与上述哈希一致：确认运行的是已经在 RK1828 上成功生成文本的同一产物；
- 不再根据 `strings` 输出决定是否部署。

离线重新加载生成的模型，排除不完整输出文件：

```bash
source ~/rk1828-work/.venv/bin/activate

python - <<'PY'
from rknn.api import RKNN

model_dir = (
    "/home/barry/rk1828-work/rknn3-model-zoo/"
    "examples/Qwen2_5/model/llm"
)
rknn = RKNN(verbose=False)
ret = rknn.load_rknn(
    f"{model_dir}/Qwen2.5-0.5B-Instruct.rknn",
    f"{model_dir}/Qwen2.5-0.5B-Instruct.weight",
)
print(f"load_rknn return code: {ret}")
rknn.release()
raise SystemExit(ret)
PY
```

本次实测返回：

```text
load_rknn return code: 0
```

## 11. 交叉编译 RK3588 ARM64 Demo

### 11.1 补齐 RKNN3 Runtime 开发文件

公开 Model Zoo 仓库的 `3rdparty/rknpu3` 目录默认只有一个提示联系 Rockchip 的
`README.md`，不包含 C++ Demo 构建所需的头文件和动态库。直接运行构建脚本会出现：

```text
fatal error: rknn3_api.h: No such file or directory
No rule to make target '.../3rdparty/rknpu3/Linux/aarch64/librknn3_api.so'
```

当前机器保存的完整 RK1820/RK1828 SDK 已包含 Model Zoo 配套文件。归档文件扩展名虽然是
`.tar.gz`，实际压缩格式是 bzip2，因此应使用 `tar -xjf`：

```bash
SDK_ARCHIVE='/media/barry/Qin专属备用盘/Linux/RK1820/RK1820_RK1828/RELEASE_V1.0.5b10/RK1820_1828_RELEASE_V1.0.5B10.tar.gz'

file "$SDK_ARCHIVE"
tar -xjf "$SDK_ARCHIVE" \
  --strip-components=3 \
  -C ~/rk1828-work/rknn3-model-zoo \
  rel_182x/rknn/rknn3-model-zoo/3rdparty/rknpu3/include \
  rel_182x/rknn/rknn3-model-zoo/3rdparty/rknpu3/Linux/aarch64
```

只需恢复以下构建依赖，不要解压或覆盖其他 Model Zoo 源码：

```text
3rdparty/rknpu3/include/rknn3_api.h
3rdparty/rknpu3/include/float16.h
3rdparty/rknpu3/Linux/aarch64/librknn3_api.so
3rdparty/rknpu3/Linux/aarch64/librknn3_api_rkcp.so
3rdparty/rknpu3/Linux/aarch64/librknn3_api_native.so
```

虽然这些文件取自 V1.0.5b10 完整 SDK，实际用于 RK3588 Host 的两个库与本机保存的
V1.0.4 M.2 安装包内容完全相同。实测 SHA-256 如下：

```text
113ec97719e04f82e51fcb8badeb18461070ac55ca9a5da87f887f3110b4fcbe  librknn3_api.so
5ea77749f44be1f0c2ad0347242d4b431d3907d03eac11d265496ddd80cfd210  librknn3_api_rkcp.so
```

因此本次补齐开发文件没有升级或覆盖 RK3588 上已经安装的 1.0.4 Runtime、传输服务或
RK1828 固件。

### 11.2 为什么不能直接在 Ubuntu 24.04 上构建

x86 宿主机是 Ubuntu 24.04，系统提供的 ARM64 交叉 sysroot 为 glibc 2.39。直接执行：

```bash
cd ~/rk1828-work/rknn3-model-zoo

export GCC_COMPILER=/usr/bin/aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
```

虽然构建可以成功，但生成的程序引用了端侧 Ubuntu 22.04 / glibc 2.35 不提供的符号：

```bash
DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo

aarch64-linux-gnu-readelf --dyn-syms --wide \
  "$DEMO_DIR/rknn_qwen2_5_demo" | \
  grep GLIBC_2.38
```

本次 Ubuntu 24.04 构建的实际输出为：

```text
__isoc23_strtoul@GLIBC_2.38
```

该引用来自 `main.cc` 中的 `strtoul()`，在新 glibc 头文件下会解析为 ISO C23 版本。
这不是 RKNN3 模型或 Runtime 错误，而是应用程序构建基线高于端侧运行基线。

不要升级或手工替换端侧 glibc。正确做法是在 Ubuntu 22.04 / glibc 2.35 环境重新构建
应用程序。

### 11.3 使用 Ubuntu 22.04 Docker 重新构建

先记录四个模型文件哈希并保留 Ubuntu 24.04 的 CMake 构建目录：

```bash
cd ~/rk1828-work/rknn3-model-zoo

DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo
sha256sum \
  "$DEMO_DIR/model/Qwen2.5-0.5B-Instruct.rknn" \
  "$DEMO_DIR/model/Qwen2.5-0.5B-Instruct.weight" \
  "$DEMO_DIR/model/Qwen2.5-0.5B-Instruct.tokenizer.gguf" \
  "$DEMO_DIR/model/Qwen2.5-0.5B-Instruct.embed.bin" \
  > /tmp/qwen2_5-model-sha256.before

BUILD_BACKUP_TAG=$(date +%Y%m%d-%H%M%S)
BUILD_DIR=build/build_rknn_Qwen2_5_demo_rk3588_linux_aarch64_Release

if [ -d "$BUILD_DIR" ]; then
  mv "$BUILD_DIR" "${BUILD_DIR}.ubuntu24-${BUILD_BACKUP_TAG}"
fi
```

使用一次性 Ubuntu 22.04 容器安装 GCC 11 ARM64 交叉工具链并完整重编译：

```bash
docker run --rm \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  ubuntu:22.04 \
  bash -lc '
    set -e
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cmake make gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    export GCC_COMPILER=/usr/bin/aarch64-linux-gnu
    ./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
    chown -R "${HOST_UID}:${HOST_GID}" \
      build/build_rknn_Qwen2_5_demo_rk3588_linux_aarch64_Release \
      install/rk3588_linux_aarch64/rknn_Qwen2_5_demo
  '
```

如果 Docker daemon 提示权限不足，将 `docker run` 改为 `sudo docker run`。传入
`HOST_UID` 和 `HOST_GID` 并在构建结束后执行 `chown`，可以避免挂载目录中的新文件归
`root` 所有。

本次容器实测使用：

```text
Ubuntu 22.04
GCC/G++ 11.4.0 for aarch64-linux-gnu
libc6-dev-arm64-cross 2.35
```

原 Ubuntu 24.04 构建目录已保留为：

```text
build/build_rknn_Qwen2_5_demo_rk3588_linux_aarch64_Release.ubuntu24-20260818-120317
```

构建日志中的 `find: 'tested_models': No such file or directory` 只是缺少可选目录的非致命
提示，不影响 `Qwen2_5` Demo 构建。

### 11.4 检查 GLIBC 兼容性和部署包

先确认程序架构，并执行用户态兼容性的快速检查：

```bash
DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo

file "$DEMO_DIR/rknn_qwen2_5_demo"

aarch64-linux-gnu-readelf --dyn-syms --wide \
  "$DEMO_DIR/rknn_qwen2_5_demo" | \
  grep GLIBC_2.38
```

`file` 输出应包含 `ARM aarch64`，最后一条命令应当没有输出。

不要只检查 `GLIBC_2.38`。以下检查会扫描主程序和部署包内所有动态库，并在发现任何高于
glibc 2.35 的需求时退出：

```bash
for ELF_FILE in \
  "$DEMO_DIR/rknn_qwen2_5_demo" \
  "$DEMO_DIR"/lib/*.so
do
  printf '%s: ' "$ELF_FILE"
  aarch64-linux-gnu-readelf --version-info "$ELF_FILE" | \
    grep -o 'GLIBC_[0-9][0-9.]*' | sort -Vu | paste -sd, -

  if aarch64-linux-gnu-readelf --version-info "$ELF_FILE" | \
      grep -Eq 'GLIBC_2\.(3[6-9]|[4-9][0-9])|GLIBC_[3-9]\.'
  then
    echo "不兼容 glibc 2.35: $ELF_FILE" >&2
    exit 1
  fi
done
```

本次实测结果：

| ELF 文件 | 最高 GLIBC 需求 |
| --- | --- |
| `rknn_qwen2_5_demo` | `GLIBC_2.34` |
| `lib/librga.so` | `GLIBC_2.17` |
| `lib/librknn3_api.so` | `GLIBC_2.17` |
| `lib/librknn3_api_rkcp.so` | `GLIBC_2.17` |

新程序信息：

```text
大小：794,896 bytes
SHA-256：3d07e2480c79ec2636aa8be02fd44c5898ce3878436f1360cf217491c7216c15
```

`make install` 会自动复制可执行文件、Runtime 动态库和四个模型文件。确认模型副本未被
重编译过程改变：

```bash
sha256sum -c /tmp/qwen2_5-model-sha256.before
```

四项都应显示 `OK`。最后检查部署包：

```bash
find "$DEMO_DIR" -maxdepth 2 -type f -printf '%s %p\n' | sort
du -sh "$DEMO_DIR"
```

本次生成的完整部署包约为 `610M`，包含：

```text
rknn_qwen2_5_demo
lib/librknn3_api.so
lib/librknn3_api_rkcp.so
lib/librga.so
model/Qwen2.5-0.5B-Instruct.rknn
model/Qwen2.5-0.5B-Instruct.weight
model/Qwen2.5-0.5B-Instruct.tokenizer.gguf
model/Qwen2.5-0.5B-Instruct.embed.bin
```

## 12. 传输到 RK3588

### 12.1 首次传输完整部署包

先确认网络和 SSH：

```bash
ping -c 3 <RK3588_IP>
ssh ubuntu@<RK3588_IP> 'uname -m && df -h /home/ubuntu/userdata'
```

传输完整 Demo：

```bash
cd ~/rk1828-work/rknn3-model-zoo

scp -r install/rk3588_linux_aarch64/rknn_Qwen2_5_demo \
  ubuntu@<RK3588_IP>:/home/ubuntu/userdata/
```

### 12.2 重编译后只传输程序和库

如果四个模型文件已经传输到端侧，Ubuntu 22.04 重编译后不需要再次传输约 600 MB 的模型
数据，只更新程序和动态库：

```bash
cd ~/rk1828-work/rknn3-model-zoo

DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo
RK3588_HOST=ubuntu@<RK3588_IP>
REMOTE_DIR=/home/ubuntu/userdata/rknn_Qwen2_5_demo

ssh "$RK3588_HOST" "mkdir -p '$REMOTE_DIR/lib'"

scp "$DEMO_DIR/rknn_qwen2_5_demo" \
  "${RK3588_HOST}:${REMOTE_DIR}/"

scp "$DEMO_DIR"/lib/*.so \
  "${RK3588_HOST}:${REMOTE_DIR}/lib/"
```

## 13. 在 RK3588 上运行

### 13.1 固定已验证运行模式

`rknn-smi` 当前支持的模式编号如下：

| 配置 | 编号 | 含义 | 当前结论 |
| --- | ---: | --- | --- |
| Work Mode | `0` | `EFFICIENT` | 已验证可运行 |
| Work Mode | `1` | `NORMAL` | 当前 1.0.4 环境会触发 RK1828 掉线 |
| Work Mode | `2` | `PERFORMANCE` | 尚未验证，不用于当前流程 |
| Prefill Mode | `0` | `EFFICIENT` | 可配置，但不是本次成功组合 |
| Prefill Mode | `1` | `PERFORMANCE` | 已验证可运行 |

模式应在没有 Demo 运行时设置。先确认设备和两个 transfer proxy 进程都在线，再固定为
`EFFICIENT/PERFORMANCE`：

```bash
sudo rknn-smi info -l
ps -ef | grep '[r]knn3_transfer'

sudo rknn-smi set -t work_mode -d 0 -c 0 -s 0
sudo rknn-smi set -t prefill_mode -d 0 -c 0 -s 1

sudo rknn-smi info -t work_mode
sudo rknn-smi info -t prefill_mode
```

必须看到：

```text
Chip Work Mode    : EFFICIENT
Chip Prefill Mode : PERFORMANCE
```

不要把 Work Mode 改为 `NORMAL`。在当前 Runtime/Firmware `1.0.4` 上，该模式会在首次 8 核
LLM 推理时使 per-device transfer proxy 退出，并导致 RK1828 变为 not alive。

### 13.2 执行推理

```bash
ssh ubuntu@<RK3588_IP>
cd /home/ubuntu/userdata/rknn_Qwen2_5_demo

export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sudo rknn-smi info -l

./rknn_qwen2_5_demo \
  model/Qwen2.5-0.5B-Instruct.rknn \
  model/Qwen2.5-0.5B-Instruct.weight \
  model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
  model/Qwen2.5-0.5B-Instruct.embed.bin \
  0xff \
  "Hi"
```

这个模型声明使用 8 个 NPU 核，因此 Core Mask 必须为 `0xff`。使用 `0x1` 会在模型初始化阶段
直接返回：

```text
core_mask 1 is not match with npu core number 8
```

已知基线 `Hi` 跑通后，再替换为中文 Prompt。稳定性隔离阶段不要同时运行连续的
`rknn-smi info -w`，只在推理前后执行一次状态检查，避免增加额外通信变量：

```bash
sudo rknn-smi info -l
sudo rknn-smi info -t health
```

Tokenizer GGUF 中显示的 `qwen2.context_length=32768` 是源模型元数据，不是当前 RKNN3 产物的
实际运行上限。Runtime 日志显示本次编译模型的 `max context length(final)` 和
`kvcache_buffer_len` 均为 `1024`，应用侧应按 1024 token 上下文限制处理。

验证成功的最低标准：

- Demo 成功加载 `.rknn` 和 `.weight`；
- 能持续输出有意义的 token；
- `rknn-smi` 显示 NPU 利用率和显存占用上升；
- 没有固件、PCIe、Runtime 或版本不匹配错误。

### 13.3 `ERROR_PIPE` 后恢复 RK1828

出现以下组合时，对端 RK1828 已经掉线，继续重试 Demo 没有意义：

```text
The connection has been closed
ERROR_PIPE
ERROR_NO_DEVICE
PCIe device 0 is not alive
magic=ffffffff
```

`systemctl status rk182x.service` 可能仍显示 `active (running)`，因为父 proxy 还在；如果
`rknn3_transfer_proxy_b98e6c51 -s 0000:01:00.0` 子进程消失，设备实际已经不可用。先尝试：

```bash
sudo systemctl stop rk182x.service
sudo killall rknn3_transfer_proxy 2>/dev/null || true
sleep 3

sudo systemctl reset-failed rk182x.service
sudo systemctl start rk182x.service
sleep 10

sudo systemctl status rk182x.service --no-pager -l
ps -ef | grep '[r]knn3_transfer'
sudo rknn-smi info -l
```

如果仍然读取到 `0xffffffff` 或 `rknn-smi` 初始化失败，执行 `sudo reboot`；重启仍不能恢复时，
对 RK3588 和 RK1828 模块冷断电后重新上电。`pcie_upgrade_tool ... rd` 在设备不处于 loader mode
时不会完成恢复，不要把 `Device is not in loader mode` 当作模型错误。

恢复 Online 后必须重新执行第 13.1 节，确认 Work Mode 已回到 `EFFICIENT`、Prefill Mode 为
`PERFORMANCE`，再运行 Demo。

## 14. 当前进度

| 阶段 | 状态 |
| --- | --- |
| RK3588/RK1828 硬件检查 | 已完成；掉线后可通过服务重启、系统重启或冷上电恢复 |
| RKNN3 Runtime/固件检查 | 已完成，版本 1.0.4 |
| x86 基础工具安装 | 已完成 |
| Python 3.12 虚拟环境 | 已完成 |
| Toolkit 和 Model Zoo 获取 | 已完成 |
| Toolkit 1.0.4 安装和导入 | 已完成 |
| PyTorch/CUDA 检查 | 已完成，2.7.0+cu128 已验证 RTX 5070 `sm_120` 内核 |
| Qwen2.5-0.5B 下载 | 已完成，`model.safetensors` 约 988 MB |
| Model Zoo 额外依赖 | 已完成，AutoGPTQ 使用无可选 CUDA 扩展模式 |
| GRQ 量化及 ONNX/配置导出 | 已完成，实测约 2 分 37 秒 |
| RKNN3 编译 | 已完成；`rk1820` 与 `rk1828` 实测生成相同哈希的 RK182X 产物 |
| ARM64 Demo 交叉编译 | 已完成，Ubuntu 22.04 构建，最高依赖 `GLIBC_2.34` |
| 端侧推理 | 已跑通；稳定组合为 Work `EFFICIENT`、Prefill `PERFORMANCE`、Core Mask `0xff` |
| 运行模式隔离 | Work `NORMAL` 已确认会在首次推理时触发 RK1828 掉线，禁止用于当前流程 |
| 接入 `RK_LLM` | 完成稳定模式连续回归后进行 |

当前最近的一步是在固定模式下连续执行单次 Demo 回归，记录成功率和 Prefill/Generate 性能；
无需重新下载 Hugging Face 权重，也无需重复执行 GRQ、ONNX 导出或 RKNN3 编译。

```bash
sudo rknn-smi info -t work_mode
sudo rknn-smi info -t prefill_mode
```

## 15. 注意事项

1. 不要使用 RKNN-Toolkit、RKNN-Toolkit2 或 RKLLM-Toolkit 替代 RKNN3 Toolkit。
2. 不要在 RK3588 上安装 x86 Toolkit wheel；模型转换只在 x86 宿主机执行。
3. 不要随意覆盖端侧 Runtime、驱动或固件；当前端到端版本已经匹配为 1.0.4。
4. 不要使用脚本的默认 3B 参数；首轮验证固定使用本地 0.5B 模型。
5. 模型转换完成不代表硬件推理成功，必须运行 ARM64 Demo 并观察 `rknn-smi`。
6. 官方 Demo 完整跑通后，再把推理链路接入当前 `RK_LLM` 项目。
7. `librknn3_api`、端侧传输服务和 RK1828 固件必须保持兼容版本；替换构建依赖前应先
   与端侧安装版本或官方安装包核对哈希。
8. 不要为运行宿主机生成的程序而升级端侧 glibc；应用程序必须在不高于目标端侧的 glibc
   基线上构建，并在传输前检查全部部署 ELF 的符号版本。
9. 当前 Toolkit 1.0.4 对 `rk1820` 与 `rk1828` 生成相同的 RK182X 产物，流程固定使用官方
   示例中的 `target_platform='rk1820'`；不要再用模型内的 `RK1820` 字符串判断硬件兼容性。
10. 当前端侧稳定配置是 Work Mode `EFFICIENT`、Prefill Mode `PERFORMANCE`、Core Mask
    `0xff`；Work Mode `NORMAL` 会导致 RK1828 掉线。
11. `rk182x.service` 为 active 只说明父 transfer proxy 存活；必须同时检查 per-device 子进程和
    `rknn-smi info -l`。
12. Tokenizer 元数据中的 32768 上下文长度不能覆盖 RKNN3 产物的 1024 token 实际上限。

## 16. 官方参考

- [RKNN3 Toolkit](https://github.com/airockchip/rknn3-toolkit)
- [RKNN3 SDK 1.0.4 中文用户指南](https://github.com/airockchip/rknn3-toolkit/blob/main/doc/02_Rockchip_RKNPU3_User_Guide_RKNN3_SDK_V1.0.4_CN.pdf)
- [RKNN3 Toolkit 1.0.4 API 参考](https://github.com/airockchip/rknn3-toolkit/blob/main/doc/03_Rockchip_RKNPU3_API_Reference_RKNN3_Toolkit_V1.0.4_EN.pdf)
- [RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)
- [Model Zoo 中文部署说明](https://github.com/airockchip/rknn3-model-zoo/blob/main/README_CN.md)
- [Qwen2.5 导出脚本](https://github.com/airockchip/rknn3-model-zoo/blob/main/examples/Qwen2_5/python/export_llm.py)
- [Qwen2.5 RKNN 转换脚本](https://github.com/airockchip/rknn3-model-zoo/blob/main/examples/Qwen2_5/python/export_rknn.py)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
