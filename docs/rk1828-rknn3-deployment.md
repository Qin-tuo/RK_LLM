# RK3588 + RK1828 的 RKNN3 模型部署流程

本文记录当前设备检查结果，以及在 x86 Ubuntu 宿主机上将
`Qwen/Qwen2.5-0.5B-Instruct` 转换并部署到 RK3588 + RK1828 的完整流程。

> 状态日期：2026-08-18
>
> 当前检查点：RKNN3 Toolkit 1.0.4、Model Zoo 依赖和 RTX 5070 CUDA 环境均已验证；
> Qwen2.5-0.5B 的 GRQ 量化及 ONNX/配置/Tokenizer/Embedding 导出已完成，下一步是
> 使用 `--platform rk1820` 编译 RKNN3 模型。端侧推理尚未开始。

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
| 端侧部署 | Demo 和四个模型文件 | 复制到 RK3588 的 `/userdata` | 可运行的完整部署包 |
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
- 实测状态：`Online`、`Health OK`。

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

### 10.1 为什么 RK1828 使用 `--platform rk1820`

实际 PCIe 协处理器是 **RK1828**，`rknn-smi` 显示的芯片型号无需也不应该改变。
但在 RKNN3 Toolkit 1.0.4 中，`target_platform` 是编译器目标标识，不是板卡型号探测参数。
官方 API 文档同时将 RK1820 和 RK1828 列为适用芯片，但该版本当前支持的
`target_platform` 值只有 `rk1820`；官方 `export_rknn.py` 也使用 `rk1820` 作为示例。

因此：

- RK1828 是真实硬件型号；
- `--platform rk1820` 是 Toolkit 1.0.4 对 RK1820/RK1828 使用的编译目标；
- 不要把参数改成 `--platform rk1828`，该版本可能报告不支持的目标平台；
- 生成的 `.rknn` 仍部署到当前 RK3588 + RK1828 设备。

这里的两个构建平台参数含义也不同：

- `export_rknn.py --platform rk1820`：RK1820/RK1828 协处理器的 RKNN3 编译目标；
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

## 11. 交叉编译 RK3588 ARM64 Demo

```bash
cd ~/rk1828-work/rknn3-model-zoo

export GCC_COMPILER=/usr/bin/aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5

DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo
file "$DEMO_DIR/rknn_qwen2_5_demo"
```

`file` 输出应包含 `ARM aarch64`。

将模型文件放入 Demo 包：

```bash
MODEL_OUT=examples/Qwen2_5/model/llm
DEMO_DIR=install/rk3588_linux_aarch64/rknn_Qwen2_5_demo

mkdir -p "$DEMO_DIR/model"
cp "$MODEL_OUT"/Qwen2.5-0.5B-Instruct.{rknn,weight,tokenizer.gguf,embed.bin} \
  "$DEMO_DIR/model/"
```

## 12. 传输到 RK3588

先确认网络和 SSH：

```bash
ping -c 3 <RK3588_IP>
ssh ubuntu@<RK3588_IP> 'uname -m && df -h /userdata'
```

传输完整 Demo：

```bash
cd ~/rk1828-work/rknn3-model-zoo

scp -r install/rk3588_linux_aarch64/rknn_Qwen2_5_demo \
  ubuntu@<RK3588_IP>:/userdata/
```

## 13. 在 RK3588 上运行

```bash
ssh ubuntu@<RK3588_IP>
cd /userdata/rknn_Qwen2_5_demo

export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
sudo rknn-smi info -l

./rknn_qwen2_5_demo \
  model/Qwen2.5-0.5B-Instruct.rknn \
  model/Qwen2.5-0.5B-Instruct.weight \
  model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
  model/Qwen2.5-0.5B-Instruct.embed.bin \
  0xff \
  "你好，请介绍一下你自己。"
```

在另一个端侧终端监控 RK1828：

```bash
sudo rknn-smi info -w
```

验证成功的最低标准：

- Demo 成功加载 `.rknn` 和 `.weight`；
- 能持续输出有意义的 token；
- `rknn-smi` 显示 NPU 利用率和显存占用上升；
- 没有固件、PCIe、Runtime 或版本不匹配错误。

## 14. 当前进度

| 阶段 | 状态 |
| --- | --- |
| RK3588/RK1828 硬件检查 | 已完成 |
| RKNN3 Runtime/固件检查 | 已完成，版本 1.0.4 |
| x86 基础工具安装 | 已完成 |
| Python 3.12 虚拟环境 | 已完成 |
| Toolkit 和 Model Zoo 获取 | 已完成 |
| Toolkit 1.0.4 安装和导入 | 已完成 |
| PyTorch/CUDA 检查 | 已完成，2.7.0+cu128 已验证 RTX 5070 `sm_120` 内核 |
| Qwen2.5-0.5B 下载 | 已完成，`model.safetensors` 约 988 MB |
| Model Zoo 额外依赖 | 已完成，AutoGPTQ 使用无可选 CUDA 扩展模式 |
| GRQ 量化及 ONNX/配置导出 | 已完成，实测约 2 分 37 秒 |
| RKNN3 编译 | 待执行，目标参数为 `--platform rk1820` |
| ARM64 Demo 交叉编译 | 未开始 |
| 端侧推理 | 未开始 |
| 接入 `RK_LLM` | 官方 Demo 验证后进行 |

当前最近的一步是进入第 10 节，将已经生成的 ONNX 和配置编译为 RKNN3 模型。开始前
可以先确认输入文件和剩余磁盘空间：

```bash
cd ~/rk1828-work/rknn3-model-zoo/examples/Qwen2_5/python
ls -lh ../model/llm/Qwen2.5-0.5B-Instruct.{onnx,config.pkl,tokenizer.gguf,embed.bin}
df -h /
```

## 15. 注意事项

1. 不要使用 RKNN-Toolkit、RKNN-Toolkit2 或 RKLLM-Toolkit 替代 RKNN3 Toolkit。
2. 不要在 RK3588 上安装 x86 Toolkit wheel；模型转换只在 x86 宿主机执行。
3. 不要随意覆盖端侧 Runtime、驱动或固件；当前端到端版本已经匹配为 1.0.4。
4. 不要使用脚本的默认 3B 参数；首轮验证固定使用本地 0.5B 模型。
5. 模型转换完成不代表硬件推理成功，必须运行 ARM64 Demo 并观察 `rknn-smi`。
6. 官方 Demo 完整跑通后，再把推理链路接入当前 `RK_LLM` 项目。

## 16. 官方参考

- [RKNN3 Toolkit](https://github.com/airockchip/rknn3-toolkit)
- [RKNN3 Toolkit 1.0.4 API 参考](https://github.com/airockchip/rknn3-toolkit/blob/main/doc/03_Rockchip_RKNPU3_API_Reference_RKNN3_Toolkit_V1.0.4_EN.pdf)
- [RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)
- [Model Zoo 中文部署说明](https://github.com/airockchip/rknn3-model-zoo/blob/main/README_CN.md)
- [Qwen2.5 导出脚本](https://github.com/airockchip/rknn3-model-zoo/blob/main/examples/Qwen2_5/python/export_llm.py)
- [Qwen2.5 RKNN 转换脚本](https://github.com/airockchip/rknn3-model-zoo/blob/main/examples/Qwen2_5/python/export_rknn.py)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
