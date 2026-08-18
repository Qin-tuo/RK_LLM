# Qwen3-4B RK1828 Quick Deployment Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise, executable guide for deploying `Qwen/Qwen3-4B` after the existing Qwen2.5-0.5B RKNN3 workflow has already been validated.

**Architecture:** Create one focused Markdown guide beside the full deployment manual. Reuse the verified RKNN3 1.0.4 environment and preserve only the Qwen3-4B delta: preflight, model conversion, RKNN3 compilation, ARM64 demo build, transfer, inference, validation, and high-value failure handling.

**Tech Stack:** Markdown, Hugging Face CLI, Python 3.12, PyTorch/CUDA, RKNN3 Toolkit 1.0.4, RKNN3 Model Zoo Qwen3 example, Docker Ubuntu 22.04, AArch64 CMake toolchain, SSH/SCP.

---

### Task 1: Write the Qwen3-4B incremental quick guide

**Files:**
- Create: `docs/rk1828-qwen3-4b-quick-deployment.md`
- Reference: `docs/rk1828-rknn3-deployment.md`
- Reference: `/home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/python/export_llm.py`
- Reference: `/home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/python/export_rknn.py`
- Reference: `/home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/cpp/rknn_qwen3_llm.h`

- [ ] **Step 1: Create the guide with the approved scope**

Create `docs/rk1828-qwen3-4b-quick-deployment.md` with these exact sections and operational details:

````markdown
# Qwen3-4B 在 RK3588 + RK1828 上的快速部署

本文是在 [完整部署手册](rk1828-rknn3-deployment.md) 已跑通
`Qwen/Qwen2.5-0.5B-Instruct` 的基础上，将模型切换为 `Qwen/Qwen3-4B` 的增量步骤。
不重复安装 Toolkit、Model Zoo、CUDA、交叉工具链或端侧 Runtime。

## 1. 固定环境和路径

要求：

- RKNN3 Toolkit、RKNN3 Runtime 和 RK1828 固件保持 `1.0.4`；
- 使用 `Qwen/Qwen3-4B`，不要替换为 Base、FP8、GGUF、VL 或 2507 变体；
- RKNN3 编译目标仍为 `rk1820`，实际推理硬件仍为 RK1828；
- 首次推理保持 Qwen3 Demo 默认的 `MAX_CONTEXT_LEN=1024`；
- 当前 30 GiB RAM、8 GiB Swap 和 8 GiB 显存属于临界转换配置，先关闭其他高内存或显存任务。

```bash
export RK1828_WORK="$HOME/rk1828-work"
export ZOO_DIR="$RK1828_WORK/rknn3-model-zoo"
export MODEL_DIR="$RK1828_WORK/models/Qwen3-4B"
export LLM_DIR="$ZOO_DIR/examples/Qwen3/model/llm"
export DEMO_DIR="$ZOO_DIR/install/rk3588_linux_aarch64/rknn_Qwen3_demo"

source "$RK1828_WORK/.venv/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

free -h
swapon --show
df -h "$RK1828_WORK"
nvidia-smi

python -c "from rknn.api import RKNN; print('RKNN3 Toolkit OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

test -f "$ZOO_DIR/examples/Qwen3/python/export_llm.py"
test -f "$ZOO_DIR/examples/Qwen3/python/export_rknn.py"
test -f "$ZOO_DIR/3rdparty/rknpu3/include/rknn3_api.h"
test -f "$ZOO_DIR/3rdparty/rknpu3/Linux/aarch64/librknn3_api.so"
```

上述命令必须确认 CUDA 为 `True`，并且磁盘有足够空间保存原始权重、中间 ONNX、编译
缓存和最终部署包。若资源不足，先停止转换并扩充内存、Swap 或磁盘。

## 2. 下载 Qwen3-4B

```bash
mkdir -p "$MODEL_DIR"

hf download Qwen/Qwen3-4B \
  --local-dir "$MODEL_DIR"

test -s "$MODEL_DIR/config.json"
find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -ls

if find "$MODEL_DIR" -name '*.incomplete' -print -quit | grep -q .; then
  echo "模型下载不完整" >&2
  exit 1
fi
```

## 3. 导出并执行 GRQ 量化

```bash
cd "$ZOO_DIR"
export PYTHONPATH="$ZOO_DIR"
cd examples/Qwen3/python
set -o pipefail

python export_llm.py --quant \
  --model_path "$MODEL_DIR" \
  --export_llm_path ../model/llm/Qwen3-4B.onnx \
  2>&1 | tee qwen3-4b-export.log
```

成功日志必须包含 `GRQ quantization success!`，并且以下文件均非空：

```bash
test -s "$LLM_DIR/Qwen3-4B.onnx"
test -s "$LLM_DIR/Qwen3-4B.config.pkl"
test -s "$LLM_DIR/Qwen3-4B.tokenizer.gguf"
test -s "$LLM_DIR/Qwen3-4B.embed.bin"
```

出现 CUDA OOM、进程被系统终止或缺少任一文件都视为失败。不要继续使用部分生成的文件。

## 4. 编译 RKNN3 模型

```bash
cd "$ZOO_DIR/examples/Qwen3/python"
export PYTHONPATH="$ZOO_DIR"
set -o pipefail

python export_rknn.py \
  --onnx_path ../model/llm/Qwen3-4B.onnx \
  --config ../model/llm/Qwen3-4B.config.pkl \
  --rknn_path ../model/llm/Qwen3-4B.rknn \
  --dataset_path ../../../datasets/CMMLU/dataset.txt \
  --platform rk1820 \
  2>&1 | tee qwen3-4b-rknn.log

test -s "$LLM_DIR/Qwen3-4B.rknn"
test -s "$LLM_DIR/Qwen3-4B.weight"
```

`--platform rk1820` 是 Toolkit 1.0.4 对 RK1820/RK1828 使用的编译目标，不要改成
`rk1828`。当前 Qwen3 转换脚本使用 `W4A16 + GRQ + group32`，保持脚本配置不变。

重新加载最终模型：

```bash
python - "$LLM_DIR" <<'PY'
from pathlib import Path
import sys

from rknn.api import RKNN

model_dir = Path(sys.argv[1])
rknn = RKNN(verbose=False)
ret = rknn.load_rknn(
    str(model_dir / "Qwen3-4B.rknn"),
    str(model_dir / "Qwen3-4B.weight"),
)
print(f"load_rknn return code: {ret}")
rknn.release()
raise SystemExit(ret)
PY
```

预期：`load_rknn return code: 0`。

## 5. 使用 Ubuntu 22.04 构建 ARM64 Demo

构建脚本会替换 Qwen3 的安装目录。先保留已有 Qwen3 构建产物：

```bash
cd "$ZOO_DIR"

BUILD_DIR="$ZOO_DIR/build/build_rknn_Qwen3_demo_rk3588_linux_aarch64_Release"
INSTALL_DIR="$ZOO_DIR/install/rk3588_linux_aarch64/rknn_Qwen3_demo"
BACKUP_SUFFIX="$(date +%Y%m%d-%H%M%S)"

if [ -d "$BUILD_DIR" ]; then
  mv "$BUILD_DIR" "${BUILD_DIR}.backup-${BACKUP_SUFFIX}"
fi
if [ -d "$INSTALL_DIR" ]; then
  mv "$INSTALL_DIR" "${INSTALL_DIR}.backup-${BACKUP_SUFFIX}"
fi
```

使用与端侧 Ubuntu 22.04 / glibc 2.35 匹配的容器构建：

```bash
docker run --rm \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "$ZOO_DIR:$ZOO_DIR" \
  -w "$ZOO_DIR" \
  ubuntu:22.04 \
  bash -lc '
    set -e
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cmake make gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    export GCC_COMPILER=/usr/bin/aarch64-linux-gnu
    ./build-linux.sh -t rk3588 -a aarch64 -d Qwen3
    chown -R "${HOST_UID}:${HOST_GID}" \
      build/build_rknn_Qwen3_demo_rk3588_linux_aarch64_Release \
      install/rk3588_linux_aarch64/rknn_Qwen3_demo
  '
```

## 6. 检查部署包

```bash
test -s "$DEMO_DIR/rknn_qwen3_demo"
test -s "$DEMO_DIR/model/Qwen3-4B.rknn"
test -s "$DEMO_DIR/model/Qwen3-4B.weight"
test -s "$DEMO_DIR/model/Qwen3-4B.tokenizer.gguf"
test -s "$DEMO_DIR/model/Qwen3-4B.embed.bin"

file "$DEMO_DIR/rknn_qwen3_demo"

for ELF_FILE in "$DEMO_DIR/rknn_qwen3_demo" "$DEMO_DIR"/lib/*.so; do
  printf '%s: ' "$ELF_FILE"
  aarch64-linux-gnu-readelf --version-info "$ELF_FILE" | \
    grep -o 'GLIBC_[0-9][0-9.]*' | sort -Vu | paste -sd, -

  if aarch64-linux-gnu-readelf --version-info "$ELF_FILE" | \
      grep -Eq 'GLIBC_2\.(3[6-9]|[4-9][0-9])|GLIBC_[3-9]\.'; then
    echo "不兼容 glibc 2.35: $ELF_FILE" >&2
    exit 1
  fi
done

du -sh "$DEMO_DIR"
```

主程序必须显示为 ARM64 ELF，且循环检查不得报告高于 glibc 2.35 的依赖。

## 7. 传输到 RK3588

```bash
read -rp "RK3588 IP: " RK3588_IP
export RK3588_HOST="ubuntu@$RK3588_IP"
export REMOTE_DIR="/home/ubuntu/userdata/rknn_Qwen3_demo"

ssh "$RK3588_HOST" 'uname -m; df -h /home/ubuntu/userdata; sudo rknn-smi info -l'

scp -r "$DEMO_DIR" \
  "$RK3588_HOST:/home/ubuntu/userdata/"
```

## 8. 板端运行

登录 RK3588：

```bash
ssh "$RK3588_HOST"
cd /home/ubuntu/userdata/rknn_Qwen3_demo
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"

sudo rknn-smi info -l

./rknn_qwen3_demo \
  model/Qwen3-4B.rknn \
  model/Qwen3-4B.weight \
  model/Qwen3-4B.tokenizer.gguf \
  model/Qwen3-4B.embed.bin \
  0xff \
  "你好，请用三句话介绍你自己。"
```

当前官方 Qwen3 C++ Demo 设置 `enable_thinking=false`，首次验证使用非思考模式。在另一个
端侧终端监控：

```bash
sudo rknn-smi info -w
```

## 9. 成功标准

- 四个 Qwen3-4B 部署文件均存在且非空；
- `load_rknn return code: 0`；
- Demo 为 ARM64 ELF，所有部署 ELF 兼容 glibc 2.35；
- 模型加载后能持续输出有意义的 token；
- `rknn-smi` 显示 RK1828 利用率和显存占用变化；
- 没有 Runtime、固件、PCIe 或版本不匹配错误。

## 10. 最短故障处理

| 现象 | 处理 |
| --- | --- |
| CUDA OOM、进程被系统终止 | 停止转换，释放主机内存和显存并扩充 Swap；将不完整产物移入备份目录后重新执行当前阶段 |
| 缺少 `rknn3_api.h` 或 `librknn3_api.so` | 按完整手册第 11.1 节补齐 `3rdparty/rknpu3` 开发文件 |
| ELF 需要 `GLIBC_2.38` | 不升级端侧 glibc；删除本次无效构建目录或移入备份，再按第 5 节使用 Ubuntu 22.04 容器重编译 |
| `rknn3_model_init` 或 session 初始化失败 | 确认 Runtime/固件均为 1.0.4、没有其他 RK1828 任务，并保持 `MAX_CONTEXT_LEN=1024` |
| 版本或 PCIe 错误 | 不覆盖端侧组件；回到完整手册核对 Runtime、传输服务、驱动和固件状态 |

首次跑通后再记录当前设备上的转换耗时、四个产物大小、端侧显存、TTFT 和 Decode TPS。
````

- [ ] **Step 2: Inspect the rendered structure and command continuity**

Run:

```bash
sed -n '1,360p' docs/rk1828-qwen3-4b-quick-deployment.md
```

Expected: the document flows from prerequisites through board inference without referring readers to an undefined variable or omitted intermediate artifact.

### Task 2: Verify and commit the guide

**Files:**
- Test: `docs/rk1828-qwen3-4b-quick-deployment.md`

- [ ] **Step 1: Verify required model and platform parameters**

Run:

```bash
rg -n 'Qwen/Qwen3-4B|Qwen3-4B\.onnx|--platform rk1820|W4A16 \+ GRQ \+ group32|-d Qwen3|MAX_CONTEXT_LEN=1024|rknn_qwen3_demo' \
  docs/rk1828-qwen3-4b-quick-deployment.md
```

Expected: every required model identifier, output, platform setting, build target, context limit, and executable name appears.

- [ ] **Step 2: Verify referenced local files and documentation paths**

Run:

```bash
test -f docs/rk1828-rknn3-deployment.md
test -f /home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/python/export_llm.py
test -f /home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/python/export_rknn.py
test -f /home/barry/rk1828-work/rknn3-model-zoo/examples/Qwen3/cpp/rknn_qwen3_llm.h
```

Expected: all commands exit `0`.

- [ ] **Step 3: Check Markdown diff quality and scope**

Run:

```bash
git diff --check
git status --short
git diff -- docs/rk1828-qwen3-4b-quick-deployment.md
```

Expected: `git diff --check` exits `0`, only the planned quick guide is newly modified during implementation, and the original full manual is unchanged.

- [ ] **Step 4: Commit the quick guide**

```bash
git add docs/rk1828-qwen3-4b-quick-deployment.md
git commit -m "docs: add Qwen3-4B RK1828 quick deployment guide"
```

- [ ] **Step 5: Confirm the final repository state**

Run:

```bash
git status --short --branch
git log -2 --oneline
```

Expected: the branch contains the design commit followed by the quick-guide commit, with no uncommitted files from this task.
