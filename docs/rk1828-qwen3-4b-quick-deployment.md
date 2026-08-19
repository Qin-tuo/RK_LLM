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

## 2. 测速并下载 Qwen3-4B

每次下载模型前都重新测试当前网络。脚本只接受同一大文件完整的 1 MiB Range 响应，自动
选择 Hugging Face、HF Mirror 和 ModelScope 中当前最快的公开源。运行前先停止同一模型
的旧下载任务，避免多个下载器同时写入 `MODEL_DIR`。

```bash
(
export MODEL_REPO="Qwen/Qwen3-4B"
export PROBE_FILE="model-00001-of-00003.safetensors"
export PROBE_BYTES=1048576

for COMMAND_NAME in curl hf modelscope pgrep awk sort flock; do
  if ! command -v "$COMMAND_NAME" >/dev/null 2>&1; then
    echo "缺少命令: $COMMAND_NAME" >&2
    exit 1
  fi
done

RUNNING_DOWNLOADS="$(
  pgrep -af '(^|/)(hf|modelscope) download ' || true
)"
if [ -n "$RUNNING_DOWNLOADS" ] && \
    grep -Fq -- "$MODEL_REPO" <<<"$RUNNING_DOWNLOADS"; then
  echo "检测到同一模型仍在下载，请先停止以下进程：" >&2
  grep -F -- "$MODEL_REPO" <<<"$RUNNING_DOWNLOADS" >&2
  exit 1
fi

if ! mkdir -p "$MODEL_DIR"; then
  echo "无法创建模型目录: $MODEL_DIR" >&2
  exit 1
fi

DOWNLOAD_LOCK="${MODEL_DIR}.download.lock"
if ! exec 9>"$DOWNLOAD_LOCK"; then
  echo "无法创建下载锁: $DOWNLOAD_LOCK" >&2
  exit 1
fi
if ! flock -n 9; then
  echo "检测到同一模型下载脚本已在运行，请等待其完成" >&2
  exit 1
fi

PROBE_RESULTS="$(mktemp)"
trap 'rm -f "$PROBE_RESULTS"' EXIT

probe_source() {
  local SOURCE_NAME="$1"
  local SOURCE_URL="$2"
  local RESOLVE_METRICS RESOLVE_HTTP_CODE RESOLVE_SIZE_DOWNLOAD RESOLVED_URL
  local METRICS HTTP_CODE SPEED_DOWNLOAD SIZE_DOWNLOAD SPEED_MIB

  if ! RESOLVE_METRICS="$(
    curl --location --silent --show-error \
      --connect-timeout 8 \
      --max-time 15 \
      --limit-rate 64K \
      --range 0-0 \
      --output /dev/null \
      --write-out $'%{http_code}\t%{size_download}\t%{url_effective}' \
      "$SOURCE_URL"
  )"; then
    printf '%-12s unavailable\n' "$SOURCE_NAME" >&2
    return 0
  fi

  IFS=$'\t' read -r RESOLVE_HTTP_CODE RESOLVE_SIZE_DOWNLOAD RESOLVED_URL \
    <<<"$RESOLVE_METRICS"
  if [ "$RESOLVE_HTTP_CODE" != "206" ] || \
      [ "$RESOLVE_SIZE_DOWNLOAD" != "1" ] || [ -z "$RESOLVED_URL" ]; then
    printf '%-12s invalid redirect response: expected HTTP 206 with 1 byte, got HTTP %s, %s bytes\n' \
      "$SOURCE_NAME" "$RESOLVE_HTTP_CODE" "$RESOLVE_SIZE_DOWNLOAD" >&2
    return 0
  fi

  if ! METRICS="$(
    curl --silent --show-error \
      --connect-timeout 8 \
      --max-time 25 \
      --range "0-$((PROBE_BYTES - 1))" \
      --max-filesize "$PROBE_BYTES" \
      --output /dev/null \
      --write-out $'%{http_code}\t%{speed_download}\t%{size_download}' \
      "$RESOLVED_URL"
  )"; then
    printf '%-12s unavailable\n' "$SOURCE_NAME" >&2
    return 0
  fi

  IFS=$'\t' read -r HTTP_CODE SPEED_DOWNLOAD SIZE_DOWNLOAD <<<"$METRICS"
  if [ "$HTTP_CODE" != "206" ] || [ "$SIZE_DOWNLOAD" != "$PROBE_BYTES" ]; then
    printf '%-12s invalid response: expected HTTP 206, got HTTP %s, %s bytes\n' \
      "$SOURCE_NAME" "$HTTP_CODE" "$SIZE_DOWNLOAD" >&2
    return 0
  fi

  if ! awk -v speed="$SPEED_DOWNLOAD" \
      'BEGIN { exit !(speed > 0) }'; then
    printf '%-12s invalid speed: %s\n' "$SOURCE_NAME" "$SPEED_DOWNLOAD" >&2
    return 0
  fi

  SPEED_MIB="$(
    awk -v speed="$SPEED_DOWNLOAD" 'BEGIN { printf "%.2f", speed / 1048576 }'
  )"
  printf '%s\t%s\n' "$SOURCE_NAME" "$SPEED_DOWNLOAD" >>"$PROBE_RESULTS"
  printf '%-12s %s MiB/s\n' "$SOURCE_NAME" "$SPEED_MIB"
}

probe_source huggingface \
  "https://huggingface.co/${MODEL_REPO}/resolve/main/${PROBE_FILE}?download=true"
probe_source hf_mirror \
  "https://hf-mirror.com/${MODEL_REPO}/resolve/main/${PROBE_FILE}?download=true"
probe_source modelscope \
  "https://www.modelscope.cn/models/${MODEL_REPO}/resolve/master/${PROBE_FILE}"

if [ ! -s "$PROBE_RESULTS" ]; then
  echo "三个公开源测速均失败，未开始下载" >&2
  exit 1
fi

echo "测速排名："
sort -t $'\t' -k2,2nr "$PROBE_RESULTS" | \
  awk -F '\t' '{ printf "  %-12s %.2f MiB/s\n", $1, $2 / 1048576 }'

FASTEST_SOURCE="$(
  sort -t $'\t' -k2,2nr "$PROBE_RESULTS" | \
    awk -F '\t' 'NR == 1 { print $1 }'
)"
echo "选择下载源: $FASTEST_SOURCE"

case "$FASTEST_SOURCE" in
  huggingface)
    if ! HF_ENDPOINT=https://huggingface.co \
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"; then
      echo "Hugging Face 下载失败" >&2
      exit 1
    fi
    ;;
  hf_mirror)
    if ! HF_ENDPOINT=https://hf-mirror.com \
        hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"; then
      echo "HF Mirror 下载失败" >&2
      exit 1
    fi
    ;;
  modelscope)
    if [ -d "$MODEL_DIR/.cache/huggingface" ]; then
      HF_CACHE_BACKUP="${MODEL_DIR}.hf-cache-$(date +%Y%m%d-%H%M%S)"
      if ! mv "$MODEL_DIR/.cache/huggingface" "$HF_CACHE_BACKUP"; then
        echo "无法保留 Hugging Face 断点数据: $HF_CACHE_BACKUP" >&2
        exit 1
      fi
      echo "已保留 Hugging Face 断点数据: $HF_CACHE_BACKUP"
    fi
    if ! modelscope download "$MODEL_REPO" \
        --local-dir "$MODEL_DIR" \
        --max-workers 4; then
      echo "ModelScope 下载失败" >&2
      exit 1
    fi
    ;;
  *)
    echo "未知下载源: $FASTEST_SOURCE" >&2
    exit 1
    ;;
esac

rm -f "$PROBE_RESULTS"
trap - EXIT

MODEL_FILES=(
  config.json
  model.safetensors.index.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  tokenizer.json
  tokenizer_config.json
)
for MODEL_FILE in "${MODEL_FILES[@]}"; do
  test -s "$MODEL_DIR/$MODEL_FILE" || {
    echo "模型文件缺失或为空: $MODEL_FILE" >&2
    exit 1
  }
done

if find "$MODEL_DIR" -name '*.incomplete' -print -quit | grep -q .; then
  echo "模型下载不完整" >&2
  exit 1
fi

du -sh "$MODEL_DIR"
)
```

测速只反映运行当时的网络状态。如果正式下载速度后来明显下降，先停止当前下载进程，再
重新运行本节。切换到 ModelScope 时只移动 Hugging Face 临时缓存到模型目录旁的时间戳
备份，不删除已下载数据。

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

## 6. 导入并生成项目部署包

外部的 `$DEMO_DIR` 是本次导入源，不是最终传输目录。回到统一项目，把已固定的源模型、
生成文件和 Demo 导入项目；导入不会修改外部工作区：

```bash
cd /home/barry/AI_Infra/RK_LLM
make host-import MODEL=qwen3_4b WORKSPACE=/home/barry/rk1828-work
make host-package MODEL=qwen3_4b

IMPORTED_DEMO="$PWD/artifacts/work/qwen3_4b/install/rknn_Qwen3_demo"
PACKAGE_ROOT="$PWD/artifacts/packages/qwen3_4b"
PACKAGE_DIR="$(find "$PACKAGE_ROOT" -mindepth 1 -maxdepth 1 -type d -print -quit)"

test -d "$IMPORTED_DEMO"
test -n "$PACKAGE_DIR"
.host-venv/bin/rk-llm package-validate --package "$PACKAGE_DIR"
```

导入后的证据目录固定为
`artifacts/work/qwen3_4b/install/rknn_Qwen3_demo`；最终可传输目录固定为
`artifacts/packages/qwen3_4b/<package_id>`。打包器只复制 Demo 可运行载荷，并验证
AArch64、GLIBC 2.35、GLIBCXX 3.4.30、大小和 SHA-256。再次运行两条 `make` 命令时，
导入应报告三类 `reused`，打包应报告 `reused`。

## 7. 传输到 RK3588

先确认端侧已通过 Git 同步同一项目，并已按 `docs/board-setup.md` 创建 `.venv`。只传输
`PACKAGE_DIR` 这一份不可变包，不传 `.vendor/`、外部工作区或整个 `artifacts/work/`：

```bash
RK3588_HOST=ubuntu@<RK3588_IP>
REMOTE_PROJECT=/home/ubuntu/RK_LLM
PACKAGE_ID="$(basename "$PACKAGE_DIR")"
REMOTE_INCOMING="$REMOTE_PROJECT/artifacts/deploy/.incoming-$PACKAGE_ID"
REMOTE_RELEASE="$REMOTE_PROJECT/artifacts/deploy/releases/$PACKAGE_ID"

ssh "$RK3588_HOST" "test ! -e '$REMOTE_INCOMING' && test ! -e '$REMOTE_RELEASE' && mkdir -p '$REMOTE_PROJECT/artifacts/deploy/releases'"
rsync -a --protect-args "$PACKAGE_DIR/" "$RK3588_HOST:$REMOTE_INCOMING/"
ssh "$RK3588_HOST" "cd '$REMOTE_PROJECT' && .venv/bin/rk-llm package-validate --package '$REMOTE_INCOMING' && mv '$REMOTE_INCOMING' '$REMOTE_RELEASE' && ln -s 'releases/$PACKAGE_ID' '$REMOTE_PROJECT/artifacts/deploy/.current-$PACKAGE_ID' && mv -Tf '$REMOTE_PROJECT/artifacts/deploy/.current-$PACKAGE_ID' '$REMOTE_PROJECT/artifacts/deploy/current'"
```

第二条 SSH 命令只有在端侧清单、文件大小和 SHA-256 全部通过后才发布 release，并通过
相对 `current` 链接原子切换。已存在的 incoming 或同 ID release 会直接拒绝，避免覆盖。

## 8. 板端运行

登录 RK3588：

```bash
ssh "$RK3588_HOST"
cd /home/ubuntu/RK_LLM/artifacts/deploy/current
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"

sudo rknn-smi info -l

./bin/rknn_qwen3_demo \
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
- `package-validate` 在宿主机 staging 和端侧 incoming 上均通过；
- 端侧 `artifacts/deploy/current` 是指向 `releases/$PACKAGE_ID` 的相对链接；
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
