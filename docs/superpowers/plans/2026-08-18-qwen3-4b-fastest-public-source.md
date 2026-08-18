# Qwen3-4B Fastest Public Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Qwen3-4B quick deployment guide so every model download first benchmarks three public sources and automatically uses the fastest valid source.

**Architecture:** Keep the behavior entirely inside the guide's existing download section. A Bash function requests the same 1 MiB byte range from Hugging Face, HF Mirror, and ModelScope, records only exact HTTP 206 samples, selects the highest measured speed, and dispatches to the matching installed CLI while preserving interrupted-download state.

**Tech Stack:** Markdown, Bash, `curl`, Hugging Face CLI (`hf`), ModelScope CLI (`modelscope`), standard Unix tools.

---

## File Structure

- Modify `docs/rk1828-qwen3-4b-quick-deployment.md`: replace section 2 with dependency checks, concurrent-download protection, public-source benchmarking, automatic source dispatch, cache backup, and exact model completeness checks.
- Do not create a standalone downloader script: the requested deliverable is a concise copy-and-run deployment guide, and one embedded Bash block keeps its variables and execution order visible.
- Do not change sections 3-10: GRQ export, RKNN3 compilation, ARM64 build, transfer, and runtime behavior are outside this update.

### Task 1: Define And Demonstrate The Documentation Contract

**Files:**
- Test: `docs/rk1828-qwen3-4b-quick-deployment.md` using read-only shell assertions

- [ ] **Step 1: Run the source-selection contract against the current guide**

```bash
DOC=docs/rk1828-qwen3-4b-quick-deployment.md

test "$(rg -c '^probe_source (huggingface|hf_mirror|modelscope) ' "$DOC")" -eq 3
rg -q 'HF_ENDPOINT=https://hf-mirror.com' "$DOC"
rg -q 'modelscope download "\$MODEL_REPO"' "$DOC"
rg -q 'HTTP 206' "$DOC"
```

Expected: FAIL on the first assertion because the current guide contains no `probe_source` calls.

### Task 2: Replace The Download Section

**Files:**
- Modify: `docs/rk1828-qwen3-4b-quick-deployment.md:43`

- [ ] **Step 1: Replace section 2, stopping before the existing section 3 heading**

Use this exact Markdown content:

````markdown
## 2. 测速并下载 Qwen3-4B

每次下载模型前都重新测试当前网络。脚本只接受同一大文件完整的 1 MiB Range 响应，自动
选择 Hugging Face、HF Mirror 和 ModelScope 中当前最快的公开源。运行前先停止同一模型
的旧下载任务，避免多个下载器同时写入 `MODEL_DIR`。

```bash
export MODEL_REPO="Qwen/Qwen3-4B"
export PROBE_FILE="model-00001-of-00003.safetensors"
export PROBE_BYTES=1048576

for COMMAND_NAME in curl hf modelscope pgrep awk sort; do
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

mkdir -p "$MODEL_DIR"
PROBE_RESULTS="$(mktemp)"
trap 'rm -f "$PROBE_RESULTS"' EXIT

probe_source() {
  local SOURCE_NAME="$1"
  local SOURCE_URL="$2"
  local METRICS HTTP_CODE SPEED_DOWNLOAD SIZE_DOWNLOAD SPEED_MIB

  if ! METRICS="$(
    curl --location --silent --show-error \
      --connect-timeout 8 \
      --max-time 25 \
      --range "0-$((PROBE_BYTES - 1))" \
      --output /dev/null \
      --write-out $'%{http_code}\t%{speed_download}\t%{size_download}' \
      "$SOURCE_URL"
  )"; then
    printf '%-12s unavailable\n' "$SOURCE_NAME" >&2
    return 0
  fi

  IFS=$'\t' read -r HTTP_CODE SPEED_DOWNLOAD SIZE_DOWNLOAD <<<"$METRICS"
  if [ "$HTTP_CODE" != "206" ] || [ "$SIZE_DOWNLOAD" != "$PROBE_BYTES" ]; then
    printf '%-12s invalid response: HTTP %s, %s bytes\n' \
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
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    ;;
  hf_mirror)
    HF_ENDPOINT=https://hf-mirror.com \
      hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    ;;
  modelscope)
    if [ -d "$MODEL_DIR/.cache/huggingface" ]; then
      HF_CACHE_BACKUP="${MODEL_DIR}.hf-cache-$(date +%Y%m%d-%H%M%S)"
      mv "$MODEL_DIR/.cache/huggingface" "$HF_CACHE_BACKUP"
      echo "已保留 Hugging Face 断点数据: $HF_CACHE_BACKUP"
    fi
    modelscope download "$MODEL_REPO" \
      --local-dir "$MODEL_DIR" \
      --max-workers 4
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
```

测速只反映运行当时的网络状态。如果正式下载速度后来明显下降，先停止当前下载进程，再
重新运行本节。切换到 ModelScope 时只移动 Hugging Face 临时缓存到模型目录旁的时间戳
备份，不删除已下载数据。
````

- [ ] **Step 2: Inspect the scoped diff**

```bash
git diff -- docs/rk1828-qwen3-4b-quick-deployment.md
```

Expected: only section 2 changes; headings and content in sections 1 and 3-10 remain unchanged.

### Task 3: Validate The Embedded Script And Repository

**Files:**
- Test: `docs/rk1828-qwen3-4b-quick-deployment.md`

- [ ] **Step 1: Parse every Bash block without executing downloads**

```bash
awk '
  /^```bash$/ { in_bash = 1; next }
  in_bash && /^```$/ { in_bash = 0; next }
  in_bash { print }
' docs/rk1828-qwen3-4b-quick-deployment.md | bash -n
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Re-run the source-selection contract**

```bash
DOC=docs/rk1828-qwen3-4b-quick-deployment.md

test "$(rg -c '^probe_source (huggingface|hf_mirror|modelscope) ' "$DOC")" -eq 3
rg -q 'HF_ENDPOINT=https://hf-mirror.com' "$DOC"
rg -q 'modelscope download "\$MODEL_REPO"' "$DOC"
rg -q '"\$HTTP_CODE" != "206"' "$DOC"

for MODEL_FILE in \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors; do
  rg -q "$MODEL_FILE" "$DOC"
done
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Run whitespace and non-hardware regression checks**

```bash
git diff --check
python3 -m pytest -m "not hardware" -q
```

Expected: `git diff --check` exits 0; pytest reports 111 passed and 1 deselected.

- [ ] **Step 4: Commit the guide update**

```bash
git add docs/rk1828-qwen3-4b-quick-deployment.md
git commit -m "docs: select fastest public model source"
```

Expected: one commit containing only the quick-guide update.
