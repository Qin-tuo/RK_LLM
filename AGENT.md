# RK_LLM Agent 工作协议

本文件是宿主机 AI 与 RK3588 端侧 AI 的共同任务边界。任何 AI 在修改、编译、
传输、激活或运行前，必须先阅读本文件以及相关 `docs/*.md`。

## 1. 项目事实

| 项目 | 当前值 |
| --- | --- |
| Git 项目 | `/home/barry/AI_Infra/RK_LLM` |
| 宿主机外部工作区 | `/home/barry/rk1828-work` |
| RK3588 SSH | `ubuntu@172.16.51.112` |
| RK3588 项目 | `/home/ubuntu/userdata/RK_LLM` |
| 当前模型 | `qwen3_4b` (`Qwen/Qwen3-4B`) |
| 当前包 ID | `4938370a62fd618b` |
| 当前激活链接 | `artifacts/deploy/current -> releases/4938370a62fd618b` |
| 目标主 SoC | RK3588 |
| 加速器 | RK1828 |
| 构建架构 | aarch64 |
| 端侧基线 | Ubuntu 22.04 / glibc 2.35 |

`target.compiler_platform`、模型 revision、工具链 revision、文件大小和 SHA-256
都由 manifest/schema 固定。不要凭印象修改这些值；当前 Qwen3 manifest/package
记录的 compiler platform 是 `rk1820`，实际加速器目标是 RK1828。

当前包是已验证的部署基线，不代表已经完成真实硬件推理。项目的
`RKNN3Backend`/Native runner 仍是 guarded boundary；Vendor Demo 二进制可以单独
运行，但不能据此宣称 `rk-llm generate --backend rknn3` 已实现。

## 2. 总原则

1. GitHub 只同步受 Git 跟踪的代码、配置、manifest、测试和文档。
2. `.vendor/`、模型权重、转换中间文件、`artifacts/work`、`artifacts/packages`
   和 `artifacts/deploy` 都是本地或传输数据，不提交 Git。
3. 宿主机生产一个不可变 package；端侧只接收、校验、激活和运行 package。
4. 不在端侧重新下载模型、重新导出、重新编译或修改宿主机产物。
5. 不升级端侧 glibc 解决兼容性问题；遇到 `GLIBC_2.38`，回宿主机用 Ubuntu
   22.04 重新构建。
6. 不覆盖已有 release，不把新文件直接写入 `current`，不删除当前激活版本。

## 3. 宿主机 AI 职责

宿主机负责完整的输入到交付流程：

- 下载或准备 `Qwen/Qwen3-4B`，确认 manifest revision 和文件 pins；
- 运行 `make host-bootstrap`，维护 `.vendor/` 中的 RKNN3 Toolkit、Model Zoo
  和 Runtime 开发文件；
- 在外部 workspace 完成导出、GRQ、RKNN 编译和 Ubuntu 22.04/aarch64 Demo 构建；
- 将外部已验证文件导入统一项目：

  ```bash
  cd /home/barry/AI_Infra/RK_LLM
  make host-import MODEL=qwen3_4b WORKSPACE=/home/barry/rk1828-work
  ```

- 构建并验证不可变 package：

  ```bash
  make host-package MODEL=qwen3_4b
  .host-venv/bin/rk-llm package-validate \
    --package artifacts/packages/qwen3_4b/<package_id>
  ```

- 检查 package profile、entrypoint、AArch64、GLIBC/GLIBCXX ceiling、大小和
  SHA-256 后，传输唯一 package 目录给端侧；
- 记录 package ID、宿主机 Git commit、命令、输出和未解决风险。

宿主机禁止：

- 把整个 `rknn3-model-zoo`、`.vendor/` 或 `artifacts/work` 复制到端侧；
- 在端侧执行模型下载、导出、量化或交叉编译；
- 为了让 Demo 运行而修改端侧 glibc、驱动、Runtime 或固件；
- 在没有用户明确批准的情况下删除外部 workspace 或项目 artifacts。

## 4. RK3588 端侧 AI 职责

端侧负责接收后的验证、激活和运行：

- 只通过 Git 同步 tracked project code；先检查本地改动，禁止无提示覆盖；
- 不复制宿主机 `.vendor/`、外部模型目录或宿主机 build cache；
- 将 package 接收到临时目录，验证成功后再移动为 release；
- 使用项目 CLI 校验 package。当前端侧命令为：

  ```bash
  cd /home/ubuntu/userdata/RK_LLM
  .venv/bin/rk-llm package-validate \
    --package artifacts/deploy/.incoming-<package_id>
  ```

- 校验成功后原子发布并切换相对链接：

  ```bash
  PACKAGE_ID=<package_id>
  PROJECT=/home/ubuntu/userdata/RK_LLM
  mv "$PROJECT/artifacts/deploy/.incoming-$PACKAGE_ID" \
     "$PROJECT/artifacts/deploy/releases/$PACKAGE_ID"
  ln -s "releases/$PACKAGE_ID" \
     "$PROJECT/artifacts/deploy/.current-$PACKAGE_ID"
  mv -Tf "$PROJECT/artifacts/deploy/.current-$PACKAGE_ID" \
     "$PROJECT/artifacts/deploy/current"
  ```

- 从 `artifacts/deploy/current` 启动 Demo，收集 Runtime、PCIe、固件、显存、
  TTFT、Decode TPS 和完整错误日志；
- 反馈实际运行结果，不把 package validation 或 ELF 检查写成硬件推理成功。

端侧运行入口：

```bash
cd /home/ubuntu/userdata/RK_LLM/artifacts/deploy/current
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./bin/rknn_qwen3_demo \
  model/Qwen3-4B.rknn \
  model/Qwen3-4B.weight \
  model/Qwen3-4B.tokenizer.gguf \
  model/Qwen3-4B.embed.bin \
  0xff \
  "你好，请用三句话介绍你自己。"
```

端侧禁止：

- `git reset --hard`、`git checkout --`、强制 pull 或覆盖其他 AI/用户的本地改动；
- 删除 `current` 指向的 release；
- 修改 package 内文件、manifest 或 package ID；
- 直接从 `artifacts/work`、外部 workspace 或 Model Zoo 安装目录运行 Demo；
- 在没有日志和版本信息的情况下声称“板端已跑通”。

## 5. Package 生命周期

```text
宿主机:
  external workspace
      -> artifacts/work/qwen3_4b/install/rknn_Qwen3_demo
      -> artifacts/packages/qwen3_4b/<package_id>
      -> 传输

端侧:
  artifacts/deploy/.incoming-<package_id>
      -> package-validate
      -> artifacts/deploy/releases/<package_id>
      -> artifacts/deploy/current
```

- `work` 是导入/构建证据，不是部署目录。
- `packages` 是宿主机生成的不可变交付物。
- `.incoming-*` 是未验证临时目录，失败时不得激活。
- `releases/<package_id>` 是正式版本，不能原地修改。
- `current` 是相对符号链接，不是第二份 package；运行始终从这里进入。
- 新版本必须使用新 package ID；同 ID 目录如果内容不同，必须报冲突并停止。

## 6. Git 同步规则

执行同步前两侧都先运行：

```bash
git status --short --branch
git rev-parse HEAD
```

只有工作树没有需要保留的 tracked 改动时，才允许：

```bash
git pull --ff-only origin main
```

端侧已有的本地文档修改必须保留并主动报告。例如
`docs/rk1828-rknn3-deployment.md` 出现 `M` 时，不得 reset 或强制覆盖；宿主机 AI
应先询问是否需要合并该文档改动。

包、模型和日志不参与 Git 同步。不要用 Git 解决二进制 package 传输问题。

## 7. AI 交接格式

每次交接必须包含以下字段，避免两侧重复工作：

```text
ROLE: host | board
PROJECT: <absolute path>
GIT_HEAD: <40-char commit>
PACKAGE_ID: <id or none>
INPUT: <source/work/incoming/release/current path>
COMMANDS: <actual commands, not intended commands>
RESULT: imported | reused | validated | activated | ran | failed
EVIDENCE: <key output, sizes, hashes, versions, logs>
LOCAL_CHANGES: <tracked changes preserved, or clean>
NEXT_ACTION: <one concrete next step>
BLOCKER: <none or exact external requirement>
```

禁止只报告“完成”“已同步”“可运行”。必须给出实际路径、package ID、校验结果和
下一步。硬件推理未执行时，明确写 `RESULT: validated/activated; hardware inference not verified`。

## 8. 完成标准

宿主机任务只有在以下条件满足时才算完成：

- manifest pins 校验通过；
- `host-import` 首次为 `imported`、重复执行为 `reused`；
- `host-package` 首次为 `created`、重复执行为 `reused`；
- package-validate 通过且 package ID 已记录；
- 传输前外部源文件哈希未变化。

端侧任务只有在以下条件满足时才算完成：

- SSH/Git 同步状态已记录且本地改动未被覆盖；
- incoming package 校验通过；
- release 目录和 `current` 相对链接均正确；
- Demo 启动命令、Runtime/固件版本和日志已记录；
- 只有在实际产生 token 并有设备证据时，才报告硬件推理成功。

