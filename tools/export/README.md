# Model Export Boundary

- Intended environment: a supported Linux model-conversion host, separate from the RK3588 board runtime.
- Input artifact: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- Output artifact: a `.rkllm` model and its conversion manifest under `artifacts/packages/`.
- Official upstream command family: the RKLLM-Toolkit Python conversion API and examples in `airockchip/rknn-llm` release `1.3.0`.

This skeleton deliberately contains no conversion wrapper. Follow [the conversion guide](../../docs/model-export.md), verify the upstream command for the pinned release, and record every option and checksum before adding one.
