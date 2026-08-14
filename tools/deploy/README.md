# Board Deployment Boundary

- Intended environment: a deployment host with authenticated access to an RK3588 board.
- Input artifact: the verified `.rkllm` package, manifest, and externally supplied RKLLM Runtime `1.3.0` files.
- Output artifact: an explicitly versioned application directory on the RK3588 board.
- Official upstream command family: the runtime-library and model deployment layout used by the `airockchip/rknn-llm` RKLLM API examples for release `1.3.0`.

This directory has no deployment script yet. A future script must verify the manifest and checksum before transfer, must not install a driver implicitly, and must not report success until the board-side files are checked.
