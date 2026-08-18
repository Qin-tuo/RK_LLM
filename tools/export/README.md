# Model Export Boundary

This directory is reserved for a future reproducible RKNN3 export wrapper. It
contains no executable export tool in the current milestone.

The future wrapper will consume the Qwen source identity and file pins in
`configs/models/qwen2_5_0_5b.yaml`, use the Toolkit and Model Zoo revisions in
`manifests/upstream.yaml`, and write intermediate generated files only beneath
ignored `artifacts/work/`. It must record every option, environment identity,
log, size, and SHA-256 before output can become package input.

The existing host import can adopt already generated, pinned files for later
comparison, but it cannot regenerate them. The manual evidence covers source
export, GRQ, and RKNN compilation performed outside project automation; it does
not prove that this placeholder directory provides those operations. See the
[model input and export status](../../docs/model-export.md).
