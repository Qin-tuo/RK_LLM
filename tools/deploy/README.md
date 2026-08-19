# Board Deployment Boundary

This directory is reserved for a future authenticated package-transfer and
activation tool. It contains no deployment script in the current milestone.

Deployment-package schema validation exists elsewhere in the project, but no
implemented target yet builds a complete package or transfers one to RK3588.
The future tool must verify an immutable package manifest and every payload hash
before transfer, check the uploaded files before activation, retain rollback
information, and avoid changing board drivers, Runtime, transport services, or
firmware as an implicit side effect.

Git synchronization carries tracked application logic and configuration only;
binary packages will use the separate deployment path. Incremental package
transfer and the first RK3588-to-RK1828 board inference have not started and are
not verified. See the [board setup status](../../docs/board-setup.md).
