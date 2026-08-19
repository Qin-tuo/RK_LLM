# Vendor Header Boundary

This directory is intentionally empty. RKNN3 Runtime headers, including `rknn3_api.h`, are supplied externally by the official Rockchip SDK or runtime release and are never copied into this repository.

The current C++ executable does not include vendor headers, link vendor libraries, load a model, or perform inference. It is an explicit unavailable stub. A later native-runtime milestone must add documented CMake inputs for the external include and library paths before replacing that stub.
