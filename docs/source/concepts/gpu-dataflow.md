# GPU dataflow

Physics, rendering, and task logic are designed to remain on the GPU. Prepared
layout mappings identify packed simulation state; render targets provide RGB,
depth, segmentation, and ultrasound outputs.

For task-specific data, allocate shared buffers and use a custom compute pass.
With CUDA interop enabled, shared buffers can also be exposed to CUDA and
PyTorch through DLPack. Synchronize explicitly when exchanging ownership
between the runtime and CUDA.

Use host readback for inspection or logging, not as the default learning-loop
path.
