# Sensors and rendering

Visual cameras can produce RGB-D, depth-only, and segmentation-with-depth
outputs. Render targets may be used for display, readback, or device-side task
processing.

Ultrasound is a separate computational-sensor stage. It is optional at build
time and requires CUDA interop together with CRESSim-Ultrasound.

For visual output, start with `examples/graphics/camera_outputs.cpp`. For
ultrasound, start with `examples/physics/soft_particles_ultrasound_multi_env.cpp`
and use an ultrasound-enabled build.
