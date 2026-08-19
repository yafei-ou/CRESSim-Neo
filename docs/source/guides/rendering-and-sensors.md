# Rendering and Sensors

After authored state is uploaded and physics is stepped, device-side copies are
dispatched to update entity poses based on the physical state. Visual sensors
can then be produced from the current GPU scene state with
`stepVisualSensors()`, and computational sensors such as ultrasound are
executed in a separate sensor stage with `stepSimulationSensors()`. See
{doc}`runtime-lifecycle` for the staged execution model.

For visual sensing, cameras support three output products: RGB-D, depth-only,
and segmentation-with-depth. Cameras can render either to renderer-managed
targets for presentation to screen, or to explicitly defined render targets. The
latter enables cameras with matched configurations to be grouped and rendered
in batches into layered array textures spanning multiple environments. GPU-side
camera preparation and indirect drawing further reduce the cost. Deformable
soft-body surfaces and strands are rendered directly from physics-updated GPU
buffers, while fluids are composited through dedicated depth, filtering, and
color passes. The underlying physical models are described in
{doc}`physics-and-constraints`.

Ultrasound image synthesis using the COLE algorithm is the supported
computational sensor through an external CUDA extension, CRESSim-Ultrasound.
Probe geometry, scanline layout, and image dimensions are defined by the user,
and RF data are generated from the current scatterer position. The scatterers
deform together with the underlying soft-body particles, allowing the ultrasound
image to respond in real time to tissue deformation. Its CUDA integration is
described in {doc}`gpu-integration-and-custom-compute`.

```{figure} ../_static/graphics-demo.png
:alt: Camera RGB, segmentation, depth, ultrasound probe scene, and synthesized ultrasound output.
:width: 100%

Example sensor outputs. Left: camera RGB rendering with corresponding
segmentation and depth products below. Middle: RGB rendering of the ultrasound
probe scene. Right: synthesized ultrasound image.
```

Sensor outputs can be consumed either through render-target readback to the host
or directly on GPU, enabling downstream custom compute, CUDA/Torch interop, and
learning loops without unnecessary host transfers. See
{doc}`gpu-integration-and-custom-compute` for GPU-side resource access.
