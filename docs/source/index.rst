CRESSim-Neo Documentation
=========================

CRESSim-Neo is a GPU-accelerated position-based dynamics (PBD) simulation
engine for surgical robotics and robot learning. It combines position-based
simulation of rigid bodies, deformable tissues, fluids, and strands with
batched rendering, surgery-specific sensing, and a GPU-resident data pipeline.
The engine supports applications including tissue manipulation, fluid suction,
suturing, cable-driven robots, and ultrasound image synthesis.

CRESSim-Neo supports Linux, Windows, and macOS. Vulkan is the standard graphics
backend, and Direct3D 12 is also available on Windows. On macOS, Vulkan is
provided through MoltenVK; platform validation remains limited.

The engine is intended as a foundational simulation engine on which surgical
scenes and applications can be built. Its C++ API is the primary interface,
while the Python bindings expose the same runtime model for applications and
learning workflows. Direct access to physics and rendering buffers supports
custom GPU computation and, when CUDA interop is enabled, zero-copy PyTorch
integration using DLPack.

The runtime has two phases: scene authoring and frame stepping. You configure a
scene through the C++ or Python API, prepare and upload its GPU layout, then
step physics, simulation sensors, visual sensors, and optional custom compute
passes. These stages make synchronization boundaries explicit and allow custom
GPU computations to be interleaved with simulation.

Start with :doc:`Getting Started <getting-started/index>` to build the C++ SDK
or install the Python bindings. Read the :doc:`User Guide <guides/index>` for
the runtime model and scene-authoring workflow, and use the :doc:`API Reference
<reference/index>` for complete interface details.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started/index
   guides/index
   reference/index
