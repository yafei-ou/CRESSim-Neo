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

.. raw:: html

   <video controls preload="metadata" style="width: 100%; max-width: 1280px;">
     <source src="_static/CRESSim-Neo-Public.webm" type="video/webm">
     Your browser does not support the WebM video format.
   </video>

The engine is intended as a foundational simulation engine on which surgical
scenes and applications can be built. Its C++ API is the primary interface,
while the Python bindings expose the same runtime model for applications and
learning workflows. Direct access to physics and rendering buffers supports
custom GPU computation and, when CUDA interop is enabled, zero-copy PyTorch
integration using DLPack.

.. note::

   CRESSim-Neo was developed at the `Telerobotic and Biorobotic Systems (TBS)
   Group <https://www.ece.ualberta.ca/~tbs/pmwiki/>`_ at the University of
   Alberta.

.. warning::

   Coding agents assisted with the development of this project. Exercise care
   when using, reviewing, or modifying it. It is provided **"AS IS," without
   warranty of any kind**; see the :doc:`Apache 2.0 License
   <getting-started/license>`, including its disclaimer of warranty and
   limitation of liability.

The runtime has two phases: scene authoring and frame stepping. You configure a
scene through the C++ or Python API, prepare and upload its GPU layout, then
step physics, simulation sensors, visual sensors, and optional custom compute
passes. These stages make synchronization boundaries explicit and allow custom
GPU computations to be interleaved with simulation.

Start with :doc:`Getting Started <getting-started/index>` to build the C++ SDK
or install the Python bindings. Read the :doc:`User Guide <guides/index>` for
the runtime model and scene-authoring workflow. The :doc:`Developer Guide
<developers/index>` covers contributions, releases, and compliance. Use the
:doc:`API Reference <reference/index>` for complete interface details.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started/index
   guides/index
   developers/index
   reference/index
