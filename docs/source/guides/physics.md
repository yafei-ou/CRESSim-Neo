# Physics

The position-based simulation pipeline supports these authored domains:

- rigid bodies, colliders, and joints;
- deformable soft bodies from tetrahedral or meshfree particles;
- fluids using particle constraints; and
- strands, attachments, and routed cables.

Start with rigid bodies, colliders, and joints. Add particle domains and
cross-domain constraints only after the basic scene is stable. Keep scene
capacities and particle budgets explicit when scaling a batched workload.

Relevant C++ examples: `basic.cpp`, `rigid_joints.cpp`,
`soft_particles_multi_env.cpp`, `fluid_scale_container.cpp`, and
`soft_body_arc_needle_thread_kinematic.cpp`.
