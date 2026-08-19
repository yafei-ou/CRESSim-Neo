# Physics and Constraints

The physics solver is based on position-based dynamics (PBD), with compliant
XPBD-style constraints where needed. It supports both rigid collider and
particle-based dynamics. Rigid-body simulation supports sphere, box, capsule,
and proxy particle colliders, as well as ball, spherical, hinge, and slider
joints. Soft bodies can be represented either by tetrahedral particle models
with distance and volumetric constraints, or by meshfree particle models with
k-nearest-neighbor distance constraints. Fluids are handled within the same
particle framework using density constraints and viscosity, cohesion, surface
tension, and vorticity terms. The implementations are based largely on the
existing CPU-based [PositionBasedDynamics](https://animation.rwth-aachen.de/software/position-based-dynamics/)
codebase. Strands are modeled using segment, distance, bend, and twist
constraints, enabling cable- and thread-like behavior.

The engine also supports user-authored constraints for surgical task-specific
simulations. Routed cable constraints enforce a path length across guide points
attached to rigid bodies, which can be used to simulate cable-driven robots.
Rigid-particle and rigid-strand attachment constraints can connect surgical
tools, needles, and threads. Suturing simulation couples strand and rigid proxy
particles with soft tissue through path-following constraints.

```{figure} ../_static/physics-demo.png
:alt: Representative rigid-body, soft-body, joint, fluid, and suturing scenes.
:width: 100%

Representative physics scenes: (a) large-scale rigid-body simulation; (b) soft
bodies; (c) joints, from left to right: drivable spherical joints, ball joints,
hinge joints, and slider joints; (d) fluids; and (e) suturing with a needle,
thread, and soft body.
```

Because these features share the same parallel prediction, neighborhood and
contact generation, and iterative constraint projection pipeline, the physics
solver scales well to batched many-environment simulation.
