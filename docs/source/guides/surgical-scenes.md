# Surgical scenes

Surgical scenes combine core engine features rather than using a separate
simulation mode: rigid instruments manipulate soft tissue, fluid, or strands;
cameras and ultrasound observe the result; custom GPU compute can define task
logic.

Start from the focused C++ examples:

- needle, thread, and soft body for suturing;
- routed cable constraints for continuum or cable-driven robots; and
- ultrasound scenes for computational sensing.

Pre-allocate likely objects and capacities when an episode requires conditional
events, because topology changes require a scene update.
