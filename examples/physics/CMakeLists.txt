include(${PROJECT_SOURCE_DIR}/examples/helpers/example_targets.cmake)

add_cressim_example(
    NAME physics_basic
    SOURCE basic.cpp
)

add_cressim_example(
    NAME physics_pairs
    SOURCE pairs.cpp
)

add_cressim_example(
    NAME physics_reallocation_during_simulation
    SOURCE reallocation_during_simulation.cpp
)

add_cressim_example(
    NAME physics_large_array_multi_env
    SOURCE large_array_multi_env.cpp
)

add_cressim_example(
    NAME physics_mixed_shape_contacts
    SOURCE mixed_shape_contacts.cpp
)

add_cressim_example(
    NAME physics_rigid_joints
    SOURCE rigid_joints.cpp
)

add_cressim_example(
    NAME physics_rigid_proxy_particles
    SOURCE rigid_proxy_particles.cpp
)

add_cressim_example(
    NAME physics_claw_machine_grasp
    SOURCE claw_machine_grasp.cpp
)

add_cressim_example(
    NAME physics_soft_particles_multi_env
    SOURCE soft_particles_multi_env.cpp
)

add_cressim_example(
    NAME physics_soft_body_arc_needle
    SOURCE soft_body_arc_needle.cpp
)

add_cressim_example(
    NAME physics_soft_body_arc_needle_kinematic
    SOURCE soft_body_arc_needle_kinematic.cpp
)

add_cressim_example(
    NAME physics_soft_body_arc_needle_thread_kinematic
    SOURCE soft_body_arc_needle_thread_kinematic.cpp
)

add_cressim_example(
    NAME physics_strand_debug
    SOURCE strand_debug.cpp
)

add_cressim_example(
    NAME physics_strand_suturing_path_follow
    SOURCE strand_suturing_path_follow.cpp
)

add_cressim_example(
    NAME physics_soft_particles_toroid_multi_env
    SOURCE soft_particles_toroid_multi_env.cpp
)

add_cressim_example(
    NAME physics_soft_particles_ultrasound_multi_env
    SOURCE soft_particles_ultrasound_multi_env.cpp
)

add_cressim_example(
    NAME physics_fluid_debug_particles
    SOURCE fluid_debug_particles.cpp
)

add_cressim_example(
    NAME physics_fluid_material_showcase_multi_env
    SOURCE fluid_material_showcase_multi_env.cpp
)

add_cressim_example(
    NAME physics_fluid_tall_block_stress
    SOURCE fluid_tall_block_stress.cpp
)

add_cressim_example(
    NAME physics_fluid_moving_splitter
    SOURCE fluid_moving_splitter.cpp
)

add_cressim_example(
    NAME physics_fluid_lighting_showcase
    SOURCE fluid_lighting_showcase.cpp
)
