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
    NAME physics_soft_particles_multi_env
    SOURCE soft_particles_multi_env.cpp
)

add_cressim_example(
    NAME physics_soft_particles_toroid_multi_env
    SOURCE soft_particles_toroid_multi_env.cpp
)
