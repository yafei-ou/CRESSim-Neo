include(${PROJECT_SOURCE_DIR}/examples/helpers/example_targets.cmake)

add_cressim_example(
    NAME physics_viewer
    SOURCE physics_viewer.cpp
)

add_cressim_example(
    NAME physics_viewer_pairs
    SOURCE physics_viewer_pairs.cpp
)

add_cressim_example(
    NAME physics_viewer_reallocation_during_simulation
    SOURCE physics_viewer_reallocation_during_simulation.cpp
)

add_cressim_example(
    NAME physics_viewer_large_array_multi_env
    SOURCE physics_viewer_large_array_multi_env.cpp
)

add_cressim_example(
    NAME physics_viewer_skybox_ibl
    SOURCE physics_viewer_skybox_ibl.cpp
    COMPILE_DEFINITIONS CRESSIM_NEO_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_cressim_example(
    NAME physics_viewer_mixed_shape_contacts
    SOURCE physics_viewer_mixed_shape_contacts.cpp
)

add_cressim_example(
    NAME physics_viewer_rigid_joints
    SOURCE physics_viewer_rigid_joints.cpp
)

add_cressim_example(
    NAME physics_viewer_soft_particles_multi_env
    SOURCE physics_viewer_soft_particles_multi_env.cpp
)

add_cressim_example(
    NAME physics_viewer_soft_particles_toroid_multi_env
    SOURCE physics_viewer_soft_particles_toroid_multi_env.cpp
)
