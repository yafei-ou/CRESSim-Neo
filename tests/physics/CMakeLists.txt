function(add_physics_test target_name source_file)
    add_executable(${target_name}
        ${PROJECT_SOURCE_DIR}/tests/physics/${source_file}
    )

    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(${target_name} PRIVATE cressim_neo_engine)

    set_target_properties(${target_name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_physics_test(cressim_neo_physics_world_soa_layout physics_world_soa_layout.cpp)
add_test(
    NAME cressim_neo_physics_world_soa_layout
    COMMAND cressim_neo_physics_world_soa_layout
)

add_physics_test(cressim_neo_physics_solver_context_split physics_solver_context_split.cpp)
add_test(
    NAME cressim_neo_physics_solver_context_split_vulkan
    COMMAND cressim_neo_physics_solver_context_split
)

add_physics_test(cressim_neo_physics_placeholder_soa_compute physics_placeholder_soa_compute.cpp)
add_test(
    NAME cressim_neo_physics_placeholder_soa_compute_vulkan
    COMMAND cressim_neo_physics_placeholder_soa_compute
)

add_physics_test(cressim_neo_physics_solver_broad_phase_overflow physics_solver_broad_phase_overflow.cpp)
add_test(
    NAME cressim_neo_physics_solver_broad_phase_overflow_vulkan
    COMMAND cressim_neo_physics_solver_broad_phase_overflow
)

add_physics_test(cressim_neo_physics_solver_mixed_shape_contacts physics_solver_mixed_shape_contacts.cpp)
add_test(
    NAME cressim_neo_physics_solver_mixed_shape_contacts_vulkan
    COMMAND cressim_neo_physics_solver_mixed_shape_contacts
)

add_physics_test(cressim_neo_physics_body_types physics_body_types.cpp)
add_test(
    NAME cressim_neo_physics_body_types_vulkan
    COMMAND cressim_neo_physics_body_types
)

add_physics_test(cressim_neo_physics_soft_body_generation_regular_tet_grid physics_soft_body_generation_regular_tet_grid.cpp)
add_test(
    NAME cressim_neo_physics_soft_body_generation_regular_tet_grid
    COMMAND cressim_neo_physics_soft_body_generation_regular_tet_grid
)

add_physics_test(cressim_neo_physics_rigid_surface_particle_generation physics_rigid_surface_particle_generation.cpp)
add_test(
    NAME cressim_neo_physics_rigid_surface_particle_generation
    COMMAND cressim_neo_physics_rigid_surface_particle_generation
)

add_physics_test(cressim_neo_physics_soft_body_multi_env_isolation physics_soft_body_multi_env_isolation.cpp)
add_test(
    NAME cressim_neo_physics_soft_body_multi_env_isolation
    COMMAND cressim_neo_physics_soft_body_multi_env_isolation
)

add_physics_test(cressim_neo_physics_soft_body_analytic_narrow_phase physics_soft_body_analytic_narrow_phase.cpp)
add_test(
    NAME cressim_neo_physics_soft_body_analytic_narrow_phase
    COMMAND cressim_neo_physics_soft_body_analytic_narrow_phase
)

add_physics_test(cressim_neo_physics_soft_gpu_scene_upload physics_soft_gpu_scene_upload.cpp)
add_test(
    NAME cressim_neo_physics_soft_gpu_scene_upload_vulkan
    COMMAND cressim_neo_physics_soft_gpu_scene_upload
)
