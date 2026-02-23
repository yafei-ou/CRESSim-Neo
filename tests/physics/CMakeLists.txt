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
