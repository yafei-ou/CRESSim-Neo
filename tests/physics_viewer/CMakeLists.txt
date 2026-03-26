function(add_physics_viewer_test target_name source_file)
    add_executable(${target_name}
        ${PROJECT_SOURCE_DIR}/tests/physics_viewer/${source_file}
    )

    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(${target_name} PRIVATE cressim_neo_viewer)

    set_target_properties(${target_name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_physics_viewer_test(
    cressim_neo_physics_viewer_manual_vulkan
    physics_viewer.cpp
)

# Manual physics viewer executables. Intentionally not added to ctest.
add_physics_viewer_test(
    cressim_neo_physics_viewer_pairs_manual_vulkan
    physics_viewer_pairs.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_large_array_manual_vulkan
    physics_viewer_large_array.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_large_array_multi_env_manual_vulkan
    physics_viewer_large_array_multi_env.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_large_array_multi_env_mixed_lighting_manual_vulkan
    physics_viewer_large_array_multi_env_mixed_lighting.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_multi_env_light_matrix_manual_vulkan
    physics_viewer_multi_env_light_matrix.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_large_array_multi_env_ibl_manual_vulkan
    physics_viewer_large_array_multi_env_ibl.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_large_array_multi_env_shared_ibl_manual_vulkan
    physics_viewer_large_array_multi_env_shared_ibl.cpp
)

add_physics_viewer_test(
    cressim_neo_physics_viewer_mixed_shape_contacts_manual_vulkan
    physics_viewer_mixed_shape_contacts.cpp
)
