function(add_engine_test target_name source_file)
    add_executable(${target_name}
        ${PROJECT_SOURCE_DIR}/tests/engine/${source_file}
    )

    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(${target_name} PRIVATE cressim_neo_engine)

    set_target_properties(${target_name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_engine_test(cressim_neo_world_sync_incremental world_sync_incremental.cpp)
add_test(
    NAME cressim_neo_world_sync_incremental_vulkan
    COMMAND cressim_neo_world_sync_incremental
)

add_engine_test(cressim_neo_runtime_physics_sync_fields runtime_physics_sync_fields.cpp)
add_test(
    NAME cressim_neo_runtime_physics_sync_fields
    COMMAND cressim_neo_runtime_physics_sync_fields
)

add_engine_test(cressim_neo_camera_viewport_aspect_policy camera_viewport_aspect_policy.cpp)
add_test(
    NAME cressim_neo_camera_viewport_aspect_policy
    COMMAND cressim_neo_camera_viewport_aspect_policy
)

add_engine_test(cressim_neo_direct_physics_authoring direct_physics_authoring.cpp)
add_test(
    NAME cressim_neo_direct_physics_authoring
    COMMAND cressim_neo_direct_physics_authoring
)

add_engine_test(cressim_neo_world_deferred_render_reconciliation world_deferred_render_reconciliation.cpp)
add_test(
    NAME cressim_neo_world_deferred_render_reconciliation
    COMMAND cressim_neo_world_deferred_render_reconciliation
)

add_engine_test(cressim_neo_environment_ibl_state environment_ibl_state.cpp)
add_test(
    NAME cressim_neo_environment_ibl_state
    COMMAND cressim_neo_environment_ibl_state
)

add_engine_test(cressim_neo_logger logger.cpp)
add_test(
    NAME cressim_neo_logger
    COMMAND cressim_neo_logger
)

add_engine_test(cressim_neo_unified_light_slots unified_light_slots.cpp)
add_test(
    NAME cressim_neo_unified_light_slots
    COMMAND cressim_neo_unified_light_slots
)

add_engine_test(cressim_neo_local_shadow_registry local_shadow_registry.cpp)
add_test(
    NAME cressim_neo_local_shadow_registry
    COMMAND cressim_neo_local_shadow_registry
)
