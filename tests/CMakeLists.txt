include(${CMAKE_CURRENT_LIST_DIR}/helpers/test_targets.cmake)

add_library(cressim_neo_graphics_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/graphics/render_plan_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/services/texture_gpu_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/passes/material_program_registry.cpp
)
target_include_directories(cressim_neo_graphics_test_support PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(cressim_neo_graphics_test_support PUBLIC
    cressim_neo_graphics
)

add_library(cressim_neo_physics_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/physics/rigid_body_common.cpp
    ${PROJECT_SOURCE_DIR}/src/physics/physics_scene_gpu_state.cpp
)
target_include_directories(cressim_neo_physics_test_support PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(cressim_neo_physics_test_support PUBLIC
    cressim_neo_physics
)

add_subdirectory(smoke)
add_subdirectory(forward_pipeline)
add_subdirectory(graphics)
add_subdirectory(engine)
add_subdirectory(physics)
