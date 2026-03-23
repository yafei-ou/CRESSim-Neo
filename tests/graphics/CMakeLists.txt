set(TARGET_NAME "cressim_neo_render_target_resize_policy")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/graphics/render_target_resize_policy.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_engine)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(
    NAME cressim_neo_render_target_resize_policy_vulkan
    COMMAND ${TARGET_NAME}
)

set(TARGET_NAME "cressim_neo_device_presentation_policy")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/graphics/device_presentation_policy.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_engine)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(
    NAME cressim_neo_device_presentation_policy_vulkan
    COMMAND ${TARGET_NAME}
)

set(TARGET_NAME "cressim_neo_render_plan_viewport_policy")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/graphics/render_plan_viewport_policy.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_graphics)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(
    NAME cressim_neo_render_plan_viewport_policy
    COMMAND ${TARGET_NAME}
)

set(TARGET_NAME "cressim_neo_explicit_surface_viewport_policy")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/graphics/explicit_surface_viewport_policy.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_engine)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(
    NAME cressim_neo_explicit_surface_viewport_policy_vulkan
    COMMAND ${TARGET_NAME}
)

set(TARGET_NAME "cressim_neo_display_resolve_color_space")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/graphics/display_resolve_color_space.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET_NAME} PRIVATE cressim_neo_engine)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(
    NAME cressim_neo_display_resolve_color_space_vulkan
    COMMAND ${TARGET_NAME}
)
