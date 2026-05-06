set(TARGET_NAME "cressim_neo_explicit_surface_depth_readback")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/explicit_surface_depth_readback/explicit_surface_depth_readback.cpp
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
    NAME cressim_neo_explicit_surface_depth_readback_vulkan
    COMMAND ${TARGET_NAME} --backend vulkan --frames 2
)
