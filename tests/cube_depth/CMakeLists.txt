set(TARGET_NAME "cressim_neo_cube_depth")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/cube_depth/cube_depth.cpp
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
    NAME cressim_neo_cube_depth_null
    COMMAND ${TARGET_NAME} --backend null --frames 2
)
