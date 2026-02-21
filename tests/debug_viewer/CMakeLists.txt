set(TARGET_NAME "cressim_neo_viewer_integration")

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/debug_viewer/debug_viewer.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${TARGET_NAME} PRIVATE
    cressim_neo_viewer
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
