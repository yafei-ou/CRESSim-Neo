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

set(TWO_ENV_TARGET_NAME "cressim_neo_two_env_viewer")

add_executable(${TWO_ENV_TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/debug_viewer/two_env_debug_viewer.cpp
)

target_include_directories(${TWO_ENV_TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${TWO_ENV_TARGET_NAME} PRIVATE
    cressim_neo_viewer
)

set_target_properties(${TWO_ENV_TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set(TEXTURE_VIEWER_TARGET_NAME "cressim_neo_texture_viewer")

add_executable(${TEXTURE_VIEWER_TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/debug_viewer/texture_debug_viewer.cpp
)

target_include_directories(${TEXTURE_VIEWER_TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${TEXTURE_VIEWER_TARGET_NAME} PRIVATE
    cressim_neo_viewer
)

set_target_properties(${TEXTURE_VIEWER_TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
