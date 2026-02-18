set(TARGET_NAME "cressim_neo_debug_viewer")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)

if(NOT TARGET glfw)
    add_subdirectory(
        ${PROJECT_SOURCE_DIR}/extern/DiligentEngine/DiligentSamples/ThirdParty/glfw
        ${CMAKE_BINARY_DIR}/extern/glfw
        EXCLUDE_FROM_ALL
    )
endif()

add_executable(${TARGET_NAME}
    ${PROJECT_SOURCE_DIR}/tests/debug_viewer/debug_viewer.cpp
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        GLFW_EXPOSE_NATIVE_WIN32
    )
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        GLFW_EXPOSE_NATIVE_X11
    )
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    cressim_neo_engine
    glfw
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
