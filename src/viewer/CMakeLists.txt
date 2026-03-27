file(GLOB_RECURSE PROJECT_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/viewer/*.h
)

file(GLOB_RECURSE TARGET_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/viewer/*.h
)

file(GLOB_RECURSE TARGET_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/viewer/*.cpp
)

if(APPLE)
    file(GLOB_RECURSE TARGET_SOURCE_APPLE CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/src/viewer/*.mm
    )
    list(APPEND TARGET_SOURCE ${TARGET_SOURCE_APPLE})
endif()

set(TARGET_NAME "cressim_neo_viewer")

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

if(ENGINE_STATIC)
    add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE})
else()
    add_library(${TARGET_NAME} SHARED ${TARGET_SOURCE})
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_VIEWER_EXPORT)
    set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    if(WIN32)
        target_sources(${TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR}/generated/version.rc)
    endif()
endif()

target_sources(${TARGET_NAME} PRIVATE ${PROJECT_HEADER} ${TARGET_HEADER})

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${PROJECT_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src/viewer"
    PREFIX "Header Files"
    FILES ${TARGET_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src/viewer"
    PREFIX "Source Files"
    FILES ${TARGET_SOURCE}
)

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        GLFW_EXPOSE_NATIVE_WIN32
    )
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        GLFW_EXPOSE_NATIVE_X11
    )
elseif(APPLE)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        GLFW_EXPOSE_NATIVE_COCOA
    )
endif()

target_link_libraries(${TARGET_NAME}
PUBLIC
    cressim_neo_engine
PRIVATE
    glfw
)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
