file(GLOB_RECURSE PROJECT_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/gpu/*.h
)

file(GLOB_RECURSE TARGET_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/gpu/*.h
)

file(GLOB_RECURSE TARGET_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/gpu/*.cpp
)

set(TARGET_NAME "cressim_neo_gpu")

if(ENGINE_STATIC)
    add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE})
else()
    add_library(${TARGET_NAME} SHARED ${TARGET_SOURCE})
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_GPU_EXPORT)
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
    TREE "${PROJECT_SOURCE_DIR}/src/gpu"
    PREFIX "Header Files"
    FILES ${TARGET_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src/gpu"
    PREFIX "Source Files"
    FILES ${TARGET_SOURCE}
)

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/extern/DiligentEngine/DiligentCore/Common/interface
)

set(CRESSIM_NEO_SHADER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/assets/shaders")
target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_SHADER_SOURCE_DIR="${CRESSIM_NEO_SHADER_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME} PUBLIC cressim_neo_common Vulkan::Headers)

# Link with Diligent
target_link_libraries(${TARGET_NAME} PUBLIC Diligent-PublicBuildSettings)
if(TARGET Diligent-GraphicsEngineVk-static)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineVk-static)
elseif(TARGET Diligent-GraphicsEngineVk-shared)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineVk-shared)
else()
    message(FATAL_ERROR "Requires Vulkan Diligent backend target.")
endif()
if(TARGET Diligent-GraphicsTools)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsTools)
elseif(TARGET Diligent-GraphicsTools-static)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsTools-static)
elseif(TARGET Diligent-GraphicsTools-shared)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsTools-shared)
else()
    message(FATAL_ERROR "Requires Diligent GraphicsTools target.")
endif()
if(TARGET Diligent-Archiver-static)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-Archiver-static)
elseif(TARGET Diligent-Archiver-shared)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-Archiver-shared)
else()
    message(FATAL_ERROR "Requires Diligent Archiver target.")
endif()

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CRESSIM_NEO_SHADER_SOURCE_DIR}"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/shaders"
)

set(CRESSIM_NEO_DXC_RUNTIME_LIBS "")

if(WIN32)
    list(APPEND CRESSIM_NEO_DXC_RUNTIME_LIBS
        "${PROJECT_SOURCE_DIR}/extern/dxc/windows/dxcompiler.dll"
        "${PROJECT_SOURCE_DIR}/extern/dxc/windows/dxil.dll"
    )
elseif(UNIX)
    list(APPEND CRESSIM_NEO_DXC_RUNTIME_LIBS
        "${PROJECT_SOURCE_DIR}/extern/dxc/linux/libdxcompiler.so"
        "${PROJECT_SOURCE_DIR}/extern/dxc/linux/libdxil.so"
    )
endif()

foreach(CRESSIM_NEO_DXC_LIB IN LISTS CRESSIM_NEO_DXC_RUNTIME_LIBS)
    if(EXISTS "${CRESSIM_NEO_DXC_LIB}")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CRESSIM_NEO_DXC_LIB}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        )
    endif()
endforeach()
