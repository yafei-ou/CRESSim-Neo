file(GLOB_RECURSE PROJECT_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/gpu/*.h
)

file(GLOB_RECURSE TARGET_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/gpu/*.h
)

file(GLOB_RECURSE TARGET_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/gpu/*.cpp
)

if(NOT WIN32)
    list(REMOVE_ITEM TARGET_SOURCE
        ${PROJECT_SOURCE_DIR}/src/gpu/d3d12_cuda_interop.cpp
    )
endif()

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

cressim_neo_enable_clang_tidy(${TARGET_NAME})

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
target_compile_definitions(${TARGET_NAME} PUBLIC
    CRESSIM_NEO_HAS_CUDA_INTEROP=$<BOOL:${CRESSIM_NEO_HAS_CUDA_INTEROP}>
)

target_link_libraries(${TARGET_NAME} PUBLIC cressim_neo_common)
if(TARGET Vulkan::Headers)
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Headers)
endif()
if(CRESSIM_NEO_HAS_CUDA_INTEROP)
    target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)
endif()

# Link with Diligent
target_link_libraries(${TARGET_NAME} PUBLIC Diligent-PublicBuildSettings)
set(CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND OFF)
if(WIN32)
    if(TARGET Diligent-GraphicsEngineD3D12-static)
        target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineD3D12-static)
        set(CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND ON)
    elseif(TARGET Diligent-GraphicsEngineD3D12-shared)
        target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineD3D12-shared)
        set(CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND ON)
    endif()
endif()
if(TARGET Diligent-GraphicsEngineVk-static)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineVk-static)
    target_compile_definitions(${TARGET_NAME} PRIVATE DILIGENT_USE_VOLK=1)
    if(TARGET volk::volk)
        target_link_libraries(${TARGET_NAME} PRIVATE volk::volk)
    endif()
    set(CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND ON)
elseif(TARGET Diligent-GraphicsEngineVk-shared)
    target_link_libraries(${TARGET_NAME} PRIVATE Diligent-GraphicsEngineVk-shared)
    target_compile_definitions(${TARGET_NAME} PRIVATE DILIGENT_USE_VOLK=1)
    if(TARGET volk::volk)
        target_link_libraries(${TARGET_NAME} PRIVATE volk::volk)
    endif()
    set(CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND ON)
endif()
if(NOT CRESSIM_NEO_GPU_HAS_DILIGENT_BACKEND)
    message(FATAL_ERROR "Requires at least one Diligent graphics backend target.")
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

if(WIN32 AND CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/spv_dxcompiler.dll"
    )
elseif(UNIX AND CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/libdxcompiler.so"
    )
elseif(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH AND NOT EXISTS "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
    message(WARNING
        "CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH does not exist: ${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}")
endif()
