include_guard(GLOBAL)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# DXC is dynamically loaded by Diligent at runtime.  Release builds must not
# inherit the SDK version that happened to be installed on the build machine.
set(CRESSIM_NEO_DXC_PROVIDER "BUNDLED" CACHE STRING
    "DXC runtime provider: BUNDLED (pinned release), SYSTEM, or OFF")
set_property(CACHE CRESSIM_NEO_DXC_PROVIDER PROPERTY STRINGS BUNDLED SYSTEM OFF)

# DXC 1.8.2505.1 is the newest official Linux binary compatible with the
# managed wheel's GLIBC 2.34 floor. Use the same pinned DXC release on every
# supported platform so shader behavior is consistent across C++ packages.
set(CRESSIM_NEO_DXC_VERSION "v1.8.2505.1" CACHE INTERNAL "Pinned DXC release")
set(CRESSIM_NEO_DXC_RUNTIME_PATH "" CACHE FILEPATH
    "Path to the DXC runtime used for D3D12 (dxcompiler.dll on Windows)")
set(CRESSIM_NEO_DXIL_RUNTIME_PATH "" CACHE FILEPATH
    "Path to the DXIL runtime used with DXC (dxil.dll on Windows)")
set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "" CACHE FILEPATH
    "Path to the Vulkan DXC runtime copied next to CRESSim-Neo binaries")

function(cressim_neo_find_single_file root pattern result)
    file(GLOB_RECURSE matches LIST_DIRECTORIES FALSE "${root}/${pattern}")
    list(LENGTH matches match_count)
    if(match_count EQUAL 0)
        message(FATAL_ERROR "Pinned DXC archive is missing ${pattern}.")
    endif()
    list(GET matches 0 match)
    set(${result} "${match}" PARENT_SCOPE)
endfunction()

function(cressim_neo_configure_dxc_provider)
    if(CRESSIM_NEO_DXC_PROVIDER STREQUAL "OFF")
        set(CRESSIM_NEO_DXC_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        set(CRESSIM_NEO_DXIL_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        message(STATUS "DXC runtime provider: OFF")
        return()
    endif()

    if(CRESSIM_NEO_DXC_PROVIDER STREQUAL "SYSTEM")
        # Diligent resolves the system SDK paths while it is configured. They
        # are adopted by cressim_neo_finalize_dxc_system_provider() below.
        set(CRESSIM_NEO_DXC_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        set(CRESSIM_NEO_DXIL_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "" CACHE FILEPATH "" FORCE)
        message(STATUS "DXC runtime provider: SYSTEM")
        return()
    endif()

    if(NOT CRESSIM_NEO_DXC_PROVIDER STREQUAL "BUNDLED")
        message(FATAL_ERROR
            "CRESSIM_NEO_DXC_PROVIDER must be BUNDLED, SYSTEM, or OFF; got "
            "'${CRESSIM_NEO_DXC_PROVIDER}'.")
    endif()

    if(NOT (WIN32 OR (UNIX AND NOT APPLE)))
        message(FATAL_ERROR
            "The bundled DXC provider currently supplies only Windows x64 and Linux x86_64.")
    endif()
    if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR
            "The bundled DXC provider currently supports only 64-bit targets "
            "(CMAKE_SIZEOF_VOID_P='${CMAKE_SIZEOF_VOID_P}', "
            "CMAKE_GENERATOR_PLATFORM='${CMAKE_GENERATOR_PLATFORM}', "
            "CMAKE_VS_PLATFORM_NAME='${CMAKE_VS_PLATFORM_NAME}').")
    endif()

    include(FetchContent)
    if(WIN32)
        set(dxc_url
            "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip")
        set(dxc_hash "SHA256=9ad895a6b039e3a8f8c22a1009f866800b840a74b50db9218d13319e215ea8a4")
    else()
        set(dxc_url
            "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/linux_dxc_2025_07_14.x86_64.tar.gz")
        set(dxc_hash "SHA256=f2213da1fc99dc8778c8823078e16ba97c7f80f86a1d4520ab1adf4b462bc48c")
    endif()

    FetchContent_Declare(cressim_neo_dxc URL "${dxc_url}" URL_HASH "${dxc_hash}")
    FetchContent_GetProperties(cressim_neo_dxc)
    if(NOT cressim_neo_dxc_POPULATED)
        FetchContent_Populate(cressim_neo_dxc)
    endif()

    if(WIN32)
        cressim_neo_find_single_file("${cressim_neo_dxc_SOURCE_DIR}" "dxcompiler.dll" dxc_runtime)
        cressim_neo_find_single_file("${cressim_neo_dxc_SOURCE_DIR}" "dxil.dll" dxil_runtime)
        set(vulkan_dxc_runtime "${dxc_runtime}")
    else()
        cressim_neo_find_single_file("${cressim_neo_dxc_SOURCE_DIR}" "libdxcompiler.so" dxc_runtime)
        set(dxil_runtime "")
        set(vulkan_dxc_runtime "${dxc_runtime}")
    endif()
    # FORCE is intentional: the bundled provider is the project runtime
    # contract and must take precedence over a system SDK discovered by Diligent.
    set(CRESSIM_NEO_DXC_RUNTIME_PATH "${dxc_runtime}" CACHE FILEPATH "" FORCE)
    set(CRESSIM_NEO_DXIL_RUNTIME_PATH "${dxil_runtime}" CACHE FILEPATH "" FORCE)
    set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "${vulkan_dxc_runtime}" CACHE FILEPATH "" FORCE)
    message(STATUS "DXC runtime provider: BUNDLED ${CRESSIM_NEO_DXC_VERSION}")
endfunction()

function(cressim_neo_finalize_dxc_system_provider)
    if(NOT CRESSIM_NEO_DXC_PROVIDER STREQUAL "SYSTEM")
        return()
    endif()

    set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "${DILIGENT_DXCOMPILER_FOR_SPIRV_PATH}"
        CACHE FILEPATH "" FORCE)
    if(WIN32)
        set(CRESSIM_NEO_DXC_RUNTIME_PATH "${DXC_COMPILER_PATH}" CACHE FILEPATH "" FORCE)
        set(CRESSIM_NEO_DXIL_RUNTIME_PATH "${DXIL_SIGNER_PATH}" CACHE FILEPATH "" FORCE)
    endif()
endfunction()
