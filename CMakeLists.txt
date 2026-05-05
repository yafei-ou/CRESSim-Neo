cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
# Diligent claims that it doesn't work with O2/O3 using gcc
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CRESSim-Neo VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(PROJECT_VENDOR "Yafei Ou")
set(PROJECT_DESCRIPTION "")
set(PROJECT_PRODUCT_NAME "")

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
    @ONLY
)
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/version.rc.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.rc
    @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

option(ENGINE_STATIC "Build the engine as a static library"  ON)
option(CRESSIM_NEO_BUILD_VIEWER "Build viewer module and related integration tests" ON)
option(CRESSIM_NEO_ENABLE_CLANG_TIDY "Enable clang-tidy checks during build" OFF)

if(CRESSIM_NEO_ENABLE_CLANG_TIDY)
    find_program(CRESSIM_NEO_CLANG_TIDY_EXE NAMES clang-tidy)
    if(CRESSIM_NEO_CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY ${CRESSIM_NEO_CLANG_TIDY_EXE})
        message(STATUS "clang-tidy enabled: ${CRESSIM_NEO_CLANG_TIDY_EXE}")
    else()
        message(WARNING "CRESSIM_NEO_ENABLE_CLANG_TIDY is ON, but clang-tidy was not found.")
    endif()
endif()

set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

if(WIN32)
    message(STATUS "Configuring for Windows")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(APPLE)
    # Keep shipped helper libraries relocatable by resolving them relative to the
    # final binary/shared-library location at runtime on macOS.
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_RPATH "@loader_path")
elseif(UNIX)
    # Keep shipped helper libraries relocatable by resolving them relative to the
    # final binary/shared-library location at runtime on ELF platforms.
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

if(ENGINE_STATIC)
    add_compile_definitions(CRESSIM_NEO_STATIC)
endif()

# Keep the embedded Diligent build lean for engine integration.
set(DILIGENT_BUILD_TOOLS OFF CACHE BOOL "Build DiligentTools module" FORCE)
set(DILIGENT_BUILD_FX OFF CACHE BOOL "Build DiligentFX module" FORCE)
set(DILIGENT_BUILD_SAMPLES OFF CACHE BOOL "Build DiligentSamples module" FORCE)
set(DILIGENT_BUILD_DOCS OFF CACHE BOOL "Build Diligent documentation" FORCE)
set(DILIGENT_BUILD_TESTS OFF CACHE BOOL "Build Diligent tests" FORCE)
set(DILIGENT_NO_DIRECT3D11 ON CACHE BOOL "Disable Direct3D11 backend" FORCE)
set(DILIGENT_NO_DIRECT3D12 ON CACHE BOOL "Disable Direct3D12 backend" FORCE)
set(DILIGENT_NO_OPENGL ON CACHE BOOL "Disable OpenGL backend" FORCE)
set(DILIGENT_NO_METAL ON CACHE BOOL "Disable Metal backend" FORCE)
set(DILIGENT_NO_WEBGPU ON CACHE BOOL "Disable WebGPU backend" FORCE)
set(DILIGENT_NO_VULKAN OFF CACHE BOOL "Disable Vulkan backend" FORCE)
if(MSVC)
    set(DILIGENT_MSVC_RELEASE_COMPILE_OPTIONS "/arch:AVX2" CACHE STRING
        "Additional MSVC compile options for release configurations" FORCE)
endif()
add_subdirectory(extern/DiligentEngine EXCLUDE_FROM_ALL)

set(DILIGENT_INSTALL_TOOLS OFF CACHE BOOL "Install DiligentTools module headers and libraries" FORCE)
set(DILIGENT_TOOLS_DIR "DiligentTools" CACHE INTERNAL "DiligentTools relative source dir")
add_subdirectory(extern/DiligentEngine/DiligentTools/ThirdParty
    ${CMAKE_BINARY_DIR}/DiligentTools/ThirdParty EXCLUDE_FROM_ALL)
add_subdirectory(extern/DiligentEngine/DiligentTools/TextureLoader
    ${CMAKE_BINARY_DIR}/DiligentTools/TextureLoader EXCLUDE_FROM_ALL)

if(NOT DEFINED CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH)
    set(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH "${DILIGENT_DXCOMPILER_FOR_SPIRV_PATH}" CACHE FILEPATH
        "Path to the Vulkan DXC runtime library copied next to CRESSim-Neo binaries")
endif()

add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# clang-format
file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_custom_target(format
    COMMAND clang-format -i ${ALL_SOURCE_FILES}
)
