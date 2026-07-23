cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
# Diligent claims that it doesn't work with O2/O3 using gcc
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CRESSim-Neo VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# CTest defaults this option to ON when it creates it.  Define it first so a
# normal project configure does not build the large test suite unless requested.
option(BUILD_TESTING "Build the testing tree" OFF)
include(CTest)
include(CheckLanguage)

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

option(ENGINE_STATIC "Build the engine as a static library" OFF)
option(CRESSIM_NEO_BUILD_VIEWER "Build viewer module and supporting executables" ON)
option(CRESSIM_NEO_BUILD_EXAMPLES "Build standalone example executables" ON)
option(CRESSIM_NEO_BUILD_DOCS "Enable documentation build targets" OFF)
option(CRESSIM_NEO_ENABLE_CLANG_TIDY "Enable clang-tidy checks during build" OFF)
option(CRESSIM_NEO_ENABLE_CUDA_INTEROP "Enable optional CUDA interop support in GPU helpers" OFF)
option(CRESSIM_NEO_ENABLE_ULTRASOUND "Enable optional CRESSim-Ultrasound integration" OFF)
option(CRESSIM_NEO_BUILD_PYTHON "Build optional Python bindings and tests" OFF)

include(cmake/CressimNeoDXC.cmake)
cressim_neo_configure_dxc_provider()

if(CRESSIM_NEO_BUILD_PYTHON)
    # The Python extension is a shared module; keep static engine libraries PIC-capable.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(CRESSIM_NEO_ENABLE_CLANG_TIDY)
    find_program(CRESSIM_NEO_CLANG_TIDY_EXE NAMES clang-tidy)
    if(CRESSIM_NEO_CLANG_TIDY_EXE)
        message(STATUS "clang-tidy enabled: ${CRESSIM_NEO_CLANG_TIDY_EXE}")
    else()
        message(WARNING "CRESSIM_NEO_ENABLE_CLANG_TIDY is ON, but clang-tidy was not found.")
    endif()
endif()

function(cressim_neo_enable_clang_tidy target_name)
    if(CRESSIM_NEO_CLANG_TIDY_EXE)
        set_target_properties(${target_name} PROPERTIES
            CXX_CLANG_TIDY "${CRESSIM_NEO_CLANG_TIDY_EXE}"
        )
    endif()
endfunction()

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

function(cressim_neo_configure_shared_library target_name)
    if(NOT WIN32)
        set_target_properties(${target_name} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
        )
    endif()
endfunction()

# Keep the embedded Diligent build lean for engine integration.
set(DILIGENT_BUILD_TOOLS OFF CACHE BOOL "Build DiligentTools module" FORCE)
set(DILIGENT_BUILD_FX OFF CACHE BOOL "Build DiligentFX module" FORCE)
set(DILIGENT_BUILD_SAMPLES OFF CACHE BOOL "Build DiligentSamples module" FORCE)
set(DILIGENT_BUILD_DOCS OFF CACHE BOOL "Build Diligent documentation" FORCE)
set(DILIGENT_BUILD_TESTS OFF CACHE BOOL "Build Diligent tests" FORCE)
set(DILIGENT_INSTALL_CORE OFF CACHE BOOL "Install DiligentCore headers and libraries" FORCE)
set(DILIGENT_NO_DIRECT3D11 ON CACHE BOOL "Disable Direct3D11 backend" FORCE)
if(WIN32)
    set(DILIGENT_NO_DIRECT3D12 OFF CACHE BOOL "Disable Direct3D12 backend" FORCE)
else()
    set(DILIGENT_NO_DIRECT3D12 ON CACHE BOOL "Disable Direct3D12 backend" FORCE)
endif()
set(DILIGENT_NO_OPENGL ON CACHE BOOL "Disable OpenGL backend" FORCE)
set(DILIGENT_NO_METAL ON CACHE BOOL "Disable Metal backend" FORCE)
set(DILIGENT_NO_WEBGPU ON CACHE BOOL "Disable WebGPU backend" FORCE)
set(DILIGENT_NO_VULKAN OFF CACHE BOOL "Disable Vulkan backend" FORCE)
if(MSVC)
    set(DILIGENT_MSVC_RELEASE_COMPILE_OPTIONS "/arch:AVX2" CACHE STRING
        "Additional MSVC compile options for release configurations" FORCE)
endif()
add_subdirectory(extern/DiligentEngine EXCLUDE_FROM_ALL)

# SYSTEM is deliberately opt-in.  Resolve it only after Diligent has inspected
# the locally installed Windows/Vulkan SDK, while BUNDLED remains independent
# of whatever SDK happened to be installed on the build host.
cressim_neo_finalize_dxc_system_provider()

set(DILIGENT_INSTALL_TOOLS OFF CACHE BOOL "Install DiligentTools module headers and libraries" FORCE)
set(DILIGENT_TOOLS_DIR "DiligentTools" CACHE INTERNAL "DiligentTools relative source dir")
add_subdirectory(extern/DiligentEngine/DiligentTools/ThirdParty
    ${CMAKE_BINARY_DIR}/DiligentTools/ThirdParty EXCLUDE_FROM_ALL)
add_subdirectory(extern/DiligentEngine/DiligentTools/TextureLoader
    ${CMAKE_BINARY_DIR}/DiligentTools/TextureLoader EXCLUDE_FROM_ALL)

# Public CRESSim-Neo headers expose Diligent math and graphics types.  Preserve
# the required platform definitions and include root in our exported package
# without making downstream consumers depend on this in-tree target.
add_library(cressim_neo_build_settings INTERFACE)
get_target_property(CRESSIM_NEO_DILIGENT_PUBLIC_DEFINITIONS
    Diligent-PublicBuildSettings INTERFACE_COMPILE_DEFINITIONS)
if(CRESSIM_NEO_DILIGENT_PUBLIC_DEFINITIONS)
    target_compile_definitions(cressim_neo_build_settings INTERFACE
        ${CRESSIM_NEO_DILIGENT_PUBLIC_DEFINITIONS})
endif()
target_include_directories(cressim_neo_build_settings INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern/DiligentEngine/DiligentCore/ThirdParty/Vulkan-Headers/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/DiligentEngine/DiligentCore/ThirdParty/Vulkan-Headers/include>)

set(CRESSIM_NEO_HAS_CUDA_INTEROP OFF)
if(CRESSIM_NEO_ENABLE_CUDA_INTEROP)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CRESSIM_NEO_HAS_CUDA_INTEROP ON)
        message(STATUS "CUDA interop enabled with CUDAToolkit ${CUDAToolkit_VERSION}.")
    else()
        message(FATAL_ERROR
            "CRESSIM_NEO_ENABLE_CUDA_INTEROP is ON, but CUDAToolkit was not found.")
    endif()
endif()

set(CRESSIM_NEO_HAS_ULTRASOUND OFF)
if(CRESSIM_NEO_ENABLE_ULTRASOUND)
    if(NOT CRESSIM_NEO_HAS_CUDA_INTEROP)
        message(FATAL_ERROR
            "CRESSIM_NEO_ENABLE_ULTRASOUND requires CRESSIM_NEO_ENABLE_CUDA_INTEROP=ON.")
    endif()

    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "CRESSIM_NEO_ENABLE_ULTRASOUND requires a working CUDA compiler toolset.")
    endif()
    enable_language(CUDA)

    set(CRESSIM_NEO_PREV_BUILD_TESTING ${BUILD_TESTING})
    set(BUILD_TESTING OFF)
    add_subdirectory(extern/CRESSim-Ultrasound EXCLUDE_FROM_ALL)
    set(BUILD_TESTING ${CRESSIM_NEO_PREV_BUILD_TESTING})
    set(CRESSIM_NEO_HAS_ULTRASOUND ON)
endif()

if(CRESSIM_NEO_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    set(CRESSIM_NEO_PYBIND11_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/pybind11")
    if(NOT EXISTS "${CRESSIM_NEO_PYBIND11_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "CRESSIM_NEO_BUILD_PYTHON requires the pinned extern/pybind11 submodule. "
            "Run: git submodule update --init --recursive")
    endif()
    add_subdirectory("${CRESSIM_NEO_PYBIND11_SOURCE_DIR}"
                     "${CMAKE_BINARY_DIR}/extern/pybind11")
endif()

add_subdirectory(src)

if(CRESSIM_NEO_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(CRESSIM_NEO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CRESSIM_NEO_BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(NOT ENGINE_STATIC)

    set(CRESSIM_NEO_CXX_TARGETS
        cressim_neo_build_settings
        cressim_neo_common
        cressim_neo_gpu
        cressim_neo_graphics
        cressim_neo_physics
        cressim_neo_engine)
    if(TARGET cressim_neo_viewer)
        list(APPEND CRESSIM_NEO_CXX_TARGETS cressim_neo_viewer)
    endif()

    set_target_properties(cressim_neo_build_settings PROPERTIES EXPORT_NAME build_settings)
    set_target_properties(cressim_neo_common PROPERTIES EXPORT_NAME common)
    set_target_properties(cressim_neo_gpu PROPERTIES EXPORT_NAME gpu)
    set_target_properties(cressim_neo_graphics PROPERTIES EXPORT_NAME graphics)
    set_target_properties(cressim_neo_physics PROPERTIES EXPORT_NAME physics)
    set_target_properties(cressim_neo_engine PROPERTIES EXPORT_NAME engine)
    if(TARGET cressim_neo_viewer)
        set_target_properties(cressim_neo_viewer PROPERTIES EXPORT_NAME viewer)
    endif()

    install(TARGETS ${CRESSIM_NEO_CXX_TARGETS}
        EXPORT CRESSimNeoTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT CXXSDK
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT CXXSDK
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT CXXSDK
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    if(CRESSIM_NEO_HAS_ULTRASOUND AND TARGET cru_interface)
        install(FILES "$<TARGET_FILE:cru_interface>"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT CXXSDK)
    endif()
    if(WIN32 AND CRESSIM_NEO_DXC_RUNTIME_PATH AND CRESSIM_NEO_DXIL_RUNTIME_PATH)
        install(FILES
            "${CRESSIM_NEO_DXC_RUNTIME_PATH}"
            "${CRESSIM_NEO_DXIL_RUNTIME_PATH}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT CXXSDK)
        if(CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH)
            install(FILES "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                RENAME "spv_dxcompiler.dll"
                COMPONENT CXXSDK)
        endif()
    elseif(UNIX AND NOT APPLE AND CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH)
        install(FILES "${CRESSIM_NEO_VULKAN_DXC_RUNTIME_PATH}"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            RENAME "libdxcompiler.so"
            COMPONENT CXXSDK)
    endif()
    if(CRESSIM_NEO_DXC_LICENSE_FILES)
        install(FILES ${CRESSIM_NEO_DXC_LICENSE_FILES}
            DESTINATION "${CMAKE_INSTALL_DATADIR}/cressim-neo/licenses"
            COMPONENT CXXSDK)
    endif()
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT CXXSDK)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/extern/DiligentEngine/DiligentCore/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/DiligentEngine/DiligentCore"
        COMPONENT CXXSDK
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl"
    )
    set(CRESSIM_NEO_CMAKE_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/CRESSimNeo")
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/CRESSimNeoConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CRESSimNeoConfig.cmake"
        INSTALL_DESTINATION "${CRESSIM_NEO_CMAKE_INSTALL_DIR}")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CRESSimNeoConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion)
    install(EXPORT CRESSimNeoTargets
        FILE CRESSimNeoTargets.cmake
        NAMESPACE CRESSimNeo::
        DESTINATION "${CRESSIM_NEO_CMAKE_INSTALL_DIR}"
        COMPONENT CXXSDK)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/CRESSimNeoConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CRESSimNeoConfigVersion.cmake"
        DESTINATION "${CRESSIM_NEO_CMAKE_INSTALL_DIR}"
        COMPONENT CXXSDK)
endif()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/assets/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cressim-neo/assets"
    COMPONENT CXXSDK)

# clang-format
file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/include/*.h
    ${CMAKE_SOURCE_DIR}/include/*.hpp
    ${CMAKE_SOURCE_DIR}/python/*.h
    ${CMAKE_SOURCE_DIR}/python/*.hpp
    ${CMAKE_SOURCE_DIR}/python/*.c
    ${CMAKE_SOURCE_DIR}/python/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_custom_target(format
    COMMAND clang-format -i ${ALL_SOURCE_FILES}
)
