cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
# Linux builds use Clang unless the configure invocation selects another
# compiler. Windows toolchain selection belongs to the Visual Studio preset;
# forcing a compiler there would make it disagree with MSVC.
if(NOT WIN32)
    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER clang)
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER clang++)
    endif()
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CRESSim-Neo VERSION 0.1.0 LANGUAGES C CXX)

if(MINGW)
    message(FATAL_ERROR "MinGW is not a supported CRESSim-Neo toolchain.")
endif()

# Diligent changes CMAKE_INSTALL_PREFIX when CMake supplied its default.
# Choose this project's local development default first, before
# GNUInstallDirs derives the corresponding full install directories.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
        CACHE PATH "Installation path" FORCE)
endif()

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
option(CRESSIM_NEO_REQUIRE_PYTHON_STUBS
       "Fail Python package builds when pybind11-stubgen is unavailable" OFF)
set(CRESSIM_NEO_CUDA_RUNTIME_PROVIDER "AUTO" CACHE STRING
    "CUDA runtime provider for Python packages: AUTO, MANAGED, or SYSTEM")
set_property(CACHE CRESSIM_NEO_CUDA_RUNTIME_PROVIDER PROPERTY STRINGS AUTO MANAGED SYSTEM)
set(CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION "" CACHE STRING
    "Required CUDA major.minor version for a MANAGED Python wheel (for example 12.6)")
string(TOUPPER "${CRESSIM_NEO_CUDA_RUNTIME_PROVIDER}" CRESSIM_NEO_CUDA_RUNTIME_PROVIDER)
if(NOT CRESSIM_NEO_CUDA_RUNTIME_PROVIDER STREQUAL "AUTO" AND
   NOT CRESSIM_NEO_CUDA_RUNTIME_PROVIDER STREQUAL "MANAGED" AND
   NOT CRESSIM_NEO_CUDA_RUNTIME_PROVIDER STREQUAL "SYSTEM")
    message(FATAL_ERROR
        "CRESSIM_NEO_CUDA_RUNTIME_PROVIDER must be AUTO, MANAGED, or SYSTEM; got "
        "'${CRESSIM_NEO_CUDA_RUNTIME_PROVIDER}'.")
endif()

# Diligent's enabled Vulkan/SPIR-V toolchain runs Python helpers while it is
# configured. Require it here so native builds fail at the project boundary
# with a direct prerequisite error rather than inside a third-party target.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

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
    set(CMAKE_MACOSX_RPATH ON)
elseif(UNIX)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

if(ENGINE_STATIC)
    add_compile_definitions(CRESSIM_NEO_STATIC)
endif()

function(cressim_neo_configure_shared_library target_name)
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN")
    endif()

    if(NOT WIN32)
        set_target_properties(${target_name} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
        )
    endif()
endfunction()

function(cressim_neo_configure_runtime_executable target_name)
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
    elseif(UNIX)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set(CRESSIM_NEO_HAS_CUDA_INTEROP OFF)
set(CRESSIM_NEO_CUDA_RUNTIME_MAJOR 0)
set(CRESSIM_NEO_CUDA_RUNTIME_MINIMUM 0)
if(WIN32)
    set(CRESSIM_NEO_CUDA_CUFFT_SONAME "cufft64_11.dll")
    set(CRESSIM_NEO_CUDA_CURAND_SONAME "curand64_10.dll")
else()
    set(CRESSIM_NEO_CUDA_CUFFT_SONAME "libcufft.so.11")
    set(CRESSIM_NEO_CUDA_CURAND_SONAME "libcurand.so.10")
endif()
if(CRESSIM_NEO_ENABLE_CUDA_INTEROP)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CRESSIM_NEO_HAS_CUDA_INTEROP ON)
        set(CRESSIM_NEO_CUDA_RUNTIME_BIN_DIR "${CUDAToolkit_BIN_DIR}")
        if(WIN32 AND EXISTS "${CUDAToolkit_BIN_DIR}/x64")
            set(CRESSIM_NEO_CUDA_RUNTIME_BIN_DIR "${CUDAToolkit_BIN_DIR}/x64")
        endif()
        string(REPLACE "." ";" CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PARTS "${CUDAToolkit_VERSION}")
        list(GET CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PARTS 0 CRESSIM_NEO_CUDA_RUNTIME_MAJOR)
        list(LENGTH CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PARTS CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PART_COUNT)
        set(CRESSIM_NEO_CUDA_RUNTIME_MINOR 0)
        if(CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PART_COUNT GREATER 1)
            list(GET CRESSIM_NEO_CUDA_TOOLKIT_VERSION_PARTS 1 CRESSIM_NEO_CUDA_RUNTIME_MINOR)
        endif()
        math(EXPR CRESSIM_NEO_CUDA_RUNTIME_MINIMUM
            "${CRESSIM_NEO_CUDA_RUNTIME_MAJOR} * 1000 + ${CRESSIM_NEO_CUDA_RUNTIME_MINOR} * 10")
        foreach(CRESSIM_NEO_CUDA_LIBRARY_NAME IN ITEMS cufft curand)
            set(CRESSIM_NEO_CUDA_LIBRARY_SONAME "")
            if(WIN32)
                # FindCUDAToolkit's imported location is an import library on
                # Windows (for example, cufft.lib), not the DLL Python must load.
                file(GLOB CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATHS
                    "${CRESSIM_NEO_CUDA_RUNTIME_BIN_DIR}/${CRESSIM_NEO_CUDA_LIBRARY_NAME}64_*.dll")
                list(LENGTH CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATHS
                    CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_COUNT)
                if(CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_COUNT EQUAL 1)
                    list(GET CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATHS 0
                        CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATH)
                    get_filename_component(CRESSIM_NEO_CUDA_LIBRARY_SONAME
                        "${CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATH}" NAME)
                elseif(CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_COUNT GREATER 1)
                    message(FATAL_ERROR
                        "Multiple ${CRESSIM_NEO_CUDA_LIBRARY_NAME} runtime DLLs were found in "
                        "${CRESSIM_NEO_CUDA_RUNTIME_BIN_DIR}: ${CRESSIM_NEO_CUDA_RUNTIME_LIBRARY_PATHS}")
                endif()
            elseif(TARGET CUDA::${CRESSIM_NEO_CUDA_LIBRARY_NAME})
                get_target_property(CRESSIM_NEO_CUDA_LIBRARY_PATH
                    CUDA::${CRESSIM_NEO_CUDA_LIBRARY_NAME} IMPORTED_LOCATION)
                if(CRESSIM_NEO_CUDA_LIBRARY_PATH)
                    get_filename_component(CRESSIM_NEO_CUDA_LIBRARY_PATH
                        "${CRESSIM_NEO_CUDA_LIBRARY_PATH}" REALPATH)
                    find_program(CRESSIM_NEO_READELF_EXECUTABLE NAMES readelf llvm-readelf)
                    if(CRESSIM_NEO_READELF_EXECUTABLE)
                        execute_process(
                            COMMAND "${CRESSIM_NEO_READELF_EXECUTABLE}" -d
                                    "${CRESSIM_NEO_CUDA_LIBRARY_PATH}"
                            OUTPUT_VARIABLE CRESSIM_NEO_CUDA_LIBRARY_DYNAMIC_SECTION
                            ERROR_QUIET)
                        string(REGEX MATCH "Library soname: \\[([^]]+)\\]"
                            CRESSIM_NEO_CUDA_LIBRARY_SONAME_MATCH
                            "${CRESSIM_NEO_CUDA_LIBRARY_DYNAMIC_SECTION}")
                        if(CRESSIM_NEO_CUDA_LIBRARY_SONAME_MATCH)
                            string(REGEX REPLACE ".*\\[([^]]+)\\].*" "\\1"
                                CRESSIM_NEO_CUDA_LIBRARY_SONAME
                                "${CRESSIM_NEO_CUDA_LIBRARY_SONAME_MATCH}")
                        endif()
                    endif()
                    if(NOT CRESSIM_NEO_CUDA_LIBRARY_SONAME)
                        # This fallback is only used on toolchains without readelf.
                        # The CUDA imported target's filename is still loadable from
                        # its configured provider directory in managed environments.
                        get_filename_component(CRESSIM_NEO_CUDA_LIBRARY_SONAME
                            "${CRESSIM_NEO_CUDA_LIBRARY_PATH}" NAME)
                    endif()
                endif()
            endif()
            if(CRESSIM_NEO_CUDA_LIBRARY_SONAME)
                string(TOUPPER "${CRESSIM_NEO_CUDA_LIBRARY_NAME}"
                    CRESSIM_NEO_CUDA_LIBRARY_UPPER_NAME)
                set(CRESSIM_NEO_CUDA_${CRESSIM_NEO_CUDA_LIBRARY_UPPER_NAME}_SONAME
                    "${CRESSIM_NEO_CUDA_LIBRARY_SONAME}")
            endif()
        endforeach()
        if(CRESSIM_NEO_CUDA_RUNTIME_PROVIDER STREQUAL "MANAGED")
            if(NOT CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
                message(FATAL_ERROR
                    "MANAGED CUDA wheels require CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION "
                    "to name their CUDA major.minor lane (for example 12.6).")
            endif()
            if(CUDAToolkit_VERSION VERSION_LESS "${CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION}" OR
               CUDAToolkit_VERSION VERSION_GREATER_EQUAL "${CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION}.999")
                message(FATAL_ERROR
                    "MANAGED CUDA wheel requires CUDAToolkit "
                    "${CRESSIM_NEO_MANAGED_CUDA_TOOLKIT_VERSION}.x; found ${CUDAToolkit_VERSION}.")
            endif()
        endif()
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
    install(FILES "${PROJECT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cressim-neo"
        COMPONENT CXXSDK)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT CXXSDK)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/extern/DiligentEngine/DiligentCore/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/DiligentEngine/DiligentCore"
        COMPONENT CXXSDK
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl"
        # CRESSim's public headers use Diligent interfaces, but neither they nor
        # those interface headers include DiligentCore/ThirdParty. Keep vendored
        # build-only, test, and tool headers out of the installed SDK.
        PATTERN "ThirdParty" EXCLUDE
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
