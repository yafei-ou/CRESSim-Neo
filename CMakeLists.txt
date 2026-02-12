cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

project(CRESSim-Neo VERSION 0.1.0 LANGUAGES C CXX)

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
option(CRESSIM_NEO_USE_DILIGENT "Enable Diligent-based graphics backend integration" OFF)
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

if(ENGINE_STATIC)
    add_compile_definitions(CRESSIM_NEO_STATIC)
endif()

add_subdirectory(src)
