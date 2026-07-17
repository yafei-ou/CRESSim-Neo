file(GLOB_RECURSE PROJECT_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/engine/*.h
)

file(GLOB_RECURSE TARGET_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/engine/*.h
)

file(GLOB_RECURSE TARGET_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/engine/*.cpp
)

set(TARGET_NAME "cressim_neo_engine")

if(ENGINE_STATIC)
    add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE})
else()
    add_library(${TARGET_NAME} SHARED ${TARGET_SOURCE})
    cressim_neo_configure_shared_library(${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_ENGINE_EXPORT)
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
    TREE "${PROJECT_SOURCE_DIR}/src/engine"
    PREFIX "Header Files"
    FILES ${TARGET_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src/engine"
    PREFIX "Source Files"
    FILES ${TARGET_SOURCE}
)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${TARGET_NAME} PUBLIC
    cressim_neo_common cressim_neo_graphics cressim_neo_physics cressim_neo_gpu
    cressim_neo_build_settings)
if(CRESSIM_NEO_HAS_ULTRASOUND)
    # cru_interface is an in-tree runtime dependency, not part of the public
    # CRESSim-Neo CMake package.  Keep it out of the installed export while
    # still linking this library in the build tree.
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_SOURCE_DIR}/extern/CRESSim-Ultrasound/interface_c/include")
    target_link_libraries(${TARGET_NAME} PRIVATE
        "$<BUILD_INTERFACE:$<TARGET_LINKER_FILE:cru_interface>>")
endif()

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
