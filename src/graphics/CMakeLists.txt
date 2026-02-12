file(GLOB_RECURSE PROJECT_HEADER
    ${PROJECT_SOURCE_DIR}/include/graphics/*.h
)

file(GLOB_RECURSE TARGET_HEADER
    ${PROJECT_SOURCE_DIR}/src/graphics/*.h
)

file(GLOB_RECURSE TARGET_SOURCE
    ${PROJECT_SOURCE_DIR}/src/graphics/*.cpp
)

set(TARGET_NAME "cressim_neo_graphics")

if(ENGINE_STATIC)
    add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE})
else()
    add_library(${TARGET_NAME} SHARED ${TARGET_SOURCE})
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_GRAPHICS_EXPORT)
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
    TREE "${PROJECT_SOURCE_DIR}/src/graphics"
    PREFIX "Header Files"
    FILES ${TARGET_HEADER}
)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/src/graphics"
    PREFIX "Source Files"
    FILES ${TARGET_SOURCE}
)

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/extern)

target_link_libraries(${TARGET_NAME} PUBLIC cressim_neo_common)

if(CRESSIM_NEO_USE_DILIGENT)
    target_compile_definitions(${TARGET_NAME} PUBLIC CRESSIM_NEO_USE_DILIGENT=1)
else()
    target_compile_definitions(${TARGET_NAME} PUBLIC CRESSIM_NEO_USE_DILIGENT=0)
endif()

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
