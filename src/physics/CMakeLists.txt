file(GLOB_RECURSE PROJECT_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/physics/*.h
)

file(GLOB_RECURSE TARGET_HEADER CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/physics/*.h
)

file(GLOB_RECURSE TARGET_SOURCE CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/physics/*.cpp
)

set(TARGET_NAME "cressim_neo_physics")

if(ENGINE_STATIC)
    add_library(${TARGET_NAME} STATIC ${TARGET_SOURCE})
else()
    add_library(${TARGET_NAME} SHARED ${TARGET_SOURCE})
    cressim_neo_configure_shared_library(${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE CRESSIM_NEO_PHYSICS_EXPORT)
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

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/extern)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_definitions(${TARGET_NAME} PUBLIC
    CRESSIM_NEO_HAS_ULTRASOUND=$<BOOL:${CRESSIM_NEO_HAS_ULTRASOUND}>
)

target_link_libraries(${TARGET_NAME} PUBLIC cressim_neo_common cressim_neo_gpu)
# Diligent assertion helpers are referenced by inline Diligent headers used in
# this library.  Link their platform implementation into this shared object:
# it has hidden visibility and therefore cannot be resolved from another DSO.
target_link_libraries(${TARGET_NAME} PRIVATE Diligent-TargetPlatform)
if(CRESSIM_NEO_HAS_ULTRASOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE cru_interface)
endif()

set_target_properties(${TARGET_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
