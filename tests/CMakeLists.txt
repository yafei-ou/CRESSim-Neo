add_subdirectory(smoke)
add_subdirectory(cube_depth)

if(CRESSIM_NEO_BUILD_DEBUG_VIEWER)
    add_subdirectory(debug_viewer)
endif()
