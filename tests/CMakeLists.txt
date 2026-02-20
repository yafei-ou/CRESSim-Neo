add_subdirectory(smoke)
add_subdirectory(cube_depth)
add_subdirectory(forward_pipeline)
add_subdirectory(graphics)
add_subdirectory(engine)

if(CRESSIM_NEO_BUILD_VIEWER)
    add_subdirectory(debug_viewer)
endif()
