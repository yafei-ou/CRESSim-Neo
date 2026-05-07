include(${PROJECT_SOURCE_DIR}/examples/helpers/example_targets.cmake)

add_cressim_example(
    NAME graphics_texture
    SOURCE texture.cpp
)

add_cressim_example(
    NAME graphics_skybox_ibl
    SOURCE skybox_ibl.cpp
)
