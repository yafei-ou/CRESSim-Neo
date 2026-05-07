include(${PROJECT_SOURCE_DIR}/examples/helpers/example_targets.cmake)

add_cressim_example(
    NAME debug_viewer
    SOURCE debug_viewer.cpp
)

add_cressim_example(
    NAME texture_viewer
    SOURCE texture_debug_viewer.cpp
)
