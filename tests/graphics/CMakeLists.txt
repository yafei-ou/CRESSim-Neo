add_cressim_test(
    SUITE graphics
    NAME render_target_resize_policy
    SOURCE render_target_resize_policy.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME device_presentation_policy
    SOURCE device_presentation_policy.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME runtime_recreate
    SOURCE runtime_recreate.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME render_plan_viewport_policy
    SOURCE render_plan_viewport_policy.cpp
    INCLUDE_SRC
    LIBS cressim_neo_graphics
)

add_cressim_test(
    SUITE graphics
    NAME explicit_surface_depth_readback
    SOURCE explicit_surface_depth_readback.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME explicit_surface_viewport_policy
    SOURCE explicit_surface_viewport_policy.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME depth_camera_readback
    SOURCE depth_camera_readback.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME display_resolve_color_space
    SOURCE display_resolve_color_space.cpp
    LIBS cressim_neo_engine glfw
)

add_cressim_test(
    SUITE graphics
    NAME texture_material_resources
    SOURCE texture_material_resources.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME mesh_tangent_generation
    SOURCE mesh_tangent_generation.cpp
    LIBS cressim_neo_graphics
)

add_cressim_test(
    SUITE graphics
    NAME fluid_transparent_smoke
    SOURCE fluid_transparent_smoke.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE graphics
    NAME debug_particle_graph_smoke
    SOURCE debug_particle_graph_smoke.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)
