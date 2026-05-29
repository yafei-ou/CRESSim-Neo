add_cressim_test(
    SUITE engine
    NAME world_sync_incremental
    SOURCE world_sync_incremental.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_physics_sync_fields
    SOURCE runtime_physics_sync_fields.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME camera_viewport_aspect_policy
    SOURCE camera_viewport_aspect_policy.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME direct_physics_authoring
    SOURCE direct_physics_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME strand_component_authoring
    SOURCE strand_component_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME soft_body_simulated_toggle
    SOURCE soft_body_simulated_toggle.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME soft_body_runtime_update_preserves_state
    SOURCE soft_body_runtime_update_preserves_state.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME world_deferred_render_reconciliation
    SOURCE world_deferred_render_reconciliation.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME environment_ibl_state
    SOURCE environment_ibl_state.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME logger
    SOURCE logger.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME unified_light_slots
    SOURCE unified_light_slots.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME local_shadow_registry
    SOURCE local_shadow_registry.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)
