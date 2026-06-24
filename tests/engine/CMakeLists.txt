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
    NAME rigid_body_proxy_suturing_component
    SOURCE rigid_body_proxy_suturing_component.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME particle_distance_constraint_authoring
    SOURCE particle_distance_constraint_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME rigid_particle_attachment_constraint_authoring
    SOURCE rigid_particle_attachment_constraint_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME strand_rigid_attachment_constraint_authoring
    SOURCE strand_rigid_attachment_constraint_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME routed_cable_constraint_authoring
    SOURCE routed_cable_constraint_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME rigid_distance_constraint_authoring
    SOURCE rigid_distance_constraint_authoring.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME suturing_sequence_authoring
    SOURCE suturing_sequence_authoring.cpp
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

add_cressim_test(
    SUITE engine
    NAME runtime_staging
    SOURCE runtime_staging.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_ultrasound_staging
    SOURCE runtime_ultrasound_staging.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_custom_compute
    SOURCE runtime_custom_compute.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_custom_compute_render_target
    SOURCE runtime_custom_compute_render_target.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_shared_buffer
    SOURCE runtime_shared_buffer.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE engine
    NAME runtime_rigid_layout_mapping
    SOURCE runtime_rigid_layout_mapping.cpp
    LIBS cressim_neo_engine
)
