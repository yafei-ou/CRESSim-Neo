add_cressim_test(
    SUITE physics
    NAME world_soa_layout
    SOURCE physics_world_soa_layout.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME world_rigid_joints
    SOURCE physics_world_rigid_joints.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME world_joint_suppression_reallocation
    SOURCE physics_world_joint_suppression_reallocation.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME solver_context_split
    SOURCE physics_solver_context_split.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME rigid_compute_smoke
    SOURCE physics_rigid_compute_smoke.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME fluid_compute_smoke
    SOURCE physics_fluid_compute_smoke.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME solver_broad_phase_overflow
    SOURCE physics_solver_broad_phase_overflow.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME solver_mixed_shape_contacts
    SOURCE physics_solver_mixed_shape_contacts.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME rigid_restitution
    SOURCE physics_rigid_restitution.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME body_types
    SOURCE physics_body_types.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_generation_regular_tet_grid
    SOURCE physics_soft_body_generation_regular_tet_grid.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_authoring_roundtrip
    SOURCE physics_soft_body_authoring_roundtrip.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_authoring_tet_mesh
    SOURCE physics_soft_body_authoring_tet_mesh.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_authoring_tetgen_roundtrip
    SOURCE physics_soft_body_authoring_tetgen_roundtrip.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_authoring_transform
    SOURCE physics_soft_body_authoring_transform.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_body_authoring_failures
    SOURCE physics_soft_body_authoring_failures.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME strand_authoring
    SOURCE physics_strand_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME rigid_proxy_suturing_authoring
    SOURCE physics_rigid_proxy_suturing_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME particle_distance_constraint_authoring
    SOURCE physics_particle_distance_constraint_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME rigid_particle_attachment_constraint_authoring
    SOURCE physics_rigid_particle_attachment_constraint_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME strand_rigid_attachment_constraint_authoring
    SOURCE physics_strand_rigid_attachment_constraint_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME routed_cable_constraint_authoring
    SOURCE physics_routed_cable_constraint_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME rigid_distance_constraint_authoring
    SOURCE physics_rigid_distance_constraint_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME suturing_sequence_authoring
    SOURCE physics_suturing_sequence_authoring.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME suturing_runtime_updates
    SOURCE physics_suturing_runtime_updates.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_gpu_scene_upload
    SOURCE physics_soft_gpu_scene_upload.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME fluid_unified_particle_pool
    SOURCE physics_fluid_unified_particle_pool.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME particle_material_tables
    SOURCE physics_particle_material_tables.cpp
    LIBS cressim_neo_engine
)

add_cressim_test(
    SUITE physics
    NAME soft_revision_tracking
    SOURCE physics_soft_revision_tracking.cpp
    LIBS cressim_neo_engine
)
