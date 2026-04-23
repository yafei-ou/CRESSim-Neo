The bottleneck is still mostly in the soft-body solver, and there’s one especially important logic mismatch in this toroid test.

1. You are still running the full soft-soft broad phase even though this scene has `selfCollisionEnabled = false`.
At [`physics_viewer_soft_particles_toroid_multi_env.cpp:492`](/home/yafei/Code/CRESSim-Neo/tests/physics_viewer/physics_viewer_soft_particles_toroid_multi_env.cpp#L492), self-collision is disabled. But the solver enables soft-soft work purely from `softParticleCount > 1` in [`physics_solver.cpp:178`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L178). That means every substep you still do:
- particle grid build
- key build
- sort
- cell range build
- soft-soft candidate count
- scan
- finalize
- emit

Even though the soft-soft shader will reject same-phase pairs when self-collision is off in [`physics_soft_soft_candidate_query.hlsli:47`](/home/yafei/Code/CRESSim-Neo/assets/shaders/include/physics/soft/physics_soft_soft_candidate_query.hlsli#L47). So the work still happens, it just produces no useful contacts. For this toroid scene, that’s likely a major remaining waste.

2. `50` internal iterations and `50` contact iterations is still very heavy for a `~1k particle / ~4k tet` body at `40` envs.
The toroid viewer sets that in [`physics_viewer_soft_particles_toroid_multi_env.cpp:508`](/home/yafei/Code/CRESSim-Neo/tests/physics_viewer/physics_viewer_soft_particles_toroid_multi_env.cpp#L508). Inside the main loop in [`physics_solver.cpp:384`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L384), each iteration can do:
- edge solve
- tet solve
- contact generation
- contact compaction
- contact solve
- apply corrections

The tet and edge solvers are especially expensive because they use atomics into shared particle correction buffers:
- [`physics_soft_solve_tet_constraints.cs.hlsl:75`](/home/yafei/Code/CRESSim-Neo/assets/shaders/physics/soft/solver/physics_soft_solve_tet_constraints.cs.hlsl#L75)
- [`physics_soft_solve_edge_constraints.cs.hlsl:55`](/home/yafei/Code/CRESSim-Neo/assets/shaders/physics/soft/solver/physics_soft_solve_edge_constraints.cs.hlsl#L55)

That creates a lot of contention once many tets/edges from many envs hit the same global buffers.

3. You still have blocking GPU readbacks in the soft inner loop.
These are not the only bottleneck anymore, but they are still costly:
- candidate meta readback after pair build: [`physics_solver.cpp:345`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L345)
- active contact meta readback inside the iteration loop: [`physics_solver.cpp:445`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L445)
- both stall with `Flush()` + `WaitForIdle()` in [`physics_scene_gpu_state.cpp:1501`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_scene_gpu_state.cpp#L1501)

So every substep/iteration is breaking async GPU flow.

4. Soft-rigid contact refresh is still expensive per iteration.
For every contact iteration, you do:
- analytic contact generation over all soft-rigid candidates
- active-flag scan
- compaction
- readback of active counts
- solve
- apply

That path is here:
- generate: [`physics_pass_dispatcher.cpp:846`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_pass_dispatcher.cpp#L846)
- compact: [`physics_pass_dispatcher.cpp:945`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_pass_dispatcher.cpp#L945)

The narrow phase itself loops over each candidate body’s colliders in [`physics_soft_rigid_generate_contacts.cs.hlsl:74`](/home/yafei/Code/CRESSim-Neo/assets/shaders/physics/soft/narrow_phase/physics_soft_rigid_generate_contacts.cs.hlsl#L74). That’s reasonable now, but multiplied by `50` contact iterations and many envs, it adds up.

5. Rigid broad phase still has two blocking readbacks per substep.
That path is lighter than soft, but it still stalls:
- first readback for active/static sets: [`physics_solver.cpp:213`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L213)
- second readback for pair counts/overflow: [`physics_solver.cpp:246`](/home/yafei/Code/CRESSim-Neo/src/physics/physics_solver.cpp#L246)

So the main ranking for this toroid case is:

1. Unnecessary soft-soft broad phase/contact pipeline despite self-collision being off.
2. `50/50` iteration count with atomic-heavy tet/edge solvers.
3. Blocking readbacks inside the soft path.
4. Per-iteration soft-rigid contact regenerate/compact/solve.
5. Secondary rigid-side readbacks.

The single highest-value fix now is: skip soft-soft broad phase and soft-soft contacts entirely for this toroid scene. Right now that work is almost pure overhead. After that, the next biggest win is removing the inner-loop soft meta readback and driving contact solve counts fully on-GPU.

If you want, I can implement the next fix directly: make soft-soft scheduling depend on actual self-collision/inter-soft-collision possibility instead of just `softParticleCount > 1`.
