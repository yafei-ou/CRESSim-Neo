# Proper Rigid-Rigid Restitution with Pair Snapshots

## Summary
Implement rigid-rigid restitution as a dedicated GPU-friendly pair-snapshot pipeline. Each body-body candidate pair owns one restitution snapshot for the frame. During rigid contact iterations, a new pair-owned prep pass performs a fixed 4-slot reduction over that pair’s contacts, latches impact data early, and refreshes contact geometry while contacts remain valid. After the positional solve and velocity reconstruction, a restitution apply pass injects bounce once per body-body pair using the latched impact speed and the latest valid stored geometry. This restores restitution without making stability depend on contact dedup or final contacts remaining active.

## Implementation Changes
- Add a transient per-pair restitution snapshot buffer sized to `candidatePairCapacity`, indexed by `pairIndex`.
- Add a GPU-facing snapshot struct with:
  - `bodyA`, `bodyB`
  - `active/valid`
  - `latchedInitialNormalVelocity`
  - `combinedRestitution`
  - `storedNormal`
  - `storedLocalPointA`
  - `storedLocalPointB`
  - `lastSeenIteration` or equivalent validity marker
- Keep the current rigid positional contact solve unchanged:
  - `physics_rigid_solve_contacts.cs.hlsl` still resolves all active contacts for penetration and position-friction
  - no body-level contact merging or dedup is introduced
- Add a new rigid restitution prep pass that runs once per rigid contact iteration, after rigid contacts are generated and before rigid corrections are applied for that iteration.
- Restitution prep pass behavior per `pairIndex`:
  - read the pair’s fixed `kRigidContactsPerPair` contact slots from `g_RigidContacts`
  - perform a fixed-size reduction across the up to 4 active contacts
  - choose one representative contact using most negative normal relative velocity
  - compute normal relative velocity from the current predicted poses and current pre-reconstruction velocities available at that stage
  - if the snapshot is empty and the representative contact is impacting beyond the restitution threshold, initialize the snapshot
  - if the snapshot is already active and the pair still has a valid contact, refresh only geometry fields:
    - normal
    - localPointA
    - localPointB
    - material restitution if needed
  - never overwrite the latched impact speed with a later weaker value
  - if no valid contact exists this iteration, leave the snapshot unchanged
- Snapshot update policy:
  - latched fields: impact speed, body ids, active state
  - refreshable fields: normal and local contact geometry
  - refresh geometry only from a valid representative contact for that same pair
- Add a new rigid restitution apply pass after `updateRigidVelocities`.
- Restitution apply pass behavior per active snapshot:
  - reconstruct world contact points and lever arms from the stored local points and final predicted poses
  - compute current effective mass from final pose geometry
  - use the latched initial normal velocity for restitution gating and target bounce speed
  - apply one normal restitution impulse to rigid linear/angular velocity correction buffers
  - optionally suppress restitution if current reconstructed separation along the stored normal exceeds a small threshold, to avoid bouncing clearly separated pairs
- Keep the existing rigid velocity correction apply path:
  - joint target velocity corrections and restitution corrections both accumulate into the same rigid velocity correction buffers
  - the existing rigid velocity apply pass remains the single place that mutates predicted rigid velocities
- Dispatcher/solver wiring:
  - clear the restitution snapshot buffer once per frame before rigid contact iterations begin
  - during each rigid contact iteration:
    - generate contacts
    - run restitution prep pass
    - run positional rigid contact solve
    - apply rigid corrections
  - after all position iterations:
    - reconstruct rigid velocities
    - run restitution apply pass
    - run rigid velocity correction apply pass
- GPU style and performance constraints:
  - one thread owns one `pairIndex`
  - no CAS, append lists, or body-level shared contact structures
  - fixed 4-way reduction only
  - branch-light compare/select logic is preferred over variable-length global scanning

## Public Interfaces / Types
- Add one transient GPU buffer for rigid pair restitution snapshots in scene state.
- Add one shared CPU/HLSL snapshot struct with 16-byte aligned layout and static size checks.
- Add two compute passes:
  - rigid restitution prep
  - rigid restitution apply
- If needed, add one indirect dispatch slot or a simple direct dispatch path over candidate pairs for these passes. Default preference is dispatch over candidate pairs rather than total contact slots.

## Test Plan
- Head-on rigid-rigid collision with restitution:
  - visible bounce returns
  - bounce magnitude scales with restitution coefficient
- Resting stack with restitution enabled:
  - no persistent micro-bounce at rest
  - no clear stacking regression from current branch
- Box-box manifold with 2-4 active contacts:
  - only one restitution event is produced for the pair
  - all contacts still contribute to positional stability
- Mixed-shape or compound contact scene:
  - no multi-point bounce explosion
  - jitter remains no worse than current stabilization branch
- Over-corrected/separated contact case:
  - if final raw contacts disappear, stored snapshot still allows restitution
  - restitution is skipped when pair separation is clearly too large at apply time
- Joint velocity regression:
  - hinge/slider target velocity behavior remains unchanged aside from shared accumulation/apply buffers

## Assumptions and Defaults
- Scope is rigid-rigid restitution only; particle contact velocity paths remain unchanged.
- V1 uses one restitution event per body-body candidate pair, not per contact cluster.
- The representative contact is the active slot with the most negative normal relative velocity.
- Final contacts are not trusted as the sole source of impact detection.
- Stored local contact geometry is the fallback when late iterations remove the final contact.
- General-purpose stability should not rely on contact dedup; duplicate contacts are tolerated by the positional solver, while restitution is collapsed to one pair event.
