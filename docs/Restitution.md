# Rigid-Rigid Restitution via Contact Velocity Constraints

## Summary
Rigid-rigid restitution should follow the same contact model the codebase already uses for position solve: generate a 1-4 point manifold per candidate pair, treat each active contact slot as its own constraint, and let shared body velocities couple those constraints at the body level.

The current code already has the right foundations for this:
- `physics_rigid_generate_contacts.cs.hlsl` writes up to `kRigidContactsPerPair` contacts per pair into `g_RigidContacts`
- `physics_rigid_solve_contacts.cs.hlsl` already solves each active contact slot independently for position correction
- rigid velocity corrections already accumulate through shared body buffers and are applied centrally in `physics_rigid_apply_contact_velocities.cs.hlsl`

Because of that, restitution should be implemented as a real rigid contact velocity solve, not as a separate one-shot pair snapshot/apply event.

## Why This Fits the Codebase
- The narrow phase already produces manifolds in the layout we want. Box-box contacts already emit up to 4 contact points; the other shape pairs already emit a single contact.
- The position solver is already contact-based, not pair-based. Each slot in `g_RigidContacts` is effectively a contact constraint today.
- The solver architecture already expects iterative accumulation through body correction buffers. Joint target velocity passes use the same pattern and share the same velocity correction apply path.
- A pair-level bounce event would cut across that structure and create a second contact model just for restitution.

The right abstraction here is "body-level behavior emerges from solving multiple contact constraints that all write into the same body velocities."

## Implementation Changes
- Keep manifold generation as-is:
  - continue to write up to `kRigidContactsPerPair` contacts per candidate pair into `g_RigidContacts`
  - do not add pair-level restitution snapshots or representative-contact reduction
- Add a transient rigid contact velocity state buffer sized to `candidatePairCapacity * kRigidContactsPerPair`
  - one state entry per `g_RigidContacts` slot
  - this mirrors the existing per-joint lambda buffers
- Add a GPU-facing contact velocity state struct, for example:
  - `active`
  - `accumulatedNormalImpulse`
  - `targetBounceVelocity`
  - `reserved`
- Add a clear pass for the rigid contact velocity state buffer before the rigid velocity solve phase each substep
- After the positional rigid solve completes and `updateRigidVelocities` reconstructs rigid velocities from the final predicted poses, regenerate rigid contacts once more
  - this refreshes manifold points and normals from the final post-position-solve poses
  - the velocity solve should operate on this final manifold, not on stale contacts from an earlier position iteration
- Add a new rigid contact velocity solve pass
  - one thread owns one contact slot
  - dispatch over `candidatePairCount * kRigidContactsPerPair`
  - skip inactive contacts

## Velocity Solve Behavior
For each active contact slot:
- read `bodyA`, `bodyB`, contact normal, and local points from `g_RigidContacts`
- reconstruct world contact points and lever arms from the final predicted poses
- read current rigid linear/angular velocities
- compute point velocities and relative normal velocity at that contact
- combine restitution from the existing contact material (`contact.material.y`)
- gate restitution with a threshold
  - if incoming normal speed is not sufficiently negative, set target bounce to 0
  - this avoids micro-bounce and resting jitter
- otherwise compute the target bounce velocity for that point
  - `targetBounceVelocity = -restitution * normalVelocity`
- compute effective mass along the contact normal
- solve for the incremental normal impulse needed to move the current relative normal velocity toward the target bounce velocity
- clamp with accumulated impulse
  - `newAccumulated = max(oldAccumulated + deltaImpulse, 0)`
  - `appliedImpulse = newAccumulated - oldAccumulated`
- apply the resulting linear/angular velocity corrections through the existing rigid velocity correction buffers

This is the standard manifold-contact approach:
- each point gets its own bounce target from the relative velocity at that point
- the manifold behaves at the body level because all contact constraints share the same body velocities
- iterative solving prevents multi-point bounce explosion better than a single pair impulse does

## Solver Flow
Recommended rigid flow per substep:

1. Run the existing rigid position solve loop unchanged:
   - generate rigid contacts
   - solve rigid positional contacts
   - solve rigid joints
   - apply rigid position corrections
2. Run `updateRigidVelocities`
   - reconstruct rigid linear/angular velocities from the final predicted poses
3. Regenerate rigid contacts once using the final predicted poses
   - rebuild the final contact manifold used for velocity solve
4. Clear rigid contact velocity state
5. Run rigid velocity iterations:
   - solve rigid contact velocity constraints
   - solve hinge/slider target velocity constraints
   - apply accumulated rigid velocity corrections

Notes:
- V1 does not need a separate restitution prep pass.
- V1 also does not need per-pair or per-contact history across frames.
- Accumulated normal impulse is only needed across velocity iterations within the current substep.

## Public Interfaces / Types
- Add one transient GPU buffer in scene state:
  - rigid contact velocity state buffer
- Add one shared CPU/HLSL struct for per-contact velocity state
- Add two compute passes:
  - clear rigid contact velocity state
  - solve rigid contact velocities
- Update `solveRigidContactVelocities()` in the dispatcher so it actually dispatches rigid contact velocity constraints, not only joint motor velocity passes

## Scope Boundaries
- Scope is rigid-rigid restitution only
- Particle contact velocity paths remain unchanged
- Positional rigid contact solve remains unchanged
- Positional friction remains unchanged for V1
- Do not introduce body-pair restitution collapse, representative-contact selection, or contact dedup as part of this work

## Test Plan
- Head-on rigid-rigid collision with restitution:
  - visible bounce returns
  - bounce magnitude scales with restitution coefficient
- Resting stack with restitution enabled:
  - no persistent micro-bounce at rest
  - threshold suppresses jitter
- Box-box manifold with 2-4 active contacts:
  - all active manifold points participate in restitution solve
  - no obvious multi-point bounce explosion
- Mixed-shape scene such as [mixed_shape_contacts.cpp](/home/yafei/Code/CRESSim-Neo/examples/physics/mixed_shape_contacts.cpp:1):
  - restitution works across sphere/box/capsule combinations
  - stability is no worse than the current position-only branch
- Joint velocity regression:
  - hinge/slider target velocity behavior remains correct while sharing the same velocity correction buffers

## Assumptions and Defaults
- The final narrow-phase manifold is good enough for the first restitution version.
- Restitution threshold should be configurable or at least kept as a clearly named constant.
- Contact ordering only needs to remain stable within the final regenerated manifold for the duration of the velocity iterations in that substep.
- If later we need more accurate impact-speed latching, that should be added per contact slot, not as a single pair-owned restitution event.
