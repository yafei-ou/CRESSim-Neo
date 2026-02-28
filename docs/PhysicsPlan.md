A. Core design choice: one solver, two "broadphases"
====================================================

### Broadphase 1: Particle grid

Used for:

-   fluid neighbors

-   particle--particle collision candidates (if you do them)

-   rigid--particle candidates (rigid AABB → grid cells)

### Broadphase 2: Rigid broadphase

Used for:

-   rigid--rigid candidates

-   rigid--static candidates (unless you treat statics specially)

* * * * *

B. Buffers you need
===================

B1) Persistent state buffers (live across frames)
-------------------------------------------------

### Particles (SoA)

-   `float3 x[Np]` current position

-   `float3 v[Np]` velocity

-   `float invMass[Np]` (0 for pinned)

-   `float radius[Np]` (or global)

-   `int phase[Np]` (collision groups / self-collide flags / fluid vs solid markers)

-   (optional) `float3 fExt[Np]`

**Soft-body connectivity (optional)**

-   `Edge edges[Ne]` : (i, j, restLen, stiffness, compliance)

-   `Tet tets[Nt]` : indices + rest volume/compliance

-   `Cluster clusters[Nc]` : shape-matching sets (indices, rest data)

-   any other constraints lists

### Rigid bodies (pose-based)

-   `float3 p[Nr]` position

-   `quat q[Nr]` orientation

-   `float3 vLin[Nr]`, `float3 vAng[Nr]`

-   `float invMassR[Nr]`

-   `mat3 invInertiaLocal[Nr]` (or diag)

-   shape info:

    -   `Shape shapes[]` (type + params)

    -   `int shapeOfRigid[Nr]` index

-   collision material:

    -   friction, restitution, contactOffset

### Statics

-   whatever you collide against: planes, boxes, triangle meshes, SDF volume, etc.

-   if triangle mesh: BVH / grid / SDF acceleration structure

* * * * *

B2) Per-substep transient buffers (recomputed each substep)
-----------------------------------------------------------

### Predicted states

-   `float3 xPred[Np]`

-   `float3 xPrev[Np]` (for velocity update; or store old x)

-   `float3 pPred[Nr]`

-   `quat qPred[Nr]`

### Particle grid (for neighbors)

Common "sorted-by-cell" layout:

-   `uint cellKey[Np]` (hash)

-   `uint particleIdSorted[Np]` (permute)

-   `uint sortIndexToParticleId[Np]` (same as above)

-   `uint particleIdToSortIndex[Np]` (optional, if you need)

-   `uint cellStart[numCells]`, `uint cellEnd[numCells]` (or start+count)

-   (optional) `float3 xPredSorted[Np]` if you reorder positions for coherence

### Fluid/neighbor quantities

If you do PBF:

-   `float density[Np]`

-   `float lambda[Np]`

-   `float3 deltaX[Np]` (accumulator)

-   (optional) `float3 vorticity[Np]`, `float3 omega[Np]`

If you do explicit neighbor list (optional):

-   `uint neighborCount[Np]`

-   `uint neighbors[Np * MAX_NEI]` **or**

-   `uint neighborOffset[Np+1]`, `uint neighborsPacked[totalNei]` (two-pass prefix sum)

### Rigid broadphase

-   `AABB aabbR[Nr]` (at predicted pose)

-   broadphase structure (choose one):

    -   LBVH nodes, or

    -   SAP arrays, or

    -   a rigid grid (less common)

-   `Pair rigidPairs[MaxRR]` candidate list (rigid--rigid)

-   `Pair rigidStaticPairs[MaxRS]` candidate list (rigid--static)

### Candidate list for rigid--particle (optional)

If you don't solve on-the-fly:

-   `Pair rigidParticlePairs[MaxRP]` (rigidId, particleId) candidates

### Contact constraints list (recommended)

A unified contact buffer for all "shape-ish" contacts:

-   `Contact contacts[MaxC]`

    -   type: RR / RP / RS / PS / PP (if you put PP in)

    -   ids: bodyA/bodyB or body/particle etc.

    -   world normal `n`

    -   contact points / anchors:

        -   for RR: local anchor on A & B (or world point + leverage)

        -   for RP: local anchor on rigid + particle position

    -   penetration depth `phi` (negative if penetrating)

    -   friction `mu`

    -   (optional) cached Lagrange multiplier `lambdaN`, `lambdaT` (XPBD / warm start)

-   `uint contactCount`

### Per-iteration correction accumulators

**Particles**

-   `float3 corrX[Np]` (reset each constraint family or each iteration)

-   `float corrW[Np]` (optional weight accumulator)

**Rigids**\
You need accumulators for pose corrections. A robust pattern:

-   `float3 corrP[Nr]` (translation delta accumulator)

-   `float3 corrRot[Nr]` (rotation delta in axis-angle / "rotation vector" form)

-   (optional) weights / counts

* * * * *

C. Full flow: per frame → substeps → iterations
===============================================

I'll describe the "cleanest" common approach:

-   grid built once per substep

-   contacts built once per substep (often)

-   constraints solved iteratively (positions/poses)

-   then velocities updated

C0) Frame start (CPU-side)
--------------------------

-   Upload changed constraints/shapes/materials

-   Choose `dtFrame`, `substeps`, `dt = dtFrame / substeps`

* * * * *

C1) For each substep
====================

Stage 1 --- Predict (integrate)
-----------------------------

**Kernel: Np threads + Nr threads**

-   Save old positions/poses: `xPrev = x`, `pPrev/qPrev` if needed

-   Predict:

    -   particles: `v += dt * g`, `xPred = x + dt*v`

    -   rigids: integrate `pPred`, `qPred` from `(vLin, vAng)` (semi-implicit)

> At this moment, treat `xPred`, `pPred/qPred` as the "solver variables."

* * * * *

Stage 2 --- Build particle grid (neighbors)
-----------------------------------------

**Kernel(s):**

1.  compute `cellKey[i]` from `xPred[i]`

2.  sort by `cellKey` (radix sort)

3.  build `cellStart/end`

This is used for:

-   fluid density constraint neighbor loops

-   particle--particle collision candidates (if enabled)

-   rigid AABB → cell scan for rigid--particle candidates

* * * * *

Stage 3 --- Rigid broadphase (RR + RS)
------------------------------------

**Kernel(s):**

1.  compute `aabbR[r]` from `(pPred,qPred)` and shape bounds (+ contactOffset)

2.  build broadphase structure

3.  emit candidate pairs:

    -   `rigidPairs[]`

    -   `rigidStaticPairs[]`

* * * * *

Stage 4 --- Candidate generation for rigid--particle (RP)
------------------------------------------------------

Two options:

### Option 4A (recommended): Build RP candidate list

**Kernel: Nr threads (one rigid per thread block is common)**

-   For each rigid:

    -   compute its `aabbR`

    -   convert to cell range

    -   scan overlapped cells, append `(rigidId, particleId)` into `rigidParticlePairs` (atomic append)

    -   optional cheap reject (distance to AABB; phase filters)

    -   optional cap per rigid

### Option 4B (no pair buffer): Solve RP contacts on-the-fly

Skip pair buffer, but then you must do narrowphase + correction inside contact solve stage by scanning cells again.

(4A is usually easier to control + debug.)

* * * * *

Stage 5 --- Narrowphase: build contact constraints (RR, RS, RP, maybe PS)
-----------------------------------------------------------------------

**Kernel(s):**

-   For each candidate pair list:

    -   RR: narrowphase shape vs shape → one or a few manifold points → emit `Contact`s

    -   RS: rigid vs static → emit `Contact`s

    -   RP: rigid vs particle: compute signed distance / closest point to shape → emit `Contact` if within radius/contactOffset

-   For particle vs static (PS):

    -   either:

        -   do per-particle static queries (if simple planes/heightfield), OR

        -   treat statics as shapes and generate PS contacts similarly

**Output:** `contacts[]`, `contactCount`

> This is the "stage for building constraints" you suspected.\
> For fluids, density constraints are implicit; for shape contacts, it's practical to build explicit constraints.

* * * * *

C2) Iterative solver loop (repeat `I` times)
============================================

This is the heart. The order below is stable and common.

Iteration stage A --- Reset accumulators
--------------------------------------

-   set `corrX=0`, `corrP=0`, `corrRot=0`

-   (or reset per family if you apply between families)

* * * * *

Iteration stage B --- Fluid density constraints (PBF / XPBD density)
------------------------------------------------------------------

**Kernel: Np threads (fluid particles only)**

-   For each fluid particle i:

    -   iterate neighbors via grid (or neighbor list)

    -   compute density ρ_i

    -   compute lambda_i

**Kernel: Np threads**

-   For each fluid particle i:

    -   iterate neighbors again

    -   compute `Δx_i` using lambdas + kernels

    -   write to `corrX[i]` (either directly, or into `deltaX`)

**Kernel: Np threads**

-   apply: `xPred += corrX` (for fluid subset)

-   optional: apply boundary corrections if you do fluid-vs-static in this stage

**Why fluid early?**\
Because fluid wants a reasonably valid volume distribution before contacts/joints try to pin it.

* * * * *

Iteration stage C --- Soft-body internal constraints (springs/tets/clusters)
--------------------------------------------------------------------------

Pick a pattern; the safe GPU pattern is **accumulate then apply**:

### For edges (springs)

**Kernel: Ne threads**

-   compute correction for edge (i,j)

-   atomic add contributions into `corrX[i]`, `corrX[j]`

### For tets / volume constraints

**Kernel: Nt threads**

-   compute corrections

-   atomic adds into `corrX[indices]`

### Apply

**Kernel: Np threads**

-   `xPred += corrX` (for soft particles)

-   reset `corrX` for next family or next stage

You can do multiple internal families before applying, or apply after each family. Applying after each family tends to be more stable; batching reduces kernel launches.

* * * * *

Iteration stage D --- Contacts (the coupling stage)
-------------------------------------------------

This is where *everything meets*.

### D1) Normal contact projection

**Kernel: contactCount threads**\
For each contact:

-   compute positional correction along normal

-   apply to the involved DOFs:

    -   particle: add to `corrX[pid]`

    -   rigid: add to `corrP[rid]` and `corrRot[rid]`

    -   rigid--rigid: both sides

-   use inverse mass/inertia to split corrections

### D2) Apply corrections

**Kernel:**

-   particles: `xPred += corrX`

-   rigids: apply `(Δp, Δrot)` to `(pPred,qPred)` and normalize `qPred`

> This is why explicit contact list is nice: it's one uniform "contact constraint projection" kernel.

### D3) Friction (optional but common)

Do a second pass over contacts:

-   compute tangential correction based on relative displacement at contact

-   clamp by `mu * normal_lambda` (or equivalent PBD friction model)

-   accumulate/apply like above

**Ordering note:** friction typically after normals, because friction limit depends on normal resolution.

* * * * *

Iteration stage E --- Joints / motors / other rigids constraints (optional)
-------------------------------------------------------------------------

If you have joints between rigids, do them **after contact normals** (or interleave), then apply corrections.

* * * * *

Optional: extra stabilization pass
----------------------------------

If you stack many objects, you may do:

-   contacts again (normals only)

-   or contacts + joints again\
    This is common in realtime solvers to reduce drift.

* * * * *

C3) End of substep --- Velocity update & post
===========================================

After all iterations:

Stage 6 --- Update velocities
---------------------------

**Particles**

-   `v = (xPred - xPrev) / dt`

-   write `x = xPred`

**Rigids**

-   `vLin = (pPred - pPrev) / dt`

-   `vAng` from `qPrev^-1 * qPred` (convert to angular velocity)

-   write `p=qPred`, `q=qPred`

Optionally apply:

-   damping

-   restitution (often done velocity-level using contact normals)

-   viscosity / vorticity confinement for fluids

* * * * *

D. Where particle--particle contact belongs
==========================================

You said: "For contact between particles, I assume we do it together with other computation, right?"

### Yes, for fluids you usually *don't* build explicit PP contacts.

You handle "keeping particles apart" via:

-   the **density constraint itself** (PBF)

-   optional **short-range repulsion / cohesion** inside the *same neighbor loop* as density

### For non-fluid particle solids (granular / sand / self-colliding soft body)

You have two practical choices:

1.  **Implicit PP collision inside grid neighbor loops**

    -   Per particle, iterate neighbors and apply separation corrections

    -   Needs care to avoid double-counting and race conditions (usually Jacobi-style accumulate/apply)

2.  **Explicit PP contact list (like rigid contacts)**

    -   Build candidates from grid (particle pairs within radius)

    -   Emit `Contact` entries (PP)

    -   Solve in the unified contact stage

**If you already have a unified `contacts[]` buffer**, option (2) gives you consistent friction/restitution across PP/RP/RR. It costs memory and contact build time.

* * * * *

E. Contact ordering summary (what order and why)
================================================

A stable default ordering inside each iteration:

1.  **Fluids density / incompressibility**

2.  **Soft internal constraints** (springs/volume/shape-matching)

3.  **Contacts normal** (RR/RP/RS/PS and optionally PP)

4.  **Contacts friction**

5.  **Joints** (if present)

6.  (optional) **Contacts normal again** (stack stabilization)

If you only do one contact stage: do it **after** fluids+soft internals so the collision solver sees the most up-to-date particle distribution.

* * * * *

F. Minimal "shopping list" of buffers (if you want the simplest working version)
================================================================================

**Must-have for particles**

-   `x, v, invMass`

-   `xPred, xPrev`

-   particle grid: `cellKey, particleIdSorted, cellStart/end`

-   fluid: `lambda, corrX`

**Must-have for rigids**

-   `p,q,vLin,vAng, invMassR, invInertiaLocal`

-   `pPred,qPred`

-   `aabbR`

-   rigid broadphase output `rigidPairs`

-   `contacts[]` + `contactCount`

-   `corrP,corrRot`

**Plus constraints lists**

-   springs/tets etc.

That's enough to build a first unified solver.

* * * * *

G. Two recommended variants
===========================

### Variant 1: "Explicit contacts buffer" (recommended)

-   build `contacts[]` once per substep

-   solve contacts every iteration from that list

-   easiest to debug and extend (friction, manifolds, warm starting)

### Variant 2: "On-the-fly RP solve during rigid cell scan"

-   no `rigidParticlePairs` and maybe no `contacts[]` for RP

-   but you'll rescan cells each iteration (expensive) and friction/manifolds get messy

-   I only recommend for prototypes or very simple shapes