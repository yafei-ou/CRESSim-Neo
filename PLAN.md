**What To Support**
Support a compact “simulation PBR” baseline:

- Texture pipeline: `baseColor`, `normal`, `metallicRoughness`, `emissive`, `AO`
- IBL: one static environment cubemap per environment, preprocessed into irradiance + prefiltered specular + BRDF LUT
- Correct tangent-space normal mapping
- Directional light shadows
- Point/spot lights, but keep them bounded and cheap
- Filmic tone mapping with explicit exposure control

That set is enough to get most of the visual gain people expect from “modern PBR” without pushing the renderer into engine-complexity territory.

**How I’d Scope It**
For your use case, I’d recommend:

- IBL:
  One static environment map per environment is exactly the right model.
  No dynamic reflection captures.
  No probe blending unless you later prove you need it.
  For RL/data generation, stable and cheap beats fancy here.

- Lights:
  Support directional + point + spot.
  Make one directional light the “hero” shadow caster.
  For point/spot, cap shadowed lights very aggressively, or even allow only unshadowed by default.
  A good compromise is: many unshadowed local lights, very few shadowed local lights.

- Tone mapping:
  Yes to filmic + exposure.
  Keep it simple and deterministic.
  ACES-fit or another lightweight filmic curve is a good step up from Reinhard.
  Avoid auto-exposure unless you have a strong reason; fixed/manual exposure is usually better for ML reproducibility.

- Materials:
  Stick to metallic-roughness workflow.
  Packed textures are good: `ORM` or `RMA` style packing reduces bandwidth and management overhead.

**What I Would Not Support**
I would avoid these, at least for the main path:

- Clear coat
- Sheen
- Anisotropy
- Subsurface scattering
- Transmission/refraction
- Parallax/POM
- SSR/SSGI
- Bloom, lens flares, chromatic aberration
- Temporal AA-heavy pipelines
- Dynamic reflection probes
- Area lights with complex BRDFs

These features add lots of complexity per unit of value for robotics/simulation. They help marketing screenshots more than dataset quality.

**Priority Order**
If we were building this for your renderer, I’d prioritize:

1. Texture-backed materials with correct color-space handling
2. Tangent-space normal mapping
3. IBL with static per-environment cubemap
4. Better tone mapping + manual exposure
5. Point/spot lights
6. Limited local-light shadows

That order gives the biggest realism jump earliest.

**One Important ML/Simulation Note**
For data generation, I’d explicitly design for:

- deterministic outputs across runs
- stable color pipeline
- fixed exposure option
- feature toggles per batch/environment
- graceful scalability when rendering many cameras/environments

**Milestones**

For this renderer, I’d build toward a “simulation-grade PBR” stack in five milestones. The order should maximize visual gain early while keeping the pipeline stable for large-scale generation.

1. **Material Foundation**
   Add texture-backed materials first: `baseColor`, `normal`, `metallicRoughness`, `emissive`, `AO`, plus correct sRGB/linear handling and packed texture conventions. This is the biggest unlock because your current shader is mostly scalar-material driven, so most modern assets still won’t look right until this exists.

2. **Correct Surface Shading**
   Implement tangent-space normal mapping properly, including tangent basis generation/import, handedness, and double-sided normal handling. This is the next highest-value change because once textures exist, bad or missing normal mapping will make everything still look flat.

3. **Static IBL Per Environment**
   Add one static environment cubemap per environment, precomputed into irradiance, prefiltered specular, and BRDF LUT. This should come before more light types because it gives you the largest jump in realism and material readability at a relatively predictable runtime cost, and it fits your “many environments / mostly static setup” model well.

4. **Camera and Output Quality**
   Upgrade display/output to filmic tone mapping plus explicit exposure control. Keep manual exposure as the default for reproducibility. This milestone is smaller than the first three, but it matters a lot once HDR lighting and emissive/IBL are in place.

5. **Local Lights**
   Add point and spot lights with hard caps and a conservative shadow model. I’d support many unshadowed local lights first, then a very small number of shadowed ones later. For RL/data generation, this is useful, but it should not come before the material and IBL work.

**What To Delay**
Push these out of the main roadmap unless you later prove they matter: clear coat, sheen, anisotropy, transmission, dynamic reflection probes, SSR/SSGI, bloom, fancy camera effects, and broad shadow support for many local lights.

**Recommended Cut**
If you want the practical “v1 target,” I’d make it:

- Milestone 1: texture pipeline
- Milestone 2: correct normal mapping
- Milestone 3: static per-env IBL
- Milestone 4: filmic tone map + exposure
- Milestone 5: point/spot lights, unshadowed first

That gives you a renderer that is much closer to SAPIEN-class usefulness without drifting into Unity-class scope.

If you want, I can turn this into a concrete `PLAN.md` milestone breakdown with deliverables and acceptance criteria for each phase.

