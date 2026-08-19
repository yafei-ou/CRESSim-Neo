# World and components

`engine::World` owns the scene’s entities and authored components. An entity
belongs to one environment and can combine a transform with rendering, physics,
sensor, and constraint state.

| Area | Main authored components |
| --- | --- |
| Rendering | transform, mesh renderer, camera, lights |
| Rigid physics | rigid body, one or more colliders, joints |
| Deformables | tetrahedral soft body, meshfree soft body, strand, fluid |
| Surgical sensing | ultrasound probe, renderer, and scatterer source |

Constraints and joints are authored through `World` as named state records,
then resolved to device indices during `prepare()`. Use the prepared layout
mappings when custom GPU code must relate an authored object to packed buffers.
