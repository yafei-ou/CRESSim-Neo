# Surgical Robotics

To show the engine's ability to support realistic surgical scenes and robot
learning workflows, CRESSim-Neo provides three example RL tasks for surgical
robotics: `TissueRetract`, `BloodSuction`, and `UltrasoundScan`.

```{figure} ../_static/dvrk-psm.png
:alt: Simulated dVRK patient-side manipulator with large needle driver and suction-irrigator tools.
:width: 100%

Simulated PSM robot from the dVRK: (a) full robot; (b) Large Needle Driver;
and (c) Suction/Irrigator.
```

With the physics capabilities described in {doc}`physics-and-constraints`, the
engine can simulate the patient-side manipulator (PSM) from the da Vinci
Research Kit. In `TissueRetract`, the PSM tool approaches a selected target
location on a deformable tissue patch, establishes a particle-rigid attachment,
and lifts the tissue to a target height. In `BloodSuction`, a PSM with a
suction-irrigator tool interacts with fluid particles inside a deformable
container, drawing nearby particles toward the tool and removing particles that
reach the suction region. `UltrasoundScan` is a batched environment in which a
linear ultrasound probe is moved over deformable tissue to center a target dark
area in the ultrasound image. The sensor model is described in
{doc}`rendering-and-sensors`.

```{figure} ../_static/tissue-retract.png
:alt: TissueRetract surgical robotics task.
:width: 100%

`TissueRetract`.
```

```{figure} ../_static/blood-suction.png
:alt: BloodSuction surgical robotics task.
:width: 100%

`BloodSuction`.
```

```{figure} ../_static/ultrasound-scan.png
:alt: UltrasoundScan surgical robotics task.
:width: 100%

`UltrasoundScan`.
```

Beyond these RL environments, CRESSim-Neo also includes surgery-relevant scenes
for suturing with a curved needle and thread passing through deformable tissue,
cable-driven continuum robot (CDCR) demonstrations based on routed cable
constraints, and thermal coagulation with varying tissue properties under heat
transfer.

```{figure} ../_static/suturing.png
:alt: Soft-body suturing scene.
:width: 100%

Soft-body suturing.
```

```{figure} ../_static/cdcr.png
:alt: Cable-driven continuum robot approximated with spherical joints.
:width: 100%

An approximated CDCR with spherical joints modeling the backbone.
```

```{figure} ../_static/thermal-coagulation.png
:alt: Thermal coagulation simulation with varying tissue properties.
:width: 100%

Thermal coagulation with varying tissue properties and appearance under heat
transfer.
```
