# Showcases

Showcases are deterministic, capture-ready demonstrations intended for product
footage.  Unlike the focused developer examples, a showcase owns its timing,
composition, and frame export.

## Paper videos

`examples/showcases/python/record_py_scenes.py` records the seven environments discussed in
the paper: CartPole, SoftBodyPush, FluidPour, TargetCenter, TissueRetract,
BloodSuction, and UltrasoundScan.  It defaults to standard 1080p H.264 at
30 fps and can record every scene or a selected subset:

```bash
# One scene (repeat --scene to select more than one)
python examples/showcases/python/record_py_scenes.py --scene blood_suction

# All paper scenes, six seconds each, replacing previous outputs
python examples/showcases/python/record_py_scenes.py --frames 180 --overwrite
```

Videos are written to `artifacts/paper_videos/` and are generated headlessly.
FFmpeg must be on `PATH`.

UltrasoundScan includes the latest ultrasound image in the upper-left corner.
Use `--motion-scale 0.5` to halve the scripted action magnitude for any
recorded scene; `1.0` matches the scripted-demo command magnitude.
Captures reset completed episodes by default; add `--stop-on-done` for a
single-episode clip.
The UltrasoundScan capture synthesizes a higher-resolution ultrasound image
with a 640-pixel height by default; its width is derived from the probe layout.
Adjust the height with `--ultrasound-height`.

## Complete paper set

The paper also contains C++ demonstrations of large-scale rigid bodies, a
deformable toroid, joints, fluids, suturing, CDCR, and ultrasound sensing. The shared
viewer now streams its off-screen render target directly to FFmpeg, producing
an MP4 without desktop capture or temporary image files.

```bash
# Rebuild the viewer and C++ examples after pulling these changes.
cmake --build build/linux-release -j2

# Select only the remaining C++ paper scenes; repeat --scene as needed.
python examples/showcases/record_cpp_scenes.py --scene suturing --scene cdcr --overwrite

# Record all seven remaining C++ scenes.
python examples/showcases/record_cpp_scenes.py --overwrite
```

The C++ selector accepts: `rigid_body_scale`, `soft_bodies`, `joints`,
`fluid_slider`, `suturing`, `cdcr`, and `ultrasound_demo`. C++ captures use the standard
1920x1080, 30 fps defaults and simulate at a fixed 60 Hz, encoding every
second simulation frame. Keep `--simulation-fps` independent from `--fps` so
the physics solver does not receive a larger timestep merely because the video
is slower. Use `--bin-dir` when the examples are built elsewhere.

`rigid_body_scale`, `soft_bodies` (the toroid scene), `fluid_slider`, and `ultrasound_demo` support multiple
environments. For example, the following creates four environments and moves
to the next environment camera every 90 video frames (three seconds at 30 fps):

```bash
python examples/showcases/record_cpp_scenes.py --scene soft_bodies \
  --envs 4 --switch-interval 90 --overwrite
```

The other C++ paper demos are authored as a single scene, so `--envs` has no
effect on them.
