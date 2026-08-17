# Showcases

Showcases are deterministic, capture-ready demonstrations intended for product
footage.  Unlike the focused developer examples, a showcase owns its timing,
composition, and frame export.

## Paper videos

`showcases/python/record_paper_scenes.py` records the seven environments discussed in
`root.pdf`: CartPole, SoftBodyPush, FluidPour, TargetCenter, TissueRetract,
BloodSuction, and UltrasoundScan.  It defaults to standard 1080p H.264 at
30 fps and can record every scene or a selected subset:

```bash
# One scene (repeat --scene to select more than one)
python showcases/python/record_paper_scenes.py --scene blood_suction

# All paper scenes, six seconds each, replacing previous outputs
python showcases/python/record_paper_scenes.py --frames 180 --overwrite
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
