# Step a simulation

After a successful `prepare()` and `uploadWorld()`, advance a frame explicitly:

```cpp
cressim::neo::common::FrameContext frame{};
frame.deltaSeconds = 1.0f / 60.0f;

runtime.stepPhysics(frame);
runtime.stepSimulationSensors(frame); // ultrasound when enabled
runtime.stepVisualSensors(frame);     // cameras
runtime.endFrame(frame);
```

Update `frameIndex` and `timeSeconds` between frames. Insert task-specific
compute after upload and at the point in the frame sequence where its inputs
are ready. Re-run preparation and upload after topology changes.
