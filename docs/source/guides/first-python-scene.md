# First Python scene

Python binds the same runtime and world API. The method names use
snake_case, but the execution model is unchanged.

```python
import cressim_neo as neo

runtime = neo.Runtime()
if not runtime.initialize(neo.RuntimeConfig()):
    raise RuntimeError("Could not initialize CRESSim-Neo")

world = runtime.world()
entity = world.create_entity()
runtime.prepare()
if not runtime.upload_world():
    raise RuntimeError("Could not upload the world")
```

See `examples/python/runtime_frame_readback.py` for a complete direct-runtime
example, then continue with the same User Guide as a C++ application.
