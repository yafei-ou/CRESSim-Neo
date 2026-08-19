# Python bindings

The Python module is a binding layer over the engine API, not a separate
simulation model. Its class and method names follow the C++ runtime with
Python naming conventions.

Install a wheel for application use, or use the repository build while
developing the engine:

```bash
scripts/build_wheel.sh
python -m pip install dist/cressim_neo-*.whl
```

See :doc:`../guides/first-python-scene` for a direct-runtime example. Torch
environments are higher-level examples, not a prerequisite for using the
engine.
