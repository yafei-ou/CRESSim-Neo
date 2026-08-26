Python bindings API reference
=============================

``cressim_neo`` is the low-level bindings and runtime package.

.. automodule:: cressim_neo

Higher-level environments and PSM authoring helpers are provided by the sibling
``cressim_neo_envs`` package. The task modules use the paper's names:
``cartpole``, ``soft_body_push_env``, ``fluid_pour_env``,
``target_center_env``, ``tissue_retract_env``, ``blood_suction_env``, and
``ultrasound_scan_env``. Import a task environment from its defining module,
for example ``cressim_neo_envs.tissue_retract_env.TissueRetractTorchVectorEnv``.
