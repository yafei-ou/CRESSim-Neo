# CartPole

**Demonstrates:** batched rigid-body simulation, Torch observations and actions,
GPU rewards, and RGB rendering.

```bash
python examples/python/cartpole_torch_vector_env.py
```

| Change | Location |
| --- | --- |
| Batch size | `env_count` |
| Episode length | `max_episode_steps` |
| Image size | `image_width`, `image_height` |
| Task logic | custom compute passes in `cartpole.py` |

This is the recommended first reference for a GPU-resident RL loop.
