# CRESSim Neo Coding Style

## Naming

Use the following naming rules across `include/` and `src/`.

| Code Element | Convention | Example |
| --- | --- | --- |
| Namespace | `lower_case` | `cressim::neo::graphics` |
| Class / Struct / Enum / Type Alias | `PascalCase` | `RenderWorld`, `FrameContext` |
| Function / Method | `camelCase` | `createEntity`, `syncWorldToRenderWorld` |
| Local Variable | `camelCase` | `frameContext`, `renderWorld` |
| Function Parameter | `camelCase` | `entityId`, `frameContext` |
| Member Variable | `m` + `camelCase` | `mRenderer`, `mWorld` |
| Global Variable | `g` + `camelCase` | `gEngineInstance` |
| `constexpr` variable / constants | `k` + `PascalCase` | `kInvalidEntityId` |
| Macro | `UPPER_SNAKE_CASE` | `CRESSIM_NEO_ENGINE_API` |

## Notes

- Avoid global variables when possible. If needed, use `g` prefix.
- Keep naming consistent across headers and implementations.
- Favor descriptive names over abbreviations unless the abbreviation is standard (`id`, `gpu`, `api`).

## Tooling

- Naming is enforced by the repository root `.clang-tidy`.
- Enable clang-tidy during CMake builds with:

```bash
cmake -S . -B build -DCRESSIM_NEO_ENABLE_CLANG_TIDY=ON
cmake --build build -j
```

