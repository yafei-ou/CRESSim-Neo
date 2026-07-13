from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rgb_tensor_to_numpy(rgb_tensor) -> np.ndarray:
    return np.clip(rgb_tensor[..., :3].detach().cpu().numpy(), 0.0, 1.0)


def create_rgb_grid_figure(
    rgb_tensor,
    *,
    max_columns: int = 4,
    title_prefix: str = "Env",
) -> tuple["plt.Figure", np.ndarray]:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    env_count = rgb_images.shape[0]
    column_count = min(max_columns, env_count)
    row_count = int(np.ceil(env_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4 * column_count, 4 * row_count),
        squeeze=False,
    )
    image_artists: list[np.ndarray] = []
    for env_index in range(env_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        image_artist = axes[row_index, column_index].imshow(rgb_images[env_index], animated=True)
        axes[row_index, column_index].set_title(f"{title_prefix} {env_index}")
        axes[row_index, column_index].axis("off")
        image_artists.append(image_artist)
    for env_index in range(env_count, row_count * column_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        axes[row_index, column_index].axis("off")
    figure.tight_layout()
    plt.show(block=False)
    return figure, np.asarray(image_artists, dtype=object)


def update_rgb_grid_figure(image_artists: np.ndarray, rgb_tensor) -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])


class InteractiveImageCapture:
    def __init__(self, figure: "plt.Figure", script_path: str | Path, *, key: str = "c") -> None:
        self.figure = figure
        self.script_path = Path(script_path).resolve()
        self.script_name = self.script_path.stem
        self.output_dir = self.script_path.parents[2] / "artifacts" / "screenshots" / self.script_name
        self.key = key
        self._capture_index = 0
        self._step_label = "reset"
        self._image_groups: list[tuple[str, np.ndarray, str | None]] = []
        self._connection_id = self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        print(
            f"Press '{self.key}' in the figure window to save per-env texture images to {self.output_dir}."
        )

    def update(self, step_label: str, image_groups: list[tuple[str, np.ndarray, str | None]]) -> None:
        self._step_label = step_label
        self._image_groups = [
            (name, np.array(images, copy=True), cmap) for name, images, cmap in image_groups
        ]

    def close(self) -> None:
        self.figure.canvas.mpl_disconnect(self._connection_id)

    def _on_key_press(self, event) -> None:
        if event.key != self.key or not self._image_groups:
            return

        self._capture_index += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        step_slug = str(self._step_label).replace(" ", "_")

        for group_name, images, cmap in self._image_groups:
            for env_index in range(images.shape[0]):
                image = images[env_index]
                if image.ndim == 3 and image.shape[-1] > 3:
                    image = image[..., :3]
                path = self.output_dir / (
                    f"{self.script_name}_{step_slug}_cap{self._capture_index:03d}_"
                    f"{group_name}_env{env_index}.png"
                )
                if image.ndim == 3:
                    plt.imsave(path, np.clip(image, 0.0, 1.0))
                else:
                    plt.imsave(path, np.clip(image, 0.0, 1.0), cmap=cmap or "gray", vmin=0.0, vmax=1.0)
                saved_paths.append(path)

        print("Saved screenshots:")
        for path in saved_paths:
            print(f"  {path}")
