from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _cache_blit_background(event) -> None:
    canvas = event.canvas
    if canvas.supports_blit:
        canvas.figure._cressim_blit_background = canvas.copy_from_bbox(canvas.figure.bbox)


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
    figure.canvas.mpl_connect("draw_event", _cache_blit_background)
    plt.show(block=False)
    figure.canvas.draw()
    artists = np.asarray(image_artists, dtype=object)
    update_rgb_grid_figure(artists, rgb_tensor)
    return figure, artists


def update_rgb_grid_figure(image_artists: np.ndarray, rgb_tensor) -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    if image_artists.size == 0:
        return

    figure = image_artists.flat[0].figure
    canvas = figure.canvas
    for env_index, image_artist in enumerate(image_artists.tolist()):
        image_artist.set_data(rgb_images[env_index])

    if not canvas.supports_blit:
        canvas.draw_idle()
        return

    background = getattr(figure, "_cressim_blit_background", None)
    if background is None:
        canvas.draw()
        background = getattr(figure, "_cressim_blit_background", None)
    if background is None:
        return

    canvas.restore_region(background)
    for image_artist in image_artists.flat:
        image_artist.axes.draw_artist(image_artist)
    canvas.blit(figure.bbox)
    canvas.flush_events()


def create_rgb_ultrasound_grid_figure(
    rgb_tensor,
    observation_tensor,
    *,
    max_columns: int = 4,
) -> tuple["plt.Figure", np.ndarray, np.ndarray]:
    """Create a blitted RGB/ultrasound grid from a stacked ultrasound observation."""
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    ultrasound_images = observation_tensor[:, -1].detach().cpu().numpy()
    env_count = rgb_images.shape[0]
    column_count = min(max_columns, env_count)
    row_count = int(np.ceil(env_count / column_count))
    figure, axes = plt.subplots(
        row_count * 2,
        column_count,
        figsize=(4 * column_count, 6 * row_count),
        squeeze=False,
    )
    rgb_artists: list[object] = []
    ultrasound_artists: list[object] = []
    for env_index in range(env_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        rgb_axis = axes[row_index * 2, column_index]
        ultrasound_axis = axes[row_index * 2 + 1, column_index]
        rgb_artist = rgb_axis.imshow(rgb_images[env_index], animated=True)
        ultrasound_artist = ultrasound_axis.imshow(
            ultrasound_images[env_index],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            animated=True,
        )
        rgb_axis.set_title(f"Env {env_index} RGB")
        ultrasound_axis.set_title(f"Env {env_index} Ultrasound")
        rgb_axis.axis("off")
        ultrasound_axis.axis("off")
        rgb_artists.append(rgb_artist)
        ultrasound_artists.append(ultrasound_artist)
    for env_index in range(env_count, row_count * column_count):
        row_index = env_index // column_count
        column_index = env_index % column_count
        axes[row_index * 2, column_index].axis("off")
        axes[row_index * 2 + 1, column_index].axis("off")
    figure.tight_layout()
    figure.canvas.mpl_connect("draw_event", _cache_blit_background)
    plt.show(block=False)
    figure.canvas.draw()
    rgb_artist_array = np.asarray(rgb_artists, dtype=object)
    ultrasound_artist_array = np.asarray(ultrasound_artists, dtype=object)
    update_rgb_ultrasound_grid_figure(
        rgb_artist_array, ultrasound_artist_array, rgb_tensor, observation_tensor
    )
    return figure, rgb_artist_array, ultrasound_artist_array


def update_rgb_ultrasound_grid_figure(
    rgb_artists: np.ndarray,
    ultrasound_artists: np.ndarray,
    rgb_tensor,
    observation_tensor,
) -> None:
    rgb_images = rgb_tensor_to_numpy(rgb_tensor)
    ultrasound_images = observation_tensor[:, -1].detach().cpu().numpy()
    if rgb_artists.size == 0:
        return

    figure = rgb_artists.flat[0].figure
    canvas = figure.canvas
    for env_index, rgb_artist in enumerate(rgb_artists.tolist()):
        rgb_artist.set_data(rgb_images[env_index])
        ultrasound_artists[env_index].set_data(ultrasound_images[env_index])

    if not canvas.supports_blit:
        canvas.draw_idle()
        canvas.flush_events()
        return

    background = getattr(figure, "_cressim_blit_background", None)
    if background is None:
        canvas.draw()
        background = getattr(figure, "_cressim_blit_background", None)
    if background is None:
        return

    canvas.restore_region(background)
    for image_artist in (*rgb_artists.flat, *ultrasound_artists.flat):
        image_artist.axes.draw_artist(image_artist)
    canvas.blit(figure.bbox)
    canvas.flush_events()


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
