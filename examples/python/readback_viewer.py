import struct

import matplotlib.pyplot as plt
import numpy as np

import cressim_neo as neo


def color_image_from_event(event) -> np.ndarray:
    color_bytes = bytes(event.color_bytes)
    stride = event.color_row_stride_bytes
    if stride == 0:
        raise RuntimeError("Readback event does not provide a valid color row stride.")

    if event.color_format == neo.TextureFormat.RGBA16Float:
        image = np.empty((event.color_height, event.color_width, 3), dtype=np.float32)
        for row_index in range(event.color_height):
            row_start = row_index * stride
            for pixel_index in range(event.color_width):
                src = row_start + pixel_index * 8
                r, g, b, _ = struct.unpack_from("<4e", color_bytes, src)
                image[row_index, pixel_index, 0] = max(0.0, r)
                image[row_index, pixel_index, 1] = max(0.0, g)
                image[row_index, pixel_index, 2] = max(0.0, b)

        image = image / (1.0 + image)
        return np.power(np.clip(image, 0.0, 1.0), 1.0 / 2.2)

    if event.color_format in (
        neo.TextureFormat.BGRA8Unorm,
        neo.TextureFormat.BGRA8UnormSrgb,
    ):
        image = np.empty((event.color_height, event.color_width, 3), dtype=np.uint8)
        for row_index in range(event.color_height):
            row_start = row_index * stride
            row_end = row_start + (event.color_width * 4)
            bgra_row = color_bytes[row_start:row_end]
            for pixel_index in range(event.color_width):
                src = pixel_index * 4
                image[row_index, pixel_index, 0] = bgra_row[src + 2]
                image[row_index, pixel_index, 1] = bgra_row[src + 1]
                image[row_index, pixel_index, 2] = bgra_row[src + 0]
        return image

    if event.color_format not in (
        neo.TextureFormat.RGBA8Unorm,
        neo.TextureFormat.RGBA8UnormSrgb,
    ):
        raise RuntimeError(f"Unsupported readback color format for display: {event.color_format}")

    image = np.empty((event.color_height, event.color_width, 3), dtype=np.uint8)
    for row_index in range(event.color_height):
        row_start = row_index * stride
        row_end = row_start + (event.color_width * 4)
        rgba_row = color_bytes[row_start:row_end]
        for pixel_index in range(event.color_width):
            src = pixel_index * 4
            image[row_index, pixel_index, 0] = rgba_row[src + 0]
            image[row_index, pixel_index, 1] = rgba_row[src + 1]
            image[row_index, pixel_index, 2] = rgba_row[src + 2]
    return image


def show_color_event(event, title: str = "CRESSim-Neo Frame Readback") -> None:
    image = color_image_from_event(event)
    center_pixel = image[event.color_height // 2, event.color_width // 2]
    print("Center pixel:", center_pixel)
    plt.figure(title)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


class ColorFramePlayer:
    def __init__(self, title: str = "CRESSim-Neo Readback Animation", interval: float = 0.25):
        self._title = title
        self._interval = interval

    def show(self, events) -> None:
        images = [color_image_from_event(event) for event in events]
        if not images:
            raise RuntimeError("Expected at least one readback event to display.")

        plt.ion()
        figure, axes = plt.subplots(num=self._title)
        artist = axes.imshow(images[0])
        axes.axis("off")

        for frame_index, image in enumerate(images):
            artist.set_data(image)
            axes.set_title(f"Frame {frame_index}")
            figure.canvas.draw_idle()
            plt.pause(self._interval)

        plt.ioff()
        plt.show()
