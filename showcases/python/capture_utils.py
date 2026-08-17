"""Shared, headless RGB video-capture helpers for CRESSim Python scenes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def rgb_tensor_to_frame(rgb_tensor, *, env_index: int = 0, supersample: int = 1) -> np.ndarray:
    """Return one rendered environment as a CPU float RGB frame."""
    frame = rgb_tensor[env_index, ..., :3]
    if supersample > 1:
        height, width, channels = frame.shape
        if height % supersample or width % supersample:
            raise ValueError("Supersampled render dimensions must divide evenly by supersample.")
        frame = frame.reshape(height // supersample, supersample, width // supersample, supersample, channels).mean(dim=(1, 3))
    return frame.detach().cpu().numpy()


class VideoWriter:
    """Stream RGB frames to an H.264 MP4 without retaining image sequences."""

    def __init__(self, output: Path, width: int, height: int, fps: int, *, overwrite: bool = False) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required for video capture but was not found on PATH.")
        self.output, self.width, self.height = output, width, height
        self.process = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected a {self.width}x{self.height} RGB frame, got {frame.shape}.")
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin was unexpectedly unavailable.")
        image = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"FFmpeg failed while encoding {self.output}.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prepare_output(output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
