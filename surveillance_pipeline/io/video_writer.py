from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Thin context-manager wrapper around cv2.VideoWriter."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
