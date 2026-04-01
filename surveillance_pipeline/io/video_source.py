from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np

from surveillance_pipeline.config import SourceConfig


class VideoSource:
    """
    Abstracted video input. Iterates as (frame, frame_idx, timestamp_sec).
    Timestamp is derived from frame index and FPS for determinism — not wall clock.
    Supports optional looping for demo purposes.
    """

    def __init__(self, config: SourceConfig) -> None:
        self._config = config
        self._cap = cv2.VideoCapture(config.input_path)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {config.input_path}")

    @property
    def fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[tuple[np.ndarray, int, float]]:
        frame_idx = 0
        fps = self.fps

        while True:
            ret, frame = self._cap.read()

            if not ret:
                if self._config.loop_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                break

            timestamp_sec = frame_idx / fps
            yield frame, frame_idx, timestamp_sec
            frame_idx += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()
