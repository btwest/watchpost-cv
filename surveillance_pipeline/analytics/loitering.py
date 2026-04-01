from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from surveillance_pipeline.analytics.base import AlertEvent, BaseAnalytic
from surveillance_pipeline.config import LoiteringConfig, ZoneDefinition

if TYPE_CHECKING:
    from surveillance_pipeline.tracker import Track


class LoiteringDetector(BaseAnalytic):
    """
    Emits a LOITERING alert when a tracked object's centroid remains inside
    a named zone polygon for longer than `threshold_seconds`.

    Alert fires once per continuous dwelling episode. The alert is suppressed
    until the track leaves and re-enters the zone.
    """

    def __init__(self, config: LoiteringConfig, fps: float) -> None:
        self._config = config
        self._fps = max(fps, 1.0)
        self._threshold_frames = int(config.threshold_seconds * self._fps)

        # Precompute polygon arrays for cv2.pointPolygonTest
        self._polygons: dict[str, np.ndarray] = {
            z.name: np.array(z.polygon, dtype=np.int32)
            for z in config.zones
        }

        # (track_id, zone_name) → accumulated frame count inside zone
        self._dwell_frames: dict[tuple[int, str], int] = {}
        # (track_id, zone_name) → alert already fired for this episode
        self._alerted: set[tuple[int, str]] = set()

    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_sec: float,
    ) -> list[AlertEvent]:
        if not self._config.enabled or not self._polygons:
            return []

        events: list[AlertEvent] = []
        active_keys: set[tuple[int, str]] = set()

        for track in tracks:
            cx, cy = track.centroid
            point = (float(cx), float(cy))

            for zone_name, polygon in self._polygons.items():
                key = (track.track_id, zone_name)
                inside = cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0

                if inside:
                    active_keys.add(key)
                    self._dwell_frames[key] = self._dwell_frames.get(key, 0) + 1

                    if (
                        self._dwell_frames[key] >= self._threshold_frames
                        and key not in self._alerted
                    ):
                        self._alerted.add(key)
                        events.append(AlertEvent(
                            event_type="LOITERING",
                            track_id=track.track_id,
                            class_name=track.class_name,
                            zone_name=zone_name,
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            metadata={
                                "dwell_seconds": round(self._dwell_frames[key] / self._fps, 2),
                                "centroid": [round(cx, 1), round(cy, 1)],
                            },
                        ))
                else:
                    # Track left the zone — reset dwell and alert suppression
                    self._dwell_frames.pop(key, None)
                    self._alerted.discard(key)

        # Clean up state for tracks that are no longer active
        gone_keys = [k for k in self._dwell_frames if k not in active_keys]
        for k in gone_keys:
            self._dwell_frames.pop(k, None)
            self._alerted.discard(k)

        return events

    def reset(self) -> None:
        self._dwell_frames.clear()
        self._alerted.clear()

    @property
    def dwell_frames(self) -> dict[tuple[int, str], int]:
        """Expose dwell state so Annotator can shade zones proportionally."""
        return self._dwell_frames

    @property
    def threshold_frames(self) -> int:
        return self._threshold_frames
