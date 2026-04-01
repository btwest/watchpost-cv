from __future__ import annotations

from typing import TYPE_CHECKING

from surveillance_pipeline.analytics.base import AlertEvent, BaseAnalytic
from surveillance_pipeline.config import CrowdThresholdConfig

if TYPE_CHECKING:
    from surveillance_pipeline.tracker import Track


class CrowdThresholdDetector(BaseAnalytic):
    """
    Emits a CROWD_THRESHOLD alert when the number of active tracked objects
    exceeds `max_objects`.

    Uses hysteresis to prevent alert flicker: the alert fires at `max_objects`
    and only clears when the count drops below `max_objects - hysteresis_band`.
    """

    def __init__(self, config: CrowdThresholdConfig) -> None:
        self._config = config
        self._crowd_active: bool = False

    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_sec: float,
    ) -> list[AlertEvent]:
        if not self._config.enabled:
            return []

        count = len(tracks)
        events: list[AlertEvent] = []
        clear_threshold = self._config.max_objects - self._config.hysteresis_band

        if not self._crowd_active and count > self._config.max_objects:
            self._crowd_active = True
            events.append(AlertEvent(
                event_type="CROWD_THRESHOLD",
                track_id=None,
                class_name=None,
                zone_name=None,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                metadata={
                    "object_count": count,
                    "threshold": self._config.max_objects,
                },
            ))
        elif self._crowd_active and count <= clear_threshold:
            self._crowd_active = False

        return events

    @property
    def is_active(self) -> bool:
        """True while crowd condition is ongoing (for HUD / rendering)."""
        return self._crowd_active

    def reset(self) -> None:
        self._crowd_active = False
