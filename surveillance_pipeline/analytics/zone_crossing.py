from __future__ import annotations

from typing import TYPE_CHECKING

from surveillance_pipeline.analytics.base import AlertEvent, BaseAnalytic
from surveillance_pipeline.config import ZoneCrossingConfig

if TYPE_CHECKING:
    from surveillance_pipeline.tracker import Track


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """
    Return True if segment p1→p2 intersects segment p3→p4.
    Uses cross-product sign test (collinear cases treated as non-crossing).
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _crossing_direction(p1, p2, line_a, line_b) -> str:
    """
    Determine crossing direction based on which side of the line the
    movement vector ends up on.
    'AB' = crossed from A-side to B-side, 'BA' = opposite.
    """
    cross_before = _cross(line_a, line_b, p1)
    cross_after = _cross(line_a, line_b, p2)
    if cross_before > 0 and cross_after <= 0:
        return "AB"
    return "BA"


class ZoneCrossingDetector(BaseAnalytic):
    """
    Emits a ZONE_CROSSING alert when a tracked object's centroid trajectory
    intersects a named virtual line between two consecutive frames.

    A per-track cooldown prevents re-triggering on slow crossings.
    """

    _COOLDOWN_FRAMES = 20

    def __init__(self, config: ZoneCrossingConfig) -> None:
        self._config = config
        # (track_id, line_name) → frame_idx of last crossing
        self._cooldown: dict[tuple[int, str], int] = {}

    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_sec: float,
    ) -> list[AlertEvent]:
        if not self._config.enabled or not self._config.lines:
            return []

        events: list[AlertEvent] = []

        for track in tracks:
            history = track.centroid_history
            if len(history) < 2:
                continue

            # Most recent movement segment
            prev = history[-2]
            curr = history[-1]

            for line_def in self._config.lines:
                key = (track.track_id, line_def.name)
                line_a = tuple(line_def.points[0])
                line_b = tuple(line_def.points[1])

                # Cooldown check
                last_cross = self._cooldown.get(key)
                if last_cross is not None and (frame_idx - last_cross) < self._COOLDOWN_FRAMES:
                    continue

                if not _segments_intersect(prev, curr, line_a, line_b):
                    continue

                direction = _crossing_direction(prev, curr, line_a, line_b)

                # Direction filter
                if line_def.direction != "any" and direction != line_def.direction:
                    continue

                self._cooldown[key] = frame_idx
                events.append(AlertEvent(
                    event_type="ZONE_CROSSING",
                    track_id=track.track_id,
                    class_name=track.class_name,
                    zone_name=line_def.name,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    metadata={
                        "direction": direction,
                        "centroid": [round(curr[0], 1), round(curr[1], 1)],
                    },
                ))

        return events

    def reset(self) -> None:
        self._cooldown.clear()
