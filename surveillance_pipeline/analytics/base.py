from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surveillance_pipeline.tracker import Track


@dataclass
class AlertEvent:
    event_type: str                   # "LOITERING" | "ZONE_CROSSING" | "CROWD_THRESHOLD"
    frame_idx: int
    timestamp_sec: float
    track_id: int | None = None       # None for scene-level events (crowd)
    class_name: str | None = None
    zone_name: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":    self.event_type,
            "track_id":      self.track_id,
            "class_name":    self.class_name,
            "zone_name":     self.zone_name,
            "frame_idx":     self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "metadata":      self.metadata,
        }


class BaseAnalytic(ABC):
    """Common interface for all per-frame analytics."""

    @abstractmethod
    def process(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp_sec: float,
    ) -> list[AlertEvent]:
        """Evaluate tracks for this frame and return any triggered events."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (e.g. between video loops)."""
        ...
