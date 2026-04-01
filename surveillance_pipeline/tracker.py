from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from surveillance_pipeline.config import TrackerConfig
from surveillance_pipeline.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    class_name: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    age_frames: int = 0
    centroid_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=60)
    )

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TrackManager:
    """
    Wraps supervision.ByteTracker. Maintains Track objects with persistent
    state (centroid history, age, first-seen frame) across frames.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=config.track_buffer,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=1,
        )
        self._tracks: dict[int, Track] = {}  # track_id → Track

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        """
        Feed new detections into ByteTracker and return the current list of
        active Track objects with updated state.
        """
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array([d.bbox for d in detections], dtype=float)
            confidences = np.array([d.confidence for d in detections], dtype=float)
            class_ids = np.array([d.class_id for d in detections], dtype=int)

            # Filter by minimum box area
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            mask = areas >= self._config.min_box_area
            sv_dets = sv.Detections(
                xyxy=xyxy[mask],
                confidence=confidences[mask],
                class_id=class_ids[mask],
            )
            # Carry class names through via data dict
            class_names = np.array([d.class_name for d in detections])[mask]
            sv_dets.data["class_name"] = class_names

        tracked = self._tracker.update_with_detections(sv_dets)

        # Build lookup from class_id → class_name using the original detections
        class_id_to_name: dict[int, str] = {d.class_id: d.class_name for d in detections}

        active_ids: set[int] = set()

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = (int(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cid = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            cname = class_id_to_name.get(cid, "unknown")

            active_ids.add(tid)

            if tid not in self._tracks:
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    class_name=cname,
                    confidence=conf,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                )
            else:
                t = self._tracks[tid]
                t.bbox = (x1, y1, x2, y2)
                t.confidence = conf
                t.last_seen_frame = frame_idx
                t.age_frames = frame_idx - t.first_seen_frame

            # Append current centroid to history
            self._tracks[tid].centroid_history.append(self._tracks[tid].centroid)

        # Prune tracks that were not updated this frame
        gone = [tid for tid in self._tracks if tid not in active_ids]
        for tid in gone:
            del self._tracks[tid]

        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self._config.track_buffer,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=1,
        )
