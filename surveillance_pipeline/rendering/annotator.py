from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from surveillance_pipeline.analytics.base import AlertEvent
from surveillance_pipeline.config import DisplayConfig, LineDefinition, ZoneDefinition

if TYPE_CHECKING:
    from surveillance_pipeline.tracker import Track


# Alert flash period in frames (box border alternates every N frames)
_FLASH_PERIOD = 6
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    """
    Centralises every OpenCV draw call. Nothing outside this class draws.
    This keeps rendering concerns isolated and allows headless runs by simply
    not calling render().
    """

    def __init__(self, config: DisplayConfig) -> None:
        self._config = config
        # Convert class color lists to BGR tuples for cv2
        self._class_colors: dict[str, tuple[int, int, int]] = {
            name: (c[2], c[1], c[0])   # RGB → BGR
            for name, c in config.class_colors.items()
        }
        self._default_color: tuple[int, int, int] = (200, 200, 200)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        events: list[AlertEvent],
        zones: list[ZoneDefinition],
        lines: list[LineDefinition],
        fps: float,
        alert_count: int,
        frame_idx: int,
        loitering_dwell: dict[tuple[int, str], int] | None = None,
        loitering_threshold_frames: int = 1,
        crowd_active: bool = False,
    ) -> np.ndarray:
        out = frame.copy()

        alerted_track_ids: set[int] = {
            e.track_id for e in events if e.track_id is not None
        }

        if self._config.draw_zones:
            self._draw_zones(out, zones, loitering_dwell or {}, loitering_threshold_frames)
            self._draw_lines(out, lines)

        if self._config.draw_tracks:
            for track in tracks:
                flashing = track.track_id in alerted_track_ids
                self._draw_track(out, track, frame_idx, flashing)

        if self._config.draw_alerts and events:
            self._draw_alert_banner(out, events)

        if self._config.show_hud:
            self._draw_hud(out, fps, len(tracks), alert_count, crowd_active)

        return out

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    def _draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        frame_idx: int,
        flashing: bool,
    ) -> None:
        x1, y1, x2, y2 = track.bbox
        color = self._class_colors.get(track.class_name, self._default_color)

        # Flash alert tracks by alternating to red
        if flashing and (frame_idx // _FLASH_PERIOD) % 2 == 0:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_zones(
        self,
        frame: np.ndarray,
        zones: list[ZoneDefinition],
        dwell: dict[tuple[int, str], int],
        threshold_frames: int,
    ) -> None:
        overlay = frame.copy()
        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32)

            # Any track dwelling in this zone? Compute max dwell ratio.
            zone_dwell_values = [v for (_, zn), v in dwell.items() if zn == zone.name]
            max_ratio = max((v / max(threshold_frames, 1) for v in zone_dwell_values), default=0.0)
            max_ratio = min(max_ratio, 1.0)

            # Blend from blue (idle) to red (loitering threshold reached)
            b = int(255 * (1 - max_ratio))
            r = int(255 * max_ratio)
            fill_color = (b, 0, r)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=fill_color, thickness=2)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Draw zone name labels
        for zone in zones:
            pts = np.array(zone.polygon, dtype=np.int32)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(frame, zone.name, (cx - 20, cy), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_lines(self, frame: np.ndarray, lines: list[LineDefinition]) -> None:
        for line in lines:
            p1 = tuple(line.points[0])
            p2 = tuple(line.points[1])
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

            # Direction arrow at midpoint
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(frame, line.name, (mx + 4, my - 4), _FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, frame: np.ndarray, events: list[AlertEvent]) -> None:
        """Draw a small red alert tag for each unique event type this frame."""
        unique_types = {e.event_type for e in events}
        x, y = 10, frame.shape[0] - 10
        for etype in sorted(unique_types):
            label = f"! {etype}"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 2)
            y -= th + 8
            cv2.rectangle(frame, (x, y - 2), (x + tw + 8, y + th + 4), (0, 0, 200), -1)
            cv2.putText(frame, label, (x + 4, y + th), _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        track_count: int,
        alert_count: int,
        crowd_active: bool,
    ) -> None:
        """Semi-transparent HUD bar at the top of the frame."""
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        hud_text = (
            f"FPS: {fps:5.1f}  |  Tracks: {track_count:3d}  |  "
            f"Alerts: {alert_count:3d}"
            + ("  |  [CROWD]" if crowd_active else "")
        )
        cv2.putText(frame, hud_text, (8, 22), _FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
