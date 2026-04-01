from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from surveillance_pipeline.analytics.base import AlertEvent
from surveillance_pipeline.analytics.crowd_threshold import CrowdThresholdDetector
from surveillance_pipeline.analytics.loitering import LoiteringDetector
from surveillance_pipeline.analytics.zone_crossing import ZoneCrossingDetector
from surveillance_pipeline.config import PipelineConfig
from surveillance_pipeline.detector import Detector
from surveillance_pipeline.io.event_logger import EventLogger
from surveillance_pipeline.io.video_source import VideoSource
from surveillance_pipeline.io.video_writer import VideoWriter
from surveillance_pipeline.rendering.annotator import Annotator
from surveillance_pipeline.tracker import Track, TrackManager


@dataclass
class FrameResult:
    """Hand-off object between _process_frame and the render/write layer."""
    raw_frame: object
    annotated_frame: object
    tracks: list[Track]
    events: list[AlertEvent] = field(default_factory=list)


class Pipeline:
    """
    Central orchestrator. Owns the main processing loop and wires all
    components together. Business logic lives in the components — this class
    sequences their calls per frame and manages resource lifetime.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self._config

        with VideoSource(cfg.source) as source:
            fps = source.fps
            frame_size = source.frame_size

            # Instantiate all components
            detector = Detector(cfg.model)
            tracker = TrackManager(cfg.tracker)
            loitering = LoiteringDetector(cfg.analytics.loitering, fps)
            crossing = ZoneCrossingDetector(cfg.analytics.zone_crossing)
            crowd = CrowdThresholdDetector(cfg.analytics.crowd_threshold)
            annotator = Annotator(cfg.display)

            # IO writers
            out_dir = Path(cfg.output.output_dir)
            video_writer: VideoWriter | None = None
            event_logger: EventLogger | None = None

            if cfg.output.save_video:
                video_path = str(out_dir / f"annotated_{self._run_ts}.mp4")
                video_writer = VideoWriter(video_path, fps, frame_size, cfg.output.video_codec)

            if cfg.output.save_events:
                events_path = str(out_dir / "events" / f"event_log_{self._run_ts}.json")
                event_logger = EventLogger(events_path)

            # Rolling FPS counter (last 30 frames)
            fps_deque: deque[float] = deque(maxlen=30)
            total_alert_count = 0

            detect_every = cfg.tracker.detect_every_n_frames
            last_detections: list = []

            try:
                for frame, frame_idx, timestamp_sec in source:
                    t0 = time.perf_counter()

                    # Only run YOLO every N frames; reuse last detections between runs
                    if frame_idx % detect_every == 0:
                        last_detections = detector.detect(frame)

                    result = self._process_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        detections=last_detections,
                        tracker=tracker,
                        loitering=loitering,
                        crossing=crossing,
                        crowd=crowd,
                    )

                    total_alert_count += len(result.events)

                    # Log events
                    if event_logger:
                        for event in result.events:
                            event_logger.log(event)

                    # Compute rolling FPS
                    fps_deque.append(time.perf_counter())
                    current_fps = (
                        len(fps_deque) / (fps_deque[-1] - fps_deque[0])
                        if len(fps_deque) > 1 else 0.0
                    )

                    # Render annotations
                    annotated = annotator.render(
                        frame=result.raw_frame,
                        tracks=result.tracks,
                        events=result.events,
                        zones=cfg.analytics.loitering.zones,
                        lines=cfg.analytics.zone_crossing.lines,
                        fps=current_fps,
                        alert_count=total_alert_count,
                        frame_idx=frame_idx,
                        loitering_dwell=loitering.dwell_frames,
                        loitering_threshold_frames=loitering.threshold_frames,
                        crowd_active=crowd.is_active,
                    )

                    if video_writer:
                        video_writer.write(annotated)

                    if cfg.output.display_live:
                        cv2.imshow("Surveillance Pipeline", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[Pipeline] Quit signal received.")
                            break

            finally:
                self._teardown(video_writer, event_logger)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame,
        frame_idx: int,
        timestamp_sec: float,
        detections: list,
        tracker: TrackManager,
        loitering: LoiteringDetector,
        crossing: ZoneCrossingDetector,
        crowd: CrowdThresholdDetector,
    ) -> FrameResult:
        tracks = tracker.update(detections, frame_idx)

        events: list[AlertEvent] = []
        events.extend(loitering.process(tracks, frame_idx, timestamp_sec))
        events.extend(crossing.process(tracks, frame_idx, timestamp_sec))
        events.extend(crowd.process(tracks, frame_idx, timestamp_sec))

        if events:
            for e in events:
                print(
                    f"[{timestamp_sec:7.2f}s | f{frame_idx:05d}] "
                    f"{e.event_type:<20} track={e.track_id}  zone={e.zone_name}  "
                    f"{e.metadata}"
                )

        return FrameResult(raw_frame=frame, annotated_frame=None, tracks=tracks, events=events)

    def _teardown(
        self,
        video_writer: VideoWriter | None,
        event_logger: EventLogger | None,
    ) -> None:
        if video_writer:
            video_writer.release()
        if event_logger:
            event_logger.finalize()
        cv2.destroyAllWindows()
        print("[Pipeline] Done.")
