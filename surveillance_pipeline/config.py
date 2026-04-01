from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class SourceConfig(BaseModel):
    input_path: str = "assets/sample_scene.mp4"
    loop_video: bool = False


class ModelConfig(BaseModel):
    weights_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.5
    inference_imgsz: int = 1280   # YOLO inference resolution — lower = faster, higher = better small-object detection
    target_classes: list[str] = Field(
        default=["person", "car", "truck", "motorcycle", "bus"]
    )


class TrackerConfig(BaseModel):
    track_buffer: int = 30       # frames before a lost track is dropped
    min_box_area: float = 400.0  # px² — filters out tiny false positives
    detect_every_n_frames: int = 2  # run YOLO every N frames; tracker interpolates between


class ZoneDefinition(BaseModel):
    name: str
    polygon: list[list[int]]  # [[x, y], ...]


class LineDefinition(BaseModel):
    name: str
    points: list[list[int]]               # [[x1, y1], [x2, y2]]
    direction: Literal["any", "AB", "BA"] = "any"


class LoiteringConfig(BaseModel):
    enabled: bool = True
    threshold_seconds: float = 7.0
    zones: list[ZoneDefinition] = Field(default_factory=list)


class ZoneCrossingConfig(BaseModel):
    enabled: bool = True
    lines: list[LineDefinition] = Field(default_factory=list)


class CrowdThresholdConfig(BaseModel):
    enabled: bool = True
    max_objects: int = 8
    hysteresis_band: int = 2


class AnalyticsConfig(BaseModel):
    loitering: LoiteringConfig = Field(default_factory=LoiteringConfig)
    zone_crossing: ZoneCrossingConfig = Field(default_factory=ZoneCrossingConfig)
    crowd_threshold: CrowdThresholdConfig = Field(default_factory=CrowdThresholdConfig)


class OutputConfig(BaseModel):
    output_dir: str = "output"
    save_video: bool = True
    video_codec: str = "mp4v"
    save_events: bool = True
    display_live: bool = True


class DisplayConfig(BaseModel):
    draw_tracks: bool = True
    draw_zones: bool = True
    draw_alerts: bool = True
    show_hud: bool = True
    class_colors: dict[str, list[int]] = Field(default_factory=lambda: {
        "person":     [0, 255, 0],
        "car":        [255, 128, 0],
        "truck":      [0, 128, 255],
        "motorcycle": [255, 0, 255],
        "bus":        [0, 255, 255],
    })


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load config from a YAML file and apply any CLI overrides on top."""
    data: dict[str, Any] = {}

    if path is not None:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

    # Apply flat CLI overrides (e.g. {"loitering_threshold": 10})
    if overrides:
        _apply_overrides(data, overrides)

    return PipelineConfig(**data)


def _apply_overrides(data: dict, overrides: dict) -> None:
    """Merge CLI override values into the raw config dict before Pydantic parses it."""
    mapping = {
        "input":               ("source", "input_path"),
        "output_dir":          ("output", "output_dir"),
        "confidence":          ("model", "confidence_threshold"),
        "loitering_threshold": ("analytics", "loitering", "threshold_seconds"),
        "crowd_threshold":     ("analytics", "crowd_threshold", "max_objects"),
        "no_display":          ("output", "display_live"),
        "no_save":             ("output", "save_video"),
    }
    for key, value in overrides.items():
        if key not in mapping or value is None:
            continue
        keys = mapping[key]
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # --no-display flag inverts the bool
        if key in ("no_display", "no_save"):
            node[keys[-1]] = not value
        else:
            node[keys[-1]] = value
