from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from surveillance_pipeline.config import ModelConfig


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


class Detector:
    """
    Thin wrapper around YOLOv8 inference.
    Runs model.predict() (not model.track()) — tracking is handled separately
    by TrackManager, which keeps the two concerns decoupled.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = YOLO(config.weights_path)
        # Build a set of target class names for fast membership testing
        self._target_classes: set[str] = set(config.target_classes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame and return filtered detections."""
        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.inference_imgsz,
            verbose=False,
        )

        detections: list[Detection] = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]

            if class_name not in self._target_classes:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            confidence = float(box.conf[0])

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
            ))

        return detections
