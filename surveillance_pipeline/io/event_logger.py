from __future__ import annotations

import json
from pathlib import Path

from surveillance_pipeline.analytics.base import AlertEvent


class EventLogger:
    """
    Accumulates AlertEvents in memory and writes a timestamped JSON file on finalize().
    Produces a valid JSON array even when no events are fired.
    """

    def __init__(self, output_path: str) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []

    def log(self, event: AlertEvent) -> None:
        self._events.append(event.to_dict())

    def finalize(self) -> None:
        with open(self._output_path, "w") as f:
            json.dump(self._events, f, indent=2)
        print(f"[EventLogger] {len(self._events)} event(s) written → {self._output_path}")

    @property
    def count(self) -> int:
        return len(self._events)
