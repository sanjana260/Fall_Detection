from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np

from hazard.types import PersonTrack, HazardFrameResult
from hazard.attribution import HazardAttributor
from hazard.water_detector import WaterDetector, WaterConfig


def _severity_rank(sev: str) -> int:
    order = {"unknown": -1, "low": 0, "medium": 1, "high": 2, "extreme": 3}
    return order.get((sev or "unknown").lower(), -1)


class HazardModule:
    """
    Water-only hazard module.
    - Runs WaterDetector (heuristic by default)
    - Attributes hazards to people via HazardAttributor
    - Creates scene_alerts for water based on max severity seen in frame
    """

    def __init__(self, water_cfg: Optional[WaterConfig] = None):
        self.attributor = HazardAttributor()
        self.water = WaterDetector(water_cfg or WaterConfig(mode="heuristic", device="cpu"))

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        people: List[PersonTrack],
        t: float,
        fall_events: Optional[Dict[int, bool]] = None,
    ) -> HazardFrameResult:

        hazards = []
        hazards.extend(self.water.detect(frame_bgr))

        per_person = self.attributor.attribute(hazards, people, fall_events=fall_events)

        scene_alerts: Dict[str, str] = {}
        if hazards:
            best = max(hazards, key=lambda h: _severity_rank(getattr(h, "severity", "unknown")))
            scene_alerts["water"] = getattr(best, "severity", "unknown")
            ar = getattr(best, "area_ratio", None)
            if ar is not None:
                scene_alerts["water_area_ratio"] = f"{float(ar):.4f}"

        return HazardFrameResult(
            t=t,
            hazards=hazards,
            per_person=per_person,
            scene_alerts=scene_alerts,
        )
