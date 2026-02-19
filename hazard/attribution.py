from __future__ import annotations
from typing import List, Dict, Optional, Any
import numpy as np

from .types import PersonTrack, HazardDetection, PersonRisk
from .utils import (
    feet_point_from_keypoints,
    feet_point_from_bbox,
    bbox_height,
    distance_point_to_bbox,
    mask_distance_to_point,
    proximity_label,
    severity_boost_by_proximity,
)


class HazardAttributor:
    """
    Produces per-person hazard proximity + explanation + impact severity.

    Camera-agnostic:
      - proximity uses normalized pixel distance by person's bbox height
      - severity is derived from hazard visible area in frame, then boosted by proximity
    """

    def __init__(self, water_trust_requires_near_feet: bool = True):
        # Water is noisy, so only "present" when it is close to feet by default.
        self.water_trust_requires_near_feet = water_trust_requires_near_feet

    def attribute(
        self,
        hazards: List[HazardDetection],
        people: List[PersonTrack],
        fall_events: Optional[Dict[int, bool]] = None,
    ) -> List[PersonRisk]:

        fall_events = fall_events or {}
        per_person: List[PersonRisk] = []

        for p in people:
            feet = feet_point_from_keypoints(p.keypoints) or feet_point_from_bbox(p.bbox)
            h_p = bbox_height(p.bbox)

            hazard_labels: Dict[str, str] = {}          # hazard_type -> proximity label
            hazard_scores: Dict[str, float] = {}        # hazard_type -> numeric contribution
            impact_severity: Dict[str, str] = {}        # hazard_type -> severity label (boosted)
            present: List[str] = []

            debug: Dict[str, Any] = {"feet": feet, "bbox_h": h_p}

            for hz in hazards:
                d_norm: Optional[float] = None

                # Compute distance from hazard to feet
                if hz.mask is not None:
                    d_px = mask_distance_to_point(hz.mask, feet)
                    d_norm = None if d_px is None else (d_px / h_p)
                elif hz.bbox is not None:
                    d_px = distance_point_to_bbox(feet, hz.bbox)
                    d_norm = d_px / h_p
                else:
                    d_norm = None

                label = proximity_label(d_norm)
                hazard_labels[hz.hazard_type] = self._closest_label(
                    hazard_labels.get(hz.hazard_type), label
                )

                # Compute severity impact label (base severity boosted by proximity)
                base_sev = getattr(hz, "severity", "unknown")
                impact_severity[hz.hazard_type] = severity_boost_by_proximity(base_sev, label)

                # Score shaping: closer hazard increases contribution
                base = float(hz.score)
                if d_norm is None:
                    contrib = 0.2 * base
                else:
                    # closer => higher; decay with distance
                    contrib = base * float(np.exp(-2.0 * d_norm))

                hazard_scores[hz.hazard_type] = max(hazard_scores.get(hz.hazard_type, 0.0), contrib)

            # Decide which hazards are "present" for this person
            for htype, sc in hazard_scores.items():
                if htype == "water" and self.water_trust_requires_near_feet:
                    # only claim water is present if close/very_close
                    if hazard_labels.get("water") in ("very_close", "close"):
                        present.append("water")
                else:
                    # for fire/smoke, allow presence even if not super close, as long as score is decent
                    if sc >= 0.15:
                        present.append(htype)

            fall_now = bool(fall_events.get(p.track_id, False))

            # Compose explanation
            parts = []
            if fall_now:
                parts.append("fall/slip event flagged")

            if "water" in present:
                prox = hazard_labels.get("water", "unknown")
                sev = impact_severity.get("water", "unknown")
                parts.append(f"wet surface {prox} to feet (slip risk), severity={sev}")

            if "fire" in present:
                prox = hazard_labels.get("fire", "unknown")
                sev = impact_severity.get("fire", "unknown")
                parts.append(f"fire {prox} to person, severity={sev}")

            if "smoke" in present:
                prox = hazard_labels.get("smoke", "unknown")
                sev = impact_severity.get("smoke", "unknown")
                parts.append(f"smoke {prox} near person, severity={sev}")

            explanation = "; ".join(parts) if parts else "no strong hazard evidence"

            overall = float(min(1.0, sum(hazard_scores.values())))

            per_person.append(
                PersonRisk(
                    track_id=p.track_id,
                    hazards_present=present,
                    score=overall,
                    proximity=hazard_labels,
                    severity=impact_severity,
                    explanation=explanation,
                    debug=debug,
                )
            )

        return per_person

    @staticmethod
    def _closest_label(existing: Optional[str], new_label: str) -> str:
        """
        Choose the "closest" of two proximity labels.
        Order: very_close < close < medium < far < unknown
        """
        order = {"very_close": 0, "close": 1, "medium": 2, "far": 3, "unknown": 4}
        if existing is None:
            return new_label
        return existing if order.get(existing, 4) <= order.get(new_label, 4) else new_label
