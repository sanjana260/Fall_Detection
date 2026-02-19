from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

BBox = Tuple[float, float, float, float]


@dataclass
class PersonTrack:
    track_id: int
    bbox: BBox
    keypoints: Optional[Any] = None


@dataclass
class HazardDetection:
    hazard_type: str
    score: float
    bbox: Optional[BBox] = None
    mask: Optional[Any] = None

    severity: str = "unknown"
    area_ratio: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonRisk:
    track_id: int
    hazards_present: List[str]
    score: float
    proximity: Dict[str, str]
    severity: Dict[str, str]
    explanation: str
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardFrameResult:
    t: float
    hazards: List[HazardDetection]
    per_person: List[PersonRisk]
    scene_alerts: Dict[str, str] = field(default_factory=dict)  # hazard_type -> severity
