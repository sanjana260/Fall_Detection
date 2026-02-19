from __future__ import annotations
from typing import List, Any
from dataclasses import dataclass
import numpy as np

from .types import HazardDetection


@dataclass
class WaterConfig:
    mode: str = "auto"  # "auto" | "gdino_sam" | "heuristic"
    device: str = "cuda"
    prompts: List[str] = None
    score_thresh: float = 0.25
    min_area: int = 500


class GroundedDinoSamWaterDetector:
    """
    Optional: requires GroundingDINO + SAM packages available in your environment.
    This is the most flexible for "puddle/wet floor/water spill" across random videos.

    NOTE:
    This is scaffolded. You must wire:
      - GroundingDINO config path
      - GroundingDINO weights path
      - SAM weights path
    """
    def __init__(self, cfg: WaterConfig):
        self.cfg = cfg
        if cfg.prompts is None:
            cfg.prompts = ["puddle", "water spill", "wet floor"]

        # Lazy imports so repo runs without these deps
        try:
            from groundingdino.util.inference import Model  # type: ignore
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "GroundingDINO/SAM not installed. Use heuristic mode or install those deps."
            ) from e

        self._GDINO_Model = Model
        self._sam_model_registry = sam_model_registry
        self._SamPredictor = SamPredictor

        # TODO: wire these paths for your environment
        self.gdino_config_path = None
        self.gdino_weights_path = None
        self.sam_type = "vit_h"
        self.sam_weights_path = None

        raise RuntimeError(
            "GroundedDinoSamWaterDetector is scaffolded but needs weight/config paths wired. "
            "For now, use mode='heuristic'. If you want this fully working, tell me your setup "
            "(pip/conda, CUDA, where you want to store weights) and I’ll fill it in."
        )

    def detect(self, frame_bgr: np.ndarray) -> List[HazardDetection]:
        # This will never run until you wire the paths and remove the RuntimeError above.
        return []


class SimpleWetFloorHeuristicDetector:
    """
    Always runs: estimates 'wet/shiny region' on floor-like area using simple image cues.

    Important:
    - Not perfect. This is a "wet surface candidate" mask, not guaranteed puddle truth.
    - Works best when combined with attribution "near feet" logic.
    """
    def __init__(self, cfg: WaterConfig):
        self.cfg = cfg
        if cfg.prompts is None:
            cfg.prompts = ["wet floor"]

    def detect(self, frame_bgr: np.ndarray) -> List[HazardDetection]:
        import cv2  # local import to keep dependencies flexible

        h, w = frame_bgr.shape[:2]

        # Focus on bottom half (more likely ground)
        y0 = int(0.45 * h)
        roi = frame_bgr[y0:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Shiny/wet often has strong highlights: high intensity + local smoothing
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        high = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)[1]

        # Close holes / connect blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)

        mask_roi = np.zeros_like(high, dtype=np.uint8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= int(self.cfg.min_area):
                mask_roi[labels == i] = 255

        if mask_roi.sum() == 0:
            return []

        # Map ROI mask back to full frame
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:, :] = mask_roi

        # NEW: compute area_ratio + severity
        from .utils import mask_area, severity_from_area_ratio

        ar = mask_area(full_mask) / float(h * w)
        sev = severity_from_area_ratio(ar)

        return [
            HazardDetection(
                hazard_type="water",
                score=0.35,            # heuristic confidence
                bbox=None,
                mask=full_mask,
                severity=sev,          # NEW
                area_ratio=float(ar),  # NEW
                meta={"mode": "heuristic"},
            )
        ]


class WaterDetector:
    def __init__(self, cfg: WaterConfig):
        self.cfg = cfg
        self.impl: Any = None

        mode = cfg.mode
        if mode == "auto":
            # try open-vocab first, fallback to heuristic
            try:
                self.impl = GroundedDinoSamWaterDetector(cfg)
                self.mode = "gdino_sam"
            except Exception:
                self.impl = SimpleWetFloorHeuristicDetector(cfg)
                self.mode = "heuristic"
        elif mode == "gdino_sam":
            self.impl = GroundedDinoSamWaterDetector(cfg)
            self.mode = "gdino_sam"
        elif mode == "heuristic":
            self.impl = SimpleWetFloorHeuristicDetector(cfg)
            self.mode = "heuristic"
        else:
            raise ValueError(f"Unknown water detector mode: {mode}")

    def detect(self, frame_bgr: np.ndarray) -> List[HazardDetection]:
        import cv2

        h, w = frame_bgr.shape[:2]

        # Focus on bottom ~65% (ground/water region)
        y0 = int(0.35 * h)
        roi = frame_bgr[y0:, :]

        # --- 1) Non-vegetation mask (water should be low "excess green") ---
        b, g, r = cv2.split(roi)
        exg = (2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32))  # can be negative

        # threshold: vegetation has high exg; water/mud usually low
        # tune 15..40 depending on video
        non_veg = (exg < 25.0).astype(np.uint8) * 255

        # --- 2) Smoothness / low-texture mask (water often smoother than grass/rocks) ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)

        # blur lap to get "texture energy" map
        tex = cv2.GaussianBlur(lap_abs, (9, 9), 0)

        # threshold: low texture => candidate water-ish surface
        # tune 6..20 depending on resolution
        smooth = (tex < 10.0).astype(np.uint8) * 255

        # --- 3) Optional: low saturation helps separate water from many materials ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.uint8)

        # water often lower saturation than vegetation
        low_sat = cv2.inRange(sat, 0, 120)  # tune 80..140

        # --- Combine masks ---
        cand = cv2.bitwise_and(non_veg, smooth)
        cand = cv2.bitwise_and(cand, low_sat)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        mask_roi = np.zeros_like(cand, dtype=np.uint8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= int(self.cfg.min_area):
                mask_roi[labels == i] = 255

        if mask_roi.sum() == 0:
            return []

        # Map back to full frame
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:, :] = mask_roi

        from .utils import mask_area, severity_from_area_ratio

        ar = mask_area(full_mask) / float(h * w)
        sev = severity_from_area_ratio(ar)

        return [
            HazardDetection(
                hazard_type="water",
                score=0.45,            # heuristic confidence
                bbox=None,
                mask=full_mask,
                severity=sev,
                area_ratio=float(ar),
                meta={"mode": "heuristic_exg_smooth"},
            )
        ]

