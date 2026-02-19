from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np

def bbox_height(bbox) -> float:
    return max(1.0, float(bbox[3] - bbox[1]))

def feet_point_from_bbox(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)

def feet_point_from_keypoints(keypoints: Any) -> Optional[Tuple[float, float]]:
    """
    Supports dict-style keypoints:
      {"left_ankle": (x,y,conf), "right_ankle": (x,y,conf)}
    Accepts (x,y) too.
    """
    if keypoints is None:
        return None
    if isinstance(keypoints, dict):
        la = keypoints.get("left_ankle")
        ra = keypoints.get("right_ankle")
        if la is None or ra is None:
            return None
        lax, lay = la[0], la[1]
        rax, ray = ra[0], ra[1]
        return ((float(lax) + float(rax)) / 2.0, (float(lay) + float(ray)) / 2.0)
    return None

def distance_point_to_bbox(pt, bbox) -> float:
    x, y = pt
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)

def mask_distance_to_point(mask: np.ndarray, pt) -> Optional[float]:
    if mask is None:
        return None
    mask_bin = (mask > 0) if mask.dtype != np.bool_ else mask
    if not mask_bin.any():
        return None
    ys, xs = np.where(mask_bin)
    x, y = pt
    n = xs.shape[0]
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    return float(np.sqrt(dx * dx + dy * dy).min())

def proximity_label(d_norm: Optional[float]) -> str:
    if d_norm is None:
        return "unknown"
    if d_norm < 0.20:
        return "very_close"
    if d_norm < 0.50:
        return "close"
    if d_norm < 1.00:
        return "medium"
    return "far"

def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

def mask_area(mask: np.ndarray) -> float:
    if mask is None:
        return 0.0
    return float((mask > 0).sum())

def severity_from_area_ratio(r: Optional[float]) -> str:
    if r is None:
        return "unknown"
    if r < 0.002:
        return "low"
    if r < 0.01:
        return "medium"
    if r < 0.04:
        return "high"
    return "extreme"

def severity_boost_by_proximity(base_sev: str, proximity: str) -> str:
    order = ["low", "medium", "high", "extreme"]
    if base_sev not in order:
        return base_sev

    boost = 0
    if proximity == "very_close":
        boost = 2
    elif proximity == "close":
        boost = 1

    idx = min(len(order) - 1, order.index(base_sev) + boost)
    return order[idx]
