import cv2
import numpy as np

from hazard_module import HazardModule
from hazard.types import PersonTrack
from hazard.water_detector import WaterConfig


VIDEO_PATH = "data/inwater.mp4"   # change if needed
OUT_PATH = "hazards_annotated.mp4"


def mask_to_bbox(mask: np.ndarray):
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35):
    if mask is None:
        return frame_bgr
    overlay = frame_bgr.copy()
    m = (mask > 0)
    if not m.any():
        return frame_bgr
    color = np.zeros_like(frame_bgr, dtype=np.uint8)
    color[m] = (255, 0, 0)  # blue tint in BGR
    out = cv2.addWeighted(overlay, 1.0, color, alpha, 0)
    return out


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outv = cv2.VideoWriter(
        OUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    hz = HazardModule(
        water_cfg=WaterConfig(
            mode="heuristic",     # "auto" will try GDINO+SAM then fallback, but your GDINO class currently raises
            device="cpu",
            min_area=500,
        )
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps

        # Placeholder person until you plug in your tracker
        people = [PersonTrack(track_id=1, bbox=(w * 0.35, h * 0.2, w * 0.65, h * 0.95))]

        result = hz.process_frame(frame, people, t=t, fall_events=None)

        vis = frame.copy()

        for det in result.hazards:
            if (det.hazard_type or "").lower() != "water":
                continue

            mask = det.mask
            vis = overlay_mask(vis, mask, alpha=0.35)

            bbox = det.bbox
            if bbox is None and mask is not None:
                bbox = mask_to_bbox(mask)

            sev = getattr(det, "severity", "unknown")
            ar = getattr(det, "area_ratio", None)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if ar is None:
                    label = f"water sev={sev}"
                else:
                    label = f"water sev={sev} ar={float(ar):.4f}"

                cv2.putText(
                    vis,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

        # Optional: show scene alert text
        if "water" in result.scene_alerts:
            cv2.putText(
                vis,
                f"SCENE ALERT: WATER ({result.scene_alerts['water']})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        outv.write(vis)
        frame_idx += 1

    cap.release()
    outv.release()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
