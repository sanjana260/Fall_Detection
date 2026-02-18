"""
Falling Object Detector
========================
Detects objects that are falling in a video, whether they were caused
by a person, and whether they hit a person.

Logic:
  1. Detect all non-person objects every frame using YOLOv8
  2. Track each object across frames using a simple IoU-based tracker
  3. For each tracked object, maintain a history of bbox center positions
  4. If the object's Y position drops rapidly over N frames → it is falling
  5. If a person was near the object just before it started falling → person caused it
  6. If the falling object's bbox overlaps the person bbox → it hit the person

Usage:
  python falling_objects.py --video path/to/video.mp4

Place yolov8n.pt in the same folder as this script.
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH = "fall_detection_output.mp4"   # default, overridden by --video

OBJ_CONF_THRESHOLD    = 0.35   # min YOLO confidence to count an object
TRACK_IOU_THRESHOLD   = 0.3    # IoU to match object to existing track
MAX_MISSED_FRAMES     = 10     # frames before a track is dropped

# Falling detection
HISTORY_FRAMES        = 10     # how many frames of position history to keep
MIN_FALL_VELOCITY     = 4.0    # min pixels/frame downward to count as falling
FALL_CONFIRM_FRAMES   = 5      # must be falling for this many frames to confirm

# Cause / hit detection
CAUSE_PROXIMITY_PX    = 120    # person within this distance just before fall → caused it
CAUSE_LOOKBACK_FRAMES = 15     # how many frames before fall to check for person proximity
HIT_IOU_THRESHOLD     = 0.05   # object bbox overlaps person bbox → hit

# Risk table (same as experiment.py)


# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────

def bbox_iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def bbox_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def center_dist(a, b):
    ca, cb = bbox_center(a), bbox_center(b)
    return np.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)

def aspect_ratio(box):
    w = box[2]-box[0]
    h = box[3]-box[1]
    return w/h if h > 0 else 1.0

# ── SIMPLE IoU TRACKER ────────────────────────────────────────────────────────
# No external dependency — matches detections to existing tracks by IoU.

class Track:
    def __init__(self, track_id, label, box, frame_idx):
        self.track_id      = track_id
        self.label         = label
        self.box           = box
        self.missed        = 0
        self.active        = True

        # Position history: list of (frame_idx, cx, cy, box, aspect_ratio)
        cx, cy = bbox_center(box)
        self.history       = [(frame_idx, cx, cy, box, aspect_ratio(box))]

        # Fall state
        self.fall_velocity = []   # recent per-frame Y velocities
        self.is_falling    = False
        self.fall_frame    = None
        self.fall_logged   = False

    def update(self, box, frame_idx):
        self.box     = box
        self.missed  = 0
        cx, cy       = bbox_center(box)
        self.history.append((frame_idx, cx, cy, box, aspect_ratio(box)))
        if len(self.history) > HISTORY_FRAMES:
            self.history.pop(0)

    def get_velocity_y(self):
        """Average downward velocity (pixels/frame) over recent history."""
        if len(self.history) < 2:
            return 0.0
        velocities = []
        for i in range(1, len(self.history)):
            df = self.history[i][0] - self.history[i-1][0]
            dy = self.history[i][2] - self.history[i-1][2]  # cy diff
            if df > 0:
                velocities.append(dy / df)
        return float(np.mean(velocities)) if velocities else 0.0

    def aspect_ratio_changed(self):
        """Returns True if object went from tall to wide (tipping over)."""
        if len(self.history) < HISTORY_FRAMES:
            return False
        first_ar = self.history[0][4]
        last_ar  = self.history[-1][4]
        return first_ar < 0.8 and last_ar > 1.2  # was tall, now wide


class Tracker:
    def __init__(self):
        self.tracks   = []
        self.next_id  = 0

    def update(self, detections, frame_idx):
        """
        detections: list of {"label": str, "box": [x1,y1,x2,y2]}
        Returns list of active Track objects.
        """
        # Match detections to existing tracks by IoU
        unmatched_dets   = list(range(len(detections)))
        matched_track_ids = set()

        for track in self.tracks:
            if not track.active:
                continue
            best_iou  = TRACK_IOU_THRESHOLD
            best_det  = None
            for di in unmatched_dets:
                if detections[di]["label"] != track.label:
                    continue
                score = bbox_iou(track.box, detections[di]["box"])
                if score > best_iou:
                    best_iou = score
                    best_det = di

            if best_det is not None:
                track.update(detections[best_det]["box"], frame_idx)
                matched_track_ids.add(track.track_id)
                unmatched_dets.remove(best_det)
            else:
                track.missed += 1
                if track.missed > MAX_MISSED_FRAMES:
                    track.active = False

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            t = Track(self.next_id, detections[di]["label"],
                      detections[di]["box"], frame_idx)
            self.tracks.append(t)
            self.next_id += 1

        return [t for t in self.tracks if t.active]


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_box(frame, box, label, color, thickness=2):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1,y1),(x2,y2), color, thickness)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1,y1-th-6),(x1+tw+6,y1), color, -1)
    cv2.putText(frame, label, (x1+3,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(video_path: str):
    video_path = Path(video_path)
    script_dir = Path(__file__).parent

    print("Loading model...")
    obj_model = YOLO(script_dir / "yolov8n.pt")
    print("Model loaded.\n")

    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path.name}  |  {total} frames @ {fps:.1f}fps\n")

    tracker  = Tracker()
    incidents = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = obj_model(frame, verbose=False)[0]

        # Separate people and objects
        person_boxes = []
        detections   = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = obj_model.names[cls_id]
                conf   = float(box.conf[0])
                b      = box.xyxy[0].tolist()

                if label == "person":
                    if conf > 0.4:
                        person_boxes.append(b)
                elif conf > OBJ_CONF_THRESHOLD:
                    # Filter out objects that heavily overlap a person
                    if all(bbox_iou(b, pb) < 0.4 for pb in person_boxes):
                        detections.append({"label": label, "box": b})

        # Update tracker
        active_tracks = tracker.update(detections, frame_idx)

        # ── Check each track for falling ─────────────────────────────────────
        for track in active_tracks:
            if track.fall_logged:
                continue

            vy = track.get_velocity_y()

            # Count falling frames
            if vy > MIN_FALL_VELOCITY:
                track.fall_velocity.append(vy)
            else:
                track.fall_velocity = []

            # Confirm fall
            if len(track.fall_velocity) >= FALL_CONFIRM_FRAMES and not track.is_falling:
                track.is_falling  = True
                track.fall_frame  = frame_idx
                track.fall_logged = True
                fall_time         = round(frame_idx / fps, 2)

                # ── Did a person cause it? ────────────────────────────────
                # Look back in track history for person proximity
                person_caused = False
                for (hf, hcx, hcy, hbox, _) in track.history:
                    if frame_idx - hf > CAUSE_LOOKBACK_FRAMES:
                        continue
                    for pb in person_boxes:
                        if center_dist(hbox, pb) < CAUSE_PROXIMITY_PX:
                            person_caused = True
                            break

                # ── Did it hit a person? ──────────────────────────────────
                hit_person = any(
                    bbox_iou(track.box, pb) > HIT_IOU_THRESHOLD
                    for pb in person_boxes
                )

                incident = {
                    "object"        : track.label,
                    "track_id"      : track.track_id,
                    "fell_at_frame" : frame_idx,
                    "fell_at_sec"   : fall_time,
                    "avg_velocity_px_per_frame": round(float(np.mean(track.fall_velocity)), 2),
                    "aspect_ratio_changed": track.aspect_ratio_changed(),
                    "person_caused" : person_caused,
                    "hit_person"    : hit_person,
                }
                incidents.append(incident)

                print(f"⚠️  FALLING OBJECT: {track.label} (track {track.track_id}) @ {fall_time}s")
                print(f"   velocity     : {incident['avg_velocity_px_per_frame']} px/frame")
                print(f"   tipped over  : {incident['aspect_ratio_changed']}")
                print(f"   person caused: {person_caused}")
                print(f"   hit person   : {hit_person}\n")

        # ── Draw ─────────────────────────────────────────────────────────────
        for pb in person_boxes:
            draw_box(frame, pb, "person", (0, 200, 0), 2)

        for track in active_tracks:
            if track.is_falling:
                color = (0, 0, 220)
                label = f"{track.label} FALLING"
            else:
                color = (160, 160, 160)
                label = f"{track.label} [{track.track_id}]"
            draw_box(frame, track.box, label, color, 1)

            # Draw velocity arrow
            if len(track.history) >= 2:
                cx1, cy1 = int(track.history[-2][1]), int(track.history[-2][2])
                cx2, cy2 = int(track.history[-1][1]), int(track.history[-1][2])
                cv2.arrowedLine(frame, (cx1,cy1), (cx2,cy2), (0,255,255), 2)

        cv2.putText(frame, f"Frame {frame_idx}/{total}  |  Falling objects: {len(incidents)}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Falling Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    print(f"\nDone. {len(incidents)} falling object(s) detected.")
    print("\n" + "="*60)
    for inc in incidents:
        print(f"\n{inc['object'].upper()} fell @ {inc['fell_at_sec']}s")
        print(f"  Person caused it : {inc['person_caused']}")
        print(f"  Hit person       : {inc['hit_person']}")
        print(f"  Tipped over      : {inc['aspect_ratio_changed']}")

    print("\nFull JSON:")
    print(json.dumps(incidents, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    run(args.video)
