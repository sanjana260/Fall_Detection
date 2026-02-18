"""
Falling Object Detector
========================
Detects objects that are falling in a video, whether they were caused
by a person, and whether they hit a person — and if so, which person.

Two detection methods:
  1. YOLO — detects known objects (chairs, bottles, etc.)
  2. Background subtraction — detects ANY large moving object (trees, unknown items)

Outputs:
  - Terminal summary
  - Short video clip saved for each fall event (before + after)

Usage:
  python falling_objects.py --video path/to/video.mp4

Place yolov8n.pt in the same folder as this script.
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
OBJ_CONF_THRESHOLD    = 0.35
TRACK_IOU_THRESHOLD   = 0.3
MAX_MISSED_FRAMES     = 10

HISTORY_FRAMES        = 10
MIN_FALL_VELOCITY     = 4.0
FALL_CONFIRM_FRAMES   = 5
FALL_COOLDOWN_SEC     = 2.0

CAUSE_PROXIMITY_PX    = 120
CAUSE_LOOKBACK_FRAMES = 15
HIT_IOU_THRESHOLD     = 0.05

CLIP_BUFFER_SEC       = 1.5
BG_MIN_AREA           = 5000

# How close two person boxes need to be (in pixels) to count as the same person
PERSON_MATCH_DIST     = 80

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

def detect_large_moving_regions(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > BG_MIN_AREA:
            x, y, ww, hh = cv2.boundingRect(cnt)
            regions.append([x, y, x+ww, y+hh])
    return regions

# ── PERSON TRACKER ────────────────────────────────────────────────────────────
# Lightweight — just assigns consistent IDs to people across frames
# by matching bbox centers within PERSON_MATCH_DIST pixels.

class PersonTracker:
    def __init__(self):
        self.persons  = {}   # person_id → last known box
        self.next_id  = 0

    def update(self, boxes):
        """
        boxes: list of [x1,y1,x2,y2]
        Returns list of {"id": int, "box": [...]}
        """
        updated = {}
        unmatched = list(range(len(boxes)))

        for pid, last_box in self.persons.items():
            best_dist = PERSON_MATCH_DIST
            best_i    = None
            for i in unmatched:
                d = center_dist(last_box, boxes[i])
                if d < best_dist:
                    best_dist = d
                    best_i    = i
            if best_i is not None:
                updated[pid] = boxes[best_i]
                unmatched.remove(best_i)

        for i in unmatched:
            updated[self.next_id] = boxes[i]
            self.next_id += 1

        self.persons = updated
        return [{"id": pid, "box": box} for pid, box in self.persons.items()]


# ── OBJECT TRACKER ────────────────────────────────────────────────────────────

class Track:
    def __init__(self, track_id, label, box, frame_idx):
        self.track_id      = track_id
        self.label         = label
        self.box           = box
        self.missed        = 0
        self.active        = True
        cx, cy             = bbox_center(box)
        self.history       = [(frame_idx, cx, cy, box, aspect_ratio(box))]
        self.fall_velocity = []
        self.is_falling    = False
        self.fall_frame    = None
        self.fall_logged   = False

    def update(self, box, frame_idx):
        self.box    = box
        self.missed = 0
        cx, cy      = bbox_center(box)
        self.history.append((frame_idx, cx, cy, box, aspect_ratio(box)))
        if len(self.history) > HISTORY_FRAMES:
            self.history.pop(0)

    def get_velocity_y(self):
        if len(self.history) < 2:
            return 0.0
        velocities = []
        for i in range(1, len(self.history)):
            df = self.history[i][0] - self.history[i-1][0]
            dy = self.history[i][2] - self.history[i-1][2]
            if df > 0:
                velocities.append(dy / df)
        return float(np.mean(velocities)) if velocities else 0.0

    def aspect_ratio_changed(self):
        if len(self.history) < HISTORY_FRAMES:
            return False
        return self.history[0][4] < 0.8 and self.history[-1][4] > 1.2


class ObjectTracker:
    def __init__(self):
        self.tracks  = []
        self.next_id = 0

    def update(self, detections, frame_idx):
        unmatched_dets = list(range(len(detections)))

        for track in self.tracks:
            if not track.active:
                continue
            best_iou = TRACK_IOU_THRESHOLD
            best_det = None
            for di in unmatched_dets:
                if detections[di]["label"] != track.label:
                    continue
                score = bbox_iou(track.box, detections[di]["box"])
                if score > best_iou:
                    best_iou = score
                    best_det = di

            if best_det is not None:
                track.update(detections[best_det]["box"], frame_idx)
                unmatched_dets.remove(best_det)
            else:
                track.missed += 1
                if track.missed > MAX_MISSED_FRAMES:
                    track.active = False

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

    obj_tracker    = ObjectTracker()
    person_tracker = PersonTracker()
    bg_subtractor  = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    incidents     = []
    last_fall_sec = -999.0
    max_buffer    = int(CLIP_BUFFER_SEC * fps) * 2 + 10
    frame_buffer  = []
    active_clips  = {}
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = obj_model(frame, verbose=False)[0]

        # ── Detect people and objects ─────────────────────────────────────────
        raw_person_boxes = []
        detections       = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = obj_model.names[cls_id]
                conf   = float(box.conf[0])
                b      = box.xyxy[0].tolist()
                if label == "person":
                    if conf > 0.4:
                        raw_person_boxes.append(b)
                elif conf > OBJ_CONF_THRESHOLD:
                    if all(bbox_iou(b, pb) < 0.4 for pb in raw_person_boxes):
                        detections.append({"label": label, "box": b})

        # ── Background subtraction ────────────────────────────────────────────
        unknown_regions = detect_large_moving_regions(frame, bg_subtractor)
        for region in unknown_regions:
            already   = any(bbox_iou(region, d["box"]) > 0.3 for d in detections)
            is_person = any(bbox_iou(region, pb) > 0.3 for pb in raw_person_boxes)
            if not already and not is_person:
                detections.append({"label": "unknown_object", "box": region})

        # ── Update trackers ───────────────────────────────────────────────────
        persons       = person_tracker.update(raw_person_boxes)  # [{"id", "box"}, ...]
        active_tracks = obj_tracker.update(detections, frame_idx)

        # ── Draw ──────────────────────────────────────────────────────────────
        annotated = frame.copy()

        for p in persons:
            draw_box(annotated, p["box"], f"Person {p['id']}", (0, 200, 0), 2)

        for track in active_tracks:
            if track.is_falling:
                color = (0, 0, 220)
                label = f"{track.label} FALLING"
            elif track.label == "unknown_object":
                color = (0, 200, 255)
                label = f"unknown [{track.track_id}]"
            else:
                color = (160, 160, 160)
                label = f"{track.label} [{track.track_id}]"
            draw_box(annotated, track.box, label, color, 1)

            if len(track.history) >= 2:
                cx1,cy1 = int(track.history[-2][1]), int(track.history[-2][2])
                cx2,cy2 = int(track.history[-1][1]), int(track.history[-1][2])
                cv2.arrowedLine(annotated, (cx1,cy1), (cx2,cy2), (0,255,255), 2)

        cv2.putText(annotated, f"Frame {frame_idx}/{total}  |  Falls: {len(incidents)}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # ── Rolling frame buffer ──────────────────────────────────────────────
        frame_buffer.append((frame_idx, annotated.copy()))
        if len(frame_buffer) > max_buffer:
            frame_buffer.pop(0)

        # ── Write active clips ────────────────────────────────────────────────
        for tid, clip in list(active_clips.items()):
            clip["writer"].write(annotated)
            if frame_idx >= clip["end_frame"]:
                clip["writer"].release()
                print(f"   clip saved   : {clip['path']}")
                del active_clips[tid]

        # ── Check each track for falling ──────────────────────────────────────
        for track in active_tracks:
            if track.fall_logged:
                continue

            vy        = track.get_velocity_y()
            fall_time = round(frame_idx / fps, 2)

            if vy > MIN_FALL_VELOCITY:
                track.fall_velocity.append(vy)
            else:
                track.fall_velocity = []

            if (len(track.fall_velocity) >= FALL_CONFIRM_FRAMES
                    and not track.is_falling
                    and fall_time - last_fall_sec > FALL_COOLDOWN_SEC):

                track.is_falling  = True
                track.fall_frame  = frame_idx
                track.fall_logged = True
                last_fall_sec     = fall_time

                # Did a person cause it?
                person_caused    = False
                caused_by_person = None
                for (hf, hcx, hcy, hbox, _) in track.history:
                    if frame_idx - hf > CAUSE_LOOKBACK_FRAMES:
                        continue
                    for p in persons:
                        if center_dist(hbox, p["box"]) < CAUSE_PROXIMITY_PX:
                            person_caused    = True
                            caused_by_person = p["id"]
                            break
                    if person_caused:
                        break

                # Did it hit a person — and which one?
                hit_person    = False
                hit_person_id = None
                for p in persons:
                    if bbox_iou(track.box, p["box"]) > HIT_IOU_THRESHOLD:
                        hit_person    = True
                        hit_person_id = p["id"]
                        break

                incident = {
                    "object"                   : track.label,
                    "track_id"                 : track.track_id,
                    "fell_at_frame"            : frame_idx,
                    "fell_at_sec"              : fall_time,
                    "avg_velocity_px_per_frame": round(float(np.mean(track.fall_velocity)), 2),
                    "aspect_ratio_changed"     : track.aspect_ratio_changed(),
                    "person_caused"            : person_caused,
                    "caused_by_person_id"      : caused_by_person,
                    "hit_person"               : hit_person,
                    "hit_person_id"            : hit_person_id,
                }
                incidents.append(incident)

                print(f"\n⚠️  FALLING OBJECT: {track.label} (track {track.track_id}) @ {fall_time}s")
                print(f"   velocity     : {incident['avg_velocity_px_per_frame']} px/frame")
                print(f"   tipped over  : {incident['aspect_ratio_changed']}")
                print(f"   person caused: {person_caused}" +
                      (f" (Person {caused_by_person})" if caused_by_person is not None else ""))
                print(f"   hit person   : {hit_person}" +
                      (f" (Person {hit_person_id})" if hit_person_id is not None else ""))

                # ── Save clip ─────────────────────────────────────────────────
                clip_path = str(video_path.parent /
                                f"clip_{track.label}_t{fall_time}s.mp4")
                writer = cv2.VideoWriter(
                    clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                clip_start = frame_idx - int(CLIP_BUFFER_SEC * fps)
                for (fi, bf) in frame_buffer:
                    if fi >= clip_start:
                        writer.write(bf)
                active_clips[track.track_id] = {
                    "writer"   : writer,
                    "end_frame": frame_idx + int(CLIP_BUFFER_SEC * fps),
                    "path"     : clip_path,
                }

        cv2.imshow("Falling Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Close any remaining clip writers
    for tid, clip in active_clips.items():
        clip["writer"].release()
        print(f"   clip saved   : {clip['path']}")

    cap.release()
    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    print(f"\nDone. {len(incidents)} falling object(s) detected.")
    print("\n" + "="*60)
    for inc in incidents:
        print(f"\n{inc['object'].upper()} fell @ {inc['fell_at_sec']}s")
        print(f"  Person caused it : {inc['person_caused']}" +
              (f" (Person {inc['caused_by_person_id']})" if inc['caused_by_person_id'] is not None else ""))
        print(f"  Hit person       : {inc['hit_person']}" +
              (f" (Person {inc['hit_person_id']})" if inc['hit_person_id'] is not None else ""))
        print(f"  Tipped over      : {inc['aspect_ratio_changed']}")

    print("\nFull JSON:")
    print(json.dumps(incidents, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    run(args.video)
