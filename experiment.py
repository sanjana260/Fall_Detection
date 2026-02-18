"""
Object Interaction Experiment
==============================
Convert from experiment.ipynb — run directly with python3.
Change VIDEO_PATH at the top to test different videos.
Nothing is saved to disk.

Changes from notebook:
  - show_frame() replaced with cv2.imshow() + waitKey
  - sitting detection now uses POSE KEYPOINTS (hip/knee angle)
    instead of bounding box aspect ratio — fixes false "horizontal"
    classification of sitting people
  - sitting_log now captured EVERY frame regardless of bbox shape,
    gated by pose-based upright check instead
  - duplicate fall (Person 0 + Person 1 same frame) suppressed
    by only firing once per fall_frame
"""

import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

# ── CHANGE THIS ───────────────────────────────────────────────────────────────
VIDEO_PATH = "/Users/narayanipemmaraju/Documents/Code/DELL_NVDIA_hackathon/Fall_Detection/data/fall-08-cam0_fall_from_chair.mp4"
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG ────────────────────────────────────────────────────────────────────
FALL_ASPECT_THRESHOLD    = 0.75
FALL_CONFIRM_FRAMES      = 8
RECOVER_FRAMES           = 30
TOUCH_IOU_THRESHOLD      = 0.05
PROXIMITY_PX             = 100
PRE_FALL_BUFFER_SEC      = 5.0
OBJ_CONF_THRESHOLD       = 0.50
SITTING_HORIZONTAL_TOL   = 0.6
TOUCH_MARGIN_PX          = 25   # px: keypoint within this distance of obj box = touching

# Pose keypoint indices (COCO format)
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16
KP_CONF_THRESHOLD = 0.3   # min keypoint confidence to use it

FLOOR_TOLERANCE_PX  = 30   # ky >= floor_y - tolerance → keypoint is at floor level
FLAT_FALL_MIN_PARTS = 4    # this many parts at floor simultaneously → flat fall
# Feet/ankles are always near floor when standing — exclude them so we detect
# the first *interesting* impact: head, knee, hip, palm, shoulder, etc.
FLOOR_EXCLUDED_PARTS = {"left_ankle", "right_ankle"}

# Falling object detection
OBJ_TRACK_MAX_DIST   = 100  # px: max centroid distance to match same object across frames
OBJ_HISTORY_FRAMES   = 20   # frames of bbox-center history per tracked object
FALLING_OBJ_VEL_PX   = 5    # min downward cy velocity (px/frame) to qualify as falling
FALLING_OBJ_CONFIRM  = 4    # consecutive frames needed to confirm object fall

# Names for all 17 COCO keypoints
KP_NAMES = {
    0:  "nose",
    1:  "left_eye",      2:  "right_eye",
    3:  "left_ear",      4:  "right_ear",
    5:  "left_shoulder", 6:  "right_shoulder",
    7:  "left_elbow",    8:  "right_elbow",
    9:  "left_wrist",    10: "right_wrist",
    11: "left_hip",      12: "right_hip",
    13: "left_knee",     14: "right_knee",
    15: "left_ankle",    16: "right_ankle",
}

HIGH_RISK   = {"chair", "dining table", "bench", "bed", "sofa", "couch",
               "toilet", "sink", "oven", "refrigerator", "scissors", "knife", "fork"}
MEDIUM_RISK = {"bottle", "cup", "vase", "potted plant", "tv", "laptop",
               "suitcase", "backpack", "remote", "keyboard", "microwave"}
LOW_RISK    = {"book", "cell phone", "mouse", "clock", "pillow", "teddy bear"}

def get_risk(label):
    if label in HIGH_RISK:   return "HIGH"
    if label in MEDIUM_RISK: return "MEDIUM"
    if label in LOW_RISK:    return "LOW"
    return "UNKNOWN"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def center_dist(a, b):
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return np.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)

def is_horizontal(box):
    w, h = box[2]-box[0], box[3]-box[1]
    return (w/h > FALL_ASPECT_THRESHOLD) if h > 0 else False

def get_kp(keypoints, idx):
    """Returns (x, y, conf) for a keypoint, or None if not confident enough."""
    if keypoints is None or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    # kp is [x, y] from .xy, conf from .conf
    return kp if kp[0] > 0 and kp[1] > 0 else None

def is_sitting_pose(keypoints_xy, keypoints_conf):
    """
    Uses hip and knee keypoints to detect sitting.
    When sitting: hips and knees are at roughly the same HEIGHT (y value),
    and knees are BELOW hips only slightly — unlike standing where knees
    are well below hips.

    Returns True if person appears to be sitting.
    """
    try:
        # Get hip and knee positions
        lh_conf = keypoints_conf[KP_LEFT_HIP]
        rh_conf = keypoints_conf[KP_RIGHT_HIP]
        lk_conf = keypoints_conf[KP_LEFT_KNEE]
        rk_conf = keypoints_conf[KP_RIGHT_KNEE]
        ls_conf = keypoints_conf[KP_LEFT_SHOULDER]
        rs_conf = keypoints_conf[KP_RIGHT_SHOULDER]

        # Need at least hips + one knee visible
        if max(lh_conf, rh_conf) < KP_CONF_THRESHOLD:
            return False
        if max(lk_conf, rk_conf) < KP_CONF_THRESHOLD:
            return False

        # Average hip y and knee y
        hip_ys  = []
        knee_ys = []
        shoulder_ys = []

        if lh_conf > KP_CONF_THRESHOLD: hip_ys.append(keypoints_xy[KP_LEFT_HIP][1])
        if rh_conf > KP_CONF_THRESHOLD: hip_ys.append(keypoints_xy[KP_RIGHT_HIP][1])
        if lk_conf > KP_CONF_THRESHOLD: knee_ys.append(keypoints_xy[KP_LEFT_KNEE][1])
        if rk_conf > KP_CONF_THRESHOLD: knee_ys.append(keypoints_xy[KP_RIGHT_KNEE][1])
        if ls_conf > KP_CONF_THRESHOLD: shoulder_ys.append(keypoints_xy[KP_LEFT_SHOULDER][1])
        if rs_conf > KP_CONF_THRESHOLD: shoulder_ys.append(keypoints_xy[KP_RIGHT_SHOULDER][1])

        avg_hip_y  = np.mean(hip_ys)
        avg_knee_y = np.mean(knee_ys)

        # In image coords, y increases downward
        # Standing: knee_y >> hip_y (knees much lower than hips)
        # Sitting:  knee_y ≈ hip_y  (knees roughly same height as hips)
        # Lying:    knee_y ≈ hip_y  but torso is also horizontal

        knee_hip_diff = avg_knee_y - avg_hip_y  # small when sitting

        # Also check torso is upright using shoulders vs hips
        torso_upright = True
        if shoulder_ys:
            avg_shoulder_y = np.mean(shoulder_ys)
            torso_vertical = avg_hip_y - avg_shoulder_y  # positive = shoulders above hips
            torso_upright  = torso_vertical > 20  # shoulders clearly above hips

        # Sitting: knees close to hip height AND torso upright
        is_sitting = (knee_hip_diff < 80) and torso_upright

        return is_sitting

    except Exception:
        return False

def is_sitting_on(person_box, obj_box, kp_xy=None, kp_conf=None):
    """
    Geometric check: person's hip level overlaps the object vertically,
    horizontal centers aligned. Uses hip keypoints when available so that
    the person's feet (which extend below the seat) don't cause a false
    negative — the old person_bottom check failed whenever feet were visible.
    """
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    person_width = px2 - px1

    # Use hip keypoint y as the seat level; fall back to bbox midpoint
    seat_y = (py1 + py2) / 2
    if kp_xy is not None and kp_conf is not None:
        hip_ys = []
        if kp_conf[KP_LEFT_HIP]  > KP_CONF_THRESHOLD: hip_ys.append(kp_xy[KP_LEFT_HIP][1])
        if kp_conf[KP_RIGHT_HIP] > KP_CONF_THRESHOLD: hip_ys.append(kp_xy[KP_RIGHT_HIP][1])
        if hip_ys:
            seat_y = float(np.mean(hip_ys))

    # Hip should sit somewhere within the object's bounding box (with slack)
    vertical_match   = (oy1 - 30 <= seat_y <= oy2 + 10)
    person_cx        = (px1 + px2) / 2
    obj_cx           = (ox1 + ox2) / 2
    horizontal_match = abs(person_cx - obj_cx) < person_width * SITTING_HORIZONTAL_TOL
    obj_is_below = (oy1 + oy2) / 2 > (py1 + py2) / 2


    return vertical_match and horizontal_match and obj_is_below

def is_fallen_pose(kp_xy, kp_conf, box):
    try:
        ls_conf = kp_conf[KP_LEFT_SHOULDER]
        rs_conf = kp_conf[KP_RIGHT_SHOULDER]
        lh_conf = kp_conf[KP_LEFT_HIP]
        rh_conf = kp_conf[KP_RIGHT_HIP]

        if max(ls_conf, rs_conf) < KP_CONF_THRESHOLD: return is_horizontal(box)
        if max(lh_conf, rh_conf) < KP_CONF_THRESHOLD: return is_horizontal(box)

        shoulder_ys, hip_ys = [], []
        shoulder_xs, hip_xs = [], []

        if ls_conf > KP_CONF_THRESHOLD:
            shoulder_ys.append(kp_xy[KP_LEFT_SHOULDER][1])
            shoulder_xs.append(kp_xy[KP_LEFT_SHOULDER][0])
        if rs_conf > KP_CONF_THRESHOLD:
            shoulder_ys.append(kp_xy[KP_RIGHT_SHOULDER][1])
            shoulder_xs.append(kp_xy[KP_RIGHT_SHOULDER][0])
        if lh_conf > KP_CONF_THRESHOLD:
            hip_ys.append(kp_xy[KP_LEFT_HIP][1])
            hip_xs.append(kp_xy[KP_LEFT_HIP][0])
        if rh_conf > KP_CONF_THRESHOLD:
            hip_ys.append(kp_xy[KP_RIGHT_HIP][1])
            hip_xs.append(kp_xy[KP_RIGHT_HIP][0])

        dy = abs(np.mean(shoulder_ys) - np.mean(hip_ys))
        dx = abs(np.mean(shoulder_xs) - np.mean(hip_xs))

        return dx > dy * 0.8  # torso more horizontal than vertical

    except Exception:
        return is_horizontal(box)

def draw_box(frame, box, label, color, thickness=2):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1,y1),(x2,y2), color, thickness)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1,y1-th-6),(x1+tw+6,y1), color, -1)
    cv2.putText(frame, label, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def show_frame(frame, title="Frame"):
    cv2.imshow(title, frame)
    cv2.waitKey(1)

def body_parts_touching_obj(kp_xy, kp_conf, obj_box):
    """
    Returns list of body-part names whose keypoint is strictly inside obj_box.
    Skips low-confidence or undetected (0,0) keypoints.
    """
    ox1, oy1, ox2, oy2 = obj_box
    touching = []
    for idx, name in KP_NAMES.items():
        if idx >= len(kp_conf) or kp_conf[idx] < KP_CONF_THRESHOLD:
            continue
        kx, ky = float(kp_xy[idx][0]), float(kp_xy[idx][1])
        if kx == 0.0 and ky == 0.0:
            continue
        if ox1 <= kx <= ox2 and oy1 <= ky <= oy2:
            touching.append(name)
    return touching

def draw_touch_snapshot(frame, pbox, detected_objects, kp_xy, kp_conf, phase_label):
    """
    Returns an annotated copy of frame showing body-part touches.
    Touching keypoints: large yellow dot + name label.
    Objects with touches: cyan box with touching part names.
    All other keypoints: small grey dot.
    """
    vis = frame.copy()

    # Recompute which parts touch which objects (so snapshot is always consistent)
    obj_touching = {}
    for obj in detected_objects:
        parts = body_parts_touching_obj(kp_xy, kp_conf, obj["box"])
        if parts:
            obj_touching[obj["label"]] = parts
    all_touching_parts = {bp for pts in obj_touching.values() for bp in pts}

    # Draw objects
    for obj in detected_objects:
        parts = obj_touching.get(obj["label"], [])
        if parts:
            draw_box(vis, obj["box"],
                     f"{obj['label']}: {', '.join(parts)}", (0, 220, 220), 3)
        else:
            draw_box(vis, obj["box"], obj["label"], (160, 160, 160), 1)

    # Draw person bounding box
    draw_box(vis, pbox, "Person", (255, 255, 255), 2)

    # Draw keypoints
    for idx, name in KP_NAMES.items():
        if idx >= len(kp_conf) or kp_conf[idx] < KP_CONF_THRESHOLD:
            continue
        kx, ky = int(kp_xy[idx][0]), int(kp_xy[idx][1])
        if kx == 0 and ky == 0:
            continue
        if name in all_touching_parts:
            cv2.circle(vis, (kx, ky), 8, (0, 255, 255), -1)
            cv2.putText(vis, name, (kx + 10, ky + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.circle(vis, (kx, ky), 4, (180, 180, 180), -1)

    # Phase banner
    cv2.putText(vis, phase_label, (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
    return vis

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    script_dir = Path(__file__).parent
    obj_model  = YOLO(script_dir / "yolov8n.pt")
    pose_model = YOLO(script_dir / "yolov8n-pose.pt")
    print("Models loaded.")

    # Output directory — created now so snapshots can be saved during the loop
    video_stem = Path(VIDEO_PATH).stem
    json_dir   = script_dir / "jsondescriptions"
    json_dir.mkdir(exist_ok=True)

    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH}  |  {total} frames @ {fps:.1f}fps\n")

    buffer_frames  = int(PRE_FALL_BUFFER_SEC * fps)

    horiz_counts     = {}
    upright_counts   = {}
    fallen_ids       = set()
    sitting_log      = []   # (frame_idx, tid, label, box)
    pre_fall_log     = []   # (frame_idx, label, box)
    incidents        = []
    fall_frames      = []   # (frame_idx, annotated_frame)
    fired_frames     = set()  # prevent duplicate falls on same frame
    sitting_frames        = []   # (frame_idx, annotated_frame) — one per person, first sit
    sitting_seen_ids      = set()  # tids we've already snapshotted
    fall_touch_log        = []   # (tid, {frame,time_sec,phase,object,risk,body_parts})
    active_fall_incidents = {}   # tid -> incident dict (mutable ref; after_fall appended live)
    wf_snap_paths         = {}   # tid -> filename of while_falling snapshot
    after_fall_snapped    = set()  # tids whose after_fall snapshot has been saved
    ankle_y_history       = {}   # tid -> list of y samples taken when person is upright
    floor_y_est           = {}   # tid -> estimated floor y coordinate
    first_floor_hit       = {}   # tid -> floor contact info dict
    pre_fall_posture      = {}   # tid -> last confirmed posture before fall ("sitting"/"standing"/"lying")

    obj_tracks            = {}   # obj_track_id -> {label, history, falling, hit_person}
    obj_track_next_id     = 0    # auto-increment counter for object tracks
    falling_obj_incidents = []   # logged falling-object events
    fired_obj_falls       = set()  # obj_track_ids already fired
    active_falling_objs   = {}   # obj_track_id -> incident dict ref (for hit_person updates)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ─────────────────────────────────────────────────────────
        obj_results  = obj_model(frame, verbose=False)[0]
        pose_results = pose_model(frame, verbose=False)[0]

        # Build object list — skip persons, skip anything that heavily
        # overlaps a detected person (avoids chair being labelled as person)
        person_boxes = [b.xyxy[0].tolist() for b in pose_results.boxes] \
                       if pose_results.boxes is not None else []
        detected_objects = []
        if obj_results.boxes is not None:
            for box in obj_results.boxes:
                cls_id  = int(box.cls[0])
                label   = obj_model.names[cls_id]
                if label == "person": continue
                conf    = float(box.conf[0])
                if conf < OBJ_CONF_THRESHOLD: continue
                obj_box = box.xyxy[0].tolist()
                if any(iou(obj_box, pb) >= 0.3 for pb in person_boxes):
                    continue
                detected_objects.append({"label": label, "box": obj_box, "conf": conf})

        # Rolling pre-fall log
        for obj in detected_objects:
            pre_fall_log.append((frame_idx, obj["label"], obj["box"]))
        cutoff = frame_idx - buffer_frames - 1
        pre_fall_log = [(f,l,b) for f,l,b in pre_fall_log if f > cutoff]

        # ── Object tracking ───────────────────────────────────────────────────
        matched_obj_tids = set()
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj["box"]
            cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            label  = obj["label"]

            # Match to the closest existing track of same label
            best_tid, best_dist = None, float("inf")
            for otid, track in obj_tracks.items():
                if track["label"] != label or not track["history"]:
                    continue
                lx, ly = track["history"][-1][1], track["history"][-1][2]
                d = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                if d < best_dist and d < OBJ_TRACK_MAX_DIST:
                    best_dist = d
                    best_tid  = otid

            if best_tid is not None and best_tid not in matched_obj_tids:
                matched_obj_tids.add(best_tid)
                track = obj_tracks[best_tid]
            else:
                # New object track
                best_tid = obj_track_next_id
                obj_track_next_id += 1
                track = {"label": label, "history": [], "falling": False, "hit_person": False}
                obj_tracks[best_tid] = track

            track["history"].append((frame_idx, cx, cy, list(obj["box"])))
            if len(track["history"]) > OBJ_HISTORY_FRAMES:
                track["history"] = track["history"][-OBJ_HISTORY_FRAMES:]

        # Expire stale tracks not seen recently
        obj_tracks = {otid: t for otid, t in obj_tracks.items()
                      if t["history"] and t["history"][-1][0] >= frame_idx - OBJ_HISTORY_FRAMES}

        # ── Falling object detection ───────────────────────────────────────────
        for otid, track in obj_tracks.items():
            hist = track["history"]
            if not hist:
                continue

            # Update hit_person for already-confirmed falling objects
            if track["falling"] and not track["hit_person"] and otid in active_falling_objs:
                last_box = hist[-1][3]
                if any(iou(last_box, pb) > TOUCH_IOU_THRESHOLD
                       or center_dist(last_box, pb) < PROXIMITY_PX
                       for pb in person_boxes):
                    track["hit_person"] = True
                    active_falling_objs[otid]["hit_person"] = True
                continue

            if track["falling"] or otid in fired_obj_falls or len(hist) < FALLING_OBJ_CONFIRM:
                continue

            # Check last N frames for consistent downward (positive cy) velocity
            recent = hist[-FALLING_OBJ_CONFIRM:]
            vels   = [recent[i+1][2] - recent[i][2] for i in range(len(recent) - 1)]
            if all(v >= FALLING_OBJ_VEL_PX for v in vels):
                track["falling"] = True
                fired_obj_falls.add(otid)

                last_box      = hist[-1][3]
                person_caused = any(
                    iou(last_box, pb) > TOUCH_IOU_THRESHOLD
                    or center_dist(last_box, pb) < PROXIMITY_PX
                    for pb in person_boxes
                )
                fo_incident = {
                    "object"          : track["label"] if track["label"] else "unknown",
                    "fell_at_frame"   : frame_idx,
                    "fell_at_time_sec": round(frame_idx / fps, 2),
                    "person_caused_it": person_caused,
                    "hit_person"      : False,
                    "injury_risk"     : get_risk(track["label"]),
                }
                falling_obj_incidents.append(fo_incident)
                active_falling_objs[otid] = fo_incident
                print(f"\n  [FALLING OBJECT] {fo_incident['object']} @ {fo_incident['fell_at_time_sec']}s"
                      f"  person_caused={person_caused}  risk={fo_incident['injury_risk']}")

        # ── Pose loop ─────────────────────────────────────────────────────────
        if pose_results.boxes is not None:
            for i, box in enumerate(pose_results.boxes):
                tid  = int(box.id[0]) if box.id is not None else i
                pbox = box.xyxy[0].tolist()

                # Get keypoints for this person
                kp_xy   = None
                kp_conf = None
                if pose_results.keypoints is not None and i < len(pose_results.keypoints.xy):
                    kp_xy   = pose_results.keypoints.xy[i].cpu().numpy()
                    kp_conf = pose_results.keypoints.conf[i].cpu().numpy()

                horiz_counts.setdefault(tid, 0)
                upright_counts.setdefault(tid, 0)

                # ── Sitting detection ─────────────────────────
                # Use pose keypoints — works even when bbox looks wide
                person_is_sitting = False
                if kp_xy is not None:
                    person_is_sitting = is_sitting_pose(kp_xy, kp_conf)

                if person_is_sitting:
                    matched_objs = []
                    for obj in detected_objects:
                        if is_sitting_on(pbox, obj["box"], kp_xy, kp_conf):
                            sitting_log.append((frame_idx, tid, obj["label"], obj["box"]))
                            matched_objs.append(obj)

                    # Snapshot the first frame per person where sitting is confirmed
                    if matched_objs and tid not in sitting_seen_ids:
                        sitting_seen_ids.add(tid)
                        vis = frame.copy()

                        # Draw all detected objects (grey = not sat on, green = sat on)
                        for obj in detected_objects:
                            if obj in matched_objs:
                                draw_box(vis, obj["box"], f"SITTING ON: {obj['label']}", (0, 220, 0), 3)
                            else:
                                draw_box(vis, obj["box"],
                                         f"{obj['label']} [{get_risk(obj['label'])}]",
                                         (160, 160, 160), 1)

                        # Draw person box
                        draw_box(vis, pbox, f"SITTING Person {tid}", (0, 200, 0), 2)

                        # Draw hip keypoint — the seat_y point driving is_sitting_on()
                        if kp_xy is not None and kp_conf is not None:
                            hip_xs, hip_ys = [], []
                            if kp_conf[KP_LEFT_HIP]  > KP_CONF_THRESHOLD:
                                hip_xs.append(int(kp_xy[KP_LEFT_HIP][0]))
                                hip_ys.append(int(kp_xy[KP_LEFT_HIP][1]))
                            if kp_conf[KP_RIGHT_HIP] > KP_CONF_THRESHOLD:
                                hip_xs.append(int(kp_xy[KP_RIGHT_HIP][0]))
                                hip_ys.append(int(kp_xy[KP_RIGHT_HIP][1]))
                            if hip_xs:
                                cx = int(np.mean(hip_xs))
                                cy = int(np.mean(hip_ys))
                                cv2.circle(vis, (cx, cy), 10, (0, 255, 255), -1)
                                cv2.putText(vis, "seat_y (hip)", (cx + 14, cy + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                        cv2.putText(vis, f"SITTING @ {round(frame_idx/fps, 2)}s", (20, 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 0), 3)
                        sitting_frames.append((frame_idx, vis))

                # Trim sitting log
                sitting_log = [(f,t,l,b) for f,t,l,b in sitting_log if f > cutoff]

                # ── Floor y estimation (only when upright and not sitting) ────
                if horiz_counts[tid] == 0 and not person_is_sitting:
                    # Use bbox bottom as proxy; prefer ankle keypoints if available
                    y_sample = pbox[3]
                    if kp_xy is not None:
                        for ak_idx in (KP_LEFT_ANKLE, KP_RIGHT_ANKLE):
                            if kp_conf[ak_idx] > KP_CONF_THRESHOLD:
                                ay = float(kp_xy[ak_idx][1])
                                if ay > 0:
                                    y_sample = max(y_sample, ay)
                    history = ankle_y_history.setdefault(tid, [])
                    history.append(y_sample)
                    if len(history) > 60:
                        ankle_y_history[tid] = history[-60:]
                    floor_y_est[tid] = max(history) + 5  # small buffer below lowest point

                # ── Fall detection ────────────────────────────
                if is_fallen_pose(kp_xy, kp_conf, pbox) if kp_xy is not None else is_horizontal(pbox):

                    horiz_counts[tid]   += 1
                    upright_counts[tid]  = 0
                else:
                    upright_counts[tid] += 1
                    horiz_counts[tid]    = 0

                # ── Pre-fall posture (freeze when horiz_counts goes to 0) ────
                if horiz_counts[tid] == 0:
                    if person_is_sitting:
                        pre_fall_posture[tid] = "sitting"
                    elif is_horizontal(pbox) or (
                            kp_xy is not None and is_fallen_pose(kp_xy, kp_conf, pbox)):
                        pre_fall_posture[tid] = "lying"
                    else:
                        pre_fall_posture[tid] = "standing"

                # ── Body-part touch logging (WHILE_FALLING / AFTER_FALL) ──────
                fall_phase = None
                if tid in fallen_ids:
                    fall_phase = "AFTER_FALL"
                elif horiz_counts[tid] >= 1:
                    fall_phase = "WHILE_FALLING"

                if fall_phase and kp_xy is not None:
                    any_touch_this_frame = False
                    for obj in detected_objects:
                        parts = body_parts_touching_obj(kp_xy, kp_conf, obj["box"])
                        if not parts:
                            continue
                        any_touch_this_frame = True
                        entry = {
                            "frame"     : frame_idx,
                            "time_sec"  : round(frame_idx / fps, 2),
                            "phase"     : fall_phase,
                            "object"    : obj["label"],
                            "risk"      : get_risk(obj["label"]),
                            "body_parts": parts,
                        }
                        fall_touch_log.append((tid, entry))
                        if fall_phase == "AFTER_FALL" and tid in active_fall_incidents:
                            af = active_fall_incidents[tid]["body_part_touches"]["after_fall"]
                            existing = next((e for e in af if e["object"] == obj["label"]), None)
                            if existing is None:
                                existing = {"object": obj["label"],
                                            "risk"  : get_risk(obj["label"]),
                                            "body_parts": []}
                                af.append(existing)
                            seen_bp = set(existing["body_parts"])
                            for bp in parts:
                                if bp not in seen_bp:
                                    existing["body_parts"].append(bp)
                                    seen_bp.add(bp)

                    # ── Touch snapshots ───────────────────────────────────────
                    if any_touch_this_frame:
                        if fall_phase == "WHILE_FALLING" and tid not in wf_snap_paths:
                            snap = draw_touch_snapshot(
                                frame, pbox, detected_objects, kp_xy, kp_conf,
                                f"WHILE FALLING — Person {tid} @ {round(frame_idx/fps,2)}s")
                            fname = f"{video_stem}_p{tid}_while_falling.jpg"
                            cv2.imwrite(str(json_dir / fname), snap)
                            wf_snap_paths[tid] = fname

                        elif (fall_phase == "AFTER_FALL"
                              and tid not in after_fall_snapped
                              and tid in active_fall_incidents):
                            snap = draw_touch_snapshot(
                                frame, pbox, detected_objects, kp_xy, kp_conf,
                                f"AFTER FALL — Person {tid} @ {round(frame_idx/fps,2)}s")
                            fname = f"{video_stem}_p{tid}_after_fall.jpg"
                            cv2.imwrite(str(json_dir / fname), snap)
                            after_fall_snapped.add(tid)
                            active_fall_incidents[tid]["body_part_touches"]["after_fall_snapshot"] = fname

                # ── First floor contact ──────────────────────────────────────
                if fall_phase and tid not in first_floor_hit and kp_xy is not None:
                    fh       = frame.shape[0]
                    floor_y  = floor_y_est.get(tid, fh * 0.85)

                    at_floor = []
                    for idx, name in KP_NAMES.items():
                        if idx >= len(kp_conf) or kp_conf[idx] < KP_CONF_THRESHOLD:
                            continue
                        ky = float(kp_xy[idx][1])
                        if ky <= 0:
                            continue
                        if ky >= floor_y - FLOOR_TOLERANCE_PX:
                            at_floor.append((name, ky))

                    # Exclude feet/ankles — they're always near the floor;
                    # we want the first non-foot part to make impact.
                    at_floor_impact = [(n, y) for n, y in at_floor
                                       if n not in FLOOR_EXCLUDED_PARTS]
                    if at_floor_impact:
                        at_floor_impact.sort(key=lambda x: x[1], reverse=True)
                        is_flat  = len(at_floor_impact) >= FLAT_FALL_MIN_PARTS
                        primary  = "flat" if is_flat else at_floor_impact[0][0]
                        hit_info = {
                            "body_part" : primary,
                            "all_parts" : [n for n, _ in at_floor_impact],
                            "flat_fall" : is_flat,
                            "frame"     : frame_idx,
                            "time_sec"  : round(frame_idx / fps, 2),
                            "snapshot"  : None,
                        }

                        # Annotated floor-contact snapshot
                        vis_fc    = frame.copy()
                        floor_set = {n for n, _ in at_floor_impact}
                        draw_box(vis_fc, pbox, f"Person {tid}", (255, 255, 255), 2)
                        for idx2, name2 in KP_NAMES.items():
                            if idx2 >= len(kp_conf) or kp_conf[idx2] < KP_CONF_THRESHOLD:
                                continue
                            kx2, ky2 = int(kp_xy[idx2][0]), int(kp_xy[idx2][1])
                            if kx2 == 0 and ky2 == 0:
                                continue
                            if name2 in floor_set:
                                cv2.circle(vis_fc, (kx2, ky2), 10, (0, 100, 255), -1)
                                cv2.putText(vis_fc, name2, (kx2 + 10, ky2 + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
                            else:
                                cv2.circle(vis_fc, (kx2, ky2), 4, (180, 180, 180), -1)
                        cv2.line(vis_fc, (0, int(floor_y)), (vis_fc.shape[1], int(floor_y)),
                                 (0, 160, 160), 2)
                        cv2.putText(vis_fc, "est. floor", (8, int(floor_y) - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 160), 1)
                        banner = (f"FLOOR CONTACT: {'FLAT' if is_flat else primary}"
                                  f" — Person {tid} @ {round(frame_idx/fps, 2)}s")
                        cv2.putText(vis_fc, banner, (20, 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 100, 255), 3)
                        fc_fname = f"{video_stem}_p{tid}_floor_contact.jpg"
                        cv2.imwrite(str(json_dir / fc_fname), vis_fc)
                        hit_info["snapshot"] = fc_fname

                        first_floor_hit[tid] = hit_info
                        # Also update the incident if it has already been fired
                        if tid in active_fall_incidents:
                            active_fall_incidents[tid]["first_floor_contact"] = hit_info

                # ── New fall event ────────────────────────────
                if (tid not in fallen_ids
                        and horiz_counts[tid] >= FALL_CONFIRM_FRAMES
                        and frame_idx not in fired_frames):

                    fallen_ids.add(tid)
                    fired_frames.add(frame_idx)
                    fall_time = round(frame_idx / fps, 2)

                    # What was person sitting on?
                    pre_sitting = [
                        {"object": l, "last_seen_frame": f,
                         "seconds_before_fall": round((frame_idx - f) / fps, 2)}
                        for (f, t, l, b) in sitting_log
                        if t == tid
                    ]
                    seen = {}
                    for s in sorted(pre_sitting, key=lambda x: x["last_seen_frame"], reverse=True):
                        if s["object"] not in seen:
                            seen[s["object"]] = s
                    sitting_summary = list(seen.values())

                    # WHILE_FALLING: body-part touches from frames leading up to this fall
                    wf_entries = [e for (t, e) in fall_touch_log
                                  if t == tid and e["phase"] == "WHILE_FALLING"]
                    wf_by_obj  = {}
                    for e in wf_entries:
                        o = e["object"]
                        if o not in wf_by_obj:
                            wf_by_obj[o] = {"object": o, "risk": e["risk"], "body_parts": set()}
                        wf_by_obj[o]["body_parts"].update(e["body_parts"])
                    while_falling_touches = [
                        {"object": v["object"], "risk": v["risk"],
                         "body_parts": sorted(v["body_parts"])}
                        for v in wf_by_obj.values()
                    ]

                    # DURING_FALL: body-part touches at this exact frame
                    during_touches = []
                    if kp_xy is not None:
                        for obj in detected_objects:
                            parts = body_parts_touching_obj(kp_xy, kp_conf, obj["box"])
                            if parts:
                                during_touches.append({
                                    "object"    : obj["label"],
                                    "risk"      : get_risk(obj["label"]),
                                    "body_parts": parts,
                                })

                    # Object interactions at fall moment
                    interactions = []
                    for obj in detected_objects:
                        overlap  = iou(pbox, obj["box"])
                        distance = center_dist(pbox, obj["box"])
                        touching = overlap > TOUCH_IOU_THRESHOLD
                        nearby   = distance < PROXIMITY_PX
                        if not (touching or nearby): continue

                        present_before = any(
                            lbl == obj["label"] and 0 < (frame_idx - f) <= buffer_frames
                            for (f, lbl, _) in pre_fall_log
                        )
                        interactions.append({
                            "object"              : obj["label"],
                            "touching"            : touching,
                            "proximity_px"        : round(distance, 1),
                            "present_before_fall" : present_before,
                            "timing"              : "BEFORE_FALL" if present_before else "AFTER_FALL",
                            "injury_risk"         : get_risk(obj["label"]),
                            "body_parts"          : body_parts_touching_obj(kp_xy, kp_conf, obj["box"])
                                                    if kp_xy is not None else [],
                        })

                    risk_order = {"HIGH":0,"MEDIUM":1,"LOW":2,"UNKNOWN":3}
                    interactions.sort(key=lambda x: (not x["touching"], risk_order[x["injury_risk"]]))

                    # DURING_FALL snapshot
                    dur_snap = None
                    if during_touches and kp_xy is not None:
                        snap = draw_touch_snapshot(
                            frame, pbox, detected_objects, kp_xy, kp_conf,
                            f"DURING FALL — Person {tid} @ {fall_time}s")
                        dur_snap = f"{video_stem}_p{tid}_during_fall.jpg"
                        cv2.imwrite(str(json_dir / dur_snap), snap)

                    # ── Build summary ─────────────────────────────────────────
                    _risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}

                    # Objects present/touching before the fall
                    _before_objs = list(dict.fromkeys(
                        [s["object"] for s in sitting_summary] +
                        [obj["object"] for obj in interactions
                         if obj.get("timing") == "BEFORE_FALL"]
                    ))

                    # Highest-risk body part at fall moment (after_fall added later)
                    _highest_risk = None
                    _best_rank    = 999
                    for _entries in [while_falling_touches, during_touches]:
                        for _e in _entries:
                            _rank = _risk_order.get(_e.get("risk", "UNKNOWN"), 3)
                            if _rank < _best_rank and _e.get("body_parts"):
                                _best_rank    = _rank
                                _highest_risk = {
                                    "body_parts": _e["body_parts"],
                                    "object"    : _e["object"],
                                    "risk"      : _e["risk"],
                                }

                    _ffc_data  = first_floor_hit.get(tid)
                    _ffc_label = ("flat" if _ffc_data["flat_fall"] else _ffc_data["body_part"]) \
                                 if _ffc_data else None

                    summary = {
                        "fell"                  : True,
                        "fall_time_sec"         : fall_time,
                        "pre_fall_posture"      : pre_fall_posture.get(tid, "unknown"),
                        "was_sitting_on"        : [s["object"] for s in sitting_summary],
                        "objects_touching"      : {
                            "before_fall"   : _before_objs,
                            "while_falling" : [e["object"] for e in while_falling_touches],
                            "during_fall"   : [e["object"] for e in during_touches],
                            "after_fall"    : [],   # populated in later frames
                        },
                        "first_floor_contact"        : _ffc_label,
                        "highest_risk_body_part"     : _highest_risk,
                        "falling_objects_during_fall": [],  # populated at end of run
                    }

                    incidents.append({
                        "person_track_id"    : tid,
                        "fall_frame"         : frame_idx,
                        "fall_time_sec"      : fall_time,
                        "was_sitting_on"     : sitting_summary,
                        "pre_fall_posture"   : pre_fall_posture.get(tid, "unknown"),
                        "first_floor_contact": first_floor_hit.get(tid),
                        "object_interactions": interactions,
                        "body_part_touches"  : {
                            "while_falling"         : while_falling_touches,
                            "while_falling_snapshot": wf_snap_paths.get(tid),
                            "during_fall"           : during_touches,
                            "during_fall_snapshot"  : dur_snap,
                            "after_fall"            : [],   # populated in subsequent frames
                            "after_fall_snapshot"   : None, # set when first after-fall touch seen
                        },
                        "summary"            : summary,
                    })
                    active_fall_incidents[tid] = incidents[-1]

                    # ── Print ONLY the summary ─────────────────────────────────
                    _ot = summary["objects_touching"]
                    print(f"\n⚠️  FALL — Person {tid} @ {fall_time}s")
                    print(f"  Pre-fall posture   : {summary['pre_fall_posture']}")
                    print(f"  Was sitting on     : {', '.join(summary['was_sitting_on']) or 'nothing detected'}")
                    print(f"  Objects touching   :")
                    print(f"    Before fall      : {', '.join(_ot['before_fall'])   or 'none'}")
                    print(f"    While falling    : {', '.join(_ot['while_falling']) or 'none'}")
                    print(f"    During fall      : {', '.join(_ot['during_fall'])   or 'none'}")
                    print(f"  First floor contact: {_ffc_label or 'not yet detected'}")
                    if _highest_risk:
                        print(f"  Highest-risk touch : {', '.join(_highest_risk['body_parts'])}"
                              f" → {_highest_risk['object']} ({_highest_risk['risk']})")
                    else:
                        print(f"  Highest-risk touch : none detected")
                    # Annotate fall frame
                    vis = frame.copy()
                    RISK_COLORS = {"HIGH":(0,0,220),"MEDIUM":(0,140,255),"LOW":(0,200,150),"UNKNOWN":(160,160,160)}
                    draw_box(vis, pbox, f"FALL Person {tid}", (0,0,220), 3)
                    for obj in detected_objects:
                        draw_box(vis, obj["box"],
                                 f"{obj['label']} [{get_risk(obj['label'])}]",
                                 RISK_COLORS[get_risk(obj["label"])], 1)
                    for s in sitting_summary:
                        for obj in detected_objects:
                            if obj["label"] == s["object"]:
                                draw_box(vis, obj["box"], f"WAS SITTING ON: {s['object']}", (255,100,0), 3)
                    cv2.putText(vis, f"FALL @ {fall_time}s", (20,48),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3)
                    fall_frames.append((frame_idx, vis))

                # Recover
                if tid in fallen_ids and upright_counts[tid] >= RECOVER_FRAMES:
                    fallen_ids.discard(tid)
                    active_fall_incidents.pop(tid, None)
                    fall_touch_log = [(t, e) for (t, e) in fall_touch_log if t != tid]

        frame_idx += 1

    cap.release()

    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print(f"\nDone. {len(incidents)} fall(s) detected.")

    # Save + show sitting snapshots
    for fidx, vis in sitting_frames:
        out_path = str(script_dir / f"sitting_frame_{fidx}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"Saved sitting snapshot: {out_path}")
        cv2.imshow(f"Sitting @ frame {fidx}", vis)

    # Show annotated fall frames
    for fidx, vis in fall_frames:
        cv2.imshow(f"Fall @ frame {fidx}", vis)

    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # ── Finalize summaries with after_fall + falling-object data ─────────────
    _risk_order_final = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    for inc in incidents:
        s   = inc.get("summary", {})
        bpt = inc.get("body_part_touches", {})
        after_fall = bpt.get("after_fall", [])

        # Fill in after_fall objects now that we have them
        s.setdefault("objects_touching", {})["after_fall"] = \
            list(dict.fromkeys(e["object"] for e in after_fall))

        # Recompute highest-risk body part across all three phases
        _best_rank = 999
        _highest   = None
        for _entries in [bpt.get("while_falling", []),
                         bpt.get("during_fall",   []),
                         after_fall]:
            for _e in _entries:
                _rank = _risk_order_final.get(_e.get("risk", "UNKNOWN"), 3)
                if _rank < _best_rank and _e.get("body_parts"):
                    _best_rank = _rank
                    _highest   = {"body_parts": _e["body_parts"],
                                  "object"    : _e["object"],
                                  "risk"      : _e["risk"]}
        s["highest_risk_body_part"] = _highest

        # Update first_floor_contact label (may have been detected after incident fired)
        _ffc = inc.get("first_floor_contact")
        if _ffc:
            s["first_floor_contact"] = "flat" if _ffc["flat_fall"] else _ffc["body_part"]

        # Populate falling objects that happened near this person fall (±buffer window)
        fall_frm = inc["fall_frame"]
        s["falling_objects_during_fall"] = [
            {"object"    : fo["object"],
             "hit_person": fo["hit_person"],
             "injury_risk": fo["injury_risk"]}
            for fo in falling_obj_incidents
            if abs(fo["fell_at_frame"] - fall_frm) <= buffer_frames
        ]

    # Flag background falling objects: fell independently with no person involvement
    for fo in falling_obj_incidents:
        fo["is_background_fall"] = not fo["person_caused_it"] and not fo["hit_person"]

    background_falls = [fo for fo in falling_obj_incidents if fo["is_background_fall"]]

    # Add background falls list to every person-fall summary
    bg_summary = [
        {"object"         : fo["object"],
         "fell_at_time_sec": fo["fell_at_time_sec"],
         "injury_risk"    : fo["injury_risk"]}
        for fo in background_falls
    ]
    for inc in incidents:
        inc["summary"]["background_falling_objects"] = bg_summary

    print("\n" + "="*60)
    print("FALL SUMMARY REPORT")
    print("="*60)
    for inc in incidents:
        s  = inc.get("summary", {})
        ot = s.get("objects_touching", {})
        hr = s.get("highest_risk_body_part")
        fo_list = s.get("falling_objects_during_fall", [])
        print(f"\nFALL — Person {inc['person_track_id']} @ {inc['fall_time_sec']}s")
        print(f"  Pre-fall posture   : {s.get('pre_fall_posture', 'unknown')}")
        print(f"  Was sitting on     : {', '.join(s.get('was_sitting_on', [])) or 'nothing detected'}")
        print(f"  Objects touching   :")
        print(f"    Before fall      : {', '.join(ot.get('before_fall',   [])) or 'none'}")
        print(f"    While falling    : {', '.join(ot.get('while_falling', [])) or 'none'}")
        print(f"    During fall      : {', '.join(ot.get('during_fall',   [])) or 'none'}")
        print(f"    After fall       : {', '.join(ot.get('after_fall',    [])) or 'none'}")
        print(f"  First floor contact: {s.get('first_floor_contact') or 'not detected'}")
        if hr:
            print(f"  Highest-risk touch : {', '.join(hr['body_parts'])}"
                  f" → {hr['object']} ({hr['risk']})")
        else:
            print(f"  Highest-risk touch : none detected")
        if fo_list:
            fo_str = ", ".join(
                f"{f['object']} (hit_person={f['hit_person']}, risk={f['injury_risk']})"
                for f in fo_list
            )
            print(f"  Falling objects    : {fo_str}")
        else:
            print(f"  Falling objects    : none near this fall")
        bg = s.get("background_falling_objects", [])
        if bg:
            print(f"  Background falls   : {', '.join(f['object'] for f in bg)}"
                  f"  (no person involved)")
        else:
            print(f"  Background falls   : none detected")

    if background_falls:
        print(f"\nBACKGROUND FALLING OBJECTS ({len(background_falls)} — no person involved):")
        for fo in background_falls:
            print(f"  {fo['object']:20s} @ {fo['fell_at_time_sec']}s  risk={fo['injury_risk']}")
    elif falling_obj_incidents:
        print(f"\nNo background falling objects detected"
              f" ({len(falling_obj_incidents)} falling object(s) all involved a person).")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "person_fall_incidents"    : incidents,
        "falling_object_incidents" : falling_obj_incidents,
        "background_falling_objects": background_falls,
    }
    json_path = json_dir / f"{video_stem}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved: {json_path}")


if __name__ == "__main__":
    run()
