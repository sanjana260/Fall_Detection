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
OBJ_CONF_THRESHOLD       = 0.35
SITTING_HORIZONTAL_TOL   = 0.6

# Pose keypoint indices (COCO format)
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_CONF_THRESHOLD = 0.3   # min keypoint confidence to use it

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

    return vertical_match and horizontal_match

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

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    script_dir = Path(__file__).parent
    obj_model  = YOLO(script_dir / "yolov8n.pt")
    pose_model = YOLO(script_dir / "yolov8n-pose.pt")
    print("Models loaded.")

    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH}  |  {total} frames @ {fps:.1f}fps\n")

    buffer_frames  = int(PRE_FALL_BUFFER_SEC * fps)

    horiz_counts   = {}
    upright_counts = {}
    fallen_ids     = set()
    sitting_log    = []   # (frame_idx, tid, label, box)
    pre_fall_log   = []   # (frame_idx, label, box)
    incidents      = []
    fall_frames    = []   # (frame_idx, annotated_frame)
    fired_frames   = set()  # prevent duplicate falls on same frame

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        obj_results      = obj_model(frame, verbose=False)[0]
        detected_objects = []
        if obj_results.boxes is not None:
            for box in obj_results.boxes:
                cls_id = int(box.cls[0])
                label  = obj_model.names[cls_id]
                if label == "person": continue
                conf = float(box.conf[0])
                if conf < OBJ_CONF_THRESHOLD: continue
                detected_objects.append({"label": label, "box": box.xyxy[0].tolist(), "conf": conf})

        # Rolling pre-fall log
        for obj in detected_objects:
            pre_fall_log.append((frame_idx, obj["label"], obj["box"]))
        cutoff = frame_idx - buffer_frames - 1
        pre_fall_log = [(f,l,b) for f,l,b in pre_fall_log if f > cutoff]

        # Pose detection
        pose_results = pose_model(frame, verbose=False)[0]
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
                    for obj in detected_objects:
                        if is_sitting_on(pbox, obj["box"], kp_xy, kp_conf):
                            sitting_log.append((frame_idx, tid, obj["label"], obj["box"]))

                # Trim sitting log
                sitting_log = [(f,t,l,b) for f,t,l,b in sitting_log if f > cutoff]

                # ── Fall detection ────────────────────────────
                if is_fallen_pose(kp_xy, kp_conf, pbox) if kp_xy is not None else is_horizontal(pbox):

                    horiz_counts[tid]   += 1
                    upright_counts[tid]  = 0
                else:
                    upright_counts[tid] += 1
                    horiz_counts[tid]    = 0

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
                            "likely_cause"        : present_before and touching,
                            "injury_risk"         : get_risk(obj["label"]),
                        })

                    risk_order = {"HIGH":0,"MEDIUM":1,"LOW":2,"UNKNOWN":3}
                    interactions.sort(key=lambda x: (not x["touching"], risk_order[x["injury_risk"]]))

                    incidents.append({
                        "person_track_id"    : tid,
                        "fall_frame"         : frame_idx,
                        "fall_time_sec"      : fall_time,
                        "was_sitting_on"     : sitting_summary,
                        "object_interactions": interactions,
                    })

                    print(f"\n⚠️  FALL — Person {tid} @ {fall_time}s (frame {frame_idx})")
                    print(f"  Sitting on before fall: {[s['object'] for s in sitting_summary] or 'nothing detected'}")
                    for obj in interactions:
                        cause = " ← LIKELY CAUSE" if obj["likely_cause"] else ""
                        print(f"  {obj['object']:20s}  touch={str(obj['touching']):5s}  "
                              f"timing={obj['timing']:12s}  risk={obj['injury_risk']}{cause}")

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

        frame_idx += 1

    cap.release()

    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print(f"\nDone. {len(incidents)} fall(s) detected.")

    # Show annotated fall frames
    for fidx, vis in fall_frames:
        cv2.imshow(f"Fall @ frame {fidx}", vis)

    for _ in range(5):
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # Print full summary
    print("\n" + "="*60)
    for inc in incidents:
        print(f"\nFALL — Person {inc['person_track_id']} @ {inc['fall_time_sec']}s")
        print(f"  Sitting on (before fall):")
        if inc["was_sitting_on"]:
            for s in inc["was_sitting_on"]:
                print(f"    {s['object']:20s}  {s['seconds_before_fall']}s before fall")
        else:
            print("    Nothing detected.")
        print(f"  Object interactions:")
        if inc["object_interactions"]:
            for obj in inc["object_interactions"]:
                cause = " ← LIKELY CAUSE" if obj["likely_cause"] else ""
                print(f"    {obj['object']:20s}  touch={str(obj['touching']):5s}  "
                      f"timing={obj['timing']:12s}  risk={obj['injury_risk']}{cause}")
        else:
            print("    None.")

    #print("\nFull JSON:")
    #print(json.dumps(incidents, indent=2))


if __name__ == "__main__":
    run()
