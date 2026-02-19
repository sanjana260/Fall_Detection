import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import os
import torch
import json
import time
mps_device = torch.device('mps')
# Configure tracker
tracker_params = {
    "max_age": 60,
    "min_hits": 3,
    "iou_threshold": 0.5,
    "match_thresh": 0.9,
}
model = YOLO("yolov8l-pose.pt").to(mps_device)
obj_model=YOLO("yolov8n.pt").to(mps_device)

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


def run_fall_detection_live(input_video_path, output_filename, min_area = 2000):
    """
    Runs fall detection on a given video and writes the annotated output to the 'Outputs' folder.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_name (str): (Optional) Name of the output video file. Defaults to 'fall_detection_output.mp4'.

    Returns:
        str: Path to the saved annotated video.
    """

    # Ensure Outputs directory exists
    os.makedirs("Outputs", exist_ok=True)
    # input_video_filename = os.path.basename(input_video_path)
    output_path = os.path.join("Outputs", output_filename)
    people_images_folder = os.path.splitext(output_filename)[0]
    people_output_dir = os.path.join('Outputs', people_images_folder + "/people")
    if os.path.exists(people_output_dir):
        shutil.rmtree(people_output_dir)
    os.makedirs(people_output_dir)

    fall_output_dir = os.path.join('Outputs', people_images_folder + "/falls")
    if os.path.exists(fall_output_dir):
        shutil.rmtree(fall_output_dir)
    os.makedirs(fall_output_dir)

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_AVFOUNDATION)
    obj_tracker = ObjectTracker()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    object_incidents = []
    object_fall_details = []
    last_fall_sec = -999.0
    frame_buffer = []
    active_clips = {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    max_buffer = int(CLIP_BUFFER_SEC * fps) * 2 + 10





    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    people = {}
    prev_timestamp = 0
    frame_id=0
    while cap.isOpened():
        ret, frame = cap.read()
        recovery = False
        if not ret:
            break
        frame_org = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        # results = model(frame)
        results = model.track(frame, persist=True, tracker='./configs/bytetrack.yaml' , conf=0.35)
        raw_person_boxes = []
        obj_detections = []

        obj_results = obj_model(frame, verbose=False)[0]

        if obj_results.boxes is not None:
            for box in obj_results.boxes:
                cls_id = int(box.cls[0])
                label  = obj_model.names[cls_id]
                conf   = float(box.conf[0])
                b      = box.xyxy[0].tolist()
                if label == "person":
                    if conf > 0.4:
                        raw_person_boxes.append(b)
                elif conf > OBJ_CONF_THRESHOLD:
                    if all(bbox_iou(b, pb) < 0.4 for pb in raw_person_boxes):
                        obj_detections.append({"label": label, "box": b})

        unknown_regions = detect_large_moving_regions(frame, bg_subtractor)
        for region in unknown_regions:
            already   = any(bbox_iou(region, d["box"]) > 0.3 for d in obj_detections)
            is_person = any(bbox_iou(region, pb) > 0.3 for pb in raw_person_boxes)
            if not already and not is_person:
                obj_detections.append({"label": "unknown_object", "box": region})

        active_tracks = obj_tracker.update(obj_detections, frame_id)

        frame_buffer.append((frame_id, frame.copy()))
        if len(frame_buffer) > max_buffer:
            frame_buffer.pop(0)

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0
        if timestamp_ms==0:
            timestamp_sec = time.time()  # seconds since epoch
        
        dt = timestamp_sec - prev_timestamp

        # Get current frame position (0-indexed)
        frame_id +=1
        print(f"Frame ID: {frame_id}")
        fall_ongoing = False

        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            # print(keypoints_conf)
            ids = result.boxes.id
            if ids is None:
                continue
            ids = ids.cpu().numpy()
            # print(ids)

            for i, kps in enumerate(keypoints):
                box = boxes[i]
                person_id = int(ids[i])
                confs = keypoints_conf[i]
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area < min_area:
                    continue

                left_shoulder = kps[5]
                right_shoulder = kps[6]

                shoulder_conf = (confs[5] + confs[6])/2
                if shoulder_conf<0.5:
                    continue

                left_hip = kps[11]
                right_hip = kps[12]
                hip_conf = (confs[11] + confs[12])/2
                if hip_conf<0.5:
                    continue

                nose = kps[0]
                left_eye = kps[1]
                right_eye = kps[2]

                head_y = np.mean([nose[1], left_eye[1], right_eye[1]])

                # Compute torso center
                shoulder_center = (left_shoulder + right_shoulder) / 2
                # print("Shoulder center confidence: ",shoulder_center[2])
                hip_center = (left_hip + right_hip) / 2

                # Identifying and storing info about the people in the frame
                if person_id not in people:
                    # INSERT_YOUR_CODE
                    frame_filename = os.path.join(people_output_dir, f"{person_id}_{int(frame_id)}.jpg")
                    person_image = frame_org.copy()
                    # Draw bounding box & person
                    cv2.rectangle(person_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(person_image, tuple(shoulder_center.astype(int)), 5, (255,0,0), -1)
                    cv2.circle(person_image, tuple(hip_center.astype(int)), 5, (0,255,0), -1)
                    cv2.putText(
                            person_image,
                            str(person_id),
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255,255,255), 2
                        )
                    print(frame_filename)
                    saved = cv2.imwrite(frame_filename, person_image)
                    print(saved)
                    people[person_id] = {
                        'falls': [],
                        'frames':[],
                        'fall_ongoing': False,
                        'fallen':False,
                        'fall_frame':0,
                        'fall_done':0,
                        'angle_changed':False,
                        'prev_angle':0,
                        'prev_head':head_y,
                        'fall_angle_data':[],
                        'angle_changes':[],
                        'horizontal':False,
                        'vertical':False,
                        'image':frame_filename,
                    }

                # Draw bounding box & person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, tuple(shoulder_center.astype(int)), 5, (255,0,0), -1)
                cv2.circle(frame, tuple(hip_center.astype(int)), 5, (0,255,0), -1)
                cv2.putText(
                        frame,
                        str(person_id),
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255,255,255), 2
                    )


                # Compute vertical drop
                torso_length = np.linalg.norm(shoulder_center - hip_center)
                if torso_length > 0:

                    # TORSO ANGLE 
                    dx = shoulder_center[0] - hip_center[0]
                    dy = shoulder_center[1] - hip_center[1]

                    angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
                    angle_change = abs(angle - people[person_id]['prev_angle'])
                    people[person_id]['prev_angle'] = angle
                    
                    is_horizontal = angle > 40
                    people[person_id]['horizontal']=is_horizontal

                    # BOX RATIO
                    box_ratio = (x2 - x1) / (y2 - y1)

                    is_wide = box_ratio > 1.2

                    # HEAD LOW
                    head_low = head_y > frame_height * 0.7

                    is_vertical = angle < 5 and not head_low
                    people[person_id]['vertical'] = is_vertical

                    head_change = head_y - people[person_id]['prev_head']
                    people[person_id]['prev_head'] = head_y

                    angle_changed = angle_change > 10
                    # people[person_id]['angle_changed'] = angle_changed

                    is_fall = is_horizontal and is_wide and head_low 
                    print("Head change and angle change: ", head_change, angle_change)

                    if angle_change > 10 or head_change < -5:
                        people[person_id]['fall_ongoing'] = True
                    if people[person_id]['fall_ongoing']:
                        print("falling")
                        people[person_id]['fall_angle_data'].append({
                            'timestamp':float(timestamp_sec),
                            'frame_id':frame_id,
                            'horizontal':str(is_horizontal),
                            'vertical':str(is_vertical),
                            'angle':float(angle),
                            'angle_change':float(angle_change),
                            'head_y':float(head_y),
                            'shoulder_y':float(shoulder_center[1]),
                            'head_change':float(head_change),
                            'fall_ongoing':str(people[person_id]['fall_ongoing']),
                            # 'frame':frame.copy()
                            })
                        
                    
                    if is_vertical:
                        if people[person_id]['fall_ongoing']:
                            print("Back up")
                        if people[person_id]['fallen']:
                            print("Recovered")
                            people[person_id]['fallen'] = False
                            recovery = True
                            with open("recoveries.txt",'a') as file:
                                file.write(f"{person_id} Recovered \n")
                        people[person_id]['angle_changed'] = False
                        people[person_id]['fall_ongoing'] = False
                        people[person_id]['fall_angle_data'] = []

                    # people[person_id]['angle_changes'].append((angle, angle_change, head_change, is_fall))
                    people[person_id]['frames'].append({
                            'timestamp':float(timestamp_sec),
                            'frame_id':frame_id,
                            'horizontal':str(is_horizontal),
                            'vertical':str(is_vertical),
                            'angle':float(angle),
                            'angle_change':float(angle_change),
                            'head_y':float(head_y),
                            'shoulder_y':float(shoulder_center[1]),
                            'head_change':float(head_change),
                            'fall_ongoing':str(people[person_id]['fall_ongoing']),
                            'frame':frame.copy()
                            })

                    if people[person_id]['horizontal'] and people[person_id]['fallen']:
                        continue

                    if is_fall:
                        print("Fell")
                        people[person_id]['fallen'] = True
                        people[person_id]['fall_frame'] = frame_id
                        fall_ongoing = True

                        frames = people[person_id]['frames'][-15:]

                        accel_head = []
                        # INSERT_YOUR_CODE
                        # Calculate acceleration of head from fall_angle_data
                        # We'll use the difference of head_change between consecutive frames divided by the time difference
                        if len(frames) > 2:
                            for i in range(1, len(frames)):
                                dt = frames[i]['timestamp'] - frames[i-1]['timestamp']
                                if dt == 0:
                                    accel_head.append(0)
                                else:
                                    delta_v = frames[i]['head_change'] - frames[i-1]['head_change']
                                    accel = delta_v / dt
                                    accel_head.append(accel)
                            accel_head = sum(accel_head)/len(accel_head)
                        else:
                            # Not enough data for meaningful acceleration
                            accel_head = '-'
                        


                        if people[person_id]['fall_angle_data'] ==[]:
                            people[person_id]['fall_angle_data'].append({
                                'timestamp':float(timestamp_sec),
                                'frame_id':frame_id,
                                'horizontal':str(is_horizontal),
                                'vertical':str(is_vertical),
                                'angle':float(angle),
                                'angle_change':float(angle_change),
                                'head_y':float(head_y),
                                'shoulder_y':float(shoulder_center[1]),
                                'head_change':float(head_change),
                                'fall_ongoing':str(people[person_id]['fall_ongoing']),
                                # 'frame':frame.copy()
                                })
                        fall_angle_data = people[person_id]['fall_angle_data']
                        dy = frames[-1]['timestamp'] - frames[0]['timestamp']
                        if dy == 0:
                            dy = 0.0001

                        # INSERT_YOUR_CODE
                        # Save video snippet of fall
                        fall_video_filename = os.path.join(fall_output_dir, f"{person_id}_{len(people[person_id]['falls'])}.mp4")
                        # Assume all frames same size/shape
                        example_frame = frames[0]['frame']
                        height, width = example_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 15  # or choose correct FPS for your data
                        out = cv2.VideoWriter(fall_video_filename, fourcc, fps, (width, height))
                        for item in frames:
                            out.write(item['frame'])
                        out.release()
                        
                        fall_data = {
                            'fall_id':len(people[person_id]['falls']),
                            'person_id':str(person_id),
                            'fall_frame':people[person_id]['fall_frame'],
                            'head_speed':(frames[0]['head_y'] - frames[-1]['head_y'])/dy,
                            'head_accel':accel_head,
                            'shoulder_speed':(frames[0]['shoulder_y'] - frames[-1]['shoulder_y'])/dy,
                            'fall_start':(frames[0]['frame_id'],frames[0]['timestamp']),
                            'fall_end':(frames[-1]['frame_id'],frames[-1]['timestamp']),
                            'fall_video':fall_video_filename,
                            'frames':fall_angle_data,
                        }
                        print(fall_data)
                        people[person_id]['falls'].append(fall_data)
                        fall_filename = str(person_id) + "_" + str(len(people[person_id]['falls'])) + '.json'
                        with open(os.path.join(fall_output_dir, fall_filename), 'w') as file:
                            json.dump(json.dumps(fall_data), file, indent=4)
                        people[person_id]['fall_angle_data'] = []
                        people[person_id]['fall_ongoing'] = False
                        label = "FALL [{}°]".format(angle)
                        color = (0,0,255)
                    elif angle_changed:
                        label = "FALLING [{}°]".format(angle)
                        color = (0, 165, 255)
                    else:
                        label = "No Fall [{}°]".format(angle)
                        color = (0,255,0)
                    # if people[person_id]['horizontal']:
                    #     fall_done = people[person_id]['fall_done']
                    #     if fall_done>10:
                    #         fall_ongoing = False
                    #         people[person_id]['fall_ongoing'] = False
                    #         people[person_id]['angle_changed'] = False
                    #         fall_angle_data = people[person_id]['fall_angle_data']
                    #         fall_data = {
                    #             'person_id':str(person_id),
                    #             'angle_changes':fall_angle_data,
                    #             'fall_frame':people[person_id]['fall_frame'],
                    #             # 'fall_start':(fall_angle_data[0]['frame_id'],fall_angle_data[0]['timestamp']),
                    #             # 'fall_end':(fall_angle_data[-1]['frame_id'],fall_angle_data[-1]['timestamp'])
                    #         }
                    #         people[person_id]['falls'].append(fall_data)
                    #         fall_filename = str(person_id) + "_" + str(len(people[person_id]['falls']) + '.json')
                    #         with open(os.path.join(fall_output_dir, fall_filename), 'w') as file:
                    #             json.dump(json.dumps(fall_data), file, indent=4)
                    #         people[person_id]['fall_angle_data'] = []
                    #         people[person_id]['fall_done'] = 0
                    #     else:
                    #         people[person_id]['fall_done'] +=1

                    cv2.putText(
                        frame,
                        label,
                        (x1-5, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Draw falling objects ──────────────────────────────────────────────────
        for track in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track.box]
            if track.is_falling:
                color = (0, 0, 220)
                label = f"FALLING OBJ: {track.label}"
            elif track.label == "unknown_object":
                color = (0, 200, 255)
                label = f"unknown [{track.track_id}]"
            else:
                color = (160, 160, 160)
                label = f"{track.label} [{track.track_id}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # ── Check each track for falling object ──────────────────────────────────
            for track in active_tracks:
                if track.fall_logged:
                    continue

                vy = track.get_velocity_y()
                fall_time = round(frame_id / fps, 2)

                if vy > MIN_FALL_VELOCITY:
                    track.fall_velocity.append(vy)
                else:
                    track.fall_velocity = []

                if (len(track.fall_velocity) >= FALL_CONFIRM_FRAMES
                        and not track.is_falling
                        and fall_time - last_fall_sec > FALL_COOLDOWN_SEC):

                    track.is_falling  = True
                    track.fall_frame  = frame_id
                    track.fall_logged = True
                    last_fall_sec     = fall_time

                    # Did a person cause it?
                    person_caused    = False
                    caused_by_person = None
                    for (hf, hcx, hcy, hbox, _) in track.history:
                        if frame_id - hf > CAUSE_LOOKBACK_FRAMES:
                            continue
                        for pb in raw_person_boxes:
                            if center_dist(hbox, pb) < CAUSE_PROXIMITY_PX:
                                person_caused    = True
                                caused_by_person = pb
                                break
                        if person_caused:
                            break

                    # Did it hit a person?
                    hit_person    = False
                    hit_person_id = None
                    for pb in raw_person_boxes:
                        if bbox_iou(track.box, pb) > HIT_IOU_THRESHOLD:
                            hit_person    = True
                            hit_person_id = pb
                            break

                    incident = {
                        "object"                   : track.label,
                        "track_id"                 : track.track_id,
                        "fell_at_frame"            : frame_id,
                        "fell_at_sec"              : fall_time,
                        "avg_velocity_px_per_frame": round(float(np.mean(track.fall_velocity)), 2),
                        "aspect_ratio_changed"     : track.aspect_ratio_changed(),
                        "person_caused"            : person_caused,
                        "hit_person"               : hit_person,
                    }
                    object_incidents.append(incident)

                    fall_detail = {
                        "object":        track.label,
                        "track_id":      track.track_id,
                        "fell_at_frame": frame_id,
                        "fell_at_sec":   fall_time,
                        "person_caused": person_caused,
                        "hit_person":    hit_person,
                        "avg_velocity_px_per_frame":    round(float(np.mean(track.fall_velocity)), 2),
                        "aspect_ratio_changed":         track.aspect_ratio_changed(),
                        "fall_velocities_px_per_frame": [round(v, 3) for v in track.fall_velocity],
                        "trajectory": [
                            {
                                "frame_idx":    h[0],
                                "center_x":     round(h[1], 2),
                                "center_y":     round(h[2], 2),
                                "box":          [round(v, 2) for v in h[3]],
                                "aspect_ratio": round(h[4], 3),
                            }
                            for h in track.history
                        ],
                    }
                    object_fall_details.append(fall_detail)

                    print(f"\n⚠️  FALLING OBJECT: {track.label} (track {track.track_id}) @ {fall_time}s")

                    # ── Save clip ─────────────────────────────────────────────────────
                    clip_path = os.path.join(fall_output_dir, f"obj_{track.label}_t{fall_time}s.mp4")
                    clip_writer = cv2.VideoWriter(
                        clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    clip_start = frame_id - int(CLIP_BUFFER_SEC * fps)
                    for (fi, bf) in frame_buffer:
                        if fi >= clip_start:
                            clip_writer.write(bf)
                    active_clips[track.track_id] = {
                        "writer"   : clip_writer,
                        "end_frame": frame_id + int(CLIP_BUFFER_SEC * fps),
                        "path"     : clip_path,
                    }

            # ── Write and close any active clips ─────────────────────────────────────
            for tid, clip in list(active_clips.items()):
                clip["writer"].write(frame)
                if frame_id >= clip["end_frame"]:
                    clip["writer"].release()
                    print(f"   obj clip saved: {clip['path']}")
                    del active_clips[tid]
                    
        if fall_ongoing:
            # Create a color overlay
            overlay = np.zeros_like(frame)
            overlay[:] = (0, 0, 255)  # Red tint

            # Blend: alpha controls transparency
            alpha = 0.3
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

            
        if recovery:
            overlay = np.zeros_like(frame)
            overlay[:] = (255, 0, 0) 

            alpha = 0.3
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        out.write(frame)
        print('here')
        cv2.imshow("Live Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
           break

    # for person_id in people:
    #     if people[person_id]['fall_ongoing']:
    #         fall_angle_data = people[person_id]['fall_angle_data']
    #         people[person_id]['falls'].append({
    #                                 'person_id':person_id,
    #                                 'angle_changes':fall_angle_data,
    #                                 'fall_start':(fall_angle_data[0]['frame_id'],fall_angle_data[0]['timestamp']),
    #                                 'fall_end':(fall_angle_data[-1]['frame_id'],fall_angle_data[-1]['timestamp'])
    #                             })
        # Store each person's photo 

    # ── Close any remaining clip writers ─────────────────────────────────────
    for tid, clip in active_clips.items():
        clip["writer"].release()

    # ── Save object incidents to JSON ─────────────────────────────────────────
    if object_incidents:
        incidents_path = os.path.join(fall_output_dir, "object_incidents.json")
        with open(incidents_path, 'w') as f:
            json.dump(object_incidents, f, indent=4)

    # ── Save detailed falling object data to JSON ──────────────────────────────
    if object_fall_details:
        details_path = os.path.join(fall_output_dir, "object_fall_details.json")
        with open(details_path, 'w') as f:
            json.dump(object_fall_details, f, indent=4)
    cap.release()
    out.release()
    print(f"Fall detection video saved to {output_path}")
    return output_path,people


output_dir, people = run_fall_detection_live(0, output_filename='test0.mp4')