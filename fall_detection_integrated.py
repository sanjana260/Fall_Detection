from numpy._core.numeric import True_
from sympy.logic import true
from sympy.ntheory import continued_fraction, continued_fraction_convergents
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import math
import os
import torch
import json
import time
mps_device = torch.device('mps')
import argparse
import time
from llm import llm
from hazard_module import HazardModule
from hazard.types import PersonTrack
from hazard.water_detector import WaterConfig

from hazard.test_video_hazards import *

import cv2

from integration1 import *
from experiment import *
from scene_context.scene_context import *
from ultralytics.trackers import BYTETracker
# Configure tracker
tracker_params = {
    "max_age": 60,
    "min_hits": 3,
    "iou_threshold": 0.5,
    "match_thresh": 0.9,
}
model = YOLO("yolov8l-pose.pt").to(mps_device)
obj_model  = YOLO("yolov8l.pt").to(mps_device)

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

def _fmt_topk(topk):
    return ", ".join([f"{d['label']}:{d['prob']:.2f}" for d in topk])

output_path = os.path.join("Outputs", 'vigilens_live.mp4')
people_images_folder = os.path.splitext('vigilens_live.mp4')[0]
people_output_dir = os.path.join('Outputs', people_images_folder + "/people")
if os.path.exists(people_output_dir):
    shutil.rmtree(people_output_dir)
os.makedirs(people_output_dir)

fall_output_dir = os.path.join('Outputs', people_images_folder + "/falls")
if os.path.exists(fall_output_dir):
    shutil.rmtree(fall_output_dir)
os.makedirs(fall_output_dir)

json_dir = os.path.join('Outputs', people_images_folder, "objects")
print(json_dir)
if os.path.exists(json_dir):
    shutil.rmtree(json_dir)
os.makedirs(json_dir)


def run_fall_detection_live(input_video_path, min_area = 2000):
    # global fall_output_dir, json_dir, output_path, people_output_dir, people_images_folder
    """
    Runs fall detection on a given video and writes the annotated output to the 'Outputs' folder.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_name (str): (Optional) Name of the output video file. Defaults to 'fall_detection_output.mp4'.

    Returns:
        str: Path to the saved annotated video.
    """
    os.makedirs("Outputs", exist_ok=True)
    print('START')
    # input_video_filename = os.path.basename(input_video_path)
    

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_AVFOUNDATION)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ############ EXPERIMENT.py ############
    script_dir = Path(__file__).parent
    # pose_model = YOLO(script_dir / "yolov8n-pose.pt")
    print("Models loaded.")

    # Output directory — created now so snapshots can be saved during the loop
    video_stem = Path(VIDEO_PATH).stem
    # json_dir   = script_dir / "jsondescriptions"
    # json_dir.mkdir(exist_ok=True)

    # cap   = cv2.VideoCapture(VIDEO_PATH)
    # fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    # total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Video: {VIDEO_PATH}  |  {total} frames @ {fps:.1f}fps\n")

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
    sitting_confidence = {}

    frame_idx = 0
    ##################################################

    ####################### integration1.py #########################
    obj_tracker = ObjectTracker()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    object_incidents = []
    object_fall_details = []
    last_fall_sec = -999.0
    frame_buffer = []
    active_clips = {}
    max_buffer = int(CLIP_BUFFER_SEC * fps) * 2 + 10
    #################################################################

    ######################### Scene context ###########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or path to a video file")
    parser.add_argument("--every_n", type=int, default=10, help="run scene model every N frames")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    ctx = SceneContext(device=args.device)
    ctx.reset()

    last_pred = None
    frame_i = 0
    t0 = time.perf_counter()
    # fps = 0.0
    last_print_scene = None
    last_print_io = None
    #######################################################################

    # Ensure Outputs directory exists

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    people = {}
    prev_timestamp = 0
    frame_id=0

    ################## Water detection ##################
    # hz = HazardModule(
    #     water_cfg=WaterConfig(
    #         mode="heuristic",     # "auto" will try GDINO+SAM then fallback, but your GDINO class currently raises
    #         device="cpu",
    #         min_area=500,
    #     )
    # )
    

    
    ######################################################
    while cap.isOpened():
        ret, frame = cap.read()
        recovery = False
        if not ret:
            break


        frame_org = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        ###################### Water detection ###########################
        # result = hz.process_frame(frame, people, t = frame_id/fps, fall_events=None)

        # vis = frame.copy()

        # for det in result.hazards:
        #     if (det.hazard_type or "").lower() != "water":
        #         continue

        #     mask = det.mask
        #     vis = overlay_mask(vis, mask, alpha=0.35)

        #     bbox = det.bbox
        #     if bbox is None and mask is not None:
        #         bbox = mask_to_bbox(mask)

        #     sev = getattr(det, "severity", "unknown")
        #     ar = getattr(det, "area_ratio", None)

        #     if bbox is not None:
        #         x1, y1, x2, y2 = map(int, bbox)
        #         cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #         if ar is None:
        #             label = f"water sev={sev}"
        #         else:
        #             label = f"water sev={sev} ar={float(ar):.4f}"

        #         cv2.putText(
        #             frame,
        #             label,
        #             (x1, max(0, y1 - 8)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6,
        #             (255, 0, 0),
        #             2,
        #         )
        # # Optional: show scene alert text
        # if "water" in result.scene_alerts:
        #     cv2.putText(
        #         frame,
        #         f"SCENE ALERT: WATER ({result.scene_alerts['water']})",
        #         (20, 35),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1.0,
        #         (255, 0, 0),
        #         2,
        #     )
        ###################################################################
        # results = model(frame)
        results = model.track(frame, persist=True, tracker='./configs/bytetrack.yaml' , conf=0.35)

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0
        if timestamp_ms==0:
            timestamp_sec = time.time()  # seconds since epoch
        
        dt = timestamp_sec - prev_timestamp

        # Get current frame position (0-indexed)
        frame_id +=1
        print(f"Frame ID: {frame_id}")
        fall_ongoing = False

        ################ Experiment.py #####################
        # ── Detection ─────────────────────────────────────────────────────────
        obj_results  = obj_model(frame, verbose=False)[0]
        # pose_results = pose_model(frame, verbose=False)[0]

        # Build object list — skip persons, skip anything that heavily
        # overlaps a detected person (avoids chair being labelled as person)
        person_boxes = [b.xyxy[0].tolist() for b in results[0].boxes] \
                       if results[0].boxes is not None else []
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
        ###################################################

        ######################## integration1.py ####################
        raw_person_boxes = []
        obj_detections = []
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
        ##############################################################

        #################### Scene context ################################
        if frame_i % args.every_n == 0 or last_pred is None:
            start = time.perf_counter()
            last_pred = ctx.predict_bgr(frame, topk=args.topk)
            infer_ms = (time.perf_counter() - start) * 1000.0
            last_pred["infer_ms"] = infer_ms

            io = last_pred["indoor_outdoor"]
            scene = last_pred["scene_type"]

            # Print only if it changed (reduces spam). Comment this if you want always.
            if io != last_print_io or scene != last_print_scene:
                print(
                    f"[scene] io={last_pred['indoor_outdoor']}({last_pred['io_conf']:.2f}) "
                    f"scene={last_pred['scene_type']} raw={last_pred['scene_type_raw']}({last_pred['scene_conf_raw']:.2f}) "
                    f"top3={_fmt_topk(last_pred['topk'][:3])} "
                    f"device={last_pred['device']} infer_ms={infer_ms:.1f}"
                )
                last_print_io = io
                last_print_scene = scene

        # FPS estimate
        # dt = time.perf_counter() - t0
        # if dt >= 1.0:
        #     fps = frame_i / dt
        #     t0 = time.perf_counter()
        #     frame_i = 0

        # Overlay
        if last_pred is not None:
            text1 = f"{last_pred['indoor_outdoor']} ({last_pred['io_conf']:.2f})"
            text2 = f"{last_pred['scene_type']} ({last_pred['scene_conf']:.2f})"
            text3 = f"device={last_pred['device']} infer={last_pred.get('infer_ms', 0.0):.1f}ms FPS~{fps:.1f}"

            cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, text3, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #######################################################

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
                        'image':frame_filename
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
                            'head_y': float(head_y),
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
                            del item['frame']
                        out.release()
                        fall_filename = str(person_id) + "_" + str(len(people[person_id]['falls'])) + '.json'
                        fall_data = {
                            'fall_id':len(people[person_id]['falls']),
                            'person_id':str(person_id),
                            'fall_frame':people[person_id]['fall_frame'],
                            'head_speed':round((frames[0]['head_y'] - frames[-1]['head_y'])/dy,2),
                            'head_accel':round(accel_head,2) if type(accel_head)==float else "-",
                            'shoulder_speed':round((frames[0]['shoulder_y'] - frames[-1]['shoulder_y'])/dy,2),
                            'fall_start':(frames[0]['frame_id'],frames[0]['timestamp']),
                            'fall_end':(frames[-1]['frame_id'],frames[-1]['timestamp']),
                            'fall_video':fall_video_filename,
                            'frames':frames,
                            'fall_json':fall_filename,
                            'scene_context':{
                                'Indoor/Outdoor':last_pred['indoor_outdoor'],
                                'Scene Type':last_pred['scene_type'],
                            }
                        }
                        print(fall_data)
                        people[person_id]['falls'].append(fall_data)
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

                    cv2.putText(
                        frame,
                        label,
                        (x1-5, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    ############### Experiment.py ####################
                    tid  =  person_id
                    print(box)
                    # pbox = result.boxes[i].xyxy.tolist()
                    pbox = box

                    # Get keypoints for this person
                    kp_xy   = None
                    kp_conf = None
                    if result.keypoints is not None and i < len(result.keypoints.xy):
                        kp_xy   = result.keypoints.xy[i].cpu().numpy()
                        kp_conf = result.keypoints.conf[i].cpu().numpy()

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
                    if is_horizontal:

                        horiz_counts[tid]   += 1
                        upright_counts[tid]  = 0
                    else:
                        upright_counts[tid] += 1
                        horiz_counts[tid]    = 0

                    # ── Pre-fall posture (freeze when horiz_counts goes to 0) ────
                    if horiz_counts[tid] == 0:
                        if person_is_sitting:
                            sitting_confidence[tid] = sitting_confidence.get(tid, 0) + 1
                        else:
                            sitting_confidence[tid] = max(0, sitting_confidence.get(tid, 0) - 1)

                        if sitting_confidence.get(tid, 0) >= 5:
                            pre_fall_posture[tid] = "sitting"
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
                                cv2.imwrite(os.path.join(json_dir, fname), snap)
                                wf_snap_paths[tid] = fname

                            elif (fall_phase == "AFTER_FALL"
                                and tid not in after_fall_snapped
                                and tid in active_fall_incidents):
                                snap = draw_touch_snapshot(
                                    frame, pbox, detected_objects, kp_xy, kp_conf,
                                    f"AFTER FALL — Person {tid} @ {round(frame_idx/fps,2)}s")
                                fname = f"{video_stem}_p{tid}_after_fall.jpg"
                                cv2.imwrite(os.path.join(json_dir, fname), snap)
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
                            cv2.imwrite(os.path.join(json_dir, fc_fname), vis_fc)
                            hit_info["snapshot"] = fc_fname

                            first_floor_hit[tid] = hit_info
                            # Also update the incident if it has already been fired
                            if tid in active_fall_incidents:
                                active_fall_incidents[tid]["first_floor_contact"] = hit_info

                    # ── New fall event ────────────────────────────
                    if (tid not in fallen_ids
                            and people[tid]['fallen']
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
                            cv2.imwrite(os.path.join(json_dir, dur_snap), snap)

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
                        with open(os.path.join(fall_output_dir, people[tid]['falls'][-1]['fall_json']), 'r') as file:
                            fall_outputs_stored = json.loads(json.load(file))
                        print(fall_outputs_stored)
                        def convert_to_str(obj):
                            if isinstance(obj, dict):
                                return {str(k): convert_to_str(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_to_str(i) for i in obj]
                            elif obj is None:
                                return "None"
                            else:
                                return str(obj)

                        fall_outputs_stored['fall_object_interaction'] = convert_to_str({
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
                        # INSERT_YOUR_CODE
                        # Write the updated dictionary with added fall_object_interaction to the same JSON file
                        json_path = os.path.join(fall_output_dir, people[tid]['falls'][-1]['fall_json'])
                        # summary = llm(json_path)
                        # fall_outputs_stored['Summary'] = summary
                        with open(os.path.join(fall_output_dir, people[tid]['falls'][-1]['fall_json']), 'w') as file:
                            json.dump(convert_to_str(fall_outputs_stored), file, indent=4)
                            
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
                            
                    ######################################################################

        
        #################### Experiment.py #####################

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
        ##########################################################################

        ########################### integration.py #################################
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
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

             # ── Save object incidents to JSON ─────────────────────────────────────────
            if object_incidents:
                incidents_path = os.path.join(json_dir, "object_incidents.json")
                with open(incidents_path, 'w') as f:
                    json.dump(object_incidents, f, indent=4)
        #########################################################


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
    cap.release()
    out.release()
    print(f"Fall detection video saved to {output_path}")
    return output_path,people


# output_dir, people = run_fall_detection_live(0, output_filename='test0.mp4')