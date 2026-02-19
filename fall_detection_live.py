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
from ultralytics.trackers import BYTETracker
# Configure tracker
tracker_params = {
    "max_age": 60,
    "min_hits": 3,
    "iou_threshold": 0.5,
    "match_thresh": 0.9,
}
model = YOLO("yolov8l-pose.pt").to(mps_device)


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
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


output_dir, people = run_fall_detection_live(0, output_filename='test0.mp4')