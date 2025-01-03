import torch
import cv2
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a video for 3D hand pose estimation.")
parser.add_argument("video_path", type=str, help="Path to the video file")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0
hand_poses = {}

with tqdm(total=frame_count, desc="Processing video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = pipe.predict(frame_rgb)

        left_hands = []
        right_hands = []

        for output in outputs:
            # Convert keypoints array to list of xyz objects
            keypoints = output['wilor_preds']['pred_keypoints_3d']
            global_orient = output['wilor_preds']['global_orient']
            
            # Convert nested arrays to list of xyz dictionaries
            joints = []
            for joint in keypoints[0]:  # keypoints is triple nested
                joints.append({
                    'x': float(joint[0]),
                    'y': float(joint[1]),
                    'z': float(joint[2])
                })
            
            # Convert global_orient to xyz dictionary
            wrist_orient = {
                'x': float(global_orient[0][0][0]),
                'y': float(global_orient[0][0][1]),
                'z': float(global_orient[0][0][2])
            }

            hand_data = {
                'joints': joints,
                'wrist_orientation': wrist_orient
            }

            if output['is_right'] == 0.0:
                left_hands.append(hand_data)
            else:
                right_hands.append(hand_data)

        hand_poses[frame_number] = {
            'left_hands': left_hands,
            'right_hands': right_hands
        }

        frame_number += 1
        pbar.update(1)

cap.release()
cv2.destroyAllWindows()

# Output the results to a JSON file
output_path = os.path.join(os.path.dirname(video_path), 'hand_poses.json')
with open(output_path, 'w') as f:
    json.dump(hand_poses, f, indent=4)
