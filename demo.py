import torch
import cv2
import argparse
import json
import os
from tqdm import tqdm
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# Define finger connections
FINGER_CONNECTIONS = [
    # Thumb
    [1, 2, 3, 4],
    # Index
    [5, 6, 7, 8],
    # Middle
    [9, 10, 11, 12],
    # Ring
    [13, 14, 15, 16],
    # Pinky
    [17, 18, 19, 20]
]

# Define palm connections (MCP joints + wrist)
PALM_CONNECTIONS = [
    [0, 1],     # Wrist -> Thumb CMC
    [1, 5],     # Thumb CMC -> Index MCP
    [5, 9],     # Index MCP -> Middle MCP
    [9, 13],    # Middle MCP -> Ring MCP
    [13, 17],   # Ring MCP -> Pinky MCP
    [17, 0]     # Pinky MCP -> Wrist
]

def draw_hand_skeleton(frame, keypoints_2d, color):
    # Draw fingers
    for finger in FINGER_CONNECTIONS:
        for i in range(len(finger) - 1):
            start_point = tuple(map(int, keypoints_2d[finger[i]]))
            end_point = tuple(map(int, keypoints_2d[finger[i + 1]]))
            cv2.line(frame, start_point, end_point, color, 2)

    # Draw palm
    for connection in PALM_CONNECTIONS:
        start_point = tuple(map(int, keypoints_2d[connection[0]]))
        end_point = tuple(map(int, keypoints_2d[connection[1]]))
        cv2.line(frame, start_point, end_point, color, 2)

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a video for 3D hand pose estimation.")
parser.add_argument("video_path", type=str, help="Path to the video file")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
output_video_path = os.path.splitext(video_path)[0] + '_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
            # Get 2D keypoints
            keypoints_2d = output['wilor_preds']['pred_keypoints_2d'][0]  # (21, 2)
            is_right = output['is_right']
            
            # Draw skeleton
            color = (0, 0, 255) if is_right == 0.0 else (255, 0, 0)  # Red for left, Blue for right
            draw_hand_skeleton(frame, keypoints_2d, color)
            
            # Draw keypoints
            for kp in keypoints_2d:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, color, -1)

            # Convert keypoints array to list of xyz objects
            keypoints = output['wilor_preds']['pred_keypoints_3d'] # (1, 21, 3)
            global_orient = output['wilor_preds']['global_orient'] # (1, 1, 3)
            pred_cam_t_full = output['wilor_preds']['pred_cam_t_full'] # (1, 3)
            pred_cam = output['wilor_preds']['pred_cam'] # (1, 3)
            
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

            # Convert pred_cam_t_full to xyz dictionary
            cam_t_full = {
                'x': float(pred_cam_t_full[0][0]),
                'y': float(pred_cam_t_full[0][1]),
                'z': float(pred_cam_t_full[0][2])
            }

            # Convert pred_cam to xyz dictionary
            cam = {
                'x': float(pred_cam[0][0]),
                'y': float(pred_cam[0][1]),
                'z': float(pred_cam[0][2])
            }

            hand_data = {
                'joints': joints,
                'wrist_orientation': wrist_orient,
                'camera_translation_full': cam_t_full,
                'camera_parameters': cam
            }

            if output['is_right'] == 0.0:
                left_hands.append(hand_data)
            else:
                right_hands.append(hand_data)

        hand_poses[frame_number] = {
            'left_hands': left_hands,
            'right_hands': right_hands
        }

        out.write(frame)
        frame_number += 1
        pbar.update(1)

cap.release()
out.release()
cv2.destroyAllWindows()

# Output the results to a JSON file
output_path = os.path.join(os.path.dirname(video_path), 'hand_poses.json')
with open(output_path, 'w') as f:
    json.dump(hand_poses, f, indent=4)
