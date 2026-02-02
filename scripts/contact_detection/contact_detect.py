import cv2
import pandas as pd
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R

def find_intrinsics(group):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset) and item.shape == (3, 3):
            return item[()]
        elif isinstance(item, h5py.Group):
            res = find_intrinsics(item)
            if res is not None: return res
    return None

def project_3d_to_2d(p_world, cam_pos, cam_rot_euler, K):
    r = R.from_euler('xyz', cam_rot_euler, degrees=True).as_matrix()
    t = np.array(cam_pos).reshape(3, 1)
    T_cam_to_world = np.eye(4)
    T_cam_to_world[:3, :3] = r
    T_cam_to_world[:3, 3] = t.flatten()
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    p_w_homo = np.append(p_world, 1.0)
    p_cam = T_world_to_cam @ p_w_homo
    if p_cam[2] <= 0: return None 
    coords_2d = K @ p_cam[:3]
    return (int(coords_2d[0] / coords_2d[2]), int(coords_2d[1] / coords_2d[2]))

def create_color_wash_video(video_path, h5_path, hand_csv, cam_csv, output_path):
    df_hand = pd.read_csv(hand_csv)
    df_cam = pd.read_csv(cam_csv)

    with h5py.File(h5_path, 'r') as h5:
        K = find_intrinsics(h5)

    # Calculate states
    df_hand['velocity'] = np.sqrt(df_hand['rightHand_x'].diff()**2 + df_hand['rightHand_y'].diff()**2 + df_hand['rightHand_z'].diff()**2) / (1/30.0)
    df_hand['aperture'] = np.sqrt((df_hand['rightThumbTip_x']-df_hand['rightIndexFingerTip_x'])**2 + (df_hand['rightThumbTip_y']-df_hand['rightIndexFingerTip_y'])**2 + (df_hand['rightThumbTip_z']-df_hand['rightIndexFingerTip_z'])**2)
    
    t_contact = int(df_hand[df_hand['aperture'] < df_hand['aperture'].quantile(0.3)]['velocity'].idxmin())
    after_contact = df_hand[df_hand['frame'] > t_contact]
    t_goal = int(after_contact[after_contact['rightHand_y'].diff() > 0.001]['frame'].iloc[0])

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

    for idx in range(len(df_hand)):
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Logic for State and Color
        if idx < t_contact:
            state, color = "SEARCHING", (0, 255, 255) # Yellow
        elif t_contact <= idx < t_goal:
            state, color = "MANIPULATING", (0, 0, 255) # Red
        else:
            state, color = "GOAL ACHIEVED", (0, 255, 0) # Green

        # 2. CREATE THE SCREEN TINT
        # Create a solid color overlay
        overlay = np.full(frame.shape, color, dtype=np.uint8)
        # Blend the frame with the overlay (0.8 alpha for frame, 0.2 for color tint)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # 3. Add HUD Text
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"SYSTEM STATE: {state}", (30, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

        # 4. Optional: Keep the projection box for precision
        p_hand = df_hand.iloc[idx][['rightHand_x', 'rightHand_y', 'rightHand_z']].values
        c_pos = df_cam.iloc[idx][['pos_x', 'pos_y', 'pos_z']].values
        c_rot = df_cam.iloc[idx][['roll', 'pitch', 'yaw']].values
        pixel_pos = project_3d_to_2d(p_hand, c_pos, c_rot, K)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Color-coded state video saved: {output_path}")

# Run the script
create_color_wash_video(
    "video_learning_samples/add_remove_lid/0.mp4", 
    "video_learning_samples/add_remove_lid/0.hdf5", 
    "outputs/extracted_metrics_csv/add_remove_lid/0_hand_poses.csv", 
    "outputs/extracted_metrics_csv/add_remove_lid/0_camera_trajectory.csv",  
    "outputs/contact_detect/contact_detect_viz.mp4"
)