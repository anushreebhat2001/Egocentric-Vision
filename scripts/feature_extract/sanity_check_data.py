import h5py
import pandas as pd
import numpy as np

# 1. Load your Data
h5_path = "video_learning_samples/add_remove_lid/1.hdf5"
csv_cam_path = "extracted_metrics_csv/add_remove_lid/0_camera_trajectory.csv"
csv_hand_path = "extracted_metrics_csv/add_remove_lid/0_hand_poses.csv"

# 2. Extract from HDF5
with h5py.File(h5_path, 'r') as h5:
    # Camera Position (Frame 0)
    h5_cam = h5['transforms/camera'][0][:3, 3]
    # Right Hand Position (Frame 0)
    h5_hand = h5['transforms/rightHand'][0][:3, 3]

# 3. Extract from CSV
df_cam = pd.read_csv(csv_cam_path)
csv_cam = df_cam.iloc[0][['pos_x', 'pos_y', 'pos_z']].values

df_hand = pd.read_csv(csv_hand_path)
csv_hand = df_hand.iloc[0][['rightHand_x', 'rightHand_y', 'rightHand_z']].values

# 4. Print & Compare
print("--- CAMERA TRAJECTORY CHECK ---")
print(f"HDF5: {h5_cam}")
print(f"CSV:  {csv_cam}")
print(f"Match: {np.allclose(h5_cam, csv_cam)}")

print("\n--- HAND POSE CHECK (Right Hand) ---")
print(f"HDF5: {h5_hand}")
print(f"CSV:  {csv_hand}")
print(f"Match: {np.allclose(h5_hand, csv_hand)}")