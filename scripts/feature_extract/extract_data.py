import os
import h5py
import csv
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def extract_to_csv(h5_path, output_root):
    """Extracts 6-DoF Ego-motion and 3D Hand Joints to separate CSVs."""
    h5 = h5py.File(h5_path, 'r')
    task_name = os.path.basename(os.path.dirname(h5_path))
    file_id = os.path.splitext(os.path.basename(h5_path))[0]
    
    # Create Task Folder in the results directory
    task_out_dir = os.path.join(output_root, task_name)
    os.makedirs(task_out_dir, exist_ok=True)

    # --- 1. Extract Camera Trajectory (Ego-motion) ---
    cam_csv = os.path.join(task_out_dir, f"{file_id}_camera_trajectory.csv")
    cam_transforms = h5['transforms/camera']
    
    with open(cam_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw'])
        for i, mat in enumerate(cam_transforms):
            pos = mat[:3, 3]
            rot = R.from_matrix(mat[:3, :3]).as_euler('xyz', degrees=True)
            writer.writerow([i, *pos, *rot])

    # --- 2. Extract 3D Hand Joints (21 Joints per hand) ---
    hand_csv = os.path.join(task_out_dir, f"{file_id}_hand_poses.csv")
    
    # Identify all hand joint keys present in this file
    joint_keys = sorted([k for k in h5['transforms'].keys() if any(side in k for side in ['left', 'right'])])
    
    with open(hand_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: frame, joint1_x, joint1_y, joint1_z, joint2_x...
        header = ['frame']
        for k in joint_keys:
            header.extend([f"{k}_x", f"{k}_y", f"{k}_z"])
        writer.writerow(header)
        
        num_frames = cam_transforms.shape[0]
        for i in range(num_frames):
            row = [i]
            for k in joint_keys:
                pos = h5[f'transforms/{k}'][i][:3, 3]
                row.extend(pos)
            writer.writerow(row)

    h5.close()

def run_extraction_batch(base_path):
    output_root = "extracted_metrics_csv"
    tasks = ['add_remove_lid', 'assemble_disassemble_furniture_bench_stool', 'basic_pick_and_place', 'open_close']
    
    print(f"Starting Data Extraction into {output_root}...")
    
    for task in tasks:
        task_path = os.path.join(base_path, task)
        if not os.path.exists(task_path): continue
        
        h5_files = [f for f in os.listdir(task_path) if f.endswith('.hdf5')]
        for h5_file in tqdm(h5_files, desc=f"Processing {task}"):
            extract_to_csv(os.path.join(task_path, h5_file), output_root)

# --- EXECUTION ---
run_extraction_batch("./")