import h5py
import numpy as np
import cv2

# --- CONFIGURATION ---
H5_FILE = '/Users/utx/Desktop/code/video_learning_samples/open_close/2.hdf5'

VIDEO_FILE = '/Users/utx/Desktop/code/video_learning_samples/open_close/2.mp4'
OUTPUT_FILE = 'output_overlay.mp4'

# We will focus on the Right Hand (the one unscrewing the lid)
JOINT_NAME = 'rightHand'
AXIS_LENGTH = 0.1  # Length of the arrows in meters (10cm)

def project_point(world_pos, cam_pose, K):
    """Standard 3D -> 2D projection we used before"""
    view_matrix = np.linalg.inv(cam_pose)
    point_h = np.append(world_pos, 1)
    point_cam = view_matrix @ point_h
    
    if point_cam[2] <= 0: return None # Behind camera

    pixel_coords_homo = K @ point_cam[:3]
    u = int(pixel_coords_homo[0] / pixel_coords_homo[2])
    v = int(pixel_coords_homo[1] / pixel_coords_homo[2])
    return (u, v)

with h5py.File(H5_FILE, 'r') as f:
    K = f['camera']['intrinsic'][:]
    cap = cv2.VideoCapture(VIDEO_FILE)
    
    # Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Visualizing Rotation Matrix for {JOINT_NAME}...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= 94: break
            
        cam_pose = f['transforms']['camera'][frame_idx]
        
        if JOINT_NAME in f['transforms']:
            # 1. GET THE FULL 4x4 MATRIX
            mat = f['transforms'][JOINT_NAME][frame_idx]
            
            # 2. EXTRACT POSITION (The "Origin" of the arrows)
            origin_3d = mat[:3, 3]
            
            # 3. EXTRACT ROTATION COLUMNS (The Axes directions)
            # Column 0 = Local X (Red), Column 1 = Local Y (Green), Column 2 = Local Z (Blue)
            x_axis_vec = mat[:3, 0]
            y_axis_vec = mat[:3, 1]
            z_axis_vec = mat[:3, 2]
            
            # 4. CALCULATE END POINTS OF ARROWS
            # origin + (direction * length)
            end_x_3d = origin_3d + (x_axis_vec * AXIS_LENGTH)
            end_y_3d = origin_3d + (y_axis_vec * AXIS_LENGTH)
            end_z_3d = origin_3d + (z_axis_vec * AXIS_LENGTH)
            
            # 5. PROJECT TO 2D
            origin_2d = project_point(origin_3d, cam_pose, K)
            end_x_2d = project_point(end_x_3d, cam_pose, K)
            end_y_2d = project_point(end_y_3d, cam_pose, K)
            end_z_2d = project_point(end_z_3d, cam_pose, K)
            
            if origin_2d:
                # Draw Center Dot
                cv2.circle(frame, origin_2d, 5, (255, 255, 255), -1)
                
                # Draw Axes (BGR Colors in OpenCV)
                if end_x_2d: cv2.arrowedLine(frame, origin_2d, end_x_2d, (0, 0, 255), 2)  # Red (X)
                if end_y_2d: cv2.arrowedLine(frame, origin_2d, end_y_2d, (0, 255, 0), 2)  # Green (Y)
                if end_z_2d: cv2.arrowedLine(frame, origin_2d, end_z_2d, (255, 0, 0), 2)  # Blue (Z)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("✅ Done! Watch the output video to see the axes twist.")