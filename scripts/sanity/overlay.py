import h5py
import numpy as np
import cv2

# --- CONFIGURATION ---
H5_FILE = '/Users/utx/Desktop/code/video_learning_samples/add_remove_lid/0.hdf5'

VIDEO_FILE = '/Users/utx/Desktop/code/video_learning_samples/add_remove_lid/0.mp4'
OUTPUT_FILE = 'output_overlay.mp4'

# Define the bone connections (Parent -> Child)
# Based on the specific keys found in your file
FINGER_NAMES = ['Index', 'Middle', 'Ring', 'Little']
SKELETON_MAP = []

def build_hand_map(side):
    """Generates the connectivity list for one hand (left or right)"""
    hand_root = f'{side}Hand'
    connections = []
    
    # 1. Thumb Chain (Thumb usually lacks a distinct Metacarpal key in some datasets, checking your list...)
    # Based on your file: Hand -> Knuckle -> IntermediateBase -> IntermediateTip -> Tip
    connections.append((hand_root, f'{side}ThumbKnuckle'))
    connections.append((f'{side}ThumbKnuckle', f'{side}ThumbIntermediateBase'))
    connections.append((f'{side}ThumbIntermediateBase', f'{side}ThumbIntermediateTip'))
    connections.append((f'{side}ThumbIntermediateTip', f'{side}ThumbTip'))

    # 2. Fingers Chain
    for finger in FINGER_NAMES:
        # Hand -> Metacarpal
        connections.append((hand_root, f'{side}{finger}FingerMetacarpal'))
        # Metacarpal -> Knuckle
        connections.append((f'{side}{finger}FingerMetacarpal', f'{side}{finger}FingerKnuckle'))
        # Knuckle -> IntermediateBase
        connections.append((f'{side}{finger}FingerKnuckle', f'{side}{finger}FingerIntermediateBase'))
        # IntermediateBase -> IntermediateTip
        connections.append((f'{side}{finger}FingerIntermediateBase', f'{side}{finger}FingerIntermediateTip'))
        # IntermediateTip -> Tip
        connections.append((f'{side}{finger}FingerIntermediateTip', f'{side}{finger}FingerTip'))
        
    return connections

# Build full map
SKELETON_CONNECTIONS = build_hand_map('left') + build_hand_map('right')

def project_point(world_pos, cam_pose, K):
    """projects 3D world coord to 2D pixels"""
    # 1. Transform World -> Camera
    view_matrix = np.linalg.inv(cam_pose)
    point_h = np.append(world_pos, 1) # Homogeneous
    point_cam = view_matrix @ point_h
    
    # 2. Filter points behind camera
    if point_cam[2] <= 0: return None

    # 3. Project Camera -> Image
    pixel_coords_homo = K @ point_cam[:3]
    
    # 4. Normalize
    u = int(pixel_coords_homo[0] / pixel_coords_homo[2])
    v = int(pixel_coords_homo[1] / pixel_coords_homo[2])
    return (u, v)

# --- MAIN LOOP ---
with h5py.File(H5_FILE, 'r') as f:
    K = f['camera']['intrinsic'][:]
    
    cap = cv2.VideoCapture(VIDEO_FILE)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print("Drawing Skeleton...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= 94: break
            
        cam_pose = f['transforms']['camera'][frame_idx]

        # Draw Bones
        for start_joint, end_joint in SKELETON_CONNECTIONS:
            # Ensure both keys exist in file
            if start_joint in f['transforms'] and end_joint in f['transforms']:
                
                # Get Positions
                p1_3d = f['transforms'][start_joint][frame_idx][:3, 3]
                p2_3d = f['transforms'][end_joint][frame_idx][:3, 3]
                
                # Project to 2D
                pt1 = project_point(p1_3d, cam_pose, K)
                pt2 = project_point(p2_3d, cam_pose, K)
                
                if pt1 and pt2:
                    # Color Logic: Left = Green, Right = Blue
                    color = (0, 255, 0) if 'left' in start_joint else (255, 100, 0)
                    
                    # Draw Line (Bone)
                    cv2.line(frame, pt1, pt2, color, 2)
                    # Draw Joint (Circle)
                    cv2.circle(frame, pt2, 4, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Skeleton video saved to {OUTPUT_FILE}")