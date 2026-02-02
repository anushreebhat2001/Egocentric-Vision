import h5py
import numpy as np
import cv2
import torch
from transformers import pipeline
from PIL import Image

# --- CONFIGURATION ---
H5_FILE = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.hdf5'
VIDEO_FILE = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.mp4'
OUTPUT_FILE = 'output_sensor_fusion.mp4'


# --- SETUP SKELETON (Same as before) ---
FINGER_NAMES = ['Index', 'Middle', 'Ring', 'Little']
def build_hand_map(side):
    hand_root = f'{side}Hand'
    connections = []
    connections.append((hand_root, f'{side}ThumbKnuckle'))
    connections.append((f'{side}ThumbKnuckle', f'{side}ThumbIntermediateBase'))
    connections.append((f'{side}ThumbIntermediateBase', f'{side}ThumbIntermediateTip'))
    connections.append((f'{side}ThumbIntermediateTip', f'{side}ThumbTip'))
    for finger in FINGER_NAMES:
        connections.append((hand_root, f'{side}{finger}FingerMetacarpal'))
        connections.append((f'{side}{finger}FingerMetacarpal', f'{side}{finger}FingerKnuckle'))
        connections.append((f'{side}{finger}FingerKnuckle', f'{side}{finger}FingerIntermediateBase'))
        connections.append((f'{side}{finger}FingerIntermediateBase', f'{side}{finger}FingerIntermediateTip'))
        connections.append((f'{side}{finger}FingerIntermediateTip', f'{side}{finger}FingerTip'))
    return connections
SKELETON_CONNECTIONS = build_hand_map('left') + build_hand_map('right')

def main():
    print("⏳ Initializing Mission Control Dashboard...")
    
    # 1. Load AI Depth
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device if device != 'mps' else -1)

    with h5py.File(H5_FILE, 'r') as f:
        K = f['camera']['intrinsic'][:]
        cap = cv2.VideoCapture(VIDEO_FILE)
        
        # Original Dims
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # --- LAYOUT DESIGN ---
        # Main Canvas: 1.5x width of original video
        canvas_w = w + int(w * 0.5)
        canvas_h = h
        out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h))

        frame_idx = 0
        total_frames = f['transforms']['camera'].shape[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames: break

            # --- A. PREPARE RGB + SKELETON (Main View) ---
            main_view = frame.copy()
            cam_pose = f['transforms']['camera'][frame_idx]
            view_matrix = np.linalg.inv(cam_pose)
            
            # Draw Skeleton on RGB (Should align better)
            for start, end in SKELETON_CONNECTIONS:
                if start in f['transforms'] and end in f['transforms']:
                    p1 = f['transforms'][start][frame_idx][:3, 3]
                    p2 = f['transforms'][end][frame_idx][:3, 3]
                    
                    # Project
                    def proj(p):
                        pc = view_matrix @ np.append(p, 1)
                        if pc[2] <= 0: return None
                        pix = K @ pc[:3]
                        return (int(pix[0]/pix[2]), int(pix[1]/pix[2]))
                    
                    u1, u2 = proj(p1), proj(p2)
                    if u1 and u2:
                        cv2.line(main_view, u1, u2, (0, 255, 0), 2)
                        cv2.circle(main_view, u2, 3, (0, 0, 255), -1)

            # --- B. PREPARE DEPTH MAP (Top Right) ---
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth_raw = np.array(depth_estimator(img_pil)["depth"])
            depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
            depth_color = cv2.applyColorMap(np.uint8(depth_norm), cv2.COLORMAP_MAGMA)
            
            # Resize for sidebar (half width of main, half height)
            sidebar_w = int(w * 0.5)
            sidebar_h = int(h * 0.5)
            depth_small = cv2.resize(depth_color, (sidebar_w, sidebar_h))
            
            # --- C. PREPARE DATA PANEL (Bottom Right) ---
            data_panel = np.zeros((sidebar_h, sidebar_w, 3), dtype=np.uint8)
            # Add some "Matrix" style text
            cv2.putText(data_panel, "ROBOT STATE:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate Real Metric Z
            try:
                z_val = f['transforms']['rightIndexFingerTip'][frame_idx][:3, 3][2] # Raw World Z
                # Or relative to camera if you prefer
                cv2.putText(data_panel, f"End Effector Z: {z_val:.3f}m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Fake "Grasp Confidence" (just for visuals)
                conf = np.mean(f['confidences']['rightHand'][frame_idx])
                cv2.putText(data_panel, f"Tracking Conf: {conf*100:.1f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except: pass

            # --- D. COMPOSE CANVAS ---
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Place Main View
            canvas[0:h, 0:w] = main_view
            
            # Place Sidebar Items
            canvas[0:sidebar_h, w:canvas_w] = depth_small
            canvas[sidebar_h:h, w:canvas_w] = data_panel
            
            # Draw Borders
            cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)
            cv2.line(canvas, (w, sidebar_h), (canvas_w, sidebar_h), (255, 255, 255), 2)

            out.write(canvas)
            print(f"Processed Frame {frame_idx}")
            frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Mission Control Video Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()