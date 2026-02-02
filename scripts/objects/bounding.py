import h5py
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
H5_FILE = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.hdf5'
VIDEO_FILE = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.mp4'
OUTPUT_FILE = 'output_smart_bbox.mp4'

# Memory settings: How many frames to remember a lost object
MEMORY_PATIENCE = 60  # 60 frames (~2 seconds)

def project_wrist(f, frame_idx, K, cam_pose, side='right'):
    """Projects 3D wrist to 2D for Safety Bubble"""
    try:
        wrist_world = f['transforms'][f'{side}Hand'][frame_idx][:3, 3]
        view_matrix = np.linalg.inv(cam_pose)
        point_cam = view_matrix @ np.append(wrist_world, 1)
        if point_cam[2] <= 0: return None
        pix_homo = K @ point_cam[:3]
        return (int(pix_homo[0] / pix_homo[2]), int(pix_homo[1] / pix_homo[2]))
    except:
        return None

def main():
    print("⏳ Loading YOLOv8-Seg with ByteTrack...")
    model = YOLO('yolov8m-seg.pt') 

    # --- MEMORY SYSTEM ---
    # Stores last known position: { 'class_id': (mask_points, box, frames_since_seen) }
    object_memory = {} 

    with h5py.File(H5_FILE, 'r') as f:
        K = f['camera']['intrinsic'][:]
        cap = cv2.VideoCapture(VIDEO_FILE)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_idx = 0
        total_frames = f['transforms']['camera'].shape[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= total_frames: break

            # 1. RUN TRACKING (persist=True enables ID tracking)
            # classes=[39, 41, 45] -> Bottle, Cup, Bowl
            results = model.track(frame, persist=True, verbose=False, classes=[39, 41, 45], tracker="bytetrack.yaml")
            
            overlay = frame.copy()
            current_detected_classes = set()

            # 2. UPDATE MEMORY WITH NEW DETECTIONS
            if results[0].masks:
                for mask, box in zip(results[0].masks.xy, results[0].boxes):
                    cls_id = int(box.cls[0])
                    current_detected_classes.add(cls_id)
                    
                    # Save to memory (Reset counter to 0)
                    # We store the polygon points and the bounding box
                    object_memory[cls_id] = {
                        'mask': np.int32([mask]),
                        'box': list(map(int, box.xyxy[0])),
                        'label': results[0].names[cls_id],
                        'lost_frames': 0
                    }

                    # DRAW LIVE OBJECT (Green)
                    cv2.fillPoly(overlay, [np.int32(mask)], (0, 255, 0))
                    cv2.polylines(frame, [np.int32(mask)], True, (0, 255, 0), 2)

            # 3. HANDLE "GHOST" OBJECTS (Occluded)
            # Check every object in memory. If we didn't see it this frame, increase 'lost_frames'
            for cls_id in list(object_memory.keys()):
                if cls_id not in current_detected_classes:
                    object_memory[cls_id]['lost_frames'] += 1
                    
                    # If lost for too long, forget it
                    if object_memory[cls_id]['lost_frames'] > MEMORY_PATIENCE:
                        del object_memory[cls_id]
                        continue

                    # DRAW GHOST OBJECT (Grey/White dashed)
                    # This visualizes "I know it's here somewhere"
                    mem = object_memory[cls_id]
                    alpha = max(0, 1 - (mem['lost_frames'] / MEMORY_PATIENCE)) # Fade out
                    
                    # Draw Ghost Mask
                    ghost_color = (200, 200, 200) # Grey
                    cv2.polylines(frame, mem['mask'], True, ghost_color, 1, cv2.LINE_AA)
                    
                    # Label it "OCCLUDED"
                    x1, y1, x2, y2 = mem['box']
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    cv2.putText(frame, f"{mem['label']} (MEMORY)", (cx-40, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, ghost_color, 1)

            # 4. SAFETY BUBBLES
            cam_pose = f['transforms']['camera'][frame_idx]
            for side in ['right', 'left']:
                wrist_uv = project_wrist(f, frame_idx, K, cam_pose, side)
                if wrist_uv:
                    cv2.circle(overlay, wrist_uv, 50, (0, 0, 255), -1) 
                    cv2.circle(frame, wrist_uv, 50, (0, 0, 255), 2)
                    cv2.circle(frame, wrist_uv, 4, (255, 255, 255), -1)

            # 5. COMPOSITE
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # HUD
            cv2.putText(frame, f"TRACKING + MEMORY | Patience: {MEMORY_PATIENCE} frames", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            if frame_idx % 20 == 0: print(f"Processing {frame_idx}...")
            frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Object Permanence Video Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()