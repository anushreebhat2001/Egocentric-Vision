import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

# --- CONFIGURATION ---
# Use the video you already have
VIDEO_PATH = '/Users/utx/Desktop/code/egodexrobotics/video_learning_samples/add_remove_lid/0.mp4'
OUTPUT_PATH = '/Users/utx/Desktop/code/egodexrobotics/outputs/output_depth_vis.mp4'


def main():
    print("⏳ Loading Depth Model... (This might take a minute the first time)")
    
    # Check for Mac GPU (MPS) or fallback to CPU
    # standard pipeline handles this, but explicit device helps if compatible
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Running on: {device.upper()}")

    # Load the "Depth Anything" model (State-of-the-Art Monocular Depth)
    # We use the 'small' version for speed on your laptop
    depth_estimator = pipeline(task="depth-estimation", 
                               model="LiheYoung/depth-anything-small-hf", 
                               device=device if device != 'mps' else -1) # Pipeline sometimes prefers -1 for CPU if MPS is buggy

    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # We will stack images side-by-side (RGB + Depth), so double the width
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Prepare Image for AI (OpenCV BGR -> PIL RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 2. INFER DEPTH (The Magic)
        # result is a dict with 'depth' key containing a PIL image
        prediction = depth_estimator(pil_image)
        depth_map = np.array(prediction["depth"])

        # 3. Process for Visualization
        # The raw depth is a float map. We need to normalize it to 0-255 for video.
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        
        # Apply a colormap (Magma or Inferno looks very "Sci-Fi/Research")
        # Dark = Close, Bright = Far (or vice versa depending on map)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        # Resize depth to match original video exactly (sometimes models output different sizes)
        depth_colored = cv2.resize(depth_colored, (width, height))

        # 4. Combine Side-by-Side
        combined_view = np.hstack((frame, depth_colored))
        
        # Add Label
        cv2.putText(combined_view, "Original RGB", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_view, "AI Estimated Depth (Z-Buffer)", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(combined_view)
        
        print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Depth video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()