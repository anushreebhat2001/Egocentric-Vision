import cv2
import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial.transform import Rotation as R

device = "cuda" if torch.cuda.is_available() else "cpu"
florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True, attn_implementation="eager").to(device).eval()
florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

sam_predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint="scripts/sam_vit_h_4b8939.pth").to(device))

def run_florence_inference(image, text_query):
    """Uses Florence-2 to find a bounding box via text prompt."""
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{text_query}"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            use_cache=False
        )
    
    results = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(results, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.shape[1], image.shape[0]))
    
    box = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'][0]
    return np.array(box)

def create_grounded_segmentation_video(video_path, hand_csv, output_path):
    df_hand = pd.read_csv(hand_csv)
    # Calculate contact frame dynamically = t_contact
    df_hand['velocity'] = np.sqrt(df_hand['rightHand_x'].diff()**2 + df_hand['rightHand_y'].diff()**2 + df_hand['rightHand_z'].diff()**2) / (1/30.0)
    df_hand['aperture'] = np.sqrt((df_hand['rightThumbTip_x']-df_hand['rightIndexFingerTip_x'])**2 + (df_hand['rightThumbTip_y']-df_hand['rightIndexFingerTip_y'])**2 + (df_hand['rightThumbTip_z']-df_hand['rightIndexFingerTip_z'])**2)

    t_contact = int(df_hand[df_hand['aperture'] < df_hand['aperture'].quantile(0.3)]['velocity'].idxmin())
    print(f"Detected contact at frame: {t_contact}")

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

    mask_lid, mask_jar = None, None

    for idx in range(len(df_hand)):
        ret, frame = cap.read()
        if not ret: break

        if idx == t_contact:
            print("Running Florence-2 Grounding...")
            box_lid = run_florence_inference(frame, "the jar lid")
            box_jar = run_florence_inference(frame, "the jar body")
            
            sam_predictor.set_image(frame)
            masks_l, _, _ = sam_predictor.predict(box=box_lid, multimask_output=False)
            masks_j, _, _ = sam_predictor.predict(box=box_jar, multimask_output=False)
            mask_lid, mask_jar = masks_l[0], masks_j[0]

        if mask_lid is not None:
            frame[mask_lid] = cv2.addWeighted(frame[mask_lid], 0.5, np.full(frame[mask_lid].shape, (255, 100, 0), dtype=np.uint8), 0.5, 0)
            frame[mask_jar] = cv2.addWeighted(frame[mask_jar], 0.5, np.full(frame[mask_jar].shape, (0, 100, 255), dtype=np.uint8), 0.5, 0)
            
            lift = df_hand.iloc[t_contact]['rightHand_y'] - df_hand.iloc[idx]['rightHand_y']
            status = "GOAL ACHIEVED: OPEN" if lift > 0.05 else "STATE: MANIPULATING"
            color = (0, 255, 0) if lift > 0.05 else (0, 255, 255)
        else:
            status, color = "STATE: SEARCHING", (255, 255, 255)

        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Grounded video saved to {output_path}")

create_grounded_segmentation_video("video_learning_samples/add_remove_lid/0.mp4", "outputs/extracted_metrics_csv/add_remove_lid/0_hand_poses.csv", "florence_sam_output.mp4")