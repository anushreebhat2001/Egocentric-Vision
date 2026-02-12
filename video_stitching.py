import cv2
import glob
import os

image_folder = 'my_object/depth_zoe_viz'
video_name = 'rgb_d.mp4'

images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
print(images)

if not images:
    print("No images found! Check if the tracker actually saved files.")
else:
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    print(f"detailed stitching {len(images)} frames into video...")

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video created successfully: {video_name}")