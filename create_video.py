import cv2
import os
from natsort import natsorted

image_folder = "/home/elisa/Documents/data/robosuite_automated/teleop_dataset_auto_20250702_093831/left_side_view"   # e.g. .../left_side_view
output_video_path = "left_side_view.avi"
frame_rate = 10                               

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsorted(images) 

if not images:
    raise ValueError("No images found in the folder!")

first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(img_path)
    video_writer.write(frame)

video_writer.release()
print(f"Video saved to {output_video_path}")
