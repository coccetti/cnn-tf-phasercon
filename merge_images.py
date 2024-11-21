# Description: This script merges all the images in the Downloads folder into a video file.

import os
import subprocess

# Path to the Downloads folder
image_folder = os.path.expanduser('~/Downloads/training3')
video_name = os.path.expanduser('~/Downloads/training3/o2.mp4')

# Duration for each image in seconds
duration = 3

# Get list of images in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images.sort()  # Sort images by name
print(images)

if not images:
    raise ValueError("No images found in the specified folder.")

# Create a temporary text file with the list of images
with open('images.txt', 'w') as f:
    for image in images:
        f.write(f"file '{os.path.join(image_folder, image)}'\n")
        f.write(f"duration {duration}\n")  # Set duration for each image

# Add the last image again to ensure the video ends with the last image
with open('images.txt', 'a') as f:
    f.write(f"file '{os.path.join(image_folder, images[-1])}'\n")
    f.write(f"duration {duration}\n")  # Set duration for the last image

# Use ffmpeg to create the video
ffmpeg_command = [
    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'images.txt', '-vf', 'fps=1', '-pix_fmt', 'yuv420p', video_name
]

try:
    subprocess.run(ffmpeg_command, check=True)
    print(f"Video saved as {video_name}")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

# Clean up the temporary text file
os.remove('images.txt')