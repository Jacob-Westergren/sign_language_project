import numpy as np
import cv2
import os
import scenedetect
from ultralytics import YOLO

# first, crop out a signer and lets call it signer patch. This will be used later for comparisions
# Then, randomly take out 100 frames from the video, and compare the left and right corner to the signer patch, the corner that consistently has
# the most matches over a threshhold is the corner the signer is in. 
# Then, in a similar manner as above, iterate over the video 


# first check how well the scenedetector works.
# add so it first checks if there's a human on screen, then check if there's a human in the most common spot near the edge, then check if the image 
# is blurry. 
# If all is true then there's an interpreter in a corner and the screen is zoomed out --> we want to include this in our dataset
# if not, then we skip segment

def detect_corner(video_file):
    # Load YOLO model (you need the YOLO config and weights files)
    # Load the model
    yolo_model = YOLO('yolov8m.pt')

    results = yolo_model.track(source=video_file, show=True, tracker="bytetrack.yaml")


def detect_blur2(image, margin_ratio=0.05, counter=0):
    image = image[100:-100, :]
    h, w = image.shape[:2]

    # Define margins (e.g., 80% of width/height)
    margin_h = int(h * margin_ratio)
    margin_w = int(w * margin_ratio)

    # Crop out the center, keeping only the border regions
    outer_regions = np.zeros_like(image)
    outer_regions[:margin_h, :] = image[:margin_h, :]  # Top
    outer_regions[-margin_h:, :] = image[-margin_h:, :]  # Bottom
    outer_regions[:, :margin_w] = image[:, :margin_w]  # Left
    outer_regions[:, -margin_w:] = image[:, -margin_w:]  # Right
    # 10 480
    if counter == 0:
        cv2.namedWindow("Outer Regions", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Outer Regions", 640, 480)  # Resize window to 640x480 or any desired size

        cv2.imshow("Outer Regions", outer_regions)
        cv2.waitKey(0)  # Wait for any key to continue
        cv2.destroyAllWindows()  # Close the window
    
    # Convert to grayscale and compute blur
    gray = cv2.cvtColor(outer_regions, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print(f"frame {counter}: {laplacian_var}")
    #if counter > 10 and counter < 20:
    #    print(f"early frame: {laplacian_var}")
    #elif counter > 480 and counter < 490:
    #    print(f"late frame: {laplacian_var}")
    # up to (including) frame 466 - all about 89
    # 467 and 468 - about 80
    # 470 to  564 - about 49 - 67
    # 564 to 691 - 70 to 116
    # 692 to the end - about 80 - 90
    return laplacian_var

def test_detector(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    blurred_frames = []
    blurred_frames_var = []
    sharp_frames = []
    sharp_frames_var=[]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame is blurred
        lap_var = detect_blur2(frame, counter=frame_count)
        if lap_var > 85 and lap_var < 91:
            sharp_frames.append(frame)
            sharp_frames_var.append(lap_var)
        else:
            blurred_frames.append(frame)
            blurred_frames_var.append(lap_var)
        frame_count += 1

    cap.release()
    # Display a few samples of each category
    print(f"Total frames processed: {frame_count}")
    print(f"Blurred frames detected: {len(blurred_frames)} with average {np.mean(blurred_frames_var)} blur")
    print(f"Sharp frames detected: {len(sharp_frames)} with average {np.mean(sharp_frames_var)} blur")
    return sharp_frames, blurred_frames

def show_frames(frames):
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", 640, 480)  # Resize window to 640x480 or any desired size

    for frame in frames:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)  # Wait for any key to proceed to the next frame
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows
