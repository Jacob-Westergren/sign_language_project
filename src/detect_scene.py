import os
import cv2
import time
import scenedetect
import numpy as np
from scenedetect import SceneManager, detect
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
from ultralytics import YOLO

class Detected_Person:
    def __init__(self,
                 crop: tuple[int,int,int,int],
                 start_frame: int,
                 end_frame: int = None,
                 freq: int = 1):
        self.start_frame = start_frame
        self.crop = crop
        self.freq = freq 
    
    # TODO: get mean crop from the detected frames

    def update(self, new_end_frame):
        self.end_frame = new_end_frame
        self.freq += 1

class Scene:
    def __init__(self,
                 id: int, 
                 start_frame: int,
                 end_frame: int):
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame

    def __str__(self):
        return f"Scene {self.id}: ({self.start_frame}, {self.end_frame})"

def create_scenes(subtitles, time_threshold=5):
    """
    Function to create scenes, where a scene is defined as a combination of closely following subtitles
    inputs:
        subtitles: Subtitles Object, storing subtitle informatin like start and end times.
        time_threshold: Integer defining the accepted time interval between subtitles 
    return:
        scenes: Dictionary of all scenes' start and end times
    
    example:
        sub0: 0 - 10
        sub1: 11 - 20
        sub2: 21 -28
        sub3: 37 - 45
        scenes = {Scene 1: (0,10)}
        entering for loop
            next_subtitle = (11,20)
            |10 - 11| < 5 == True --> scenes = {Scene 1: (0,20)}

            next_subtitle = (21,28)
            |20 - 21| < 5 == True --> scenes = {Scene 1: (0,28)}
            
            next_subtitle = (37,45)
            |28 - 37| < 5 == False --> scenes = {Scene 1: (0,28), Scene 2: (37, 45)}
    """
    scenes_timestamps = {"Scene 1": (subtitles[0].start, subtitles[0].end)}            
    curr_sub = subtitles[0]
    for scene_id, next_subtitle in enumerate(subtitles[1:]):
        # if the subtitles quickly follow each other 
        if np.abs(curr_sub.end - next_subtitle.start) < time_threshold:
            # move scene ending to start of end of next subtitle
            scenes_timestamps[scene_id][1] = next_subtitle.end
        else:
            scenes_timestamps[scene_id] = (next_subtitle.start, next_subtitle.end)
        curr_sub = next_subtitle
    return scenes_timestamps

def extract_scenes(video_path, scenes_timestamps, output_dir, padding=4):
    # Setup so we can read input
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # TODO
    # add logic maybe so if some scenes are super long, it's divided up here into smaller chunks.
    # TODO

    # Iterate over all scenes and store their corresponding frames in their own mp4 file
    for scene_id, (scene_start, scene_end) in scenes_timestamps.items():
        # Creating output writer
        scene_output_path = os.path.join(output_dir, f"{video_path.split()[-2]}", scene_id)
        out = cv2.VideoWriter(scene_output_path, fourcc, fps, (width, height))  

        # Convert time to frames
        start_frame, end_frame = int((scene_start - padding) * fps), int((scene_end + padding) * fps)
        
        # Move to the scene's start frame in the video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            # if we reached the end of the scene or video file --> break
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                break
            # else --> store current frame
            out.write(frame)
            
    cap.release()
 
def play_scene_folder(folder_path, fps=30):
    frame_files = sorted(os.listdir(folder_path))  # Sort to maintain order
    frame_delay = 1 / fps  # Calculate delay in seconds

    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)  # Read frame
        
        cv2.imshow("Video Playback", frame)  # Display frame

        # Wait for (1000 / fps) ms, exit if 'q' is pressed
        if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()  # Close window after playback

def extract_cropped_interpreter_frames(
        episode_dir: str, 
        video_file: str, 
        scene: Scene, 
        yolo_config: str,  
        time_jump: int=1):
    video_path = os.path.join(episode_dir, video_file)
    scene_dir = os.path.join(episode_dir, f"scene_{scene.id}")
    print(f"video_path: {video_path}")
    print(f"scene_dir: {scene_dir}")
    
    os.makedirs(scene_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    yolo_model = YOLO(yolo_config)

    frame_count = scene.start_frame
    corner_counter = {}
    # TODO: # add adaptive frame counter incrementation, so fps/2 first and last 2 seconds, and fps*3 in between
    start = time.time()
    while cap.isOpened() and frame_count < scene.end_frame: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        ret, frame = cap.read()
        if not ret:
            print(f"Exited scena at frame {frame_count} as there's no more frames. There's probably an error in the extract scenes code or subtitle metadata handling")
            break

        results = yolo_model.track(frame, persist=True)[0]  # [results] --> so we extract results
        # result is object with the following attributes: orig_im, boxes, names - dict mapping class IDs to class names

        human_detections = [result for result in results if result.boxes.cls == 0.] # person class corresponds to 0

        for detected in human_detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, detected.boxes.xyxy[0])   # (x1,y1)=bottom left corner, (x2,y2)=top right corner

            if (y1 < 350 and (x1 < 50)) or (y1 < 350 and (x2 > width - 50)):   
                box_id = detected.boxes.id
                if box_id:  # if box_id != None which can occur if it's unsure, then
                    box_id = int(box_id)
                    if box_id in corner_counter:
                        corner_counter[box_id].update(frame_count)
                    else:
                        corner_counter[box_id] = Detected_Person(
                            start_frame=frame_count,
                            crop=(x1,y1,x2,y2),
                        )

        # Move to next desired time step        
        frame_count += fps * 2  

        # for fps/2:  scene 1 - 0, 285 in 5.9 sec, scene 2 - 300,465 in 5.7 sec, scene 3 - 720-735 in 2.9 sec
        # for fps*3:  scene 1 - 0, 270 in 2 sec, scene 2 - 300-390 in 1 sec sec, scene 3 - 0.7 sec
        # having * 3 seems to signifantly speed up the process, and it still works quite well as long as the scene is not half or more background, but this shouldn't
        # be the case since I will create the scenes using the subtitles time stamps, or basically each "scene" will be one or two sentence timestamps put together.

    print(f"Finished applying YOLO to the scene in {time.time() - start} seconds.")

    interpreter = max(corner_counter.values(), key=lambda person: person.freq, default=None)
    if interpreter:
        print(f"For frame {scene.start_frame, scene.end_frame}, found interpreter at frames {interpreter.start_frame, interpreter.end_frame}")
    else:
        print(f"For frame {scene.start_frame, scene.end_frame}, found no interpreter.")
        return

    # find the mean crop size
    x1,y1,x2,y2 = interpreter.crop

    # Just store the frames and crop in a json

    # Move to first frame where the interpreter was found
    frame_count = interpreter.start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    start = time.time()
    while cap.isOpened() and frame_count < interpreter.end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Broke early for video {video_path} at frame {frame_count}")
            break
        
        crop_path = os.path.join(scene_dir, f"frame_{"{:03d}".format(frame_count)}.jpg")  # should probably zip the folder when writing it to to bucket
        cv2.imwrite(crop_path, frame[y1:y2, x1:x2]) # replace with some function i guess later using aws cli

        frame_count += 1
    print(f"Finished storing all desired frames in the scene in {time.time() - start} seconds.")
    cap.release()
 
def mask_video(video_dir, video_name, output_dir, mask_ratio=0.05):
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output
    base_filename = os.path.basename(video_path)
    print(f"video_path: {video_path}")
    print(f"base_filename: {base_filename}")
    output_path = os.path.join(output_dir, f"{video_name}_masked.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, 400))  

    mask_x = int(width * mask_ratio)
    mask_y = int(height * mask_ratio) + 100
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply white mask to the center
        frame[mask_y:-mask_y, mask_x:-mask_x] = 0.5
        out.write(frame)
        if counter == 0:
            cv2.namedWindow("Masked Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Masked Video", 640, 480)  # Resize window to 640x480 or any desired size

            cv2.imshow("Masked Video", frame)
            cv2.waitKey(0)  # Wait for any key to continue
            cv2.destroyAllWindows()  # Close the window
            counter += 1

    cap.release()
    out.release()
    return output_path






# --------
# OLD CODE
# --------

def mask_video_small(video_dir, video_name, output_dir):
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output
    base_filename = os.path.basename(video_path)
    print(f"video_path: {video_path}")
    print(f"base_filename: {base_filename}")
    output_path = os.path.join(output_dir, f"{video_name.split()[1]}_small_masked")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, 400))  

    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply white mask to the center
        top_part = frame[0:200, :]
        bottom_part = frame[-200:, :]

        # Concatenate vertically
        mini_frame = np.vstack((top_part, bottom_part))

        out.write(mini_frame)
        # for debug purposes
        if counter == 0:
            cv2.namedWindow("Masked Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Masked Video", 640, 400)  # Resize window to 640x480 or any desired size

            cv2.imshow("Masked Video", mini_frame)
            cv2.waitKey(0)  # Wait for any key to continue
            cv2.destroyAllWindows()  # Close the window
            counter += 1

    cap.release()
    out.release()
    return output_path

def detect_scenes(video_file):
    scene_list = detect(video_file, ContentDetector(threshold=0.001))
    print(scene_list)
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))
        
    return scene_list

def detect_scenes_old(video_file, threshold=30.0):
    # Create a SceneManager object to manage scene detection
    scene_manager = SceneManager()

    # Add the content detector to detect scene changes based on content
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Open the video file using the scene manager
    video = scenedetect.VideoManager([video_file])
    
    # Start scene detection process
    video.start()

    # Detect scenes in the video
    scene_manager.detect_scenes(frame_source=video)

    # Get the list of scene boundaries (start, end) in frames
    scene_list = scene_manager.get_scene_list()

    print(f"Total scenes detected: {len(scene_list)}")

    # Display each scene using OpenCV
    for scene_num, (start, end) in enumerate(scene_list, start=1):
        start_time = start / video.get_fps()  # Convert frame to time in seconds
        end_time = end / video.get_fps()      # Convert frame to time in seconds

        print(f"Scene {scene_num}: {start_time:.2f}s - {end_time:.2f}s")

        # Set the video position to the start of the scene
        video.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Read and display the frames for this scene
        while video.get(cv2.CAP_PROP_POS_FRAMES) < end:
            ret, frame = video.read()
            if not ret:
                break

            # Display the frame using OpenCV
            cv2.imshow(f"Scene {scene_num}", frame)

            # Wait for the user to press a key to proceed to the next frame
            key = cv2.waitKey(1)  # Adjust the delay to control frame rate
            if key == 27:  # ESC key to exit the scene view
                break

        # Close the window for this scene after displaying all frames
        cv2.destroyAllWindows()

    # Release resources after processing
    video.release()
