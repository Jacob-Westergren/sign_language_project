import os
import yaml
from typing import List
from .scene import Scene
from .scene_extractor import SceneExtractor
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self.load_config(config_path)
        self.scene_extractor = SceneExtractor(self.config['scene_extraction']['YOLO_model'])

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def process_episodes(self):
        program_path = os.path.join(
            self.config["data"]["base_dir"], 
            f"{self.config['data']['program_to_extract']}"
        )

        episode_files = [f for f in os.listdir(f"{program_path}\\1234") if f.endswith('.mp4')]
        print(f"Episodes: {episode_files}")

        for episode in episode_files:
            self.process_single_episode(program_path, episode)

    def process_single_episode(self, program_path: str, episode: str):
        episode_dir = os.path.join(program_path, f"{episode.split('.')[0]}")
        os.makedirs(episode_dir, exist_ok=True)

        print(f"Processing Episode {episode.split('.')[0]}")
        scenes = self.scene_extractor.extract_scenes(episode)

        for scene in scenes:
            print(f"Processing Scene {scene.id}")
            self.scene_extractor.extract_cropped_interpreter_frames(
                episode_dir,
                episode,
                scene
            )


import cv2
from typing import List
from .scene import Scene
import os

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


class SceneExtractor:
    def __init__(self, yolo_model_path: str):
        self.yolo_model_path = yolo_model_path

    def extract_scenes(self, video_path: str) -> List[Scene]:
        """
        Extract scenes from a video file
        Currently returns dummy data - implement actual scene detection later
        """
        # Temporary hardcoded scenes for testing
        return [
            Scene(id=1, start_frame=0, end_frame=30*10),
            Scene(id=2, start_frame=30*10, end_frame=30*20),
            Scene(id=3, start_frame=30*20, end_frame=30*30)
        ]

    def extract_cropped_interpreter_frames(
            self, 
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

        interpreter = max(corner_counter.values(), key=lambda person: person.freq, default=None)
        if interpreter:
            print(f"For frame {scene.start_frame, scene.end_frame}, found interpreter at frames {interpreter.start_frame, interpreter.end_frame}")
        else:
            print(f"For frame {scene.start_frame, scene.end_frame}, found no interpreter.")
            return

        # find the mean crop size
        x1,y1,x2,y2 = interpreter.crop

        # Move to first frame where the interpreter was found
        frame_count = interpreter.start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        while cap.isOpened() and frame_count < interpreter.end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Broke early for video {video_path} at frame {frame_count}")
                break
            
            crop_path = os.path.join(scene_dir, f"frame_{"{:03d}".format(frame_count)}.jpg")  # should probably zip the folder when writing it to to bucket
            cv2.imwrite(crop_path, frame[y1:y2, x1:x2]) # replace with some function i guess later using aws cli

            frame_count += 1
        cap.release()

        def play_scene_folder(self, folder_path, fps=30):
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

from core.video_processor import VideoProcessor

def main():
    processor = VideoProcessor()
    processor.process_episodes()

if __name__ == "__main__":
    # processor = VideoProcessor()
    # processor.scene_extractor.play_scene_folder("data\\programs\\2001345\\1234\\scene_2")
    main()