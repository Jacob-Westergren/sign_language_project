import cv2
import numpy as np
from typing import List, Optional, Dict
from ..structures import Scene, DetectedPerson
from pathlib import Path
from ultralytics import YOLO
import torch
from ..utils import timing
from ..structures import SceneData
import json

class SceneExtractor:
    def __init__(
        self, 
        yolo_model_path: str
    ) -> None:
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
 
        # Load model to specified device
        self.yolo_model = YOLO(yolo_model_path)#.to(self.device)

    def _detect_interpreter_in_frame(
        self, 
        frame: np.ndarray, 
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Detect interpreter in a single frame, return list of (box_id, crop_coords)"""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        interpreter_detections = []
        
        human_detections = [result for result in results if result.boxes.cls == 0.]
        for detected in human_detections:
            x1, y1, x2, y2 = map(int, detected.boxes.xyxy[0])
            # For SVT's data, the interpreter is always below 350, and noone else is for the scene's the interpreter is present. 
            # Depending on the program you chose to work with, you might have to adjust this criteria.
                # and x2 < 100         or y < 350 and (x2 > width - 100)
            if (y1 < 350):
                box_id = detected.boxes.id
                if box_id is not None:
                    interpreter_detections.append((int(box_id), (x1, y1, x2, y2)))
                    
        return interpreter_detections

    # Move this back to individual funciton cuz this was mainly when I wanted prints to debug and the print took up alot of space. 
    # Prob test it first to see if the min_freq code line is reasonable, or if it needs to change, or if even the code needs to have it.
    def _find_main_interpreter(
        self, 
        corner_counter: dict,
        scene: Scene,
        fps: int,
        time_jump: int
    ) -> Optional[DetectedPerson]:
        """Find the interpreter with highest frequency meeting minimum requirements"""
        if not corner_counter:
            print(f"For frame {scene.start_frame, scene.end_frame}, found no interpreter.")
            return None

        interpreter = max(corner_counter.values(), key=lambda person: person.freq, default=None)
        min_freq = (scene.end_frame/fps - scene.start_frame/fps) * 0.6 // time_jump
        
        if interpreter and interpreter.freq >= min_freq:
            print(f"For frame {scene.start_frame, scene.end_frame}, found interpreter at frames "
                  f"{interpreter.start_frame, interpreter.end_frame} with frequency {interpreter.freq}")
            return interpreter
        else:
            print(f"For frame {scene.start_frame, scene.end_frame}, found no interpreter, "
                  f"or too low frequency with {interpreter.freq if interpreter else 0}.")
            return None

    def extract_cropped_interpreter_frames(
        self,
        episode_path: Path,
        scene: Scene,   
        time_jump: int=2
    ) -> Optional[Dict]:
        """
        Extract and crop interpreter frames from a specific scene
        Returns scene metadata if interpreter is found in >= 60% of frames
        """
        with timing(f"Processing Scene {scene.id}"):
            cap = cv2.VideoCapture(str(episode_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            print(f"The video's fps is {fps}")
            # Hard coded value due to SVT's data being either in 25 or 50 FPS. 
            frame_increment = 2 if fps == 50 else 1
            
            frame_count = scene.start_frame
            corner_counter = {}
            
            while cap.isOpened() and frame_count <= scene.end_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Exited scene at frame {frame_count} for episode {episode_path.name} scene {scene.id} as there's no more frames. "
                          f"Probably an error in extract scenes function.")
                    break

                detections = self._detect_interpreter_in_frame(frame)
                for box_id, crop_coords in detections:
                    if box_id in corner_counter:
                        corner_counter[box_id].update(frame_count)
                    else:
                        corner_counter[box_id] = DetectedPerson(
                            start_frame=frame_count,
                            crop=crop_coords,
                        )
                # If fps is 50, then we need to jump wice as many frames to jump the desired time amount, ex time_jump=1 sec --> 1*25 = 0.5 sec if fps = 50
                frame_count += (frame_increment * time_jump) *  fps

            interpreter = self._find_main_interpreter(corner_counter, scene, fps, time_jump)
            
            if interpreter:
                return SceneData(
                    id= scene.id,
                    start_frame=interpreter.start_frame,
                    end_frame=interpreter.end_frame,
                    interpreter_crop=interpreter.crop,
                    interpreter_frequency=interpreter.freq
                )
            return None

        """
        Move to first frame where the interpreter was found
        frame_count = interpreter.start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        while cap.isOpened() and frame_count < interpreter.end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Broke early for video {episode_path} at frame {frame_count}")
                break
            
            crop_path = scene_dir / f"frame_{'{:03d}'.format(frame_count)}.jpg"
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])

            frame_count += 1
        cap.release()
        """

    def extract_scenes(
        self, 
        subtitle_metadata_path: Path
    ) -> List[Scene]:
        """
        Extract scenes from a video file
        Currently returns dummy data - implement actual scene detection later
        """
        print(f"subtitle path is: {subtitle_metadata_path}")
        with open(subtitle_metadata_path, 'r') as file:
            subtitle_metadata = json.load(file)  
        # Potentially add some more code to make it combine subtitles into one scene if the gap between subtitles are within a certain value
        scenes = [
            Scene(id=scene_id, start_frame=scene['start_frame'], end_frame=scene['end_frame'])
            for scene_id, scene in enumerate(subtitle_metadata['subtitle'])
        ]
        return scenes

