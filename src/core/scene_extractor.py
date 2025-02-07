import cv2
import numpy as np
from typing import List, Optional, Dict
from .scene import Scene
from .person import DetectedPerson
from pathlib import Path
from ultralytics import YOLO

class SceneExtractor:
    def __init__(
        self, 
        yolo_model_path: str
    ) -> None:
        self.yolo_model = YOLO(yolo_model_path)

    def _setup_video_capture(
        self, 
        episode_path: Path
    ) -> tuple[cv2.VideoCapture, int, int]:
        """Set up video capture and return capture object with basic properties"""
        cap = cv2.VideoCapture(str(episode_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return cap, fps, width

    def _detect_interpreter_in_frame(
        self, 
        frame: np.ndarray, 
        width: int 
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Detect interpreter in a single frame, return list of (box_id, crop_coords)"""
        results = self.yolo_model.track(frame, persist=True, verbose=False)[0]
        interpreter_detections = []
        
        human_detections = [result for result in results if result.boxes.cls == 0.]
        for detected in human_detections:
            x1, y1, x2, y2 = map(int, detected.boxes.xyxy[0])
            
            if (y1 < 350 and (x1 < 50)) or (y1 < 350 and (x2 > width - 50)):
                box_id = detected.boxes.id
                if box_id is not None:
                    interpreter_detections.append((int(box_id), (x1, y1, x2, y2)))
                    
        return interpreter_detections

    def _process_frame_detections(
        self, 
        detections: list[tuple[int, tuple[int, int, int, int]]], 
        corner_counter: dict,
        frame_count: int
    ) -> None:
        """Process detections and update corner_counter"""
        for box_id, crop_coords in detections:
            if box_id in corner_counter:
                corner_counter[box_id].update(frame_count)
            else:
                corner_counter[box_id] = DetectedPerson(
                    start_frame=frame_count,
                    crop=crop_coords,
                )

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

    def extract_scenes(self, 
                       video_path: Path) -> List[Scene]:
        """
        Extract scenes from a video file
        Currently returns dummy data - implement actual scene detection later
        """
        # Temporary hardcoded scenes for testing
        return [
            Scene(id=1, start_frame=0, end_frame=30*8),
            Scene(id=2, start_frame=30*8, end_frame=30*20),
            Scene(id=3, start_frame=30*20, end_frame=30*30)
        ]

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
        cap, fps, width = self._setup_video_capture(episode_path)
        
        frame_count = scene.start_frame
        corner_counter = {}
        
        while cap.isOpened() and frame_count < scene.end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Exited scene at frame {frame_count} as there's no more frames.")
                break

            detections = self._detect_interpreter_in_frame(frame, width)
            self._process_frame_detections(detections, corner_counter, frame_count)
            frame_count += fps * time_jump

        interpreter = self._find_main_interpreter(corner_counter, scene, fps, time_jump)
        
        if interpreter:
            return {
                "scene_id": scene.id,
                "start_frame": interpreter.start_frame,
                "end_frame": interpreter.end_frame,
                "interpreter_crop": interpreter.crop,
                "interpreter_frequency": interpreter.freq
            }
        else:
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

    def play_scene_folder(self, folder_path: str, fps: int=30):
        """
        Play frames from a scene folder
        """
        # frame_files = sorted(os.listdir(folder_path))  
        folder = Path(folder_path)
        frame_files = sorted(folder.iterdir())  # Sort to maintain order
        frame_delay = 1 / fps  # Calculate delay in seconds

        #for frame_file in frame_files:
        #    frame_path = os.path.join(folder_path, frame_file)
        #     frame = cv2.imread(frame_path)  # Read frame
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))  # Read frame
            
            cv2.imshow("Video Playback", frame)  # Display frame

            # Wait for (1000 / fps) ms, exit if 'q' is pressed
            if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()  # Close window after playback


