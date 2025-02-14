import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path
from ..structures import SceneData
from omegaconf import DictConfig
from ..utils import timing
from typing import Tuple
from pose_format.utils.holistic import load_holistic

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]

class KeypointExtractor:
    def __init__(
        self,
        cfg: DictConfig
    ) -> None:
        self.holistic_config = {
            "static_image_mode" : cfg.mediapipe.static_image_mode,
            "model_complexity" : cfg.mediapipe.model_complexity,
            "refine_face_landmarks" : cfg.mediapipe.refine_face_landmarks,
            "min_detection_confidence" : cfg.mediapipe.min_detection_confidence
        }


    def _load_video_frames(
        self, 
        cap: cv2.VideoCapture, 
        start_frame: int, 
        end_frame: int, 
        crop: Tuple[int, int, int, int]
    ) -> np.array:
        """Generator function that allows user to loop over a mp4 file and get the cropped frames of the interpreter."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # if fps is 50, we want to get every other frame.
        time_increment = 2 if fps == 50 else 1

        frame_cnt = start_frame
        x1, y1, x2, y2 = crop
        while frame_cnt < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            frame_cnt += time_increment
        cap.release()


    # Have to add error handling, so what it should do if it fails to detect some keypoint between frames, what it should do if it fails to detect 
    def _extract_keypoints_from_scene(
        self, 
        video_path: Path,
        output_path: Path,
        scene_data: SceneData,
        minimum_length: int = 6,    
        reduce: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """"Extract facial, hand, and pose keypoints from a video scene."""
    
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = self._load_video_frames(cap, scene_data["start_frame"], scene_data["end_frame"], scene_data["interpreter_crop"])

        x1, y1, x2, y2 = scene_data["interpreter_crop"]
        pose = load_holistic(frames, fps=fps, width=(x2-x1), height=(y2-y1), progress=True, 
                             additional_holistic_config=self.holistic_config)

        # Remove world landmarks by default
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

        # Reduce as Surrey did
        if reduce:
            pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
        
        with open(output_path, "wb") as f:
            pose.write(f)
            

    def process_scenes(
        self,
        video_path: Path,
        scenes_data: list[SceneData]
    ) -> None:
        """Process all scenes in the video."""
        poses_dir = video_path.parent / "poses"
        poses_dir.mkdir(exist_ok=True)
        for scene_data in scenes_data:
            output_path = poses_dir / f"scene_{'{:04d}'.format(scene_data['id'])}.pose"
            print(f"Extracting keypoints from video {video_path.name} scene {scene_data['id']}. Output file: {output_path}")
            with timing(f"Extracting keypoints from scene {scene_data['start_frame']} to {scene_data['end_frame']}"):
                self._extract_keypoints_from_scene(video_path, output_path, scene_data)