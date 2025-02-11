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

    """
    def shift_hand(pose: Pose, hand_component: str, wrist_name: str):
        # pylint: disable=protected-access
        wrist_index = pose.header._get_point_index(hand_component, wrist_name)
        hand = pose.body.data[:, :, wrist_index: wrist_index + 21]
        wrist = hand[:, :, 0:1]
        pose.body.data[:, :, wrist_index: wrist_index + 21] = hand - wrist

    def _pre_process_mediapipe(pose: Pose):
        # Remove legs, simplify face
        pose = reduce_holistic(pose)
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

        # Align hand wrists with body wrists
        correct_wrists(pose)
        # Adjust pose based on shoulder positions
        pose = pose.normalize(pose_normalization_info(pose.header))

        # Shift hands to origin
        (left_hand_component, right_hand_component), _, (wrist, _) = hands_components(pose.header)
        shift_hand(pose, left_hand_component, wrist)
        shift_hand(pose, right_hand_component, wrist)

        return pose
    """

    def _load_video_frames(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, crop: Tuple[int, int, int, int]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_cnt = start_frame
        x1, y1, x2, y2 = crop
        while frame_cnt < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            frame_cnt += 1
        cap.release()

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
            
        """
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= scene_data.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = frame[y1:y2, x1:x2]
            results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(results)
            

            # If frame contains all needed landmarks, extract and store them.
            if results.pose_landmarks.face_landmarks  \
                and results.pose_landmarks.multi_hand_landmarks \
                and results.pose_landmarks.pose_landmarks:
                face_landmarks = results.pose_landmarks.face_landmarks
                left_hand_landmarks = results.pose_landmarks.left_hand_landmarks
                right_hand_landmarks = results.pose_landmarks.right_hand_landmarks
                pose_landmarks = results.pose_landmarks.pose_landmarks

                face_landmarks_np = np.array([[lmk.x, lmk.y, lmk.z] for lmk in face_landmarks.landmark])
                left_hand_landmarks_np = np.array([[lmk.x, lmk.y, lmk.z] for lmk in left_hand_landmarks.landmark])
                right_hand_landmarks_np = np.array([[lmk.x, lmk.y, lmk.z] for lmk in right_hand_landmarks.landmark])
                pose_landmarks_np = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark])

                landmarks.append((face_landmarks_np, 
                                  left_hand_landmarks_np, 
                                  right_hand_landmarks_np, 
                                  pose_landmarks_np))
            else:
                # Return landmarks if valid frames processed exceeds minimum length
                if fps * cap.get(cv2.CAP_PROP_POS_FRAMES) > minimum_length:
                    cap.release()
                    return landmarks
                cap.release()
                return []

        cap.release()
        return landmarks
        """
    
    def process_scenes(
        self,
        video_path: Path,
        scenes_data: list[SceneData]
    ) -> None:
        """Process all scenes in the video."""
        poses_dir = video_path.parent / "poses"
        poses_dir.mkdir(exist_ok=True)
        for scene_data in scenes_data:
            output_path = poses_dir / f"scene_{scene_data['id']}.pose"
            print(f"Extracting keypoints from video {video_path.name} scene {scene_data['id']}. Output file: {output_path}")
            with timing(f"Extracting keypoints from scene {scene_data['start_frame']} to {scene_data['end_frame']}"):
                self._extract_keypoints_from_scene(video_path, output_path, scene_data)