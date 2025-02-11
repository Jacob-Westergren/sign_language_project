import json
from pathlib import Path
from typing import List
from .scene_extractor import SceneExtractor
from ..structures import Scene, SceneData
from omegaconf import DictConfig
from .keypoint_extractor import KeypointExtractor


class VideoProcessor:
    def __init__(
        self, 
        cfg: DictConfig
    ) -> None:
        self.cfg = cfg
        print(cfg)

        yolo_model_path = Path(self.cfg.model.yolo.model_name)
        try:
            print(f"Initializing scene extractor with model: {yolo_model_path}")
            self.scene_extractor = SceneExtractor(str(yolo_model_path))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\nError: YOLO model '{yolo_model_path.name}' not found.\n"
                f"Please check your config.yaml and ensure the model file exists.\n"
                f"Recommended models: yolo11n.pt, yolo11s.pt (n-nano, s-small)"
            )
        try:
            self.keypoint_extractor = KeypointExtractor(self.cfg.model)
        except Exception as e:
            raise Exception(f"Error initializing keypoint extractor: {e}")


    def process_episodes(
        self
    ) -> None:
        program_path = Path(self.cfg.data.base_dir) / self.cfg.data.program_to_extract
        print(f"Processing programs from: {program_path}")

        # episode_files = [f for f in os.listdir(f"{program_path}\\1234") if f.endswith('.mp4')]
        episodes_paths = list(program_path.rglob('*.mp4'))   # r for recursive glob
        print(f"The found episodes are: {episodes_paths}, where each episode is of type: {type(episodes_paths[0])}")

        for episode_path in episodes_paths:
            print(f"Processing episode {episode_path.stem}, Episode path is: {episode_path}")
            self.process_single_episode(episode_path)

    def process_single_episode(
        self, 
        episode_path: Path
    ) -> None:
        subtitle_metadata_path = episode_path.parent / "subtitle_metadata.json"
        scenes: List[Scene] = self.scene_extractor.extract_scenes(subtitle_metadata_path)
        print(f"Scenes:")
        for scene in scenes: print(f"scene {scene.id}: ({scene.start_frame}, {scene.end_frame})")

        scenes_data: List[SceneData] = []
        for scene in scenes:
            scene_data = self.scene_extractor.extract_cropped_interpreter_frames(
                episode_path,
                scene
            )
            if scene_data:
                scenes_data.append(scene_data)
        print(f"Found {len(scenes_data)} scenes with interpreters")

        # Save scenes data to JSON file next to the video file
        json_path = episode_path.parent / 'scenes.json'
        with open(json_path, 'w') as f:
            json.dump({"scenes": scenes_data}, f, indent=4)

        print(f"Saved scene information to {json_path}")

        # Extract keypoints from interpreter frames if scenes were found
        if scenes_data:
            self.keypoint_extractor.process_scenes(episode_path, scenes_data) 
            