import json
import yaml
import time
from pathlib import Path
from typing import List
from .scene_extractor import SceneExtractor
from ..models.scene import Scene, SceneData
from ..utils.config import load_config

class VideoProcessor:
    def __init__(
        self, 
        config_path: str = 'config.yaml'
    ) -> None:

        self.config = load_config(config_path)
        self.scene_extractor = SceneExtractor(self.config['scene_extraction']['YOLO_model'])

    def process_episodes(
        self
    ) -> None:
        program_path = Path(self.config["data"]["base_dir"]) / self.config['data']['program_to_extract']
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
        scenes: List[Scene] = self.scene_extractor.extract_scenes(episode_path)
        
        scenes_data: List[SceneData] = []
        for scene in scenes:
            time_start = time.time()
            print(f"Processing Scene {scene.id}")
            scene_data = self.scene_extractor.extract_cropped_interpreter_frames(
                episode_path,
                scene
            )
            if scene_data:
                scenes_data.append(scene_data)
            time_end = time.time()
            print(f"Time taken to process scene {scene.id}: {time_end - time_start} seconds")
        print(f"Found {len(scenes_data)} scenes with interpreters")

        time_start = time.time()
        # Save scenes data to JSON file next to the video file
        json_path = episode_path.parent / 'scenes.json'
        with open(json_path, 'w') as f:
            json.dump({"scenes": scenes_data}, f, indent=4)

        print(f"Saved scene information to {json_path}")
        time_end = time.time()
        print(f"Time taken to save scene information to {json_path}: {time_end - time_start} seconds")

