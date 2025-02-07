import os
import yaml
from typing import List
from .scene import Scene
from .scene_extractor import SceneExtractor

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

