import os
import yaml
from test import test_func
from detect_interpreter import test_detector, show_frames, detect_corner
from detect_scene import extract_scenes, extract_cropped_interpreter_frames, Scene, play_scene_folder

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_program(config):
    # should be the bucket path later
    program_path = os.path.join(config["data"]["base_dir"], f"{config["data"]['program_to_extract']}")

    # implement a real list_files_in_bucket function later
    episode_files = [f for f in os.listdir(f"{program_path}\\1234") if f.endswith('.mp4')]
    print(f"Episodes: {episode_files}")
    # want to store at processed_data, then like processed_data/2001345/1234
    for episode in episode_files:
        episode_dir = os.path.join(program_path, f"{episode.split(".")[0]}")
        os.makedirs(episode_dir, exist_ok=True)

        print(f"Processing Episode {episode.split(".")[0]}")
        #scenes = extract_scenes()
        scenes = [Scene(id=1, start_frame=0, end_frame=30*10), 
                  Scene(id=2, start_frame=30*10, end_frame=30*20),
                  Scene(id=3, start_frame=30*20, end_frame=30*30)] # hardcoded for testing purposes
        for scene in scenes:
            print(f"Processing Scene {scene.id}")
            extract_cropped_interpreter_frames(
                episode_dir, 
                episode, 
                scene,
                config['scene_extraction']['YOLO_model'] 
                #config['scene_extraction']['tracking_model']
            )
            
def main():
    config = load_config()
    if config:
        process_program(config)
    play_scene_folder("data\\programs\\2001345\\1234\\scene_2")

if __name__ == "__main__":
    main()