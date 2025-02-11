import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from core import VideoProcessor
from pose_format.pose_visualizer import PoseVisualizer
from pathlib import Path
import cv2
from pose_format import Pose
# Load .env file so cfg can read env variables
load_dotenv()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting the video processor.")
    processor = VideoProcessor(cfg)
    processor.process_episodes()
    print("Finished the video processor.")

if __name__ == "__main__":
    
    #main()
    file = Path("data/programs/2001345/1234/poses/scene_1.pose")
    with open(file, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())  
    
    print(pose)
    pose_visualizer = PoseVisualizer(pose)
    plain_video = pose_visualizer.draw_on_video("data/programs/2001345/1234/1234.mp4", max_frames=30*8)

    # Visualize the frames
    for frame in plain_video:
        cv2.imshow("Pose Visualization", frame)
        
        # 1ms delay between frames
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    # Release resources and close windows
    cv2.destroyAllWindows()

"""
FPS: 30.0
Data: <class 'numpy.ma.core.MaskedArray'> (240, 1, 203, 3), float32
Confidence shape: <class 'numpy.ndarray'> (240, 1, 203), float32
Duration (seconds): 8.0
"""