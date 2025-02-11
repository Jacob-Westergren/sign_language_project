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
    
    main()

    if True:
        for i in range(12):
            try:
                file = Path(f"data/programs/3001345/2345/poses/scene_{i}.pose") # this has higher x1 and x2 since it's on right side, 
                with open(file, 'rb') as pose_file:
                    pose = Pose.read(pose_file.read())  
                
                pose_visualizer = PoseVisualizer(pose)
                plain_video = pose_visualizer.draw(max_frames=30*10)
                #plain_video = pose_visualizer.draw_on_video("data/programs/3001345/2345/Rapport_Sample.mp4", max_frames=30*10)

                # Visualize the frames
                for frame in plain_video:
                    cv2.imshow("Pose Visualization", frame)
                    
                    # 1ms delay between frames
                    if cv2.waitKey(1) & 0xFF == ord('q'):  
                        break
            except:
                print(f"No scene {i}, skipping.")
                continue
        
        # Release resources and close windows
        cv2.destroyAllWindows()
