import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from core import VideoProcessor, comp_mean_std, normalize_poses
from pose_format.pose_visualizer import PoseVisualizer
from pathlib import Path
import cv2
from pose_format import Pose
import matplotlib.pyplot as plt

# Load .env file so cfg can read env variables
load_dotenv()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Extract poses from videos
    if True:
        print("Starting the video processor.")
        processor = VideoProcessor(cfg)
        print(f"Calling process_episodes.")
        processor.process_episodes()
        print("Finished the video processor.")

    # Compute mean and std for poses
    print(f"Computing mean and std of all poses")
    pose_dir = Path("data/programs/3001345/2345/poses")
    pose_norm_dir = pose_dir.parent / "poses_mean_std"
    pose_dir.mkdir(exist_ok=True)
    pose_norm_dir.mkdir(exist_ok=True)
    comp_mean_std(pose_dir, pose_norm_dir)

    # Normalize poses using computed mean and std
    print("Normalizing all poses using computed mean and std values")
    ziped_pose_norm_file = pose_dir.parent / "norm_poses_ziped"
    normalize_poses(pose_dir, pose_norm_dir, ziped_pose_norm_file)
    print(f"Done. Zipped normalized pose data can be found in {ziped_pose_norm_file}")

if __name__ == "__main__":
    print(f"Hello World!")
    main()

    if False:
        for i in range(12):
            try:
                file = Path(f"data/programs/3001345/2345/poses/scene_{i}.pose") # this has higher x1 and x2 since it's on right side, 
                with open(file, 'rb') as pose_file:
                    pose = Pose.read(pose_file.read())  
                
                pose_visualizer = PoseVisualizer(pose)
                plain_video = pose_visualizer.draw(max_frames=30*10)
                
                """
                for frame in plain_video:
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.clf()
                """
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
