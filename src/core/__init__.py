# Core package initialization
# Define what is imported when the package "core" is imported
from .structures.scene import Scene
from .structures.person import DetectedPerson
from .data_extraction.scene_extractor import SceneExtractor
from .data_extraction.video_processor import VideoProcessor
from .data_preprocessing import comp_mean_std, normalize_poses

