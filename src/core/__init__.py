# Core package initialization
# Define what is imported when the package "core" is imported
from .models.scene import Scene
from .models.person import DetectedPerson
from .processors.scene_extractor import SceneExtractor
from .processors.video_processor import VideoProcessor

