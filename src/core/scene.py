from dataclasses import dataclass
from typing import TypedDict, Tuple

@dataclass
class Scene:
    id: int
    start_frame: int
    end_frame: int

class SceneData(TypedDict):
    scene_id: int
    start_frame: int
    end_frame: int
    interpreter_crop: Tuple[int, int, int, int]
    interpreter_frequency: int


