from dataclasses import dataclass

class Scene:
    def __init__(self, id: int, start_frame: int, end_frame: int):
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame


