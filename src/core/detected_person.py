from dataclasses import dataclass

@dataclass
class Detected_Person:
    crop: tuple[int,int,int,int]
    start_frame: int
    end_frame: int = None
    freq: int = 1

    def update(self, new_end_frame):
        self.end_frame = new_end_frame
        self.freq += 1