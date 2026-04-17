import cv2
import numpy as np
from PIL import Image

class FramePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self.target_size)

    def to_pil(self, frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))