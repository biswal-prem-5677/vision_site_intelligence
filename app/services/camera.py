import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET


class Camera:
    def __init__(self, source=None):
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.source = source if source is not None else CAMERA_INDEX

    def start(self, source=None) -> bool:
        if source is not None:
            self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        if self.cap.isOpened():
            self.running = True
            return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.running or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret and isinstance(self.source, str):
            # Auto-loop video file for continuous demo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def release(self):
        self.stop()
