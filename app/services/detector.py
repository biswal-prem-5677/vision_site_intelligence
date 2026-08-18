from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from app.config import YOLO_MODEL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, PERSON_CLASS
from app.schemas import Detection


class Detector:
    def __init__(self, use_cpu: bool = False):
        self.model = YOLO(YOLO_MODEL)
        if not use_cpu:
            try:
                self.model.to("cuda")
            except Exception:
                pass  # falls back to CPU silently

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(
                Detection(
                    track_id=None,
                    class_name=self.model.names[cls_id],
                    confidence=conf,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        return detections
