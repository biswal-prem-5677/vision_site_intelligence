from typing import List, Optional
import numpy as np
from app.schemas import Detection, TrackedObject
from app.config import TRACKER_MAX_AGE, TRACKER_INIT_HIT_STREAK, PERSON_CLASS


class SimpleTracker:
    """Lightweight centroid-based tracker using IoU matching."""

    def __init__(self):
        self.next_id = 1
        self.tracks: dict[int, TrackedObject] = {}
        self.age: dict[int, int] = {}
        self.hit_streak: dict[int, int] = {}

    def _box_area(self, x1, y1, x2, y2):
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _iou(self, a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = self._box_area(x1, y1, x2, y2)
        if inter == 0:
            return 0.0
        union = self._box_area(*a) + self._box_area(*b) - inter
        return inter / union if union > 0 else 0.0

    def _centroid(self, x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections: List[Detection], frame_width: int, frame_height: int) -> List[TrackedObject]:
        IOU_THRESH = 0.3
        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        n = len(det_boxes)
        used_dets = set()
        matched_tracks: set[int] = set()

        # Build IoU cost matrix
        if n > 0 and len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            for ti, tid in enumerate(track_ids):
                tb = (self.tracks[tid].x1, self.tracks[tid].y1, self.tracks[tid].x2, self.tracks[tid].y2)
                best_di, best_iou = -1, 0.0
                for di in range(n):
                    if di in used_dets:
                        continue
                    iou = self._iou(tb, det_boxes[di])
                    if iou > best_iou:
                        best_iou = iou
                        best_di = di
                if best_di >= 0 and best_iou >= IOU_THRESH:
                    tid_match = track_ids[ti]
                    d = detections[best_di]
                    cx, cy = self._centroid(d.x1, d.y1, d.x2, d.y2)
                    prev_cx, prev_cy = self.tracks[tid_match].cx, self.tracks[tid_match].cy
                    movement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    self.tracks[tid_match].x1 = d.x1
                    self.tracks[tid_match].y1 = d.y1
                    self.tracks[tid_match].x2 = d.x2
                    self.tracks[tid_match].y2 = d.y2
                    self.tracks[tid_match].confidence = d.confidence
                    self.tracks[tid_match].prev_cx = self.tracks[tid_match].cx
                    self.tracks[tid_match].prev_cy = self.tracks[tid_match].cy
                    self.tracks[tid_match].cx = cx
                    self.tracks[tid_match].cy = cy
                    self.tracks[tid_match].movement = movement
                    self.age[tid_match] = 0
                    self.hit_streak[tid_match] = self.hit_streak.get(tid_match, 0) + 1
                    used_dets.add(best_di)
                    matched_tracks.add(tid_match)

        # Age unmatched tracks, remove stale
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.age[tid] = self.age.get(tid, 0) + 1
                self.hit_streak[tid] = self.hit_streak.get(tid, 0)
                if self.age[tid] > TRACKER_MAX_AGE:
                    del self.tracks[tid]
                    del self.age[tid]
                    if tid in self.hit_streak:
                        del self.hit_streak[tid]

        # Create new tracks for unmatched detections
        for di, d in enumerate(detections):
            if di in used_dets:
                continue
            cx, cy = self._centroid(d.x1, d.y1, d.x2, d.y2)
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackedObject(
                track_id=tid,
                class_name=d.class_name,
                confidence=d.confidence,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                cx=cx,
                cy=cy,
                prev_cx=cx,
                prev_cy=cy,
                movement=0.0,
            )
            self.age[tid] = 0
            self.hit_streak[tid] = 1

        # Return confirmed tracks
        return [
            self.tracks[tid]
            for tid in self.tracks
            if self.hit_streak.get(tid, 0) >= TRACKER_INIT_HIT_STREAK
            or tid in matched_tracks
        ]
