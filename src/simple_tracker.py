import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(boxA, boxB):
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

class SimpleTracker:
    def __init__(self, max_lost=5, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}  # id: {"box": [...], "class": id, "lost": 0}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        detections: list of tuples (bbox, class_id)
        bbox: [x1, y1, x2, y2]
        """
        updated_tracks = {}
        det_boxes = [d[0] for d in detections]
        det_classes = [d[1] for d in detections]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]

        if len(track_boxes) == 0:
            for box, cls in zip(det_boxes, det_classes):
                updated_tracks[self.next_id] = {"box": box, "class": cls, "lost": 0}
                self.next_id += 1
            self.tracks = updated_tracks
            return self.tracks

        # Compute cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(track_boxes), len(det_boxes)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                cost_matrix[i][j] = 1 - compute_iou(tb, db)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_det = set()
        matched_tracks = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < (1 - self.iou_threshold):
                tid = track_ids[r]
                updated_tracks[tid] = {
                    "box": det_boxes[c],
                    "class": det_classes[c],
                    "lost": 0
                }
                matched_det.add(c)
                matched_tracks.add(tid)

        # Unmatched detections → new track
        for i, (box, cls) in enumerate(zip(det_boxes, det_classes)):
            if i not in matched_det:
                updated_tracks[self.next_id] = {"box": box, "class": cls, "lost": 0}
                self.next_id += 1

        # Unmatched tracks → increment lost count
        for tid in self.tracks:
            if tid not in matched_tracks:
                old = self.tracks[tid]
                old["lost"] += 1
                if old["lost"] <= self.max_lost:
                    updated_tracks[tid] = old  # keep it temporarily

        self.tracks = updated_tracks
        return self.tracks
