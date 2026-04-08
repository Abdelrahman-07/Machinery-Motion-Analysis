import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch

# Set device to GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
# Map model output classes to our standard activity names
CLASS_MAP = {
    # Excavator classes
    "excavator-inactive": ("excavator", "INACTIVE", "INACTIVE"),
    "excavator-moving":   ("excavator", "ACTIVE",   "DIGGING"),
    "excavator-dumping":  ("excavator", "ACTIVE",   "DUMPING"),
    "excavator-swinging": ("excavator", "ACTIVE",   "SWINGING"),
    "excavator-digging":  ("excavator", "ACTIVE",   "DIGGING"),
    "excavator-waiting":  ("excavator", "ACTIVE",   "WAITING"),

    # Dumptruck classes
    "dumptruck-moving":   ("dump truck", "ACTIVE",   "MOVING"),
    "dumptruck-inactive": ("dump truck", "INACTIVE", "INACTIVE"),
    "dumptruck-waiting":  ("dump truck", "ACTIVE",   "WAITING"),
}

class EquipmentDetector:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "/app/models/best.pt")
        print(f"Loading local model from {model_path}...")
        self.model = YOLO(model_path)
        self.trackers = {}
        self.next_id  = 1
        print("Model loaded successfully!")
        print(f"Classes: {self.model.names}")

    def detect(self, frame):
        # model.track() = detect objects AND assign consistent IDs
        # Regular model.predict() detects but gives different IDs each frame
        # model.track() remembers "this is the same machine as last frame"
        # persist=True → keep track memory between frames
        # verbose=False → suppress per-frame console output
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.3
        )

        detections = []
        # If YOLO found nothing in this frame, return empty list
        if results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            # box.cls = class index of what was detected
            cls_id   = int(box.cls[0])
            # box.conf = confidence score between 0.0 and 1.0
            # 1.0 = completely certain, 0.0 = completely uncertain
            conf     = float(box.conf[0])
            # box.id = tracker's consistent ID for this object
            # Same physical machine gets same ID across frames
            # box.id can be None if tracking fails — default to 0
            track_id = int(box.id[0]) if box.id is not None else self.next_id

            # Discard detections below 30% confidence
            # Reduces false positives
            if conf < 0.3:
                continue
            
            # box.xyxy = bounding box as [x1, y1, x2, y2]
            # x1,y1 = top-left corner pixel
            # x2,y2 = bottom-right corner pixel
            # map(int, ...) converts float coordinates to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get raw class name from model
            raw_class = self.model.names[cls_id].lower()

            # Look up our standardized mapping
            mapped = CLASS_MAP.get(raw_class)
            if mapped is None:
                # Try partial match
                mapped = self._fuzzy_match(raw_class)
            if mapped is None:
                continue  # Skip unknown classes

            eq_class, state, activity = mapped

            detections.append({
                "bbox":     (x1, y1, x2, y2),
                "class":    eq_class,
                "confidence": conf,
                "id":       track_id,
                # Activity comes directly from model — no classifier needed
                "state":    state,
                "activity": activity,
            })

        return detections

    def _fuzzy_match(self, raw_class):
        """Try to match class name even if format is slightly different"""
        raw_lower = raw_class.lower().replace(" ", "-").replace("|", "-").replace("/", "-")
        for key, value in CLASS_MAP.items():
            key_lower = key.lower().replace(" ", "-").replace("|", "-").replace("/", "-")
            if key_lower in raw_lower or raw_lower in key_lower:
                return value
        return None