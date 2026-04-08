import cv2
import os
import time
import datetime
from detector import EquipmentDetector
from classifier import ActivityClassifier
from producer import create_producer, send_payload
import torch

# Set device to GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
# ── ENVIRONMENT VARIABLES ─────────────────────────────────────────
# These are set in docker-compose.yml and read here at runtime
# The model path points to the mounted models folder inside the container
# docker-compose.yml maps D:\...\cv_service\models → /app/models
VIDEO_PATH  = os.getenv("VIDEO_PATH", "/app/videos/sample.mp4")
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/models/best.pt")
FRAMES_DIR  = "/app/frames"

# ── BOUNDING BOX COLORS ───────────────────────────────────────────
# Each activity gets a distinct color for its bounding box
# Colors are in BGR format (Blue, Green, Red) — OpenCV uses BGR not RGB
ACTIVITY_COLORS = {
    "DIGGING":  (0, 255, 0),       # Green
    "SWINGING": (0, 165, 255),     # Orange
    "DUMPING":  (0, 0, 255),       # Red
    "WAITING":  (128, 128, 128),   # Gray
    "MOVING":   (255, 255, 0),     # Cyan
    "INACTIVE": (255, 50, 255),    # Purple
}

def draw_annotations(frame, det, is_active, activity, util_pct):
    """
    Draw bounding box and info labels on the frame for one detection
    det      = detection dict from detector.py
    is_active = True if machine is working
    activity  = string like "DIGGING", "WAITING" etc
    util_pct  = utilization percentage so far
    """
    x1, y1, x2, y2 = det["bbox"]

    # Format the equipment ID as a readable string e.g. "EX-001"
    eq_id = f"EX-{det['id']:03d}"

    # Pick color based on current activity
    # If activity not in dict, default to white
    color = ACTIVITY_COLORS.get(activity, (255, 255, 255))

    # Draw the bounding box rectangle around the machine
    # 2 = line thickness in pixels
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Three label lines to display above the bounding box
    state_text = "ACTIVE" if is_active else "INACTIVE"
    labels = [
        f"{eq_id} | {det['class'].upper()}",        # e.g. "EX-001 | DUMP TRUCK"
        f"{state_text} | {activity}",               # e.g. "ACTIVE | MOVING"
        f"Util: {util_pct}% | Conf: {det['confidence']:.0%}"  # e.g. "Util: 83.3% | Conf: 91%"
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    padding    = 4

    # Draw each label line one at a time, stacked above the bounding box
    for i, label in enumerate(labels):
        # Measure how many pixels wide and tall this text will be
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Position each line above the bounding box
        # i=0 is closest to box, i=1 above that, i=2 above that
        label_y = y1 - 10 - (i * (text_h + padding + 4))

        # If label would go above the top edge of the frame,
        # put it below the bounding box instead
        if label_y - text_h - padding < 0:
            label_y = y2 + 20 + (i * (text_h + padding + 4))

        # Draw filled colored rectangle as background behind text
        # Makes text readable regardless of what's in the background
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - padding),
            (x1 + text_w + padding, label_y + padding),
            color,
            -1    # -1 = filled rectangle
        )

        # Draw text in black on top of colored background
        cv2.putText(
            frame, label,
            (x1 + padding, label_y),
            font, font_scale,
            (0, 0, 0),    # black text
            thickness
        )

    return frame

def save_frame(frame):
    """
    Save the annotated frame to the shared Docker volume
    Streamlit's Flask stream service reads from this same location
    We always overwrite the same file so only the latest frame is kept
    This prevents the disk from filling up
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(FRAMES_DIR, "latest_frame.jpg"), frame)
def get_motion_source(equipment_class: str, activity: str)->str:
    """
    Derive the physical motion source from equipment type and activity
    
    Excavators:
      - DIGGING  → only the arm/bucket moves, tracks are stationary
      - SWINGING → full body rotation (cab swings)
      - DUMPING  → arm only, releasing bucket contents
      - WAITING  → arm may have slight movement, tracks still
      - INACTIVE → nothing moving
    
    Dump Trucks:
      - MOVING   → full body moves (whole truck drives)
      - WAITING  → engine running, whole body stationary
      - INACTIVE → nothing moving
    """
    cls = equipment_class.lower()
    act = activity.upper()
    if act == "INACTIVE":
        return "none"

    if "excavator" in cls:
        if act == "DIGGING":
            return "arm_only"
        elif act == "SWINGING":
            return "full_body"
        elif act == "DUMPING":
            return "arm_only"
        elif act == "WAITING":
            return "arm_only"
        else:
            return "arm_only"

    if "truck" in cls or "dumptruck" in cls:
        if act == "MOVING":
            return "full_body"
        elif act == "WAITING":
            return "full_body"
        else:
            return "full_body"

    # Default for unknown equipment
    return "full_body"


def process_video(cap, fps, producer, detector, classifier, time_tracker, frame_id):
    """
    Process one full pass through the video
    Returns the updated frame_id after processing all frames
    """
    # How many real seconds each frame represents
    # e.g. at 30fps: 1/30 = 0.033 seconds per frame
    frame_duration = 1.0 / fps

    while cap.isOpened():
        # Record the start time for this frame.
        frame_start = time.time()
        # Read the next frame from the video file
        # ret = True if frame was read successfully
        # ret = False when video ends
        ret, frame = cap.read()
        if not ret:
            # Video has ended — return current frame_id
            print("Video ended — restarting...")
            break

        # Run YOLO detection and tracking on this frame
        # Returns list of dicts with bbox, class, state, activity, id
        detections = detector.detect(frame)

        # Make a copy of the frame to draw annotations on
        # We keep the original clean for detection
        annotated_frame = frame.copy()

        for det in detections:
            # Format tracker ID as readable equipment ID
            # :03d = zero-padded to 3 digits e.g. 1 → "001"
            eq_id = f"EX-{det['id']:03d}"

            # Smooth the model's predictions over recent frames
            # Prevents single-frame misclassifications from showing
            is_active, activity = classifier.classify(
                eq_id,
                det["state"],
                det["activity"]
            )

            # Initialize time tracking for newly detected machines
            if eq_id not in time_tracker:
                time_tracker[eq_id] = {"active": 0.0, "total": 0.0}

            # Add this frame's duration to running totals
            time_tracker[eq_id]["total"] += frame_duration

            # Only count as active time if machine is actually working
            if is_active:
                time_tracker[eq_id]["active"] += frame_duration

            # Calculate utilization statistics
            total    = time_tracker[eq_id]["total"]
            active   = time_tracker[eq_id]["active"]
            idle     = total - active

            # Utilization % = what fraction of tracked time was active
            util_pct = round((active / total) * 100, 1) if total > 0 else 0.0

            # Draw bounding box and labels on the annotated frame
            annotated_frame = draw_annotations(
                annotated_frame, det, is_active, activity, util_pct
            )

            # Convert frame number to readable timestamp HH:MM:SS.mmm
            timestamp = str(datetime.timedelta(seconds=frame_id / fps))
            # Get the motion source of the machine
            motion_source = get_motion_source(det["class"],activity)
            # Build JSON payload matching the required Kafka format
            payload = {
                "frame_id":        frame_id,
                "equipment_id":    eq_id,
                "equipment_class": det["class"],
                "timestamp":       timestamp,
                "utilization": {
                    # State comes from the ML model via classifier
                    "current_state":    "ACTIVE" if is_active else "INACTIVE",
                    "current_activity": activity,
                    # motion_source tells us this came from ML not pixel analysis
                    "motion_source":    motion_source
                },
                "time_analytics": {
                    "total_tracked_seconds": round(total, 2),
                    "total_active_seconds":  round(active, 2),
                    "total_idle_seconds":    round(idle, 2),
                    "utilization_percent":   util_pct
                }
            }

            # Send payload to Kafka topic "equipment-events"
            send_payload(producer, payload)
            print(f"[Frame {frame_id}] {eq_id} → {activity} ({util_pct}%)")

        # Save annotated frame to shared volume for Flask stream service
        save_frame(annotated_frame)

        # Increment frame counter
        frame_id += 1

        # Throttle processing to real video speed
        # Without this the CV service would process frames as fast as possible
        # making the video feed appear sped up
        elapsed = time.time() - frame_start
        sleep_time = max(0, frame_duration - elapsed)
        time.sleep(0)

    return frame_id

def main():
    print("Starting CV Service...")

    # Wait for Kafka to fully initialize before attempting connection
    # Even after healthcheck passes, Kafka needs a moment to stabilize
    print("Waiting 15 seconds for Kafka to stabilize...")
    time.sleep(15)

    # Initialize all components
    producer   = create_producer()       # Kafka connection with retry
    detector   = EquipmentDetector()     # YOLOv8 local model
    classifier = ActivityClassifier()    # Prediction smoother

    # Track cumulative active and total time per machine
    # Persists across video loops so utilization keeps accumulating
    # e.g. {"EX-001": {"active": 12.5, "total": 15.0}}
    time_tracker = {}

    # Global frame counter — keeps incrementing across video loops
    # This ensures Kafka messages always have unique frame IDs
    frame_id = 0

    # Outer loop — restarts video when it ends
    while True:
        print(f"Opening video: {VIDEO_PATH}")

        # Open the video file for reading
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            # Video file not found or corrupted
            print(f"ERROR: Cannot open video file: {VIDEO_PATH}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Get the video's frames per second
        # Used to calculate real timestamps and throttle processing speed
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print(f"Video opened successfully — {fps:.1f} FPS")

        # Process the entire video, returns updated frame_id
        frame_id = process_video(
            cap, fps, producer, detector,
            classifier, time_tracker, frame_id
        )

        # Release the video file handle before reopening
        cap.release()

        # Brief pause before restarting to avoid hammering the disk
        print("Restarting video in 1 second...")
        time.sleep(1)

if __name__ == "__main__":
    main()