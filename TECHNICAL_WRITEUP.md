# Technical Write-Up: Equipment Utilization \& Activity Classification System

\---

## 1\. System Overview

This system is a real-time microservices pipeline designed to process video footage of construction equipment, classify machine activity states, and compute utilization metrics. The pipeline is built around Apache Kafka as a message broker, a custom-trained YOLOv8 model for computer vision, TimescaleDB for time-series storage, and Streamlit for live visualization.

The core technical challenge this system addresses is accurately determining not just whether a machine is present in a video frame, but what it is doing — and doing so reliably in real time, frame by frame.

\---

## 2\. Architecture Decisions

### 2.1 Microservices Over Monolith

The system is decomposed into five independent services: CV, Consumer, Stream, UI, and Database. Each service has a single responsibility and communicates through well-defined interfaces — either through Kafka messages or direct database queries.

**Trade-off:** A monolithic design would be simpler to build and debug. However, the microservices approach provides several practical advantages for this use case. The CV service is computationally expensive — it runs GPU inference on every frame. Decoupling it from the UI means the dashboard remains responsive even when inference is slow. Additionally, each service can be restarted, rebuilt, or scaled independently without affecting the others.

### 2.2 Apache Kafka as Message Broker

Kafka sits between the CV service and the consumer service. The CV service produces one JSON message per detected machine per frame. The consumer service reads these messages and writes them to the database.

**Trade-off:** Kafka introduces operational complexity — it requires Zookeeper, has its own configuration, and takes time to start up. A simpler alternative would be writing directly from the CV service to the database. However, Kafka provides two important guarantees: first, if the consumer crashes, no messages are lost because Kafka retains them until they are acknowledged; second, Kafka decouples the processing speed of the CV service from the write speed of the database, preventing back-pressure.

### 2.3 TimescaleDB for Storage

TimescaleDB is a PostgreSQL extension optimized for time-series data. Each row in the `equipment_events` table is automatically partitioned by timestamp through TimescaleDB's hypertable feature.

**Trade-off:** Regular PostgreSQL would work for a prototype. TimescaleDB was chosen because the data is inherently time-series — one row per frame per machine, potentially thousands of rows per minute. TimescaleDB's time-based partitioning makes queries like "give me the last 30 seconds of data for machine EX-001" significantly faster as the dataset grows.

### 2.4 Flask MJPEG Stream for Video

A separate Flask service reads the latest annotated frame from a shared Docker volume and serves it as a continuous MJPEG (Motion JPEG) stream over HTTP. The browser connects to this stream once and receives a continuous feed of JPEG frames.

**Trade-off:** The initial approach embedded the video feed directly in Streamlit using `st.image()` inside a refresh loop. This caused the browser to freeze because Streamlit's `st.rerun()` disconnected and reconnected the image on every refresh cycle. The MJPEG stream approach moves video delivery entirely to the browser's native image rendering pipeline, which handles continuous streams without interference from Streamlit's refresh mechanism.

\---

## 3\. The Articulated Motion Challenge

### 3.1 Problem Definition

A construction excavator is an articulated machine — it has multiple independently moving parts. The cab can rotate while the tracks remain stationary. The boom, arm, and bucket can move through a full digging cycle while the machine is not traveling. This means a naive motion detection approach that asks "is this bounding box moving?" will incorrectly classify a digging excavator as inactive, because the bounding box itself may not change position even though significant work is being performed.

### 3.2 Initial Approach: Optical Flow with Region Splitting

The first implementation used Farneback dense optical flow. For each detected machine, the bounding box was split horizontally into two regions: the upper half (arm, boom, bucket) and the lower half (tracks, chassis). Optical flow was computed independently for each region, producing a motion magnitude value. If only the upper region exceeded the motion threshold, the machine was classified as `arm_only` motion — indicating active work with stationary tracks.

```
Bounding Box
┌─────────────────┐
│   Upper Half    │  ← arm, boom, bucket
│  (arm region)   │
├─────────────────┤  ← mid_y split point
│   Lower Half    │  ← tracks, chassis
│ (track region)  │
└─────────────────┘

upper_motion > threshold AND lower_motion <= threshold → arm_only → ACTIVE
upper_motion > threshold AND lower_motion > threshold  → full_body → ACTIVE
upper_motion <= threshold AND lower_motion <= threshold → none → INACTIVE
```

**Why this approach was abandoned:** Optical flow is sensitive to camera shake, lighting changes, and background motion. On real construction footage, the signal-to-noise ratio was too low to reliably distinguish genuine arm movement from environmental noise. A truck driving through the scene while stationary could produce false optical flow vectors. The threshold required manual tuning per video and did not generalize.

### 3.3 Final Approach: Custom-Trained YOLOv8 Model

The final solution treats activity classification as an object detection problem. Rather than detecting a machine and then separately analyzing its motion, the model is trained to detect machines in specific activity states directly from visual appearance.

Each class label encodes both the machine type and its current activity:

```
Excavator-Digging    → excavator with arm extended downward into ground
Excavator-Swinging   → excavator cab rotated, arm in mid-swing
Excavator-Dumping    → excavator bucket tipped, releasing material
Excavator-Waiting    → excavator with arm raised but no productive motion
Excavator-Inactive   → excavator completely stationary
Dumptruck-Moving     → dump truck in motion
Dumptruck-Waiting    → dump truck stationary with engine running
Dumptruck-Inactive   → dump truck completely stopped
```

The model learns the visual signature of each state — the position and angle of the boom, the orientation of the bucket, the posture of the arm — directly from labeled training images. This eliminates the need for motion analysis entirely and correctly handles the articulated motion problem because a digging excavator looks visually distinct from an inactive one regardless of whether the bounding box position changes between frames.

**How it solves arm-only motion:** A model trained on `Excavator-Digging` examples has learned that this class is characterized by the arm in a downward extended position and the tracks in contact with the ground. It does not need to compare consecutive frames to know this — it reads the posture from a single frame. This is more robust than optical flow because it is invariant to camera motion, lighting changes, and background movement.

### 3.4 Motion Source Derivation

Since the ML model classifies activity directly, the `motion\_source` field is derived from the predicted class rather than computed from pixel differences:

```python
def get_motion_source(equipment_class, activity):
    if activity == "INACTIVE":
        return "none"
    if "excavator" in equipment_class:
        if activity in ("DIGGING", "DUMPING", "WAITING"):
            return "arm_only"   # arm moves, tracks stationary
        elif activity == "SWINGING":
            return "full_body"  # whole cab rotates
    if "truck" in equipment_class:
        return "full_body"      # whole truck moves
    return "full_body"
```

This preserves the semantic meaning of `motion_source` as a description of physical machine behavior rather than a pixel-level measurement.

\---

## 4\. Activity Classification Design

### 4.1 Model Architecture

The activity classifier is a YOLOv8s (small) model fine-tuned on a custom dataset. YOLOv8s was chosen over the nano variant for better feature extraction on the subtle visual differences between activity states, and over larger variants to keep inference fast enough for real-time processing on CPU.

YOLOv8 was selected over a pure classification model because it simultaneously produces bounding box coordinates and class predictions in a single forward pass. This means one model call per frame handles both localization (where is the machine?) and classification (what is it doing?), rather than requiring a detection model followed by a separate classification model.

### 4.2 Dataset Construction

Training images were extracted from construction site videos at approximately 2 frames per second to ensure diversity without excessive near-duplicate frames. Each frame was manually annotated in Roboflow with tight bounding boxes drawn around each visible machine, labeled with the appropriate activity class.

Minimum annotations per class: 30 images. With Roboflow's augmentation pipeline (horizontal flip, ±15° rotation, ±25% brightness adjustment, blur up to 1.5px), each labeled image produces approximately 3 effective training samples.

Dataset split: 70% training, 20% validation, 10% test.

### 4.3 Prediction Smoothing

The raw model output can flicker between adjacent states on consecutive frames — for example, briefly predicting `Excavator-Waiting` for one frame during a `Digging` sequence. A temporal smoothing step is applied using a rolling history window of 10 frames per machine:

```python
# Vote on most common state in recent history
recent_activities = list(self.history[equipment_id])
smoothed_activity = max(set(recent_activities), key=recent_activities.count)
```

This majority-vote approach ensures that brief misclassifications do not appear in the output or affect utilization calculations. A single outlier frame must be the majority in a 10-frame window to change the reported state, which requires sustained misclassification rather than a momentary one.

### 4.4 Utilization Calculation

Utilization is calculated as a running ratio of active time to total tracked time:

```
utilization_percent = (total_active_seconds / total_tracked_seconds) × 100
```

Active time accumulates whenever the smoothed activity state is anything other than `INACTIVE`. This means `Excavator-Waiting` counts as active time because the machine's engine is running and the operator is engaged, even though no material is being moved. This matches the industry definition of equipment utilization — the machine is available and engaged, not idle or powered down.

\---

## 5\. Trade-offs and Limitations

### 5.1 Single-Frame Classification

The ML model classifies activity from a single frame. It does not have memory of previous frames (beyond the smoothing window). This means it cannot distinguish between an excavator that just started digging and one that has been digging for five minutes — both look the same to the model. For the purpose of utilization tracking, this is acceptable because we care about the current state, not the history.

### 5.2 Custom Dataset Size

The training dataset is small by deep learning standards. A larger dataset with more diverse lighting conditions, camera angles, and equipment types would produce a more robust model. The current model performs well on footage similar to its training data but may struggle with significantly different visual conditions.

### 5.3 Kafka Consumer Group Offsets

In development, restarting the consumer service without resetting the Kafka consumer group offset causes the consumer to skip already-processed messages. This is by design — Kafka's offset tracking prevents double-processing in production. For development, a unique group ID per session is used to always read from the beginning of the topic.

### 5.4 Video Speed vs Inference Speed

YOLOv8 inference on CPU takes longer than the natural frame duration of the video. The CV service compensates by subtracting inference time from the sleep interval between frames, but if inference consistently exceeds the frame duration the video will play slower than real time. GPU inference eliminates this issue entirely, but due to time constraints GPU inference results are not included however I am currently testing it to ensure it works within the docker container and will update the GitHub repo as soon as possible.

\---

## 6\. Summary of Design Choices

|Decision|Choice Made|Key Reason|
|-|-|-|
|Activity detection method|Custom YOLOv8 model|Handles articulated motion without optical flow|
|Message broker|Apache Kafka|Decoupling, fault tolerance, replay|
|Database|TimescaleDB|Time-series optimized queries|
|Video streaming|Flask MJPEG|Smooth playback independent of Streamlit refresh|
|Prediction smoothing|10-frame majority vote|Eliminates single-frame misclassifications|
|Model size|YOLOv8s|Balance of accuracy and inference speed|
|Utilization definition|Active = any non-INACTIVE state|Matches industry standard|



