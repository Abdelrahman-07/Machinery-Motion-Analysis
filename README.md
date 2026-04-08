# Machinery Utilization \& Activity Classification System

A real-time, microservices-based pipeline that processes video footage of construction equipment, tracks utilization states, classifies work activities using a custom-trained machine learning model, and streams results through Apache Kafka to a live dashboard.

\---

## Table of Contents

* [Architecture Overview](#architecture-overview)
* [Project Structure](#project-structure)
* [Services](#services)
* [Prerequisites](#prerequisites)
* [Setup Instructions](#setup-instructions)
* [Configuration](#configuration)
* [Kafka Payload Format](#kafka-payload-format)
* [Dashboard](#dashboard)
* [Training the Model](#training-the-model)

\---

## Architecture Overview

```
Video File
    │
    ▼
┌─────────────────────────────┐
│       CV Service            │
│  - YOLOv8 (custom model)    │
│  - Activity Classification  │
│  - Frame annotation         │
└────────────┬────────────────┘
             │ JSON Payload
             ▼
┌─────────────────────────────┐
│      Apache Kafka           │
│  Topic: equipment-events    │
└────────────┬────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌─────────────────────────┐
│ Consumer │   │     Stream Service      │
│ Service  │   │  Flask MJPEG Server     │
│ (DB sink)│   │  Serves annotated frames│
└────┬─────┘   └───────────┬─────────────┘
     │                     │
     ▼                     ▼
┌──────────────┐   ┌───────────────────┐
│ TimescaleDB  │   │   Streamlit UI    │
│ (PostgreSQL) │◄──│   Live Dashboard  │
└──────────────┘   └───────────────────┘
```

### Data Flow

1. The **CV Service** reads a video file frame by frame, runs object detection and activity classification using a custom YOLOv8 model, annotates each frame with bounding boxes, and publishes JSON payloads to Kafka.
2. **Apache Kafka** acts as the message broker, decoupling the CV service from the rest of the system.
3. The **Consumer Service** reads from Kafka and writes each frame's analysis result to TimescaleDB.
4. The **Stream Service** serves annotated frames as a smooth MJPEG video stream over HTTP.
5. The **Streamlit UI** reads from TimescaleDB and renders a live dashboard showing equipment status, activity, and utilization metrics. The video feed is embedded directly from the Stream Service.

\---

## Project Structure

```
machinery-motion-analysis/
├── docker-compose.yml
├── README.md
├── Recording.mp4
├── TECHNICAL\_WRITEUP.md
│
├── cv_service/
│   ├── main.py            # Orchestrator — reads video, runs pipeline
│   ├── detector.py        # YOLOv8 detection and tracking
│   ├── classifier.py      # Prediction smoother
│   ├── producer.py        # Kafka producer with retry logic
│   ├── models/
│   │   └── best.pt        # Custom trained YOLOv8 weights
│   ├── Dockerfile
│   └── requirements.txt
│
├── consumer_service/
│   ├── main.py            # Kafka consumer — writes to TimescaleDB
│   ├── Dockerfile
│   └── requirements.txt
│
├── stream_service/
│   ├── app.py             # Flask MJPEG stream server
│   ├── Dockerfile
│   └── requirements.txt
│
├── streamlit_ui/
│   ├── app.py             # Live dashboard
│   ├── Dockerfile
│   └── requirements.txt
│
├── db/
│   └── init.sql           # TimescaleDB schema
│
└── videos/
    └── sample.mp4         # Input video file
```

\---

## Services

|Service|Technology|Port|Description|
|-|-|-|-|
|zookeeper|Confluent Zookeeper 7.4.0|2181|Kafka coordinator|
|kafka|Confluent Kafka 7.4.0|9092|Message broker|
|timescaledb|TimescaleDB latest-pg14|5432|Time-series database|
|cv_service|Python 3.10 + YOLOv8|—|Computer vision pipeline|
|consumer_service|Python 3.10|—|Kafka → DB sink|
|stream_service|Python 3.10 + Flask|5000|MJPEG video stream|
|streamlit_ui|Python 3.10 + Streamlit|8501|Live dashboard|

\---

## Prerequisites

* **Docker Desktop** (with WSL2 on Windows)
* **Docker Compose** v2.0+
* **4GB+ RAM** available for Docker
* A video file of construction equipment in `.mp4` format
* A trained YOLOv8 model (`best.pt`) — see [Training the Model](#training-the-model)

### Installing Docker on Windows

1. Enable WSL2:

```powershell
wsl --install
```

2. Download and install Docker Desktop from https://www.docker.com/products/docker-desktop
3. During installation, ensure **"Use WSL 2 instead of Hyper-V"** is checked.
4. Verify installation:

```powershell
docker --version
docker compose version
```

\---

## Setup Instructions

### 1\. Clone or create the project folder

```powershell
mkdir machinery-motion-analysis
cd machinery-motion-analysis
```

### 2\. Add your video file

```powershell
mkdir videos
copy C:\\path\\to\\your\\video.mp4 videos\\sample.mp4
```

### 3\. Add your trained model weights

```powershell
mkdir cv\_service\\models
copy C:\\path\\to\\best.pt cv\_service\\models\\best.pt
```

### 4\. Set your Roboflow API key (if using hosted API)

Edit `docker-compose.yml` and replace `your\_api\_key\_here`:

```yaml
environment:
  ROBOFLOW\_API\_KEY: your\_api\_key\_here
```

### 5\. Build all Docker images

```powershell
docker compose build
```

This takes 5–15 minutes on first run as it downloads base images and installs dependencies.

### 6\. Start all services

```powershell
docker compose up
```

### 7\. Open the dashboard

Navigate to **http://localhost:8501** in your browser.

The video feed is available directly at **http://localhost:5000/video**.

### 8\. Useful commands

```powershell
# View logs for a specific service
docker compose logs cv\_service --follow
docker compose logs consumer\_service --follow

# Stop all services
docker compose down

# Stop and wipe all data (full reset)
docker compose down -v

# Rebuild a single service after code changes
docker compose build cv\_service
docker compose restart cv\_service

# Check service status
docker compose ps

# Open a terminal inside a container
docker compose exec cv\_service bash

# Query the database directly
docker compose exec timescaledb psql -U eagle -d equipmentdb -c "SELECT COUNT(\*) FROM equipment\_events;"
```

\---

## Configuration

All configuration is managed through environment variables in `docker-compose.yml`.

|Variable|Service|Default|Description|
|-|-|-|-|
|`KAFKA_BROKER`|cv_service, consumer_service|`kafka:9092`|Kafka broker address|
|`VIDEO_PATH`|cv_service|`/app/videos/sample.mp4`|Path to input video inside container|
|`MODEL_PATH`|cv_service|`/app/models/best.pt`|Path to YOLOv8 weights inside container|
|`DB_HOST`|consumer_service, streamlit_ui|`timescaledb`|Database hostname|
|`DB_USER`|consumer_service, streamlit_ui|`eagle`|Database username|
|`DB_PASS`|consumer_service, streamlit_ui|`eaglepass`|Database password|
|`DB_NAME`|consumer_service, streamlit_ui|`equipmentdb`|Database name|

\---

## Kafka Payload Format

The CV service publishes one JSON message per detected machine per frame to the `equipment-events` topic:

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "0:00:15.000",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "utilization_percent": 83.3
  }
}
```

### Activity Classes

|Class|State|Description|
|-|-|-|
|Excavator-Digging|ACTIVE|Arm and bucket moving into ground|
|Excavator-Swinging|ACTIVE|Cab rotating to load truck|
|Excavator-Dumping|ACTIVE|Bucket releasing material|
|Excavator-Waiting|ACTIVE|Engine running, no productive work|
|Excavator-Inactive|INACTIVE|Completely stationary|
|Dumptruck-Moving|ACTIVE|Truck driving|
|Dumptruck-Waiting|ACTIVE|Engine running, stationary|
|Dumptruck-Inactive|INACTIVE|Completely stationary|

### Motion Source Values

|Value|Meaning|
|-|-|
|`arm_only`|Only the excavator arm is moving (tracks stationary)|
|`full_body`|Entire machine is moving|
|`none`|No movement detected|

\---

## Dashboard

The Streamlit dashboard at **http://localhost:8501** displays:

* **Live Video Feed** — annotated video stream with color-coded bounding boxes per activity
* **Equipment Status** — real-time state (ACTIVE/INACTIVE), current activity, and motion source per machine
* **Utilization Metrics** — total tracked time, active time, idle time, and utilization percentage
* **Charts** — pie chart showing active vs idle time breakdown, bar chart showing utilization vs 70% target

### Bounding Box Colors

|Color|Activity|
|-|-|
|Green|Digging|
|Orange|Swinging|
|Red|Dumping|
|Purple|Inactive|
|Cyan|Moving|
|Gray|Waiting|

\---

## Training the Model

The activity classification model is a custom YOLOv8s model trained on a labeled dataset of construction equipment images.

### Dataset

The dataset was created using Roboflow with the following classes:

* `Excavator-Inactive`, `Excavator-Digging`, `Excavator-Swinging`, `Excavator-Dumping`, `Excavator-Waiting`, `Excavator-Moving` 
* `Dumptruck-Inactive`, `Dumptruck-Moving`, `Dumptruck-Waiting`

Frames were extracted from construction site videos at 2 frames per second, then manually annotated with tight bounding boxes per machine per activity.

Dataset split: **70% train / 20% validation / 10% test**

Augmentations applied: horizontal flip, ±15° rotation, ±25% brightness, blur up to 1.5px.

### Training

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="equipment-activity/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,       # GPU
    patience=10,
)
```

Training was performed on an NVIDIA RTX 4060 (8GB VRAM) and completed in approximately 20–30 minutes.

The resulting `best.pt` weights file is placed in `cv\_service/models/` and mounted into the Docker container at runtime.

