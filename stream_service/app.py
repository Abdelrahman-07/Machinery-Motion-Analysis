import cv2
import os
import time
from flask import Flask, Response

app  = Flask(__name__)
FRAMES_DIR = "/app/frames"

def generate_frames():
    """
    Continuously read latest_frame.jpg and yield it as MJPEG stream
    MJPEG = Motion JPEG = series of JPEG frames sent over HTTP
    Browser treats this exactly like a live webcam feed
    """
    while True:
        frame_path = os.path.join(FRAMES_DIR, "latest_frame.jpg")

        if os.path.exists(frame_path):
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()

            # MJPEG multipart format — each frame is wrapped in this boundary
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )

        # Serve at ~25fps
        time.sleep(0.04)

@app.route("/video")
def video_feed():
    return Response(
        generate_frames(),
        # This content type tells the browser to expect
        # a continuous stream of JPEG images
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # 0.0.0.0 = accept connections from outside the container
    app.run(host="0.0.0.0", port=5000, threaded=True)