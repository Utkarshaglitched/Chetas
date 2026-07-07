from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
from libcamera import Transform
import time

# ----------------------------
# Load YOLO Model
# ----------------------------
model = YOLO("yolov8n.pt")      # Change if your model name is different

# ----------------------------
# Initialize Camera
# ----------------------------
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (416, 416), "format": "RGB888"},
    buffer_count=4,
    transform=Transform(vflip=True)
)

picam2.configure(config)
picam2.start()

# ----------------------------
# FPS Variables
# ----------------------------
prev_time = time.perf_counter()

# ----------------------------
# Main Loop
# ----------------------------
while True:

    # Capture frame
    frame = picam2.capture_array()

    # YOLO Inference
    result = model(
        frame,
        imgsz=416,
        conf=0.5,
        verbose=False
    )[0]

    # Draw detections
    annotated = result.plot()

    # Calculate FPS
    current_time = time.perf_counter()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show frame
    cv2.imshow("YOLO Detection", annotated)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()