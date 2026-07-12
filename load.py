from ultralytics import YOLO
from insightface.app import FaceAnalysis
from cosine import cosine
from picamera2 import Picamera2
from libcamera import Transform
import pyaudio
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad,VADIterator

facial_app=FaceAnalysis()
yolo_model=YOLO("Models/yolov8n.pt")
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (640, 640), "format": "RGB888"},
    buffer_count=4,
    transform=Transform(vflip=True)
)

picam2.configure(config)


DEVICE_INDEX = 0
RATE = 16000
CHANNELS = 1
CHUNK = 512

audio=pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)

whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)


vad_model=load_silero_vad()
vad = VADIterator(
    vad_model,
    threshold=0.5,
    sampling_rate=16000
)