import time
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from cosine import cosine
from variables import facial
from picamera2 import Picamera2
from libcamera import Transform

app=FaceAnalysis()
model=YOLO("Models/yolov8n.pt")
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (416, 416), "format": "RGB888"},
    buffer_count=4,
    transform=Transform(vflip=True)
)

picam2.configure(config)
picam2.start()

def vision():
    start = time.perf_counter()
    time.sleep(0.1)

    FrameArray = []

    while (time.perf_counter() - start) <= 1:
        frame = picam2.capture_array()
        FrameArray.append(frame)

    objectArray = []
    personArray = []

    if len(FrameArray) > 0:
        frme = FrameArray[int(len(FrameArray) / 2)]

        results = model(frme,verbose=False)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                obj = list(box.cls)

                for j in obj:

                    if int(j) == 0:
                        x1, y1, x2, y2 = box.xyxy[0]

                        f = frme[
                            int(y1):int(y2),
                            int(x1):int(x2)
                        ]

                        person = person_check(f)
                        personArray.append(person)

                    else:
                        objectArray.append(
                            result.names[int(j)]
                        )

        return {
            "persons": personArray,
            "objects": objectArray
        }
    

def person_check(face):
    if app.get(face):
        embed=app.get(face)[0].embedding

        cnt_sim=[]
  
        for i in facial:
            face_embed=facial[i]
            cnt_sim.append(cosine.consimilaritry(embed,face_embed))

        max_sim_index=cnt_sim.index(max(cnt_sim))
        if cnt_sim[max_sim_index]>0.6:  
            return list(facial.keys())[max_sim_index]
        
        return "Uknown"
