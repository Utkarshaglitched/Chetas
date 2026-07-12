import time
from retirival import cosine
from load import (yolo_model,picam2,facial_app)
from variables import facial

picam2.start()

def vision():
    start = time.perf_counter()

    FrameArray = []

    while (time.perf_counter() - start) <= 1:
        frame = picam2.capture_array()
        FrameArray.append(frame)

    objectArray = []
    personArray = []

    if len(FrameArray) > 0:
        frme = FrameArray[int(len(FrameArray) / 2)]

        results = yolo_model(frme,verbose=False)

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
    if facial_app.get(face):
        embed=facial_app.get(face)[0].embedding

        cnt_sim=[]
  
        for i in facial:
            face_embed=facial[i]
            cnt_sim.append(cosine.consimilaritry(embed,face_embed))

        max_sim_index=cnt_sim.index(max(cnt_sim))
        if cnt_sim[max_sim_index]>0.6:  
            return list(facial.keys())[max_sim_index]
        
        return "Uknown"
