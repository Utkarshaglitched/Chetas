import time
from retirival import cosine
from load import (yolo_model,picam2,facial_app)
from variables.variables import facial
from variables import state
import threading


picam2.start()

def vision():


    while state.vision_event.is_set():
        frame = picam2.capture_array()
        

        objectArray = []
        personArray = []

        results = yolo_model(frame,verbose=False)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                obj = list(box.cls)

                for j in obj:

                    if int(j) == 0:
                        x1, y1, x2, y2 = box.xyxy[0]

                        f = frame[
                            int(y1):int(y2),
                            int(x1):int(x2)
                        ]

                        person = person_check(f)
                        personArray.append(person)

                    else:
                        objectArray.append(
                            result.names[int(j)]
                        )

        state.vision_context_frame={
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
        
        return "unknown"
