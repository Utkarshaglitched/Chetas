import model.model_selector as model_selector
import time
import numpy as np
import os
from load import (stream,vad,CHUNK,whisper_model)
import torch
import speak
from variables import state,variables
import threading
from retirival.vision import vision
from data.LTM import start_ltm_process,test


voice_detected=False

state.vision_event.set()
vision_thread=threading.Thread(target=vision,daemon=True)
vision_thread.start()
print("starting camera")

ltm_thread=threading.Thread(target=start_ltm_process,daemon=True)

frames=[]
 
speaking=False
PauseTime=300.01

start_speak_timer=time.perf_counter()
while True:
    text = ""
    data=stream.read(CHUNK,exception_on_overflow=False)

    audio_np=np.frombuffer(data,dtype="int16").astype(np.float32)
    audio_np/=32768.0
    audio_tensor=torch.from_numpy(audio_np)

    result=vad(audio_tensor)
    if  result is not None:
        for i in result:
            status=i
            if status=="start":
                frames = []
                voice_detected=True
                print("Detected Voice")
                

            if status=="end":
                voice_detected=False
                print("Stopping LTM process")
                state.ltm_mem_event.clear()
                state.is_ltm=False
                start_speak_timer=time.perf_counter()
                print("starting whisper")
                
                start=time.perf_counter()
                audio_data = b"".join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0
                
                after_spoke=time.perf_counter()
                
                state.vision_event.clear()
                vision_thread.join()
                
                segments, info = whisper_model.transcribe(
                    audio_np,
                    language="en",
                    beam_size=1
                )


                for seg in segments:
                    text += seg.text + " "
                text=text.strip()
                print(repr(text))
                frames=[]

                try:
                    if text:
                        speaking=True
                        model,conf=model_selector.select_model(text)
                        print(model)
                        if model=="conversation":
                            from model.convo import process
                                       
                        
                            res=process(text.strip())
                            if res:
                                
                                print(f"\n{time.perf_counter()-after_spoke}\n")
                                
                                speech_status=speak.speech(res)
                                if speech_status:
                                    spoke_status,msg=speak.speak()
                                    time.sleep(0.1)
                                    if not spoke_status:
                                        print(msg)
                                
                                variables.convo_history.append(
                                {
                                    "role":"user",
                                    "content": text.strip()
                                }
                                )

                                variables.convo_history.append(
                                    {
                                        "role":"assistant",
                                        "content": res
                                    }
                                )

                            else:
                                print("Nothings recived")
                            

                        elif model=="DepthDetection":
                            pass
                        elif model=="ObjectDetection":
                            pass

                        else:
                            from model.convo import process
                            result = process(text)
                              
                            print(result)

                        end=time.perf_counter()
                        print(f"Time taken to complete: {(end-start):.2f} seconds")

                except RuntimeError as e:
                    print(e)
                    print()
                    continue

                
                state.vision_event.set()
                vision_thread=threading.Thread(target=vision,daemon=True)
                vision_thread.start()
                print("starting camera")
                speaking=False

    
    if voice_detected:
            frames.append(data)



    if (time.perf_counter()-start_speak_timer)>=10.00:
                if not state.is_ltm:
                    if len(variables.potential_memory)>0:
                        state.vision_event.clear()
                        vision_thread.join()
                        state.ltm_mem_event.set()

                        print("LTM memory started!!!")
                        ltm_thread.start()

                        print("LTM memory over!!!")
                        start_speak_timer=time.perf_counter()



