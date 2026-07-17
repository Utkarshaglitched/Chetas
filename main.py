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

voice_detected=False

state.vision_event.set()
vision_thread=threading.Thread(target=vision,daemon=True)
vision_thread.start()
print("starting camera")


frames=[]
os.system('clear')

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
                
                print("starting whisper")

                
                
                start=time.perf_counter()
                audio_data = b"".join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0

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
                        model,conf=model_selector.select_model(text)
                        print(model)
                        if model=="conversation":
                            from model.convo import process
                            os.system('clear')
                            
                            state.vision_event.clear()
                            vision_thread.join()

                            
                        
                            res=process(text.strip())
                            if res:
                                os.system('clear')
                                print(res)
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
                            os.system('clear')
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
                
    
    if voice_detected:
            frames.append(data)