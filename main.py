import model_selector
import time
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad,VADIterator
import torch


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
voice_detected=False
frames=[]
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
                    language="hi",
                    beam_size=1
                )


                for seg in segments:
                    text += seg.text + " "
                text=text.strip()
                print(repr(text))
                frames=[]
                # vad.reset_states()

                try:
                    if text:
                        model,conf=model_selector.select_model(text)
                        print(model)
                        if model=="conversation":
                            from convo import process
                            res=process(text)
                            if res:
                                print(res)
                            else:
                                print("Nothings recived")
                            time.sleep(0.3)

                        elif model=="DepthDetection":
                            pass
                        elif model=="ObjectDetection":
                            pass
                        elif model=="FaceRecognistion":
                            pass

                        else:
                            from convo import process
                            print(f"\n\n{process(text)}\n")

                        end=time.perf_counter()
                        print(f"Time taken to complete: {(end-start):.2f} seconds")

                except RuntimeError as e:
                    print(e)
                    print()
                    continue
                
                
    
    if voice_detected:
            frames.append(data)