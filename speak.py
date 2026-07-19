
import os
from groq import Groq
from pathlib import Path
import subprocess
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

output_path = "/home/pi/Desktop/Virtual-Chetas/speech/speaker/speech.wav"

def speech(text):
    if text:
        response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="abdullah",
        response_format="wav",
        input=str(text).strip(),
        )
        response.write_to_file(output_path)
        return True
    return False



def speak():
    try:
        subprocess.run([
        "ffplay",
        "-nodisp",
        "-autoexit",
        "/home/pi/Desktop/Virtual-Chetas/speech/speaker/speech.wav"
        ])
        return True,""
    except Exception as e:
        return False,e
    
