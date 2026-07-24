from pathlib import Path
import numpy

BASE_DIR = Path(__file__).resolve().parent.parent

facial={
    "Ankit":numpy.load("npySaves/Ankit.npy"),
    "Utkarsha":numpy.load("npySaves/utkarsha.npy"),
    "chachu":numpy.load("npySaves/chachu.npy")
}

print(len(numpy.load("npySaves/Ankit.npy")))


category_dict={
    "conversation":numpy.load("npySaves/convo.npy"),  
    "DepthDetection":numpy.load("npySaves/depth.npy"),
    "ObjectDetection":numpy.load("npySaves/obj.npy")
}

output_path = str((BASE_DIR / "speech" / "speaker" / "speech.wav").resolve())

vision_dict={

}

convo_history=[
    
]

potential_memory=[

]

ollama_model="granite4.1:3b"
rag_model="qwen3-embedding:0.6b"