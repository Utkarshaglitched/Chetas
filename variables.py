import numpy

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

ollama_model="granite4.1:3b"
rag_model="qwen3-embedding:0.6b"