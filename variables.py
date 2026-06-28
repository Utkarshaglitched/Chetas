import numpy

facial={
    "ankit":numpy.load("npySaves/Ankit.npy"),
    "utkarsha":numpy.load("npySaves/utkarsha.npy"),
    "chachu":numpy.load("npySaves/chachu.npy")
}

print(len(numpy.load("npySaves/Ankit.npy")))


category_dict={
    "conversation":numpy.load("npySaves/convo.npy"),  
    "DepthDetection":numpy.load("npySaves/depth.npy"),
    "ObjectDetection":numpy.load("npySaves/obj.npy")
}