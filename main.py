import model_selector
import time

while True:
    start=time.perf_counter()

    print("enter '/bye' to exit")
    x=input("Enter your sentance: ")

    if x.lower()=="/bye":
        break

    try:
        model,conf=model_selector.select_model(x)
        print(model)

    except RuntimeError as e:
        print(e)
        print()
        continue
    
    if model=="conversation":
        from convo import process
        print(f"\n\n{process(x)}\n")

    elif model=="DepthDetection":
        pass
    elif model=="ObjectDetection":
        pass
    elif model=="FaceRecognistion":
        pass

    else:
        from convo import process
        print(f"\n\n{process(x)}\n")

    end=time.perf_counter()
    print(f"Time taken to complete: {(end-start):.2f} seconds")