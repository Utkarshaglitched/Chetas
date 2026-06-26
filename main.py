import model_selector

while True:
    print("enter '/bye' to exit")
    x=input("Enter your sentance: ")

    if x.lower()=="/bye":
        break

    model,conf=model_selector.select_model(x)
    
    print(model)
    print(conf)
    print()