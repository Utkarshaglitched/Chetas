import ollama
from retirival.vision import vision as get_vision_context
from retirival.RAGcontext import context
from variables.variables import ollama_model
import threading 
import json
from variables import state,prompts
from data.databaseModel import add,update
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)


    
def start_storing(wh,pl,sen,emb):

    prompt=prompts.memory_prompt_builder(wh,pl,sen)

    response=ollama.chat(
        model=ollama_model,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    raw_text = response["message"]["content"]
    data=json.loads(raw_text)


    if data["action"]=="insert":
        res=add(pl,data["sentence"],emb)
        if res:
            print("New Data added")
        else:
            print("New Data addition failed")


    elif data["action"]=="update":
        up=update(data["replaces_id"],data["sentence"],emb)
        if up:
            print("data updated!!")
        else:
            print("failed to update data")


    elif data["action"]=="duplicate":
        print("No data addition needed")
    
    elif data["action"]=="ignore":
        print("No data addition needed")
    
    print()
    print(data)
    print()
    return data

def process(sentence):
    # visionContext=state.vision_context
    people=state.vision_context_frame["persons"]
    print(people)
    prompt=""
    rag_context=context(sentence,people)
    prompt=prompts.prompt_builder(state.vision_context_frame,sentence,rag_context)
    print(rag_context)
    
    # t1=threading.Thread(target=start_storing,args=(rag_context,people[0],sentence,embed))

    # t1.start()
    # if len(people)==1:
    #     pass
    # else:
        # prompt=prompt_builder(state.vision_context_frame,sentence)

    # response=ollama.chat(
    #     model=ollama_model,
    #     messages=[
    #         {
    #             "role":"user",
    #             "content":prompt
    #         }
    #     ]
    # )
    # print(visionContext)
    # print(response["message"]["content"])
    # return response["message"]["content"]
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
        {
            "role": "user",
            "content":prompt
        }
        ],
        temperature=1,
        max_completion_tokens=7000,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    parts = []

    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)

    text = "".join(parts)

    return text


