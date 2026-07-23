import ollama
from retirival.RAGcontext import context
from variables.variables import ollama_model
import json
from variables import prompts,variables,state
from data.databaseModel import add,update
from retirival.cosine import consimilaritry
import numpy
 


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

def memory_status(memory_prompt):
    try:
        print(">>> Memory process started", flush=True)

        response = ollama.chat(
        model="granite4.1:3b",   
        messages=memory_prompt,
        format=prompts.memory_schema,
        options={
        "temperature": 0.4
        }
    )

        content = response["message"]["content"].strip()


        data = json.loads(content)

        print(f"\033[91mMemory Decision: {data}\033[0m", flush=True)

        # return data

    except json.JSONDecodeError as e:
        print(f"\033[91mJSON Decode Error: {e}\033[0m")
        print(content)
        return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def ignore_check(emb):
    
    max_cnt=0
    ignore_list=numpy.load("npySaves/ignore_embeddings.npy")
    for i in ignore_list:
        sim=consimilaritry(emb,i)
        if sim>max_cnt:
            max_cnt=sim

    
    if max_cnt>=0.70:
        return False,max_cnt
    
    return True,max_cnt


def start_ltm_process():
    state.is_ltm=True
    cnt=0
    done_storage=[

    ]

    while state.ltm_mem_event.is_set():


        prompt = [
            prompts.system_memory_prompt,
            variables.potential_memory[cnt]
        ]

        st=memory_status(prompt)
        if st:
            done_storage.append(variables.potential_memory[cnt])
        cnt+=1

    for i in done_storage:
        variables.potential_memory.remove(i)
    state.is_ltm=False
    state.vision_event.clear()

def test():
    state.is_ltm=True
