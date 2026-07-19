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

def memory_status(memory_prompt):
    try:
        print(">>> Memory thread entered", flush=True)

        response = ollama.chat(
            model="granite4.1:3b",  
            messages=memory_prompt,
            format={
                "type": "object",
                "properties": {
                    "store": {"type": "boolean"},
                    "importance": {"type": "integer"},
                    "memory": {"type": "string"}
                },
                "required": ["store", "importance", "memory"]
            }
        )

        print(">>> Ollama finished", flush=True)

        content = response["message"]["content"]
        print(f"\nRaw JSON:\n{content}\n", flush=True)

        memory = json.loads(content)

        print(f"\033[91mMemory Decision: {memory}\033[0m", flush=True)

        return memory

    except json.JSONDecodeError as e:
        print(f"\033[91mJSON Decode Error: {e}\033[0m", flush=True)
        print(content, flush=True)
        return None

    except Exception:
        import traceback
        traceback.print_exc()
        return None

    except Exception as e:
        import traceback
        traceback.print_exc()

def process(sentence):

    people=state.vision_context_frame["persons"]
    prompt=""
    rag_context=context(sentence,people)
    prompt=prompts.prompt_builder(state.vision_context_frame,sentence,rag_context)

    memory_thread=None
    print(len(people))
    if len(people)==1:
        print("Starting memory thread...", flush=True)
        mem_prom=prompts.memory_prompt_builder(people,sentence,rag_context)
        memory_thread=threading.Thread(target=memory_status,args=(mem_prom,))
        memory_thread.start()
        # memory_status(mem_prom)

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=prompt,
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

    if memory_thread is not None:
        memory_thread.join()

    return text

