import ollama
from retirival.vision import vision as get_vision_context
from retirival.RAGcontext import context
from variables.variables import ollama_model
import threading 
import json
from variables import state,prompts,variables
from data.databaseModel import add,update
from groq import Groq
from dotenv import load_dotenv
import os
from data.LTM import ignore_check



load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)



def process(sentence):

    people=state.vision_context_frame["persons"]
    prompt=""
    rag_context,emd=context(sentence,people)
    prompt=prompts.prompt_builder(state.vision_context_frame,sentence,rag_context)

    memory_thread=None
    print(len(people))
    if len(people)==1:
        ig_status,simi=ignore_check(emd)
        if ig_status:
            if rag_context:
                memory=""
                for person_data in rag_context:
                    for person, (memories, _) in person_data.items():

                        memory += f"\nPerson: {person}\n"

                        if memories:
                            for m in memories:
                                memory += f"[ID:{m[1]}] {m[3]}\n"
                        else:
                            memory += "- Nothing remembered yet.\n"

            memory = memory.strip() if memory.strip() else "No relevant long-term memories."
            
            
            variables.potential_memory.append(
                                {
                                    "role":"user",
                                    "content":
                                f"""
                                Person: {people[0]}

                                Current Conversation:
                                {sentence}

                                Relevant Memories:
                                {memory}
                                """
                                })
            print(f"similarity: {simi}")

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
