import ollama
from retirival.RAGcontext import context
from variables.variables import ollama_model
import json
from variables import prompts,variables,state
from data.databaseModel import add,update
from retirival.cosine import consimilaritry
import numpy
from retirival.RAGcontext import embed_convert


def start_storing(res):
    """
    Takes the model's decision dict and commits it to memory.db.
    Returns True if the intended action was successfully applied,
    False if it failed or the input was invalid.
    """
    if not isinstance(res, dict):
        print("start_storing: invalid input, not a dict")
        return False

    action = res.get("action")
    person = res.get("person", "")
    sentence = res.get("sentence", "")
    replace_id = res.get("replace_id")

    if action == "ignore":
        return True

    if action == "duplicate":
        return True

    if action == "insert":
        if not person or not sentence:
            print("start_storing: insert requires person and sentence")
            return False
        try:
            embedding = embed_convert(sentence)
        except Exception as e:
            print("start_storing: embedding generation failed:", e)
            return False
        return add(person, sentence, embedding)

    if action == "update":
        if replace_id is None:
            print("start_storing: update requires a replace_id")
            return False
        if not sentence:
            print("start_storing: update requires a sentence")
            return False
        try:
            embedding = embed_convert(sentence)
        except Exception as e:
            print("start_storing: embedding generation failed:", e)
            return False
        success = update(replace_id, sentence, embedding)
        if not success:
            print(f"start_storing: update failed, no memory found with id={replace_id}")
        return success

    print(f"start_storing: unknown action '{action}'")
    return False



def memory_status(memory_prompt):
    try:
        print(">>> Memory process started", flush=True)

        response = ollama.chat(
        model="granite4.1:3b",   
        messages=memory_prompt,
        format=prompts.memory_schema,
        options={
        "temperature": 0.1
        }
    )

        content = response["message"]["content"].strip()


        data = json.loads(content)

        print(f"\033[91mMemory Decision: {data}\033[0m", flush=True)

        return data
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
    state.is_ltm = True
    done_storage = []
    try:
        while state.ltm_mem_event.is_set():
            if not variables.potential_memory:
                break  # nothing left to process

            item = variables.potential_memory[0]

            prompt = [
                prompts.system_memory_prompt,
                item
            ]

            st = memory_status(prompt)

            if st:
                storage_status = start_storing(st)
                if storage_status:
                    done_storage.append(item)
                else:
                    print(f"start_ltm_process: storage failed for item, skipping: {item}")
            else:
                print(f"start_ltm_process: memory_status failed for item, skipping: {item}")

            variables.potential_memory.pop(0)  

    finally:
        state.is_ltm = False
        state.vision_event.clear()
    
 