
import ollama
from retirival.cosine import consimilaritry
from variables import category_dict



def select_model(words):
    try:
        response=ollama.embed(
            model="qwen3-embedding:0.6b",
            input=words
        )
    except Exception as e:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is installed and running, then run 'ollama pull qwen3-embedding:8b' and 'ollama serve'."
        ) from e

    embedding=response["embeddings"][0]

    sim_list=[]
    for i in category_dict:
        sim_max2=0
        val=category_dict[i]
        for j in val:
            sim=consimilaritry(embedding,j)
            if sim>sim_max2:
                sim_max2=sim
            else:
                sim_max2=sim_max2
        sim_list.append(sim_max2)

    
    index=sim_list.index(max(sim_list))

    return list(category_dict.keys())[index], sim_list[index]