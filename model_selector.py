import numpy
import ollama
from cosine import cosine
import time
from variables import category_dict



def select_model(words):
    response=ollama.embed(
        model="qwen3-embedding:8b",
        input=words
    )
    embedding=response["embeddings"][0]

    sim_list=[]
    for i in category_dict:
        sim_max2=0
        val=category_dict[i]
        for j in val:
            sim=cosine.consimilaritry(embedding,j)
            if sim>sim_max2:
                sim_max2=sim
            else:
                sim_max2=sim_max2
        sim_list.append(sim_max2)

    
    index=sim_list.index(max(sim_list))

    return list(category_dict.keys())[index], sim_list[index]