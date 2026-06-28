import json
import ollama
from databaseModel import retrive
from cosine import cosine
from datetime import UTC,datetime
model="qwen3-embedding:8b"

def embed_convert(text):
    res=ollama.embed(
        model=model,
        input=text
    )
    embedding=res["embeddings"][0]

    return embedding


def context(text,ppl):
    threshold=0.6
    response=retrive(ppl)
    embed=embed_convert(text)
    max_list=[]
    for i in response:
        # print(len(json.loads(i[3])))
        sim=cosine.consimilaritry(embed,json.loads(i[3]))
        if sim>=threshold:
            max_list.append(response.index(i))
    
    context=[]
    for j in max_list:

        context.append({
            "memory":response[j][2],
            "date":response[j][4]
            })
    
    return context

# print(context("do you know what I love","Utkarsha"))