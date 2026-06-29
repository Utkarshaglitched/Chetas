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


def context(text,ppl,k=10):
    print(ppl)
    response=retrive(ppl)
    embed=embed_convert(text)
    
    sentence_list=[]
    for i in response:
        
        sim=cosine.consimilaritry(embed,json.loads(i[3]))

        sentence_list.append((sim,{
            i[2]:i[4]
        }))

    sorted_list = sorted(sentence_list, key=lambda pair: pair[0], reverse=True)
    context=sorted_list[:k]
        
    return context

def reranking(cntext):
    pass

