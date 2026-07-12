import json
import ollama
from data.databaseModel import retrive
from retirival import cosine
from datetime import UTC,datetime
from variables import rag_model

def embed_convert(text):
    res=ollama.embed(
        model=rag_model,
        input=text
    )
    embedding=res["embeddings"][0]

    return embedding


def context(text,ppl,k=10):
    response=retrive(ppl)
    embed=embed_convert(text)
    
    sentence_list=[]
    for i in response:
        
        sim=cosine.consimilaritry(embed,json.loads(i[3]))

        sentence_list.append((sim,i[0],i[1],i[2],i[4]))

    sorted_list = sorted(sentence_list, key=lambda pair: pair[0], reverse=True)
    context=sorted_list[:k]
        
    return context,embed

def reranking(cntext):
    pass

# print(context("love","Ankit")) #print("love u too","Utkarsh")