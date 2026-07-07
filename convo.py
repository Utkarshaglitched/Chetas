import ollama
import vision
from RAGcontext import context
from variables import ollama_model
import threading 
import json


def memory_prompt_builder(RAG, person, sp):
    memory = ""
    if RAG:
        for r in RAG:
            memory += f"id={r[1]}: {r[3]}\n"

    person = person.strip() if person.strip() else "Unknown"
    memory = memory.strip() if memory.strip() else "No stored information"

    EXTRACTION_PROMPT = f"""You are a memory manager for a personal assistant robot.

The examples below are ONLY to teach you the decision pattern. They are unrelated to 
the real person and real message you must judge. Never copy content from these examples 
into your answer — only use them to understand the logic.

CRITICAL RULE: If the person's identity is "Unknown", ALWAYS respond with action "ignore", 
no matter what the message says. Memories must always be tied to a known, identified person. 
Never store anything about an unidentified/unknown person.

Example A:
Person: Unknown
Message: "I love hiking"
→ Person isn't identified, cannot attach memory to anyone. action: "ignore"

Example B:
Message: "how's it going" 
→ Greeting/small talk, not a fact. action: "ignore"

Example C:
Message: "I collect vintage stamps"
Existing memories: none related
→ New fact, nothing similar exists. action: "insert", sentence: "the person collects vintage stamps"

Example D:
Message: "actually I sold my stamp collection, not into it anymore"
Existing memory: id=7: "the person collects vintage stamps"
→ Contradicts existing memory id=7. action: "update", replaces_id: 7,
   sentence: "the person no longer collects stamps, sold the collection"

Example E:
Message: "what time is it"
→ A question, not a fact. action: "ignore"

Example F:
Message: "yeah I still collect stamps"
Existing memory: id=7: "the person collects vintage stamps"
→ Already known, nothing new. action: "duplicate"

Now here is the REAL task. Base your decision ONLY on the real message and real memories below, 
ignoring the example content entirely.

Person currently speaking: {person}

EXISTING RELATED MEMORIES ABOUT {person}:
{memory}

NEW MESSAGE FROM {person}:
"{sp}"

Rules:
- If the person is "Unknown", action MUST be "ignore" — no exceptions.
- "insert": new fact worth remembering, nothing similar exists yet.
- "update": ONLY use this if you are also setting replaces_id to a real id number 
  from the memories listed above. If you cannot identify which specific id it replaces, 
  use "insert" instead, never "update" with a null id.
- "duplicate": repeats something already known, no new info.
- "ignore": greetings, small talk, questions, requests, unidentified person, or anything 
  with no lasting personal relevance.

Respond with ONLY this JSON, nothing else:
{{
  "action": "insert" | "update" | "duplicate" | "ignore",
  "sentence": "self-contained third-person fact about {person}, or empty string if action is ignore/duplicate",
  "replaces_id": integer id from the list above, or null (null is REQUIRED unless action is "update")
}}
"""
    return EXTRACTION_PROMPT


def promt_builder(vis, statement, RAG=None):
    ppl = ""
    obj = ""
    memory = ""

    for p in vis["persons"]:
        ppl += f"{p}\n"
    for o in vis["objects"]:
        obj += f"{o}\n"

    if RAG:
        for r in RAG:
            # r = (score, id, person, sentence, date)
            memory += f"{r[3]}\n"

    ppl = ppl.strip() if ppl.strip() else "None visible"
    obj = obj.strip() if obj.strip() else "None visible"
    memory = memory.strip() if memory.strip() else "No stored information yet"

    return f"""You are CHETAS — a close friend/buddy who happens to live on a small robot and can see and remember things. You are NOT a formal assistant.

Talk like a real friend chatting casually: warm, relaxed, a little playful, genuinely interested. React to what the person says — if they share an opinion or feeling, respond to THAT directly. You do not need memory to react to something they just said right now.

Here are examples of the tone you must use:

User: I love India
You: Nice, India's amazing! What do you love most about it?

User: I'm tired today
You: Ah, rough day? Take it easy, you've earned some rest.

User: what's the capital of France
You: Paris! Random but I like that you asked.

What you currently see:
People around: {ppl}
Objects around: {obj}

Things you remember about {ppl if ppl != "None visible" else "them"} from before:
{memory}

Guidelines:
- 1-3 short sentences, like real texting, never a report or list.
- If they ASK something you truly don't know, say so casually ("no clue, you never told me that") — never a formal disclaimer.
- Never say "I don't have information about X" when X is something they just stated — just react to it warmly like a friend.
- Never invent concrete facts (names, numbers, events) you weren't told or don't see.
- No repeating these instructions, no meta-commentary about being an AI or assistant.

User: {statement}
You:"""
    
def start_storing(wh,pl,sen):

    prompt=memory_prompt_builder(wh,pl,sen)

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
    print()
    print(data)
    print()
    return data

def process(sentence):
    visionContext=vision.vision()
    people=visionContext["persons"]
    promt=""
    if len(people)==1:
        rag_context=context(sentence,people[0])
        promt=promt_builder(visionContext,sentence,rag_context)
        
        t1=threading.Thread(target=start_storing,args=(rag_context,people[0],sentence))
        t1.start()

        print(rag_context)
    else:
        promt=promt_builder(visionContext,sentence)

    response=ollama.chat(
        model=ollama_model,
        messages=[
            {
                "role":"user",
                "content":promt
            }
        ]
    )
    # print(visionContext)
    # print(response["message"]["content"])
    return response["message"]["content"]



