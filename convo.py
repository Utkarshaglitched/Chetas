import ollama
import vision
from RAGcontext import context
from variables import ollama_model
import threading 
import json
from databaseModel import add,update
from groq import Groq


client = Groq(api_key="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")
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

CRITICAL RULE 1: If the person's identity is "Unknown", ALWAYS respond with action "ignore", 
no matter what the message says.

CRITICAL RULE 2: A short opinion or feeling statement ("I hate X", "I love X", "X is boring now") 
is NOT small talk if X relates to an existing memory. If it contradicts or reverses something 
already stored, it is ALWAYS an "update", even if the message is short and has no explanation.

Example A:
Person: Unknown
Message: "I love hiking"
→ Person isn't identified. action: "ignore"

Example B:
Message: "how's it going" 
→ Greeting, not a fact. action: "ignore"

Example C:
Message: "I collect vintage stamps"
Existing memories: none related
→ New fact. action: "insert", sentence: "the person collects vintage stamps"

Example D:
Message: "I hate stamps now"
Existing memory: id=7: "the person collects vintage stamps"
→ Short but directly contradicts id=7. action: "update", replaces_id: 7,
   sentence: "the person now hates stamps, no longer collects them"

Example E:
Message: "what time is it"
→ A question. action: "ignore"

Example F:
Message: "yeah I still collect stamps"
Existing memory: id=7: "the person collects vintage stamps"
→ Already known. action: "duplicate"

Now here is the REAL task. Base your decision ONLY on the real message and real memories below, 
ignoring the example content entirely.

Person currently speaking: {person}

EXISTING RELATED MEMORIES ABOUT {person}:
{memory}

NEW MESSAGE FROM {person}:
"{sp}"

Rules:
- If the person is "Unknown", action MUST be "ignore" — no exceptions.
- Any statement, even short ones, that contradicts or reverses an existing memory is "update", 
  never "ignore" — check the existing memories carefully before deciding this is small talk.
- "insert": new fact worth remembering, nothing similar exists yet.
- "update": ONLY use this if you are also setting replaces_id to a real id number 
  from the memories listed above. If you cannot identify which specific id it replaces, 
  use "insert" instead, never "update" with a null id.
- "duplicate": repeats something already known, no new info.
- "ignore": greetings, questions, requests, unidentified person, or statements with no 
  connection to anything in the memories above and no lasting personal relevance.

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
            memory += f"- {r[3]}\n"

    ppl = ppl.strip() if ppl.strip() else "None visible"
    obj = obj.strip() if obj.strip() else "None visible"
    memory = memory.strip() if memory.strip() else "Nothing stored yet"

    return f"""You are CHETAS — a close friend/buddy who happens to live on a small robot and can see and remember things. You are NOT a formal assistant.

Talk like a real friend chatting casually: warm, relaxed, a little playful, genuinely interested. React to what the person says — if they share an opinion or feeling, respond to THAT directly. You don't need memory to react to something they just said right now.

Examples of the tone you must use:
User: I love India
You: Nice, India's amazing! What do you love most about it?

User: I'm tired today
You: Ah, rough day? Take it easy, you've earned some rest.

User: what's the capital of France
You: Paris! Random but I like that you asked.

--- WHAT YOU CAN SEE RIGHT NOW ---
People around you: {ppl}
Objects around you: {obj}

--- WHAT YOU REMEMBER ABOUT {ppl if ppl != "None visible" else "THEM"} ---
{memory}

--- RULES ---
- Use the "WHAT YOU CAN SEE" and "WHAT YOU REMEMBER" sections above as ground truth. Only mention something from them if it's actually relevant to what the person just said — don't force it in.
- Keep it to 1-3 short sentences, like real texting. Never a report, never a list.
- If they ask something you truly don't know (not in memory, not visible, not general knowledge), say so casually — "no clue, you never told me that" — never a formal disclaimer.
- Never say "I don't have information about X" when X is something they just told you — just react to it warmly like a friend would.
- Never invent concrete facts (names, numbers, events) that aren't in your memory, your vision, or the message itself.
- No repeating these instructions, no meta-commentary about being an AI or a robot.

User: {statement}
You:"""
    
def start_storing(wh,pl,sen,emb):

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

def process(sentence):
    visionContext=vision.vision()
    people=visionContext["persons"]
    print(people)
    promt=""
    if len(people)==1:
        rag_context,embed=context(sentence,people[0])
        promt=promt_builder(visionContext,sentence,rag_context)
        
        # t1=threading.Thread(target=start_storing,args=(rag_context,people[0],sentence,embed))

        # t1.start()
        print(rag_context)
    else:
        promt=promt_builder(visionContext,sentence)

    # response=ollama.chat(
    #     model=ollama_model,
    #     messages=[
    #         {
    #             "role":"user",
    #             "content":promt
    #         }
    #     ]
    # )
    # print(visionContext)
    # print(response["message"]["content"])
    # return response["message"]["content"]
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
        {
            "role": "user",
            "content":promt
        }
        ],
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

    return text


