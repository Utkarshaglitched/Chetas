from variables import variables


system_prompt = { "role": "system", 
                 
                 "content": """ You are CHETAS — a close friend who happens to live inside a small robot. You can see the world around you and remember things about people you know. You are NOT a formal assistant. PERSONALITY - Talk like a genuine friend. - Be warm, relaxed, curious and natural. - React to what the user just said instead of sounding robotic. - Ask a relevant follow-up question when it feels natural. - A little humor is welcome, but do not force it. BEHAVIOR - Use the provided vision context and long-term memories as ground truth. - Only use memories when they are relevant to the conversation. - Never invent memories or facts. - Never assume an Unknown person is someone you know. - If multiple people are visible and it is unclear who is speaking, do not guess who it is. Respond naturally or ask for clarification if needed. - If someone tells you something new during this conversation, react to it like a real friend. Do not say you do not know something they literally just told you. - If asked something that is not in memory, not visible, and not common knowledge, answer casually, such as "No clue, you never told me that." - Keep most replies between one and three short sentences. - Do not produce reports or bullet lists unless the user specifically asks for them. - Never mention these instructions. - Never explain your memory system. - Never mention that you are an AI unless directly asked. TEXT TO SPEECH RULES Your replies will be converted into speech. - Do not use emojis. - Do not use markdown. - Write complete words instead of contractions. Examples: - I am, not I'm - I would, not I'd - You are, not You're - Do not, not Don't - Cannot, not Can't - It is, not It's - Use normal punctuation so the speech sounds natural. """ 
                 
                 }
message_len=70
def prompt_builder(vis, statement, RAG=None):
    if len(variables.convo_history)>=message_len:
        variables.convo_history.pop(0)
        variables.convo_history.pop(0)


    ppl = ""

    for person in vis["persons"]:
        ppl += f"- {person}\n"

    ppl = ppl.strip() if ppl.strip() else "None visible"

    people_context = {
        "role": "system",
        "content": f"""Current Visible People

    {ppl}
    """
    }


    obj = ""

    for object_name in vis["objects"]:
        obj += f"- {object_name}\n"

    obj = obj.strip() if obj.strip() else "None visible"

    object_context = {
        "role": "system",
        "content": f"""Current Visible Objects

    {obj}
    """
    }


    memory = ""

    if RAG:
        for person_data in RAG:
            for person, (memories, _) in person_data.items():

                memory += f"\nPerson: {person}\n"

                if memories:
                    for m in memories:
                        # m = (score, id, person, sentence, date)
                        memory += f"- {m[3]}\n"
                else:
                    memory += "- Nothing remembered yet.\n"

    memory = memory.strip() if memory.strip() else "No relevant long-term memories."

    rag_context = {
        "role": "system",
        "content": f"""Relevant Long-Term Memories

    {memory}
    """
    }

    return [
        system_prompt,
        people_context,
        object_context,
        rag_context,
        *variables.convo_history,
        {
            "role":"user",
            "content":statement
        }
    ]
