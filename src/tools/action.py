def force_dialogue(agent1, agent2):
    conversation_history = []
    response2 = agent1.generate_dialogue_response(f"Say sometihng to {agent2.name} about your day", conversation_history=conversation_history)
    response1 = speak_to(agent2, response2[1], conversation_history=conversation_history)
    print(response1[1])
    conversation_history.append(response2[1])
    response2 = speak_to(agent1, response1[1], conversation_history=conversation_history)
    conversation_history.append(response1[1])
    print(response2[1])

    while response1[0] or response2[0]:
        prev_response1 = response1
        prev_response2 = response2
        response1 = speak_to(agent2, response2[1], conversation_history=conversation_history)
        print(response1[1])
        conversation_history.append(response2[1])
        response2 = speak_to(agent1, response1[1], conversation_history=conversation_history)
        print(response2[1])
        conversation_history.append(response1[1])
    

def speak_to(agent, message, conversation_history=[]):
    response = agent.generate_dialogue_response(message,conversation_history=conversation_history)
    return response