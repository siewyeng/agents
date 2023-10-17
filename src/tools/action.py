from typing import List

from src.generative_agents.generative_agent import StemssGenerativeAgent

USER_NAME = "STEMSS"


def interview_agent(agent: StemssGenerativeAgent, user_name: str, message: str) -> str:
    """Help the notebook user interact with the agent."""

    new_message = f"{user_name} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def run_conversation(
    agents: List[StemssGenerativeAgent], initial_observation: str
) -> None:
    """Runs a conversation between agents and prints observations at every turn"""
    _, observation = agents[1].generate_reaction(observation=initial_observation)
    turns = 0
    history = []
    while True:
        break_dialogue = False
        for agent in agents:
            # print(agent.generate_dialogue(, observation))
            stay_in_dialogue, observation = agent.generate_dialogue(
                agent.name, observation, conversation_history=history
            )
            print(f"{agent.name}: {observation}")
            history.append({agent.name: observation})
            # if len(history)>4:
            #     history = history[-4:]
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
            # print(observation)
        if break_dialogue:
            break
        turns += 1
    return history
