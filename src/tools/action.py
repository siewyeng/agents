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
    print(initial_observation)
    _, observation = agents[1].generate_reaction(observation=initial_observation)
    print("\n", observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            # print(agent.generate_dialogue(, observation))
            stay_in_dialogue, observation = agent.generate_dialogue(
                agent.name, observation
            )
            print("\n", observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
            # print(observation)
        if break_dialogue:
            break
        turns += 1
