import re

from src.generative_agents.generative_agent import StemssGenerativeAgent


def generate_schedule(model, agent):
    """
    Generate the schedule for the agent base on the bio provided

    Args:
        name ([str]): [Name of the agent]
        age ([int]): [Age of the agent]
        background ([str]): [Background description of the agent]

    Returns:
        [list]: [Schedule generated from LLM]
    """

    prompt = f"You are a plan generating AI, and your job is to help characters make new plans based on new information. Given the character's info(bio, goals), generate ten plans for the day. The plan list should be in the order in which they should be performed. Do not mention about repeating or evaluating the plan. The first plan should be Plan to have breakfast. The last plan should be Plan to go to bed. Let's Begin! Name: {agent.name} Age: {agent.age} Background: {agent.background} "

    schedule = model(prompt)
    schedule = schedule.replace("{", "").replace("}", "")
    schedule = schedule.replace("[", "").replace("]", "")
    schedule = schedule.replace("'", "")
    pattern = r"[0-9]+.\s"
    schedule = re.sub(pattern, "", schedule)
    schedule = schedule.split("\n")
    return schedule
