from langchain.prompts import PromptTemplate
from src.generative_agents.generative_agent import StemssGenerativeAgent


def generate_schedule(model,agent):
        """
        Generate the schedule for the agent base on the bio provided

        Args:
            name ([str]): [Name of the agent]
            age ([int]): [Age of the agent]
            background ([str]): [Background description of the agent]

        Returns:
            [list]: [Schedule generated from LLM]
        """

        template = """You are a plan generating AI, and your job is to help characters make new plans based on new information. Given the character's info(bio, goals), generate a new set of plans of the day for them to carry out, such that the final set of plans and include at least 15 individual plans. The plan list should be numbered in the order in which they should be performed, with each plan containing a description with at least 10 words and never use the word /'repeat/'. Give output in this format without numbering: [Plan to take dinner with their spouse at home, Plan to thave a jog around town]. Try to collaborate with spouse if possible. The location can only be office or home. Do not mention about repeating or evaluating the plan. Always prioritize finishing any pending conversation before doing other things. Let\'s Begin! Name: {name} Age: {age} Background: {background}"""
        
        prompt_template = PromptTemplate(
                input_variables=["name","age","background"],
                template=template
                )
        
        schedule = model(prompt_template.format(name=agent.name, age=agent.age, background=agent.background))
        schedule = schedule.replace("[","").replace("]","")
        schedule = schedule.split(",")
        return schedule

