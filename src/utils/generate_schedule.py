import os
# from vertexai.preview.language_models import TextGenerationModel
from langchain.llms import VertexAI
import pandas as pd


class Generate_Schedule:

    """
    Class to return schedule
    todo : generate schedule
    
    
    """

    def __init__(self):
        pass


    def fixed_schedule(self):
           

        schedule = [
        "Joe feels like waking up.",
        "Joe plans to make some coffee.",
        "Joe plans to check his email.",
        "Joe plans to spends some time to update his resume and cover letter.",
        "Joe feels like exploring the city and look for job openings.",
        "Joe saw a sign for a job fair and plans to attend the job fair.",
        "Joe plans to meet potential employers at the job fair.",
        "Joe plans to leave the job fair.",
        "Joe plans to grab some lunch.",
        "Joe decides to apply for the job.",
        "Joe plan to continue his search for job openings", 
        "Jope thinks he should drops off his resume at several local businesses.",
        "Joe feels like going for a walk in a nearby park.",
        "A dog approaches and Joe's wants to pet the dog.",
        "Joe plans to join a group of people playing frisbee.",
        "Joe has fun playing frisbee but gets hit in the face with the frisbee and hurts his nose.",
        "Joe decides to goes back to his apartment.",
        "Joe plans to call his best friend to vent about his struggles.",
        ]

        return(schedule)

    def generated_schedule(self,name:str,bio:str):
        """
        Generate the schedule for the agent base on the bio provided

        Args:
            name ([str]): [Name of the agent]
            bio ([str]): [Bio description of the agent]

        Returns:
            [list]: [Schedule generated from LLM]
        """



        prompt = f"You are a plan generating AI, and your job is to help characters make new plans based on new information.\
        Given the character's info(bio, goals), generate a new set of plans of the day for them to carry out, such that the final\
        set of plans and include at least 15 individual plans. The plan\
        list should be numbered in the order in which they should be performed, with each plan containing a description\
        with at least 10 words and never use the word /'repeat/'\
        Example plan: ['Plan to take dinner with their spouse at home']\
        Try to collaborate with spouse if possible\
        The location can only be office or home\
        Do not mention about repeating or evaluating the plan\
        Always prioritize finishing any pending conversation before doing other things.\
        Let\'s Begin!\
        Name: {name}\
        Bio: {bio} \
        \
        "
        # print(prompt)
        llm = VertexAI(model_name='text-bison@001', max_output_tokens=500, temperature=0.7)

        return llm(prompt)


    

