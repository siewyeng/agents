from langchain.prompts import PromptTemplate 
import json

def generate_agent_name(model, num_of_agents):
    template = """
    Generate a list of {num_of_agents} of 1 word names for the characters of our game. Each name in the ouput should be seperated by a comma.

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["num_of_agents"],
        template=template
    )

    agent_names = model(prompt_template.format(num_of_agents=num_of_agents))
    return agent_names.split(", ")

def generate_characters(model, agent_name):
    template = """
    You are a character generation AI. Your job is to make believable details for the characters of our town AI Town. Given the character's name. Create interesting and believable details for our characters.

    Generate details for our character {name}.

    Example Output: {{"age":52,"traits":"curious, enthusiastic, paranoid","background":"You are an occult enthusiast who makes vlogs of you going to various allegedly haunted places. After graduating with a software engineering degree 10 years ago, you decided that ghosts were more interesting to you and now you make a decent living through your videos. In a sense, you are a small celebrity."}}

    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["name"],
        template=template
    )

    agent_details = model(prompt_template.format(name=agent_name))
    return json.loads(agent_details)