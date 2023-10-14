import argparse
import os
from typing import List
from csv import writer

from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vertexai.preview.language_models import TextGenerationModel

from src.generative_agents.generative_agent import StemssGenerativeAgent
from src.generative_agents.memory import StemssGenerativeAgentMemory
from src.retrievers.time_weighted_retriever import ModTimeWeightedVectorStoreRetriever
from src.vectorstores.chroma import EnhancedChroma
from src.generators.schedule import generate_schedule
from src.generators.agent import generate_agent_name, generate_characters

# Load the .env file
load_dotenv()

# Get the path of Google Cloud credentials from the environment variable
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Optionally: Ensure that the variable has been set correctly
if google_creds_path is None:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the .env file")
else:
    print(f"GOOGLE_APPLICATION_CREDENTIALS set to {google_creds_path}")


def load_documents() -> List[Document]:
    """ "Load memory (history) from CSV "

    Returns
    -------
    list
        A list of documents where each row of the CSV is a document
    """
    loader = CSVLoader(
        mem_file, metadata_columns=["last_accessed_at", "created_at", "importance"]
    )
    docs = loader.load()
    return docs


def create_new_memory_retriever(decay_rate: float = 0.5, k: int = 5, mem_file:str="./memory/memory.csv"):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = VertexAIEmbeddings()
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return ModTimeWeightedVectorStoreRetriever(
        vectorstore=vs, other_score_keys=["importance"], decay_rate=decay_rate, k=k, mem_file=mem_file
    )


# print("Hello World")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Game")

    parser.add_argument(
        "-d", "--decay", type=float, help="Decay rate, 0 to 1", default=0.5
    )
    parser.add_argument("-k", "--top_k", type=int, help="top_k to return", default=5)

    args = parser.parse_args()

    path='./outputs'
    all_folders = os.listdir(path)
    new = 1
    if len(all_folders)!=0:
        all_folders.sort()
        latest = all_folders[-1].replace('run_', '')
        new = int(latest) + 1
    new_path = f"{path}/run_{new}"
    os.makedirs(new_path)
    os.makedirs(f"{new_path}/memory")

    llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2)

    agents = []
    #Generate agent
    agent_names = generate_agent_name(model=llm, num_of_agents=2)
    for agent_name in agent_names:

        #generate agent details
        agent_details = generate_characters(model=llm, agent_name=agent_name)
        agent_details['name']=agent_name

        #Create csv if it doesn't exist
        mem_file = f"{new_path}/memory/{agent_name}.csv"
        if  not os.path.exists(mem_file):
            with open(mem_file, 'a') as f_object:
                List = ["created_at", "last_accessed_at", "observations", "importance"]
                writer_object = writer(f_object)
                writer_object.writerow(List)
                f_object.close()

        #Load CSV
        docs = load_documents()
        memory_retriever = create_new_memory_retriever(decay_rate=args.decay, k=args.top_k ,mem_file=mem_file)
        if(len(docs)!=0):
            memory_retriever.add_documents(docs)

        agent_memory = StemssGenerativeAgentMemory(
            llm=llm,
            memory_retriever=memory_retriever,
            reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
            verbose=True,
        )

        agent = StemssGenerativeAgent(
            name=agent_name,
            age=agent_details["age"],
            traits=agent_details["traits"],
            status="",
            memory_retriever=create_new_memory_retriever(),
            llm=llm,
            memory=agent_memory,
            background=agent_details["background"],
        )
        agents.append(agent)
        agent.schedule = generate_schedule(model=llm, agent=agent)
        print(agent.schedule)
    for agent in agents:
        print(agent.get_summary())
