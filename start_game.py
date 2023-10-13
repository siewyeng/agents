import argparse
import os
from typing import List

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

# Load the .env file
load_dotenv()

# Get the path of Google Cloud credentials from the environment variable
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Optionally: Ensure that the variable has been set correctly
if google_creds_path is None:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the .env file")
else:
    print(f"GOOGLE_APPLICATION_CREDENTIALS set to {google_creds_path}")

mem_file = "./memory/memory.csv"


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


def create_new_memory_retriever(decay_rate: float = 0.5, k: int = 5):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = VertexAIEmbeddings()
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return ModTimeWeightedVectorStoreRetriever(
        vectorstore=vs, other_score_keys=["importance"], decay_rate=decay_rate, k=k
    )


# print("Hello World")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Game")

    parser.add_argument(
        "-d", "--decay", type=float, help="Decay rate, 0 to 1", default=0.5
    )
    parser.add_argument("-k", "--top_k", type=int, help="top_k to return", default=5)

    args = parser.parse_args()

    docs = load_documents()

    memory_retriever = create_new_memory_retriever(decay_rate=args.decay, k=args.top_k)
    memory_retriever.add_documents(docs)

    llm = VertexAI(model_name="text-bison@001", max_output_tokens=256, temperature=0.2)

    joel_memory = StemssGenerativeAgentMemory(
        llm=llm,
        memory_retriever=memory_retriever,
        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
        verbose=True,
    )

    joel = StemssGenerativeAgent(
        name="Joel",
        age=52,
        traits="curious, enthusiastic, paranoid",  # You can add more persistent traits here
        status="going home for his birthday",  # When connected to a virtual world, we can have the characters update their status
        memory_retriever=create_new_memory_retriever(),
        llm=llm,
        memory=joel_memory,
    )

    print(joel.get_summary())
