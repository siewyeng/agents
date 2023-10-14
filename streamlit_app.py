import argparse
import os
from csv import writer
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_option_menu import option_menu
from vertexai.preview.language_models import TextGenerationModel

from src.generative_agents.generative_agent import StemssGenerativeAgent
from src.generative_agents.memory import StemssGenerativeAgentMemory
from src.generators.agent import generate_agent_name, generate_characters
from src.generators.schedule import generate_schedule
from src.retrievers.time_weighted_retriever import ModTimeWeightedVectorStoreRetriever
from src.vectorstores.chroma import EnhancedChroma


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


def create_new_memory_retriever(
    decay_rate: float = 0.5, k: int = 5, mem_file: str = "./memory/memory.csv"
):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = VertexAIEmbeddings()
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return ModTimeWeightedVectorStoreRetriever(
        vectorstore=vs,
        other_score_keys=["importance"],
        decay_rate=decay_rate,
        k=k,
        mem_file=mem_file,
    )


if __name__ == "__main__":
    # Model and folders to be initialized once everything the app runs
    if "initalized" not in st.session_state:
        # Load the .env file
        load_dotenv()

        # Get the path of Google Cloud credentials from the environment variable
        google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Optionally: Ensure that the variable has been set correctly
        if google_creds_path is None:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the .env file")
        else:
            print(f"GOOGLE_APPLICATION_CREDENTIALS set to {google_creds_path}")

        st.session_state["llm"] = VertexAI(
            model_name="text-bison@001", max_output_tokens=256, temperature=0.2
        )

        path = "./outputs"
        all_folders = os.listdir(path)
        new = 1
        if len(all_folders) != 0:
            all_folders.sort()
            latest = all_folders[-1].replace("run_", "")
            new = int(latest) + 1
        new_path = f"{path}/run_{new}"
        os.makedirs(new_path)
        os.makedirs(f"{new_path}/memory")
        st.session_state["new_path"] = new_path

        st.session_state["initalized"] = True

    if "selected_option" not in st.session_state:
        st.session_state["selected_option"] = "Home"

    with st.sidebar:
        selected = option_menu(
            "Navigation",
            [
                "Home",
                "Settings",
                "Initialize Agents",
                "Agents and Controls",
                "View Detailed Logs",
            ],
            icons=["house", "gear", "person", "chat-left-quote", "search"],
            menu_icon="list",
            default_index=0,
        )
        # Update session_state if a new option is selected
        if st.session_state["selected_option"] != selected:
            st.session_state["selected_option"] = selected

    # Home page
    if st.session_state["selected_option"] == "Home":
        st.header("The world with only 2 agents but 4 overlords")
        st.caption("*Smaller than Smallville")
        st.image("./img/front.webp")

    # Settings page
    if st.session_state["selected_option"] == "Settings":
        st.title("All settings")
        st.header("Retriever related settings")
        st.session_state["decay_rate"] = st.slider("Decay Rate", 0.0, 1.0, 0.05)
        st.session_state["top_k"] = st.slider("Top K", 1, 10, 1)

    if st.session_state["selected_option"] == "Initialize Agents":
        st.header("Agent Initialization")
        st.write("Agent names are generated randomly")
        if st.button("Generate Agent Names"):
            agent_names = generate_agent_name(
                model=st.session_state["llm"], num_of_agents=2
            )
            st.session_state["agent_names"] = agent_names
            st.write(f"Agent names are {agent_names[0]} and {agent_names[1]}")

            st.write("Generating agents")
            agents = []
            for agent_name in agent_names:
                # generate agent details
                agent_details = generate_characters(
                    model=st.session_state["llm"], agent_name=agent_name
                )
                agent_details["name"] = agent_name

                # Create csv if it doesn't exist
                new_path = st.session_state["new_path"]
                mem_file = f"{new_path}/memory/{agent_name}.csv"
                if not os.path.exists(mem_file):
                    with open(mem_file, "a") as f_object:
                        List = [
                            "created_at",
                            "last_accessed_at",
                            "observations",
                            "importance",
                        ]
                        writer_object = writer(f_object)
                        writer_object.writerow(List)
                        f_object.close()

                # Load CSV
                docs = load_documents()
                memory_retriever = create_new_memory_retriever(
                    decay_rate=st.session_state["decay_rate"],
                    k=st.session_state["top_k"],
                    mem_file=mem_file,
                )
                if len(docs) != 0:
                    memory_retriever.add_documents(docs)

                agent_memory = StemssGenerativeAgentMemory(
                    llm=st.session_state["llm"],
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
                    llm=st.session_state["llm"],
                    memory=agent_memory,
                    background=agent_details["background"],
                )
                # agent.schedule = generate_schedule(model=llm, agent=agent)
                observations = generate_schedule(
                    model=st.session_state["llm"], agent=agent
                )
                for observation in observations:
                    agent.memory.add_memory(observation)
                agents.append(agent)
                st.session_state["agents"] = agents

                st.write("Successfully generated agents")
                for agent in agents:
                    st.info(agent.get_summary())
        # st.radio("Select the agent you would like to initialize", agent_names)
        # mem_file = st.file_uploader("Upload memory files")
    if st.session_state["selected_option"] == "Agents and Controls":
        st.info(st.session_state["agent_names"])

        x = st.text_area("Write the memory you would like to inject")
        # st.button("Inject memories", on_click=pass)

    if st.session_state["selected_option"] == "View Interactions":
        pass
    if st.session_state["selected_option"] == "View Detailed Logs":
        st.info("WIP. spot for logs")
