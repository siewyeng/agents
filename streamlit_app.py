import logging
import os
import sys
from csv import writer
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from streamlit_extras.stateful_chat import add_message, chat
from streamlit_option_menu import option_menu

from src.generative_agents.generative_agent import StemssGenerativeAgent
from src.generative_agents.memory import StemssGenerativeAgentMemory
from src.generators.agent import generate_agent_name, generate_characters
from src.generators.schedule import generate_schedule
from src.retrievers.time_weighted_retriever import ModTimeWeightedVectorStoreRetriever
from src.utils import general_utils, streamlit_utils
from src.vectorstores.chroma import EnhancedChroma

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)


# --- Streamlit callback functions
def CB_Menu():
    st.session_state.active_page = st.session_state.selected_option


def create_new_memory_retriever(
    decay_rate: float = 0.5,
    k: int = 5,
    mem_file: str = "./memory/memory.csv",
    collection_name: str = None,
):
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = VertexAIEmbeddings()
    vs = EnhancedChroma(
        embedding_function=embeddings_model, collection_name=collection_name
    )
    return ModTimeWeightedVectorStoreRetriever(
        vectorstore=vs,
        other_score_keys=["importance"],
        decay_rate=decay_rate,
        k=k,
        mem_file=mem_file,
    )


def create_memory_bank():
    # Location of agent's memory from each run
    path = "./outputs"
    all_folders = os.listdir(path)
    new = 1
    index = []
    if len(all_folders) != 0:
        for folder in all_folders:
            index.append(int(folder.replace("run_", "")))
        index.sort()
        latest = index[-1]
        new = int(latest) + 1
    new_path = f"{path}/run_{new}"
    os.makedirs(new_path)
    os.makedirs(f"{new_path}/memory")
    return new, new_path


# def memory_fix():
#    # To resolve the list index out of range error
#    memory_retriever = create_new_memory_retriever()
#    memory_retriever.vectorstore.delete_collection()


def interview_agent(agent: StemssGenerativeAgent, user_name: str, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{user_name} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


if __name__ == "__main__":
    # To preserve streamlit session states across pages
    st.session_state.update(st.session_state)

    # Model and folders to be initialized once everything the app runs
    if "initalized" not in st.session_state:
        # Load the .env file
        load_dotenv()
        # Get the path of Google Cloud credentials from the environment variable
        google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        # Ensure that the variable has been set correctly
        if google_creds_path is None:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in the .env file")
        else:
            print(f"GOOGLE_APPLICATION_CREDENTIALS set to {google_creds_path}")
        # Instantiate and store llm to session state
        llm = VertexAI(
            model_name="text-bison@001",
            max_output_tokens=256,
            temperature=0.5,
            top_p=0.9,
            top_k=20,
        )
        st.session_state["llm"] = llm

        # Get run number, path to each run
        new, new_path = create_memory_bank()
        st.session_state["new"] = new
        st.session_state["new_path"] = new_path

        # Execute list index out of range workaround
        # memory_fix()

        # Set initialized flag to true
        st.session_state["initalized"] = True

    if "selected_option" not in st.session_state:
        st.session_state["active_page"] = "Home"
        st.session_state["settings_check"] = False
        st.session_state["generated_agents"] = False
        st.session_state["gen_count"] = 0
        st.session_state["selected_option"] = "Home"

    # Navigation Menu
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            [
                "Home",
                "Settings",
                "Initalize Agents",
                "Interact with Agents",
                "Agent-to-Agent",
                "View Detailed Logs",
            ],
            icons=["house", "gear", "person", "chat-left-dots", "search"],
            menu_icon="list",
            default_index=0,
            key="selected_option",
            on_change=CB_Menu(),
        )

    # Home page
    if st.session_state["active_page"] == "Home":
        st.header("The world with only 2 agents but 4 overlords")
        st.caption("*Smaller than Smallville")
        st.image("./img/front.webp")

    # Settings page
    if st.session_state["active_page"] == "Settings":
        st.title("All settings")
        st.header("Retriever settings")
        st.session_state["decay_rate"] = st.slider(f"**Decay Rate**", 0.0, 1.0, 0.2, 0.05)
        st.session_state["top_k"] = st.slider(
            f"**Top-k documents to retriever**", 1, 10, 5, 1
        )
        st.divider()
        st.header("LLM settings")
        st.session_state["model_name"] = st.text_input(
            f"**Which model to use?**", value="text-bison@001"
        )
        st.session_state["max_output_tokens"] = st.number_input(
            f"**Max Output Tokens**", 1, 2048, 256
        )
        st.session_state["temperature"] = st.slider(
            f"**Temperature**", 0.0, 1.0, 0.5, 0.05
        )

        st.session_state["top_k"] = st.slider(
            f"**top_k (k-number of highest probablity tokens for each step)**",
            1,
            40,
            20,
            1,
        )
        st.session_state["top_p"] = st.slider(
            f"**top_p (Higher value increase randomness)**", 0.0, 1.0, 0.9, 0.05
        )
        if st.button("Reinitialize LLM"):
            llm = VertexAI(
                model_name=st.session_state["model_name"],
                max_output_tokens=st.session_state["max_output_tokens"],
                temperature=st.session_state["temperature"],
                top_p=st.session_state["top_p"],
                top_k=st.session_state["top_k"],
            )
            st.session_state["llm"] = llm

    if st.session_state["active_page"] == "Initalize Agents":
        st.title("Agents Initialization")
        st.divider()
        st.checkbox(f"**I have visited the settings page**", key="settings_check")

        if st.session_state.settings_check == True:
            st.write("Note: Agent names are generated randomly")
            if st.button("Generate Agents"):
                st.session_state["gen_count"] += 1
                progress = 0
                my_bar = st.progress(progress, text="Generating agents. Please wait.")
                agent_names = generate_agent_name(
                    model=st.session_state["llm"], num_of_agents=2
                )
                st.session_state["agent_names"] = agent_names
                st.write(f"Agent names are **{agent_names[0]}** and **{agent_names[1]}**")

                progress += 10
                my_bar.progress(
                    progress,
                    text="Agent's name generated, generating character now...",
                )

                agents = []
                for idx, agent_name in enumerate(agent_names):
                    # Generate agent details
                    agent_details = generate_characters(
                        model=st.session_state["llm"], agent_name=agent_name
                    )
                    agent_details["name"] = agent_name

                    progress += 15
                    my_bar.progress(
                        progress, text=f"Generated personality for {agent_name}"
                    )

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
                    docs = (
                        general_utils.load_documents()
                    )  # currently loads by default the memory.csv

                    # Initialize memory retriever
                    collection_name = (
                        str(st.session_state["new"])
                        + "_"
                        + str(st.session_state["gen_count"])
                        + "_"
                        + agent_name
                    )
                    print(collection_name)
                    memory_retriever = create_new_memory_retriever(
                        decay_rate=st.session_state["decay_rate"],
                        k=st.session_state["top_k"],
                        mem_file=mem_file,
                        collection_name=collection_name,
                    )
                    if len(docs) != 0:
                        memory_retriever.add_documents(docs)

                    # Instantiate agent's memory
                    agent_memory = StemssGenerativeAgentMemory(
                        llm=st.session_state["llm"],
                        memory_retriever=memory_retriever,
                        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
                        verbose=True,
                    )
                    # Instantiate agent
                    agent_name = agent_details["name"]
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
                    # Get each agent's summary
                    agent.get_summary()
                    # Store each agent to a list
                    agents.append(agent)

                    progress += 15
                    my_bar.progress(progress, text=f"Generated agent {agent_name}")
                # Store agents into session_state
                st.session_state["agents"] = agents

                for single_agent in agents:
                    observations = generate_schedule(
                        model=st.session_state["llm"], agent=single_agent
                    )
                    progress += 10
                    my_bar.progress(
                        progress, text=f"Adding memories for {single_agent.name}"
                    )
                    with streamlit_utils.st_stdout("code"):
                        for observation in observations:
                            single_agent.memory.add_memory(observation)

                with streamlit_utils.st_stdout("info"):
                    for single_agent in agents:
                        print(single_agent.get_summary(), "\n")
                progress += 10
                my_bar.progress(progress, text="Completed")

                st.success("Successfully generated all agents")
                st.session_state["generated_agents"] = True

            # Get back generated agents's details if generated before
            elif st.session_state["generated_agents"]:
                agent_names = st.session_state["agent_names"]
                st.write(f"Agent names are **{agent_names[0]}** and **{agent_names[1]}**")
                with streamlit_utils.st_stdout("info"):
                    for single_agent in st.session_state["agents"]:
                        print(single_agent.get_summary(), "\n")
                st.success("Successfully generated all agents")

    # Interact with Agents page
    if st.session_state["active_page"] == "Interact with Agents":
        if "agent_names" not in st.session_state:
            st.header("No agents to interact with")
        else:
            agent_names = st.session_state["agent_names"]
            agents = st.session_state["agents"]
            st.header("Interact with Agents")
            st.divider()
            selected_agent = st.radio(
                "Which agent do you want to interact with?",
                [agent_name for agent_name in agent_names],
            )
            selected_idx = agent_names.index(selected_agent)
            st.divider()
            st.subheader("View summary of agent")
            if st.button("View"):
                with streamlit_utils.st_stdout("info"):
                    print(agents[selected_idx].get_summary())
            st.divider()
            st.subheader("Inject Memories")
            mem_to_inject = st.text_input("Memory to inject", value="")
            if st.button("Inject"):
                agents[selected_idx].memory.add_memory(mem_to_inject)
                st.text(f"{selected_agent} suddenly experiences deja vu")
            st.divider()
            st.subheader("Chat with Agent")
            user_name = st.text_input("What is your name?", value="")
            things_to_say = st.text_input("What do you want to ask/talk about?", value="")
            st.session_state["things_to_say"] = things_to_say
            if st.button("Chat"):
                reply = interview_agent(agents[selected_idx], user_name, things_to_say)
                st.session_state["reply"] = reply
                with chat():
                    add_message("user", things_to_say)
                    add_message("assistant", reply)

    if st.session_state["active_page"] == "View Detailed Logs":
        st.info("WIP. spot for logs")
