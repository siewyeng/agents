import streamlit as st
from streamlit_option_menu import option_menu


def some_function(x):
    pass


with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Inject Memories", "View Interactions", "View Detailed Logs"],
        icons=["house", "gear", "chat-left-quote", "search"],
        menu_icon="list",
        default_index=0,
    )
    # selected

if selected == "Home":
    st.header("The world with only 2 agents but 4 overlords")
    st.caption("*Smaller than Smallville")
    st.image("./img/front.webp")
elif selected == "Inject Memories":
    x = st.text_area("Write the memory you would like to inject")
    st.button("Inject memories", on_click=some_function(x))

elif selected == "View Interactions":
    pass
elif selected == "View Detailed Logs":
    st.info("WIP. spot for logs")


# def on_update():
#    st.write(injected_mem)


# x = st.text_area("Write the memory you would like to inject")
# injected_mem = st.button("Inject memories", on_click=on_update)
