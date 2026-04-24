import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import run_agent  # your existing agent logic

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="PM Agent",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Project Manager Agent")
st.write("Ask questions about your project documents. The agent uses your PDFs as its knowledge base.")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------
user_input = st.chat_input("Ask a question about the project...")

if user_input:
    # Add user message to history
    st.session_state.history.append(HumanMessage(content=user_input))

    # Run your LangGraph agent
    response = run_agent(user_input, st.session_state.history)

    # Add agent response to history
    st.session_state.history.append(response)

    

# ---------------------------------------------------------
# Chat Display
# ---------------------------------------------------------
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)


