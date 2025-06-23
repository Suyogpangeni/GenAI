import os
import streamlit as st
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
os.environ["GROQ_API_KEY"]="Your_Api_key"

# --- Set your Groq API Key ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY environment variable.")
    st.stop()

# --- Set up LLM ---
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="meta-llama/llama-4-maverick-17b-128e-instruct",   
    temperature=0.7,
)

# --- Define State Type ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Define the chatbot node ---
def chatbot(state: State):
    try:
        response = llm.invoke(state["messages"])
        if isinstance(response, AIMessage):
            assistant_message = {"role": "assistant", "content": response.content}
        else:
            assistant_message = {"role": "assistant", "content": str(response)}
        return {"messages": state["messages"] + [assistant_message]}
    except Exception as e:
        st.error(f"Error in chatbot node: {str(e)}")
        return {"messages": state["messages"] + [{"role": "assistant", "content": "I'm sorry, I encountered an error."}]}

# --- Build LangGraph ---
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("LangGraph Demo")

# --- Initialize message history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous messages ---
for msg in st.session_state.messages:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    user_msg = {"role": "user", "content": user_input}
    st.chat_message("user").write(user_input)

    # Run LangGraph with the updated message history
    input_state = {"messages": st.session_state.messages + [user_msg]}
    try:
        for event in graph.stream(input_state):
            for value in event.values():
                last_msg = value["messages"][-1]
                st.session_state.messages = value["messages"]
                st.chat_message("assistant").write(last_msg["content"])
    except Exception as e:
        st.error(f"Error running LangGraph: {e}")