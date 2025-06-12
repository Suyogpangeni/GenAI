import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
import os

from langchain_community.tools import TavilySearchResults
import streamlit as st
import time

# Set API keys
os.environ['TAVILY_API_KEY'] = 'your_api_key'
os.environ["GROQ_API_KEY"] = "your_api_key"

# Create Tavily search tool
search_tool = TavilySearchResults()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
    temperature=0.7
)

# Define tools
tools = [
    Tool(
        name='tavily_search',
        func=search_tool.run,
        description='Search current events or topics on the web.'
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit app
def main():
    st.title('Searching Agent')
    query = st.text_area('Enter your query')

    if st.button('Enter'):
        if not query.strip():
            st.warning("Please enter a valid query.")
            return
        
        with st.spinner("Thinking..."):
            for attempt in range(3):
                
                    result = agent.run(query)
                    st.success("Done!")
                    st.info(result)
                    break
               
            else:
                st.error("Failed after multiple attempts. Please try again later.")

if __name__ == '__main__':
    main()
