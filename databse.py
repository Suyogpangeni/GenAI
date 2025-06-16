import streamlit as st
import pandas as pd
import os
import duckdb
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType

# --- Configuration ---
st.set_page_config(page_title="CSV Chat Analyzer", layout="wide")

# Set GROQ API Key
os.environ["GROQ_API_KEY"] = "your_key"

#  Initialize the LLM
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
    temperature=0.8
)

# --- SQL Runner using DuckDB ---
def run_sql_query(query: str):
    try:
        st.code(query)  # Show the executed SQL
        result = duckdb.sql(query).df()
        return result
    except Exception as e:
        st.error(f"Query execution failed:\n{query}\n\nError: {e}")
        return pd.DataFrame()


# --- Main App ---
def main():
    st.title("SQL Query")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one CSV file to continue.")
        return

    st.subheader("Uploaded Files Preview")

    table_names = []

    for file in uploaded_files:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.replace(" ", "_").str.strip()

                table_name = file.name.replace(".", "_")
                duckdb.register(table_name, df)
                table_names.append(table_name)

                st.markdown(f"**File: {file.name} → **Registered as:** `{table_name}`")
                st.dataframe(df.head())
            except Exception as e:
                st.warning(f"Could not process '{file.name}': {e}")
        else:
            st.warning(f"Skipped file '{file.name}' — not a CSV.")

    # Only proceed if tables are registered
    if table_names:
        st.divider()

        # Update Tool with actual table names
        tool = Tool(
            name="run_sql_query",
            func=run_sql_query,
            description=(
                f"Use this to answer questions about the uploaded CSV dataset. "
                f"Available tables: {', '.join(table_names)}. "
                "Column names have spaces replaced with underscores you have to recognize the tables from the query."
                "you have to implement joins if you require results from the two or more tables"

            )
        )

        # Initialize the Langchain agent *after* the tables are registered
        agent = initialize_agent(
            tools=[tool],
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        st.subheader("Ask a Question")
        st.markdown("You can use the following table names in your SQL:")
        st.code(", ".join(table_names))

        query = st.text_area("Type your SQL question below:")

        if st.button("Run Query"):
            if query.strip():
                with st.spinner("Processing your request..."):
                    response = agent.run(query)
                    st.success("Query executed successfully.")
                    st.subheader("Agent Response")
                    st.write(response)
            else:
                st.warning("Please enter a valid SQL query.")

if __name__  ==  "__main__":
    main()
 