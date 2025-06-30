import streamlit as st
import pandas as pd
import os
import duckdb
import pathlib
import re
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

# Helper: extract SQL from LLM output (remove markdown and explanation)
def extract_sql(llm_output: str) -> str:
    code_blocks = re.findall(r"```(?:sql)?\s*([\s\S]+?)```", llm_output, re.IGNORECASE)
    if code_blocks:
        return code_blocks[0].strip()
    sql_start = re.search(r"(SELECT|WITH|INSERT|UPDATE|DELETE)", llm_output, re.IGNORECASE)
    if sql_start:
        return llm_output[sql_start.start():].strip()
    return llm_output.strip()

# Generate schema string from uploaded tables
def get_schema_string():
    schema_lines = []
    for fname, df in st.session_state.uploaded_dfs.items():
        table_name = pathlib.Path(fname).stem.replace(" ", "_").lower()
        columns = ", ".join([f"{col} ({str(df[col].dtype)})" for col in df.columns])
        schema_lines.append(f"Table: {table_name}\nColumns: {columns}")
    return "\n\n".join(schema_lines)

# Remove backticks and strip query
def clean_sql_query(query: str) -> str:
    query = query.strip()
    if query.startswith("`") and query.endswith("`"):
        query = query[1:-1].strip()
    # Replace backticks in identifiers with double quotes or nothing (DuckDB uses double quotes)
    query = query.replace("`", '"')
    return query

# Agent: Convert natural language to SQL using schema context
def nl_to_sql_agent(question: str):
    schema_info = get_schema_string()
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=api_key,
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.2,
        verbose=False
    )

    prompt = f"""
You are an expert data analyst. Write an accurate SQL query to answer the user's question using the available schema.

Use only these tables and columns:

{schema_info}

Some useful guidance:
- If the question is about books not returned, check if Return_Date IS NULL.
- If the question is about overdue books, also check if Due_Date < CURRENT_DATE.
- Assume CURRENT_DATE represents today's date.

Question: {question}

SQL Query:
"""


    sql_query_raw = llm.predict(prompt).strip()
    sql_query_clean = extract_sql(sql_query_raw)
    sql_query_clean = clean_sql_query(sql_query_clean)
    return sql_query_clean

# Run SQL query directly on DuckDB
def run_sql_query(query: str):
    try:
        st.code(query)
        result = duckdb.sql(query).df()
        return result
    except Exception as e:
        st.error(f"Query execution failed:\n{query}\n\nError: {e}")
        return pd.DataFrame()

# Combined NL → SQL → Execute → Display flow
def handle_user_question(question: str):
    st.write("### Step 1: Converting your question to SQL...")
    sql_query = nl_to_sql_agent(question)
    st.code(sql_query, language="sql")

    st.write("### Step 2: Executing SQL query and fetching results...")
    results = run_sql_query(sql_query)

    st.write("### Results:")
    if isinstance(results, pd.DataFrame) and not results.empty:
        st.dataframe(results)
    elif isinstance(results, pd.DataFrame) and results.empty:
        st.info("Query executed successfully but returned no results.")
    else:
        st.write(results)

# Streamlit app logic
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "uploaded_dfs" not in st.session_state:
        st.session_state.uploaded_dfs = {}
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Home"):
            st.session_state.page = "Home"
    with col2:
        if st.button("Query"):
            st.session_state.page = "query"

    if st.session_state.page == "Home":
        st.title("Home")
        files_uploaded = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

        if files_uploaded:
            st.success("Files uploaded successfully")
            for file in files_uploaded:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                    # Normalize columns: strip and replace spaces with underscores
                    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
                    st.session_state.uploaded_dfs[file.name] = df
                else:
                    st.error(f"{file.name} is not a CSV file.")

        if st.session_state.uploaded_dfs:
            st.write("Registered tables:")
            for fname, df in st.session_state.uploaded_dfs.items():
                table_name = pathlib.Path(fname).stem.replace(" ", "_").lower()
                duckdb.register(table_name, df)
                st.write(f"- {table_name}")
                st.dataframe(df.head())

    elif st.session_state.page == "query":
        st.title("Query")

        if st.session_state.uploaded_dfs:
            for fname, df in st.session_state.uploaded_dfs.items():
                table_name = pathlib.Path(fname).stem.replace(" ", "_").lower()
                duckdb.register(table_name, df)

        # Use a key for the text_area to ensure Streamlit tracks changes
        query = st.text_area("Enter your question in natural language:", value=st.session_state.user_query, key="user_query_text_area")

        # Only run the query if the user submits a new query (not just edits the text)
        if st.button("Run Query"):
            if query and query.strip():
                st.session_state.user_query = query
                handle_user_question(query.strip())

if __name__ == "__main__":
    main()
