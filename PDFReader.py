import streamlit as st
import tempfile
from groq import Groq
from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS # storing it to the faiss vector store
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from operator import itemgetter
def load_documents(up):
    up.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(up.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(file_path=tmp_path)
    data = loader.load()
    return data
def splitPdfFile(up):# splitting the pdf file into chunks
    data = load_documents(up)
    print("documented loaded")
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = splitter.split_documents(data)  
    return chunks
os.environ["GROQ_API_KEY"]="your_api_key"
llm=ChatGroq(                   # setting up the model
    api_key=os.environ.get("GROQ_API_KEY"),
    model='meta-llama/llama-4-maverick-17b-128e-instruct',
    temperature='0.7')
def embeeding(up):
    embeeding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vcstore= FAISS.from_documents(splitPdfFile(up),embeeding)
    return vcstore
def returner(up):
    print('vcstore')
    retriever = embeeding(up).as_retriever(search_type="similarity",search_kwargs={"k":2})
    qa = RetrievalQA.from_chain_type(llm, chain_type = "stuff", retriever=retriever)
    return qa

def main():
    st.title('Advance RAG System')
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
       
        try:
            if uploaded_file.type == "application/pdf":
               
                st.success("PDF file uploaded successfully.")
            
            else:

                st.warning("uploaded file is not pdf")

        except Exception as e:
            
            st.error(f"An unexpected error occurred: {e}")

    if uploaded_file:
        returner(uploaded_file)
        query= st.text_input('your query')
        btn=st.button("Submit")
        # if query:
        #     st.warning('click the submit button')
        try:
            if btn or query:
                
                prompt="""
                answer the question based on the uploaded pdf file.
                if you cannot answer the question reply "I do not know" or if no question is asked then say"please Enter your query"
                question:{question}
                Context:{context}
                """
                chain_prompting = ChatPromptTemplate.from_template(prompt)
                parser=StrOutputParser()
                chains = (
                    {
                        "context": itemgetter("question") | returner(uploaded_file),
                        "question": itemgetter("question")
                    }
                    | chain_prompting
                    | llm
                    | parser
                    )
                result=chains.invoke({"question":query})
                st.info(result)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")          
if __name__ == '__main__':
    main()
