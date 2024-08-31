import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import scraper as sc

url = "https://www.iitk.ac.in/doms/"
links = sc.scrape_links(url)
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv('GOOGLE_API_KEY')
os.environ["GROQ_API_KEY"]=os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader=WebBaseLoader(links)
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("DoMS GPT (Beta)")
llm=ChatGroq(model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only. If answer is not present in context, then reply: `Couldn't find answer to this query`.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input your query here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    
