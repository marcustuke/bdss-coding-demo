# Databricks notebook source
# MAGIC %md
# MAGIC # BDSS Event RAG Demo
# MAGIC %pip install -r demo_requirements.txt



#%pip install langchain-openai==0.1.25
#%pip install langchain==0.2.17
#%pip install langchain-core==0.2.43
#%pip install sqlalchemy==2.0.37 #lc dependencies
#%pip install langchain-community==0.2.19
#%pip install langchain-huggingface==0.0.3
#%pip install sentence-transformers==3.4.1
#%pip install transformers>=4.44.0
#%pip install accelerate==1.3.0
#%pip install mlflow==2.20.0
#%pip install pypdf
#%pip install faiss-cpu==1.8.0

#dbutils.library.restartPython() 


import os
import mlflow, langchain, json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter
)
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap,
    RunnableLambda,
    RunnableParallel,
)

from langchain_openai import AzureChatOpenAI # you might import other open source libraries

def load_documents(file_path):
    """Load and split documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as config_file:
        return json.load(config_file)

def initialize_llm(sec_config):
    """Initialize the Azure OpenAI LLM with the given security configuration."""
    return AzureChatOpenAI(
        openai_api_version="2023-05-15", 
        azure_deployment="gpt-4-0163",
        api_key=sec_config['azure_openai']['key'],
        azure_endpoint=sec_config['azure_openai']['endpoint'],
        temperature=0.1,
        max_tokens=512,
        timeout=None,
        max_retries=2,
        top_p=0.9,
    )

def create_text_splitter():
    """Create a text splitter for document chunking."""
    return RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50,
        separators=[
            "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
        ]
    )

def create_faiss_db(docs, embedding_model):
    """Create a FAISS database from documents and an embedding model."""
    return FAISS.from_documents(docs, embedding_model)

def get_final_prompt(prompt_template):
    """Generate the final prompt template."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

def rag_chain_with_source(retriever, prompt, llm):
    """Create a RAG Chain using langchain abstraction that returns sources o"""
    final_prompt = get_final_prompt(prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | final_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

def create_template():
    """Create a prompt template for the assistant."""
    return """
    You are an assistant for question-answering tasks. \
    Only use the following pieces of retrieved context to answer the question. \
    Be as helpful as possible and explain step by step how you come to the answer. \
    If the user asks for information not found in the below context, do not answer.\

    Be accurate and very careful when giving answers. \
    Use five sentences maximum and keep the answer concise. 
    Question: {question} 
    Context: {context} 
    Answer:
    """

def main(question):
    """Main function to process the question and return an answer."""
    try:
        docs = load_documents("gpt-4-5-system-card.pdf")
        sec_config = load_config("sec_config.json")
        llm = initialize_llm(sec_config)
        text_splitter = create_text_splitter()
        splits = text_splitter.split_documents(docs)
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        db = create_faiss_db(splits, embedding_model)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        template_2 = create_template()

        with mlflow.start_run() as run:
            mlflow.langchain.autolog()
            chain = rag_chain_with_source(retriever, template_2, llm)
            answer_with_source = chain.invoke(question)
        return answer_with_source
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# COMMAND ----------

answer_with_source = main(question="what is the advantages of using gpt-4.5 compared to other gpt models?")

# COMMAND ----------

answer_with_source = main(question="answer in bullet points of how I should be using gpt 4.5 to maximise its performance")

# COMMAND ----------

