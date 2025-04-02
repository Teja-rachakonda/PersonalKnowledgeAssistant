from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import os

# Global variables
vectorstore = None
chat_history = []

def process_document(file_path):
    global vectorstore
    
    # Load and split document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Cache the embedding model locally
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = "./model_cache"  # Temporary folder on Streamlit Cloud
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir  # Save model here
    )
    
    # Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)

def answer_question(query):
    global vectorstore, chat_history
    
    if vectorstore is None:
        return "Upload a document first!"
    
    # Free LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="distilgpt2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100}  # Updated from 30
    )
    
    # Conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False
    )
    
    # Get answer
    result = chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    
    # Update chat history
    chat_history.append((query, answer))
    if len(chat_history) > 3:
        chat_history.pop(0)
    
    return answer