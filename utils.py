from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline

# Global variables (stored in memory, not cloud)
vectorstore = None
chat_history = []

def process_document(file_path):
    global vectorstore
    
    # Load and split document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Free embedding model (downloads once, runs locally)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Store in FAISS (local vector database)
    vectorstore = FAISS.from_documents(chunks, embedding_model)

def answer_question(query):
    global vectorstore, chat_history
    
    if vectorstore is None:
        return "Upload a document first!"
    
    # Free LLM (downloads once, runs locally)
    llm = HuggingFacePipeline.from_model_id(
        model_id="distilgpt2",  # Smaller than GPT-2, works on low-end PCs
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 200}  # Short answers to save memory
    )
    
    # Conversational chain (all local)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False
    )
    
    # Get answer
    result = chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    
    # Update chat history (stored in RAM)
    chat_history.append((query, answer))
    if len(chat_history) > 3:  # Limit to 3 turns to save memory
        chat_history.pop(0)
    
    return answer