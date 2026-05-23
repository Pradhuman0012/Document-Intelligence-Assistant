from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
from .config import EMBEDDING_MODEL_NAME

# The directory to persist the database locally
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'local_files_only': True}
    )

def get_vector_store() -> Chroma:
    """
    Initialize and return the Chroma vector store.
    """
    embeddings = get_embeddings_model()
    
    # Initialize Chroma db
    vector_store = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vector_store

def add_to_vector_store(chunks: list[Document]):
    """
    Add chunks to the vector store.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

def clear_vector_store():
    """
    Clear the vector store. Useful for when user uploads new documents and wants a clean slate,
    or just for resetting state.
    """
    vector_store = get_vector_store()
    # Delete all collections/data in the persist directory by re-initializing it empty 
    # Or by using the client methods. A simple way is to delete the collection:
    try:
        vector_store.delete_collection()
    except Exception:
        pass
