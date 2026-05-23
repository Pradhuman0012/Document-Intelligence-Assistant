from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of documents into chunks using RecursiveCharacterTextSplitter.
    Metadata from the original document (e.g., page, source) is maintained in each chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # split_documents automatically preserves the metadata of the input documents
    chunks = text_splitter.split_documents(documents)
    return chunks
