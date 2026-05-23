import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_uploaded_pdfs(uploaded_files) -> list[Document]:
    """
    Load PDF files uploaded via Streamlit's file_uploader using PyPDFLoader.
    Extracts text and maintains metadata including 'source' and 'page'.
    """
    documents = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            # Load documents (each page is a separate Document)
            docs = loader.load()
            
            # Update metadata
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                if 'page' in doc.metadata:
                    # Make it 1-indexed for better readability in UI
                    doc.metadata['page'] = doc.metadata['page'] + 1
                    
            documents.extend(docs)
        finally:
            # Clean up the temporary file
            os.remove(temp_path)
            
    return documents
