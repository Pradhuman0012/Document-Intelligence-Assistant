import sys
from types import ModuleType

# Monkey-patch PyPDF2.xmp to completely bypass xml.parsers.expat which crashes on your Mac
mock_xmp = ModuleType("PyPDF2.xmp")
mock_xmp.XmpInformation = None
sys.modules["PyPDF2.xmp"] = mock_xmp

import tempfile
import os
from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_uploaded_pdfs(uploaded_files) -> list[Document]:
    """
    Load PDF files uploaded via Streamlit's file_uploader using PyPDF2.
    Extracts text and maintains metadata including 'source' and 'page'.
    """
    documents = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            reader = PdfReader(temp_path)
            # Load documents (each page is a separate Document)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": i + 1  # 1-indexed for better readability
                        }
                    )
                    documents.append(doc)
        finally:
            # Clean up the temporary file
            os.remove(temp_path)
            
    return documents
