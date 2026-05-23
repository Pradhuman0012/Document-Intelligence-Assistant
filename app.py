import streamlit as st
import os

# Ensure config is loaded first to populate env vars
from src import config
from src.document_loader import load_uploaded_pdfs
from src.text_splitter import split_documents
from src.vector_store import add_to_vector_store, clear_vector_store
from src.rag_service import ask_question

st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")

st.title("Document Intelligence Assistant")
st.markdown("Upload your PDF documents and ask questions based on their content.")

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # 1. Clear existing DB
                    clear_vector_store()
                    
                    # 2. Load PDFs
                    docs = load_uploaded_pdfs(uploaded_files)
                    st.write(f"Loaded {len(docs)} pages.")
                    
                    # 3. Split into chunks
                    chunks = split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")
                    
                    # 4. Store in Chroma
                    add_to_vector_store(chunks)
                    
                    st.session_state.documents_processed = True
                    st.success("Documents processed and added to vector store successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload at least one PDF.")
            
    st.markdown("---")
    st.header("Debug Actions")
    if st.button("Clear Vector Store"):
        clear_vector_store()
        st.session_state.documents_processed = False
        st.success("Vector store cleared.")
        
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

# ----------------- CHAT INTERFACE -----------------
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message is from the assistant and has metadata, display it
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            
            # Display unique sources
            if meta.get("unique_sources"):
                st.markdown("**Sources:** " + ", ".join(meta["unique_sources"]))
                
            # Display basic response metadata
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Response Time: {meta.get('response_time', 0):.2f}s")
            with col2:
                st.caption(f"Chunks Retrieved: {meta.get('retrieval_count', 0)}")
                
            # Display retrieved chunks in expandable section
            with st.expander("View Retrieved Context Chunks (Debug)"):
                for i, chunk in enumerate(meta.get("retrieved_chunks", [])):
                    source = chunk.metadata.get("source", "Unknown")
                    page = chunk.metadata.get("page", "Unknown")
                    st.markdown(f"**Chunk {i+1} from {source} (Page {page})**")
                    st.info(chunk.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    if not st.session_state.documents_processed:
        st.warning("Please upload and process documents first before asking questions.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    # ask_question is decorated with @time_it, so it returns (result_dict, execution_time)
                    result, execution_time = ask_question(prompt)
                    
                    answer = result["answer"]
                    sources = result["unique_sources"]
                    retrieved_chunks = result["retrieved_chunks"]
                    retrieval_count = result["retrieval_count"]
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display citations
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(sources))
                        
                    # Display metadata inline
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Response Time: {execution_time:.2f}s")
                    with col2:
                        st.caption(f"Chunks Retrieved: {retrieval_count}")
                        
                    # Expandable chunks for debug
                    with st.expander("View Retrieved Context Chunks (Debug)"):
                        for i, chunk in enumerate(retrieved_chunks):
                            source = chunk.metadata.get("source", "Unknown")
                            page = chunk.metadata.get("page", "Unknown")
                            st.markdown(f"**Chunk {i+1} from {source} (Page {page})**")
                            st.info(chunk.page_content)
                            
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": {
                            "unique_sources": sources,
                            "response_time": execution_time,
                            "retrieval_count": retrieval_count,
                            "retrieved_chunks": retrieved_chunks
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"An error occurred while generating the response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
