import streamlit as st
import os
import time

# Ensure config is loaded first to populate env vars
from src import config
from src.document_loader import load_uploaded_pdfs
from src.text_splitter import split_documents
from src.vector_store import add_to_vector_store, clear_vector_store
from src.rag_service import ask_question
from src.router import classify_query
from src.tools import calculator_tool, web_search_tool, document_summarizer_tool
from src.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

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
            
            tool = meta.get("tool")
            if tool == "calculator":
                st.markdown("**(Calculator Tool)**")
            elif tool == "web_search":
                st.markdown("**(Web Search Tool)**")
            elif tool == "summarizer":
                st.markdown("**(Summarization Tool)**")
            elif tool == "rag":
                st.markdown("**(Knowledge Base)**")
            
            # Display unique sources
            if meta.get("unique_sources"):
                st.markdown("**Sources:** " + ", ".join(meta["unique_sources"]))
                
            # Display basic response metadata
            col1, col2 = st.columns(2)
            with col1:
                if meta.get('response_time') is not None:
                    st.caption(f"Response Time: {meta.get('response_time', 0):.2f}s")
            with col2:
                if "retrieval_count" in meta:
                    st.caption(f"Chunks Retrieved: {meta.get('retrieval_count', 0)}")
                
            # Display retrieved chunks in expandable section
            if "retrieved_chunks" in meta and meta["retrieved_chunks"]:
                with st.expander("View Retrieved Context Chunks (Debug)"):
                    for i, chunk in enumerate(meta.get("retrieved_chunks", [])):
                        source = chunk.metadata.get("source", "Unknown")
                        page = chunk.metadata.get("page", "Unknown")
                        st.markdown(f"**Chunk {i+1} from {source} (Page {page})**")
                        st.info(chunk.page_content)

# ----------------- QUICK ACTIONS -----------------
st.markdown("### Quick Actions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Summarize Document", use_container_width=True):
        st.session_state.quick_action = "summarize"
with col2:
    if st.button("Extract Key Points", use_container_width=True):
        st.session_state.quick_action = "key_points"
with col3:
    if st.button("Generate FAQs", use_container_width=True):
        st.session_state.quick_action = "faqs"

# Accept user input
prompt = st.chat_input("Ask a question about the uploaded documents, calculate math, or search the web...")
quick_action = st.session_state.pop("quick_action", None)

if prompt or quick_action:
    # Set up prompt string
    if quick_action:
        action_map = {
            "summarize": "Please summarize the document.",
            "key_points": "Please extract the key points.",
            "faqs": "Please generate FAQs."
        }
        display_prompt = action_map[quick_action]
        intent = "SUMMARIZE"
    else:
        display_prompt = prompt
        with st.spinner("Classifying intent..."):
            intent = classify_query(prompt)
            
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": display_prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(display_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            if intent == "SUMMARIZE" or (intent == "DOCUMENT_QUERY" and quick_action):
                if not st.session_state.documents_processed:
                    error_msg = "Please upload and process documents first before asking document-related questions."
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    with st.spinner("Generating document summary..."):
                        start_time = time.time()
                        answer = document_summarizer_tool(quick_action if quick_action else "summarize")
                        execution_time = time.time() - start_time
                        
                        st.markdown("**(Summarization Tool)**")
                        st.markdown(answer)
                        st.caption(f"Response Time: {execution_time:.2f}s")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "metadata": {"tool": "summarizer", "response_time": execution_time}
                        })
                        
            elif intent == "MATH_QUERY":
                with st.spinner("Calculating..."):
                    start_time = time.time()
                    answer = calculator_tool(prompt)
                    execution_time = time.time() - start_time
                    
                    st.markdown("**(Calculator Tool)**")
                    st.markdown(answer)
                    st.caption(f"Response Time: {execution_time:.2f}s")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": {"tool": "calculator", "response_time": execution_time}
                    })
                    
            elif intent == "WEB_SEARCH":
                with st.spinner("Searching the web..."):
                    start_time = time.time()
                    search_result = web_search_tool(prompt)
                    
                    # Synthesize final answer using LLM
                    synth_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Use the following web search results to answer the user's query clearly and concisely.\n\nWeb Results:\n{context}"),
                        ("human", "{query}")
                    ])
                    chain = synth_prompt | get_llm()
                    answer = chain.invoke({"context": search_result, "query": prompt}).content
                    execution_time = time.time() - start_time
                    
                    st.markdown("**(Web Search Tool)**")
                    st.markdown(answer)
                    st.caption(f"Response Time: {execution_time:.2f}s")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": {"tool": "web_search", "response_time": execution_time}
                    })

            else: # DOCUMENT_QUERY
                if not st.session_state.documents_processed:
                    error_msg = "Please upload and process documents first before asking document-related questions."
                    st.warning(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    with st.spinner("Searching knowledge base..."):
                        # ask_question is decorated with @time_it, so it returns (result_dict, execution_time)
                        result, execution_time = ask_question(prompt)
                        
                        answer = result["answer"]
                        sources = result["unique_sources"]
                        retrieved_chunks = result["retrieved_chunks"]
                        retrieval_count = result["retrieval_count"]
                        
                        st.markdown("**(Knowledge Base)**")
                        st.markdown(answer)
                        
                        if sources:
                            st.markdown("**Sources:** " + ", ".join(sources))
                            
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Response Time: {execution_time:.2f}s")
                        with col2:
                            st.caption(f"Chunks Retrieved: {retrieval_count}")
                            
                        with st.expander("View Retrieved Context Chunks (Debug)"):
                            for i, chunk in enumerate(retrieved_chunks):
                                source = chunk.metadata.get("source", "Unknown")
                                page = chunk.metadata.get("page", "Unknown")
                                st.markdown(f"**Chunk {i+1} from {source} (Page {page})**")
                                st.info(chunk.page_content)
                                
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "metadata": {
                                "tool": "rag",
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
