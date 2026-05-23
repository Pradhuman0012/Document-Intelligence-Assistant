from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .vector_store import get_vector_store
from .llm import get_llm
from .utils import time_it

system_prompt = (
    "You are a Document Intelligence Assistant. Your primary goal is to provide accurate answers based on the provided context.\n"
    "1. If the question asks for facts strictly about the document and the answer is not in the context, state: 'I could not find the answer to this question in the provided documents.'\n"
    "2. If the user explicitly asks you to compare the document with outside knowledge, industry practices, or general information, you ARE ALLOWED and EXPECTED to use your own pre-trained general knowledge to perform the comparison. Clearly distinguish between what is in the document and what is general industry practice.\n\n"
    "Context: {context}"
)

def get_rag_chain():
    """
    Creates and returns the Retrieval-Augmented Generation (RAG) chain.
    """
    llm = get_llm()
    vector_store = get_vector_store()
    
    # Configure retriever
    # top_k determines how many chunks to retrieve. Increased to 10 for better accuracy on tables/lists.
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the document chain (stuff documents)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the final retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

@time_it
def ask_question(query: str):
    """
    Ask a question against the vector database.
    Returns a dictionary containing the answer, source documents, and metadata.
    """
    rag_chain = get_rag_chain()
    
    # Invoke the chain
    response = rag_chain.invoke({"input": query})
    
    # Extract source documents and format them
    source_documents = response.get("context", [])
    
    sources_info = []
    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown file")
        page_num = doc.metadata.get("page", "Unknown page")
        sources_info.append(f"{source_name} (Page {page_num})")
        
    # Deduplicate sources info for the summary
    unique_sources = list(dict.fromkeys(sources_info))
    
    return {
        "answer": response["answer"],
        "retrieved_chunks": source_documents,
        "unique_sources": unique_sources,
        "retrieval_count": len(source_documents)
    }
