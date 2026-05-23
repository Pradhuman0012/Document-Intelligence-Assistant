import numexpr as ne
from langchain_community.tools import DuckDuckGoSearchRun
from .vector_store import get_vector_store
from .llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

def calculator_tool(query: str) -> str:
    """
    Evaluates a mathematical expression safely by extracting it from natural language first.
    """
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert math parser. Extract ONLY the mathematical expression from the user's query. Return ONLY the numbers and operators (like +, -, *, /, **). Do not return any other words or formatting. If there is no math expression, return '0'."),
            ("human", "{query}")
        ])
        
        chain = prompt | llm
        clean_expr = chain.invoke({"query": query}).content.strip()
        
        # Evaluate the clean expression
        result = ne.evaluate(clean_expr)
        
        # Return a nicely formatted string
        ans = result.item() if hasattr(result, "item") else result
        return f"The result of **{clean_expr}** is **{ans}**"
    except Exception as e:
        return f"Error evaluating math expression: {str(e)}"

def web_search_tool(query: str) -> str:
    """
    Performs a web search using DuckDuckGo.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def document_summarizer_tool(action: str) -> str:
    """
    Summarizes the document based on the requested action.
    actions: 'summarize', 'key_points', 'faqs'
    """
    try:
        vector_store = get_vector_store()
        
        # Get all documents from the vector store
        # In a real large scale app we might want to sample or use a map-reduce chain
        # But for simple/lightweight, we get everything and pass to LLM (relying on large context window)
        docs = vector_store.get() 
        documents = docs.get('documents', [])
        
        if not documents:
            return "No documents found to summarize."
            
        # Combine all text (truncate to approx 25000 chars to avoid prompt limits if too large)
        full_text = "\n\n".join(documents)[:25000]
        
        llm = get_llm()
        
        if action == "summarize":
            system_prompt = "You are a helpful assistant. Provide a comprehensive summary of the following document content."
        elif action == "key_points":
            system_prompt = "You are a helpful assistant. Extract the most important key points from the following document content, formatted as a bulleted list."
        elif action == "faqs":
            system_prompt = "You are a helpful assistant. Generate 5-7 frequently asked questions (FAQs) and their answers based on the following document content."
        else:
            system_prompt = "You are a helpful assistant. Summarize the following document content."
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document Content:\n\n{context}"),
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": full_text})
        
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"
