from .llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Route(BaseModel):
    intent: str = Field(description="The category of the user query. Must be one of: DOCUMENT_QUERY, MATH_QUERY, WEB_SEARCH")

def classify_query(query: str) -> str:
    """
    Classifies a user query into one of three intents.
    Returns: 'DOCUMENT_QUERY', 'MATH_QUERY', or 'WEB_SEARCH'.
    """
    system_prompt = (
        "You are an expert intent router. Your job is to classify the user's query into one of three categories: "
        "'DOCUMENT_QUERY', 'MATH_QUERY', or 'WEB_SEARCH'.\n\n"
        "Guidelines:\n"
        "1. If the user is asking about information that might be in an uploaded document (e.g., 'What is the policy?', 'Summarize this', 'Explain the handbook'), return 'DOCUMENT_QUERY'.\n"
        "2. If the user is asking to calculate a mathematical expression (e.g., 'What is 5+5?', 'Calculate 100 * 45', '20 * 50'), return 'MATH_QUERY'.\n"
        "3. If the user is asking for general, external, or current world knowledge that is definitely NOT in their personal documents (e.g., 'Who is the president of USA?', 'Weather in London', 'Capital of France'), return 'WEB_SEARCH'.\n\n"
        "If you are unsure, default to 'DOCUMENT_QUERY'."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    llm = get_llm()
    
    try:
        # Using with_structured_output to enforce output format
        structured_llm = llm.with_structured_output(Route)
        chain = prompt | structured_llm
        result = chain.invoke({"query": query})
        return result.intent
    except Exception:
        # Fallback in case of failure (e.g. LLM doesn't support structured output properly, or API error)
        return "DOCUMENT_QUERY"
