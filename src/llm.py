from langchain_groq import ChatGroq
from .config import LLM_MODEL_NAME, LLM_TEMPERATURE

def get_llm():
    """
    Initialize and return the Groq LLM model.
    """
    return ChatGroq(
        temperature=LLM_TEMPERATURE, 
        model=LLM_MODEL_NAME
    )
