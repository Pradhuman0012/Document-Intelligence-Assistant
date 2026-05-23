import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.0

# Ensure API keys are available
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")
