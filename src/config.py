"""
Configuration settings for A_Team_Agent RAG application.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    
    # Application Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # File Paths
    BASE_DIR = Path(__file__).parent.parent
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "data" / "vector_store"))
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", str(BASE_DIR / "data" / "documents"))
    
    # Vector Store Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 4))
    
    # OpenAI Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    
    # FAISS Configuration
    INDEX_TYPE = os.getenv("INDEX_TYPE", "IndexFlatIP")
    SAVE_LOCAL = os.getenv("SAVE_LOCAL", "True").lower() == "true"
    
    # Supported file types
    SUPPORTED_FILE_TYPES = ['.pdf', '.txt', '.md', '.docx']
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        Path(cls.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
        
        return True

# Global config instance
config = Config()