"""utils.config

Application configuration utilities. Loads environment variables and exposes
a `Settings` dataclass with commonly used configuration values (API keys,
model names, encoding, and limits). A global `settings` instance is created
from the environment at module import time.
"""

from dotenv import load_dotenv
import os
from dataclasses import dataclass


# Load .env once at module level
load_dotenv()

@dataclass
class Settings:
    """Application settings loaded from environment variables"""
    
    # Google API
    google_api_key: str
    
    # Tavily API
    tavily_api_key: str
    
    # Ollama configuration
    ollama_base_url: str
    
    # Model configuration
    default_model: str
    gemini_model: str
    
    # Validation settings
    max_retries: int
    
    # File handling
    file_encoding: str
    
    # Generation parameters
    temperature: float
    max_tokens: int
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create Settings instance from environment variables"""
        return cls(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            ollama_base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            default_model=os.getenv('DEFAULT_MODEL', 'llama3.2:3b'),
            gemini_model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite'),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            file_encoding=os.getenv('FILE_ENCODING', 'utf-8'),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('MAX_TOKENS', '2048'))
        )
    
    
    def is_gemini_available(self) -> bool:
        return bool(self.google_api_key)
    
    
    def get_model_info(self) -> dict:
        return {
            "ollama": {
                "model": self.default_model,
                "base_url": self.ollama_base_url,
                "available": True  # Asumimos que Ollama está disponible
            },
            "gemini": {
                "model": self.gemini_model,
                "available": self.is_gemini_available()
            }
        }
        
    

# Create a single global instance
settings = Settings.from_env()