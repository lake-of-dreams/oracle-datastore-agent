"""
Configuration settings for Oracle Datastore Agent
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Oracle Database Configuration
    oracle_user: str = "datastore_user"
    oracle_password: str = "datastore_pass"
    oracle_host: str = "localhost"
    oracle_port: int = 1521
    oracle_service_name: Optional[str] = "FREEPDB1"
    oracle_sid: Optional[str] = None
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ollama_timeout: int = 30
    ollama_max_tokens: int = 1000
    ollama_temperature: float = 0.1
    
    # Application Configuration
    app_name: str = "oracle-datastore"
    debug: bool = True
    log_level: str = "INFO"
    
    # Vector Search Configuration
    vector_dimension: int = 1536  
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
