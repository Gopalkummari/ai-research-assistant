import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AI Research & Knowledge Assistant"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./data/assistant.db"
    VECTOR_DB_DIR: str = "./data/vector_db"
    RAW_DOCUMENTS_DIR: str = "./data/raw_documents"
    DATASET_DIR: str = "./data/dataset"
    MODEL_PATH: str = "./models/tf_classifier.h5"
    TOKENIZER_PATH: str = "./models/tokenizer.pickle"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()

# Ensure required data and model directories exist
os.makedirs(settings.RAW_DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(settings.DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
