import os

# Path settings
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Model settings
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 512

# Text splitter settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 