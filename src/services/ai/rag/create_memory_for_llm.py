import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Dynamic import to avoid module-level import errors
try:
    from config.settings import DATA_PATH, DB_FAISS_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    # Fallback configuration for Colab
    print(" Warning: Could not import config.settings, using fallback configuration")
    DATA_PATH = "data/document"
    DB_FAISS_PATH = "data/vectorstore/db_faiss" 
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(" ERROR: OpenAI API Key is missing! Set 'OPENAI_API_KEY' in your environment.")

def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path from project root."""
    if os.path.isabs(relative_path):
        return relative_path
    return str(project_root / relative_path)

def load_pdf_files(data_dir: str):
    """Loads all PDF files in the specified directory using PyPDFLoader."""
    abs_data_dir = get_absolute_path(data_dir)
    
    if not os.path.exists(abs_data_dir):
        print(f" ERROR: Data directory '{abs_data_dir}' not found.")
        print(f"   Looking for: {abs_data_dir}")
        return []

    pdf_files = [f for f in os.listdir(abs_data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f" WARNING: No PDF files found in directory: {abs_data_dir}")
        return []

    loader = DirectoryLoader(abs_data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f" Loaded {len(documents)} documents from {len(pdf_files)} PDF file(s)")
    return documents

def create_chunks(extracted_data):
    """Splits documents into smaller text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f" Created {len(text_chunks)} text chunks.")
    return text_chunks

def check_existing_faiss():
    """Checks if an existing FAISS database is available."""
    abs_db_path = get_absolute_path(DB_FAISS_PATH)
    if os.path.exists(abs_db_path):
        print("🔍 FAISS vector store already exists. Overwriting...")
        return True
    return False

def main():
    print(" Creating FAISS Vector Store for AI Health Consultant")
    print("=" * 60)
    print(f" Project root: {project_root}")
    print(f" Data path: {get_absolute_path(DATA_PATH)}")
    print(f" FAISS path: {get_absolute_path(DB_FAISS_PATH)}")
    print()
    
    # 1. Load PDF documents
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print(" ERROR: No documents to process. Exiting...")
        return

    # 2. Create text chunks
    text_chunks = create_chunks(documents)
    
    # 3. Load embedding model
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    test_vector = embedding_model.embed_query("Test query")  # Ensure embedding dimension matches FAISS
    print(f" OpenAI Embedding Model Loaded (Vector Dimension: {len(test_vector)})")

    # 4. Create FAISS vector store (overwrite if exists)
    check_existing_faiss()
    abs_db_path = get_absolute_path(DB_FAISS_PATH)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(abs_db_path), exist_ok=True)
    
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(abs_db_path)
    print(f" FAISS vector store created and saved to: {abs_db_path}")

if __name__ == "__main__":
    main()

