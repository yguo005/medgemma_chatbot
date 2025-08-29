import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Detect environment and set correct project root
current_file = Path(__file__).resolve()

# Check if we're in Colab (look for /content/ in path)
if "/content/" in str(current_file):
    # In Colab: find the project root by looking for the main directory
    project_root = None
    for parent in current_file.parents:
        if parent.name == "medgemma_chatbot" or (parent / "config").exists():
            project_root = parent
            break
    
    if project_root is None:
        # Fallback: assume standard Colab structure
        project_root = Path("/content/medgemma_chatbot")
else:
    # Local environment: go up 4 levels from src/services/ai/rag/
    project_root = current_file.parent.parent.parent.parent

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

def debug_directory_structure():
    """Debug function to show directory structure in Colab"""
    print(" Debugging directory structure:")
    
    # Show current working directory
    print(f" Current working directory: {os.getcwd()}")
    
    # Show project root contents
    if project_root.exists():
        print(f" Project root contents:")
        try:
            for item in sorted(project_root.iterdir()):
                if item.is_dir():
                    print(f"   [DIR]  {item.name}")
                else:
                    print(f"   [FILE] {item.name}")
        except Exception as e:
            print(f"   Error listing directory: {e}")
    else:
        print(f" Project root does not exist: {project_root}")
    
    # Check for data directory
    data_path = Path(get_absolute_path(DATA_PATH))
    if data_path.exists():
        print(f" Data directory contents:")
        try:
            for item in sorted(data_path.iterdir()):
                print(f"   {item.name}")
        except Exception as e:
            print(f"   Error listing data directory: {e}")
    else:
        print(f" Data directory does not exist: {data_path}")
        
        # Try to find where data might be
        possible_locations = [
            project_root / "data",
            project_root / "data" / "document",
            Path("/content/medgemma_chatbot/data"),
            Path("/content/medgemma_chatbot/data/document"),
        ]
        
        print(" Checking possible data locations:")
        for loc in possible_locations:
            exists = loc.exists()
            print(f"   {loc}: {'EXISTS' if exists else 'NOT FOUND'}")
            if exists and loc.is_dir():
                try:
                    files = list(loc.iterdir())
                    print(f"     Contains {len(files)} items: {[f.name for f in files[:5]]}")
                except:
                    pass
    print()

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
        print(" FAISS vector store already exists. Overwriting...")
        return True
    return False

def main():
    print(" Creating FAISS Vector Store for AI Health Consultant")
    print("=" * 60)
    print(f" Current script: {Path(__file__).resolve()}")
    print(f" Project root: {project_root}")
    print(f" Data path: {get_absolute_path(DATA_PATH)}")
    print(f" FAISS path: {get_absolute_path(DB_FAISS_PATH)}")
    
    # Debug: Check if directories exist
    print(f" Project root exists: {project_root.exists()}")
    print(f" Data directory exists: {Path(get_absolute_path(DATA_PATH)).exists()}")
    print()
    
    # If data directory doesn't exist, run debug function
    if not Path(get_absolute_path(DATA_PATH)).exists():
        debug_directory_structure()
    
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

