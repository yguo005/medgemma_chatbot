import os
import traceback
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from config.settings import DB_FAISS_PATH, EMBEDDING_MODEL
from src.services.ai.medgemma.medgemma_service import MedGemmaService
from src.services.ai.medgemma.model_garden import MedGemmaModelGarden
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    """
    Chatbot class that integrates a FAISS vector store with an AI service.
    This class is responsible for Retrieval-Augmented Generation (RAG).
    """
    def __init__(self, openai_api_key: str, ai_service):
        """
        Initialize the Chatbot with a vector store and an AI service.
        
        Args:
            openai_api_key: The API key for OpenAI embeddings.
            ai_service: An initialized AI service manager that handles model interactions.
        """
        try:
            # Initialize embeddings and vector store for RAG
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self.db_path = "data/vectorstore/db_faiss"
            self.vector_store = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store: {e}")
            self.vector_store = None
        
        # --- REFACTOR: Use Injected AI Service ---
        # The Chatbot now depends on an external AI service manager,
        # centralizing all AI logic and removing redundant clients.
        self.ai_service = ai_service
        logger.info("RAG service initialized to use the central AI Service Manager.")
        # -----------------------------------------

    async def get_response(self, query: str) -> str:
        """
        Get response from the chatbot, incorporating RAG.
        This is a general-purpose method driven by the input query.
        """
        if not self.vector_store:
            logger.warning("Vector store not available, proceeding without RAG context.")
            context = ""
        else:
            try:
                # 1. Retrieve relevant context from the vector store
                retrieved_docs = self.vector_store.similarity_search(query, k=3)
                context = " ".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved clinical context: {context[:200]}...")
            except Exception as e:
                logger.error(f"Failed to retrieve from vector store: {e}")
                context = ""

        # 2. Generate a response using the injected AI Service Manager
        # This respects the application's mode (e.g., cloud_first) and optimizations.
        response = await self.ai_service.generate_medical_response(
            query=query,
            context=context
        )
        
        if response.get("success"):
            return response.get("response", "I am unable to provide a response at this time.")
        else:
            # Fallback response if the AI service manager fails
            logger.error(f"AI Service Manager failed to generate a response. Error: {response.get('error')}")
            return "I apologize, but I'm having trouble processing your medical query right now."

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the RAG service."""
        return {
            "rag_service_status": "healthy",
            "vector_store_initialized": self.vector_store is not None,
            "embedding_model": "text-embedding-ada-002"
        }