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

    async def get_diagnostic_response(self, query: str) -> str:
        """
        Gets a high-quality diagnostic response using the full RAG + prioritized AI model flow.
        This should be used for the final summary where medical accuracy is paramount.
        """
        if not self.vector_store:
            logger.warning("Vector store not available, proceeding without RAG context.")
            context = ""
        else:
            try:
                retrieved_docs = self.vector_store.similarity_search(query, k=3)
                context = " ".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved clinical context for diagnosis: {context[:200]}...")
            except Exception as e:
                logger.error(f"Failed to retrieve from vector store: {e}")
                context = ""

        # Generate a response using the main, prioritized medical response generator
        response = await self.ai_service.generate_medical_response(
            query=query,
            context=context
        )
        
        if response.get("success"):
            return response.get("response", "I am unable to provide a response at this time.")
        else:
            logger.error(f"AI Service Manager failed to generate diagnostic response. Error: {response.get('error')}")
            return "I apologize, but I'm having trouble processing your medical query right now."

    async def generate_contextual_question(self, query: str) -> str:
        """
        Generates a contextual follow-up question using RAG + OpenAI for speed.
        """
        if not self.vector_store:
            logger.warning("Vector store not available, proceeding without RAG context.")
            context = ""
        else:
            try:
                retrieved_docs = self.vector_store.similarity_search(query, k=3)
                context = " ".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Retrieved conversational context for question generation: {context[:200]}...")
            except Exception as e:
                logger.error(f"Failed to retrieve from vector store: {e}")
                context = ""
        
        # Use the dedicated conversational response generator (which uses OpenAI)
        response = await self.ai_service.generate_conversational_response(
            query=query,
            context=context
        )

        if response.get("success"):
            return response.get("response", f"Can you please provide more details?")
        else:
            logger.error(f"AI Service Manager failed to generate conversational response. Error: {response.get('error')}")
            return "Could you please tell me more?"

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the RAG service."""
        return {
            "rag_service_status": "healthy",
            "vector_store_initialized": self.vector_store is not None,
            "embedding_model": "text-embedding-ada-002"
        }