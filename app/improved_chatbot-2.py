import os
import traceback
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from app.config import DB_FAISS_PATH, EMBEDDING_MODEL
from app.medgemma_service import MedGemmaService
from app.medgemma_model_garden import MedGemmaModelGarden
import logging

logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, openai_api_key: str, use_medgemma_garden: bool = False, gcp_project_id: str = None, endpoint_id: str = None):
        if not openai_api_key:
            raise ValueError("❌ ERROR: OpenAI API Key is missing!")
        
        self.openai_api_key = openai_api_key
        self.use_medgemma_garden = use_medgemma_garden
        
        # Load vector store (still using OpenAI embeddings for retrieval)
        self.vectorstore = self._get_vectorstore()
        
        # Initialize MedGemma service (either local or Model Garden)
        self.medgemma_service = self._initialize_medgemma_service(gcp_project_id, endpoint_id)
        
        # Create retriever
        self.retriever = self._create_retriever()

    def _get_vectorstore(self):
        """Loads the FAISS vector store."""
        try:
            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=self.openai_api_key)
            vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            
            # Validate index
            index = vectorstore.index
            test_vector = embedding_model.embed_query("Test")
            if index.d != len(test_vector):
                raise ValueError("FAISS index dimension mismatch.")

            logger.info("✅ FAISS vector store loaded successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS vector store: {e}")
            return None

    def _initialize_medgemma_service(self, gcp_project_id: str = None, endpoint_id: str = None):
        """Initialize MedGemma service (Model Garden or local)."""
        try:
            if self.use_medgemma_garden and gcp_project_id:
                # Use Model Garden (production)
                service = MedGemmaModelGarden(
                    project_id=gcp_project_id,
                    endpoint_id=endpoint_id or os.getenv("MEDGEMMA_ENDPOINT_ID", ""),
                    credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                )
                logger.info("✅ MedGemma Model Garden service initialized")
            else:
                # Use local Hugging Face (development)
                service = MedGemmaService(
                    model_name="google/medgemma-4b-it",  # Same 4B model for both dev and prod
                    device="auto",
                    use_quantization=True,
                    multimodal=True  # Enable multimodal capabilities
                )
                logger.info("✅ MedGemma local service initialized")
            
            return service
        except Exception as e:
            logger.error(f"❌ Failed to initialize MedGemma service: {e}")
            return None

    def _create_retriever(self):
        """Creates the retriever from vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded.")
        
        return VectorStoreRetriever(
            vectorstore=self.vectorstore,
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )

    async def get_response(self, query: str) -> str:
        """Gets a response using RAG + MedGemma architecture."""
        try:
            if not self.medgemma_service:
                return "Sorry, the medical AI service is not available right now."
            
            # Step 1: Retrieve relevant medical knowledge using OpenAI embeddings
            relevant_docs = self.retriever.get_relevant_documents(query)
            
            # Step 2: Combine retrieved context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Step 3: Generate response using MedGemma with retrieved context
            if hasattr(self.medgemma_service, 'analyze_symptoms_text'):
                # Model Garden version
                response = await self.medgemma_service.analyze_symptoms_text(query, context)
                if response["success"]:
                    return response["response"]
                else:
                    return "I apologize, but I'm having trouble processing your medical query right now."
            else:
                # Local service version
                response = await self.medgemma_service.generate_medical_response(
                    query=query,
                    context=context,
                    max_length=512,
                    temperature=0.3
                )
                if response["success"]:
                    return response["response"]
                else:
                    return "I apologize, but I'm having trouble processing your medical query right now."
                    
        except Exception as e:
            logger.error(f"❌ Error in get_response: {e}\n{traceback.format_exc()}")
            return "Sorry, I encountered an error while processing your query."

    def get_service_info(self) -> dict:
        """Get information about the current setup."""
        return {
            "vectorstore_loaded": self.vectorstore is not None,
            "medgemma_service_loaded": self.medgemma_service is not None,
            "using_model_garden": self.use_medgemma_garden,
            "service_type": "Model Garden" if self.use_medgemma_garden else "Local Hugging Face",
            "retriever_ready": self.retriever is not None
        }