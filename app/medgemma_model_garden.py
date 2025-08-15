import os
import logging
from typing import Dict, Any, Optional
import asyncio
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
import openai
import google.auth
import google.auth.transport.requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedGemmaModelGarden:
    """
    MedGemma service using Google Cloud Model Garden,
    aligned with the official quick-start guide.
    """
    
    def __init__(
        self, 
        project_id: str,
        location: str = "us-central1",
        endpoint_id: str = "",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize MedGemma Model Garden service
        
        Args:
            project_id: Google Cloud Project ID
            location: GCP region (default: us-central1)
            endpoint_id: Your Vertex AI Endpoint ID
            credentials_path: Path to service account JSON file
        """
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        self.credentials_path = credentials_path
        self.endpoint = None
        self.openai_client = None
        
        self.model_name = "medgemma-27b-it"  # Defaulting to a capable model
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize clients for Vertex AI and OpenAI SDK"""
        try:
            creds, _ = google.auth.default()
            if self.credentials_path and os.path.exists(self.credentials_path):
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            
            aiplatform.init(
                project=self.project_id,
                location=self.location,
                credentials=creds
            )
            
            self.endpoint = aiplatform.Endpoint(self.endpoint_id)
            logger.info(f"✅ Vertex AI Endpoint loaded: {self.endpoint.display_name}")

            # Set up OpenAI-compatible client
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)

            base_url = f"https://{self.location}-aiplatform.googleapis.com/v1beta1/{self.endpoint.resource_name}"
            
            self.openai_client = openai.OpenAI(base_url=base_url, api_key=creds.token)
            logger.info("✅ OpenAI-compatible client for Vertex AI initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Model Garden client: {e}")
            self.endpoint = None
            self.openai_client = None

    async def generate_medical_response(
        self, 
        messages: list, 
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate medical response using MedGemma via an OpenAI-compatible interface.
        """
        if not self.openai_client:
            return {
                "success": False,
                "response": "MedGemma client is not available.",
                "error": "Client not initialized"
            }
        
        try:
            loop = asyncio.get_event_loop()
            model_response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            response_text = model_response.choices[0].message.content
            
            logger.info("✅ MedGemma response generated successfully via OpenAI client")
            
            return {
                "success": True,
                "response": self._extract_response(response_text),
                "model_used": model_response.model,
            }
        except Exception as e:
            logger.error(f"❌ MedGemma generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your query right now.",
                "error": str(e)
            }

    async def analyze_symptoms_text(self, symptoms: str, context: str = "") -> Dict[str, Any]:
        """Analyzes text-based symptoms."""
        system_prompt = self._construct_medical_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nSymptoms: {symptoms}\n\nPlease provide an analysis."}
        ]
        return await self.generate_medical_response(messages)

    async def analyze_symptoms_multimodal(self, text_prompt: str, image_url: str) -> Dict[str, Any]:
        """Analyzes symptoms with text and an image."""
        system_prompt = self._construct_medical_prompt()
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        return await self.generate_medical_response(messages)
    
    def _construct_medical_prompt(self) -> str:
        """Constructs a standard medical prompt for MedGemma."""
        return """You are a medical AI assistant. Your role is to provide helpful, accurate medical information while always emphasizing the importance of consulting a healthcare professional for diagnosis and treatment. Be informative but not diagnostic."""
    
    def _extract_response(self, generated_text: str) -> str:
        """Extracts and cleans the response."""
        response = generated_text.replace("</s>", "").strip()
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Gets information about the Model Garden setup."""
        if not self.endpoint:
            return {"error": "Endpoint not initialized"}
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "endpoint_name": self.endpoint.display_name,
            "client_initialized": self.openai_client is not None,
            "service": "google-cloud-model-garden"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Checks if the Model Garden service is healthy."""
        try:
            test_messages = [{"role": "user", "content": "What is a fever?"}]
            test_result = await self.generate_medical_response(test_messages, max_tokens=50)
            return {
                "model_garden_available": test_result["success"],
                "endpoint_accessible": True
            }
        except Exception as e:
            logger.error(f"❌ Model Garden health check failed: {e}")
            return {
                "model_garden_available": False,
                "endpoint_accessible": False,
                "error": str(e)
            } 