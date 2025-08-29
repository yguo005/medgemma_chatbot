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
    MedGemma 4B service using Google Cloud Model Garden,
    aligned with project requirements for using MedGemma 4B specifically.
    """
    
    def __init__(
        self, 
        project_id: str,
        location: str = "us-central1",
        endpoint_id: str = "",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize MedGemma 4B Model Garden service
        
        Args:
            project_id: Google Cloud Project ID
            location: GCP region (default: us-central1)
            endpoint_id: Your Vertex AI Endpoint ID with MedGemma 4B
            credentials_path: Path to service account JSON file
        """
        self.project_id = project_id
        self.location = location
        self.endpoint_id = endpoint_id
        self.credentials_path = credentials_path
        self.endpoint = None
        self.openai_client = None
        
        # Using MedGemma 4B instruction-tuned as specified in project requirements
        self.model_name = "google/medgemma-4b-it"
        
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
            logger.info(f"✅ MedGemma 4B Vertex AI Endpoint loaded: {self.endpoint.display_name}")

            # Set up OpenAI-compatible client
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)

            base_url = f"https://{self.location}-aiplatform.googleapis.com/v1beta1/{self.endpoint.resource_name}"
            
            self.openai_client = openai.OpenAI(base_url=base_url, api_key=creds.token)
            logger.info("✅ OpenAI-compatible client for MedGemma 4B initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MedGemma 4B Model Garden client: {e}")
            self.endpoint = None
            self.openai_client = None

    async def generate_medical_response(
        self, 
        messages: list, 
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate medical response using MedGemma 4B via an OpenAI-compatible interface.
        """
        if not self.openai_client:
            return {
                "success": False,
                "response": "MedGemma 4B client is not available.",
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
            
            logger.info("✅ MedGemma 4B response generated successfully")
            
            return {
                "success": True,
                "response": self._extract_response(response_text),
                "model_used": self.model_name,
                "model_version": "4B"
            }
        except Exception as e:
            logger.error(f"❌ MedGemma 4B generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your query right now.",
                "error": str(e)
            }

    async def analyze_symptoms_text(self, symptoms: str, context: str = "") -> Dict[str, Any]:
        """Analyzes text-based symptoms using MedGemma 4B."""
        system_prompt = self._construct_medical_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nSymptoms: {symptoms}\n\nPlease provide an analysis."}
        ]
        return await self.generate_medical_response(messages)

    async def analyze_symptoms_multimodal(self, text_prompt: str, image_url: str) -> Dict[str, Any]:
        """Analyzes symptoms with text and an image using MedGemma 4B."""
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
        """Constructs a medical prompt optimized for MedGemma 4B."""
        return """You are a medical AI assistant powered by MedGemma 4B. Your role is to provide helpful, accurate medical information while always emphasizing the importance of consulting a healthcare professional for diagnosis and treatment. 

Key guidelines:
- Be informative but not diagnostic
- Use probabilistic language ("could be related to", "might indicate")
- Always recommend professional medical consultation
- Keep responses concise and clear
- Maintain a supportive, professional tone"""
    
    def _extract_response(self, generated_text: str) -> str:
        """Extracts and cleans the response."""
        response = generated_text.replace("</s>", "").strip()
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Gets information about the MedGemma 4B Model Garden setup."""
        if not self.endpoint:
            return {"error": "Endpoint not initialized"}
        return {
            "model_name": self.model_name,
            "model_version": "4B",
            "project_id": self.project_id,
            "location": self.location,
            "endpoint_name": self.endpoint.display_name,
            "client_initialized": self.openai_client is not None,
            "service": "google-cloud-model-garden-medgemma-4b"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Checks if the MedGemma 4B Model Garden service is healthy."""
        try:
            test_messages = [{"role": "user", "content": "What is a fever?"}]
            test_result = await self.generate_medical_response(test_messages, max_tokens=50)
            return {
                "medgemma_4b_available": test_result["success"],
                "endpoint_accessible": True,
                "model_version": "4B"
            }
        except Exception as e:
            logger.error(f"❌ MedGemma 4B health check failed: {e}")
            return {
                "medgemma_4b_available": False,
                "endpoint_accessible": False,
                "error": str(e)
            }