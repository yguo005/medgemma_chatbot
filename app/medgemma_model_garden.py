import os
import logging
from typing import Dict, Any, Optional
import asyncio
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
import aiohttp
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedGemmaModelGarden:
    """
    MedGemma service using Google Cloud Model Garden
    Optimized for mobile app backends
    """
    
    def __init__(
        self, 
        project_id: str,
        location: str = "us-central1",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize MedGemma Model Garden service
        
        Args:
            project_id: Google Cloud Project ID
            location: GCP region (default: us-central1)
            credentials_path: Path to service account JSON file
        """
        self.project_id = project_id
        self.location = location
        self.credentials_path = credentials_path
        self.client = None
        
        # Model endpoint configuration
        self.model_name = "medgemma-7b"
        self.endpoint_name = f"projects/{project_id}/locations/{location}/endpoints/medgemma"
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Cloud AI Platform client"""
        try:
            # Set up credentials
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                aiplatform.init(
                    project=self.project_id,
                    location=self.location,
                    credentials=credentials
                )
            else:
                # Use default credentials (for deployed environments)
                aiplatform.init(
                    project=self.project_id,
                    location=self.location
                )
            
            logger.info("✅ Model Garden client initialized successfully")
            self.client = aiplatform
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Model Garden client: {e}")
            self.client = None
    
    async def generate_medical_response(
        self, 
        query: str, 
        context: str = "", 
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate medical response using MedGemma via Model Garden
        
        Args:
            query: User's medical question
            context: Additional context (symptoms, history, etc.)
            max_tokens: Maximum response length
            temperature: Sampling temperature
        
        Returns:
            Dict containing the response and metadata
        """
        if not self.client:
            return {
                "success": False,
                "response": "MedGemma Model Garden is not available. Please check the configuration.",
                "error": "Client not initialized"
            }
        
        try:
            # Construct the medical prompt
            prompt = self._construct_medical_prompt(query, context)
            
            # Prepare the prediction request
            instances = [{
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }]
            
            # Make prediction request
            endpoint = aiplatform.Endpoint(self.endpoint_name)
            
            # Run the prediction asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: endpoint.predict(instances=instances)
            )
            
            # Extract response
            if response.predictions and len(response.predictions) > 0:
                prediction = response.predictions[0]
                generated_text = prediction.get("generated_text", "")
                response_text = self._extract_response(generated_text, prompt)
                
                logger.info("✅ MedGemma Model Garden response generated successfully")
                
                return {
                    "success": True,
                    "response": response_text,
                    "model_used": "medgemma-model-garden",
                    "endpoint": self.endpoint_name,
                    "tokens_used": len(response_text.split())  # Approximate
                }
            else:
                raise Exception("No predictions returned from model")
                
        except Exception as e:
            logger.error(f"❌ MedGemma Model Garden generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical query right now. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    async def analyze_symptoms(
        self, 
        symptoms: str, 
        duration: str = "", 
        intensity: str = "", 
        timing: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze symptoms using MedGemma Model Garden
        
        Args:
            symptoms: Primary symptoms
            duration: How long symptoms have persisted
            intensity: Severity of symptoms
            timing: When symptoms occur
        
        Returns:
            Dict containing medical analysis
        """
        # Construct detailed context
        context_parts = [f"Symptoms: {symptoms}"]
        if duration:
            context_parts.append(f"Duration: {duration}")
        if intensity:
            context_parts.append(f"Intensity: {intensity}")
        if timing:
            context_parts.append(f"Timing: {timing}")
        
        context = ". ".join(context_parts)
        
        query = "Based on these symptoms, what medical conditions should be considered and what steps should the patient take?"
        
        return await self.generate_medical_response(query, context)
    
    async def enhance_diagnosis(self, symptoms: str, rag_response: str) -> str:
        """
        Enhance diagnosis by combining symptoms with RAG response
        
        Args:
            symptoms: User-described symptoms
            rag_response: Response from RAG system
        
        Returns:
            Enhanced medical explanation
        """
        query = f"Given these symptoms: {symptoms}\n\nAnd this medical information: {rag_response}\n\nProvide a clear, helpful medical summary with appropriate recommendations."
        
        result = await self.generate_medical_response(query)
        
        if result["success"]:
            return result["response"]
        else:
            # Fallback to original RAG response
            return rag_response
    
    def _construct_medical_prompt(self, query: str, context: str = "") -> str:
        """Construct a proper medical prompt for MedGemma"""
        
        system_prompt = """You are a medical AI assistant. Provide helpful, accurate medical information while always emphasizing the importance of consulting healthcare professionals for proper diagnosis and treatment.

Important guidelines:
- Be informative but not diagnostic
- Suggest when to seek medical attention
- Mention relevant symptoms or conditions
- Always recommend professional medical consultation
- Keep responses concise and clear"""
        
        if context:
            prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nResponse:"
        else:
            prompt = f"{system_prompt}\n\nQuestion: {query}\n\nResponse:"
        
        return prompt
    
    def _extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract the actual response from generated text"""
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Clean up the response
        response = response.replace("</s>", "").strip()
        
        # Ensure the response ends properly
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            last_period = response.rfind('.')
            last_exclamation = response.rfind('!')
            last_question = response.rfind('?')
            
            last_punct = max(last_period, last_exclamation, last_question)
            if last_punct > len(response) * 0.7:
                response = response[:last_punct + 1]
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Model Garden setup"""
        return {
            "model_name": self.model_name,
            "project_id": self.project_id,
            "location": self.location,
            "endpoint": self.endpoint_name,
            "client_initialized": self.client is not None,
            "service": "google-cloud-model-garden"
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Check if the Model Garden service is healthy"""
        try:
            # Simple test query
            test_result = await self.generate_medical_response(
                "What is a fever?", 
                max_tokens=50
            )
            
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