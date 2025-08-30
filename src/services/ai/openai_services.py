import os
import base64
import tempfile
import asyncio
from typing import Optional, Dict, Any
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServices:
    def __init__(self, api_key: str, use_medgemma: bool = False, gcp_project_id: str = None):
        """Initialize AI services with OpenAI API key and optional MedGemma Model Garden"""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.use_medgemma = use_medgemma
        self.medgemma_service = None
        
        # Initialize MedGemma Model Garden if requested
        if use_medgemma:
            try:
                if not gcp_project_id:
                    gcp_project_id = os.getenv("GCP_PROJECT_ID")
                
                if not gcp_project_id:
                    raise ValueError("GCP_PROJECT_ID is required for Model Garden")
                
                from src.services.ai.medgemma.medgemma_model_garden import MedGemmaModelGarden
                self.medgemma_service = MedGemmaModelGarden(
                    project_id=gcp_project_id,
                    credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                )
                logger.info(" MedGemma Model Garden service initialized successfully")
            except ImportError:
                logger.warning(" Model Garden dependencies not available. Install: pip install google-cloud-aiplatform")
                self.use_medgemma = False
            except Exception as e:
                logger.warning(f" MedGemma Model Garden initialization failed: {e}")
                self.use_medgemma = False
        
        logger.info(" AI Services initialized successfully")
    
    async def analyze_image(self, image_data: str, context: str = "medical") -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision API
        
        Args:
            image_data: Base64 encoded image data (with or without data URL prefix)
            context: Context for analysis (default: "medical")
        
        Returns:
            Dict containing analysis results and metadata
        """
        try:
            # Clean the image data if it has data URL prefix
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1] if ',' in image_data else image_data
            
            # Prepare the prompt based on context
            prompt = self._get_vision_prompt(context)
            
            # Make API call to GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3  # Lower temperature for more consistent medical analysis
            )
            
            analysis_text = response.choices[0].message.content
            
            logger.info(" Image analysis completed successfully")
            
            return {
                "success": True,
                "analysis": analysis_text,
                "model_used": "gpt-4-vision-preview",
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            logger.error(f" Image analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "I'm sorry, I couldn't analyze the image at the moment. Please try describing your symptoms in text."
            }
    
    async def transcribe_audio(self, audio_content: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper API
        
        Args:
            audio_content: Raw audio file bytes
            filename: Original filename (for format detection)
        
        Returns:
            Dict containing transcription results and metadata
        """
        temp_file_path = None
        
        try:
            # Determine file extension
            file_extension = self._get_audio_extension(filename)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            # Transcribe using Whisper API
            with open(temp_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Get detailed response with confidence
                    language="en"  # Specify English for medical context
                )
            
            logger.info(" Audio transcription completed successfully")
            
            return {
                "success": True,
                "transcription": transcript.text,
                "language": getattr(transcript, 'language', 'en'),
                "duration": getattr(transcript, 'duration', None),
                "model_used": "whisper-1"
            }
            
        except Exception as e:
            logger.error(f" Audio transcription failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "transcription": "I'm sorry, I couldn't transcribe the audio. Please try typing your message instead."
            }
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    logger.warning(f" Failed to cleanup temp file: {cleanup_error}")
    
    async def enhance_diagnosis_with_rag(self, symptoms: str, rag_response: str) -> str:
        """
        Enhance diagnosis by combining user symptoms with RAG system response
        Uses MedGemma Model Garden if available, otherwise falls back to GPT-4
        
        Args:
            symptoms: User-described symptoms
            rag_response: Response from RAG system
        
        Returns:
            Enhanced diagnosis text
        """
        # Try MedGemma Model Garden first if available
        if self.use_medgemma and self.medgemma_service:
            try:
                enhanced_response = await self.medgemma_service.enhance_diagnosis(symptoms, rag_response)
                logger.info(" Diagnosis enhanced with MedGemma Model Garden")
                return enhanced_response
            except Exception as e:
                logger.warning(f" MedGemma Model Garden enhancement failed, falling back to GPT-4: {e}")
        
        # Fallback to GPT-4
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical AI assistant. Your role is to synthesize user symptoms with medical knowledge to provide helpful information. Always remind users to consult healthcare professionals for proper diagnosis and treatment."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Based on these symptoms: {symptoms}
                        
                        And this medical information: {rag_response}
                        
                        Provide a clear, helpful summary that:
                        1. Acknowledges the symptoms
                        2. Relates them to the medical information
                        3. Suggests appropriate next steps
                        4. Reminds the user to consult a healthcare professional
                        
                        Keep the response concise and supportive.
                        """
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            logger.info(" Diagnosis enhanced with GPT-4")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f" RAG enhancement failed: {str(e)}")
            return rag_response  # Fall back to original RAG response
    
    async def analyze_symptoms_with_medgemma(
        self, 
        symptoms: str, 
        duration: str = "", 
        intensity: str = "", 
        timing: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze symptoms using MedGemma Model Garden (mobile-optimized)
        
        Args:
            symptoms: Primary symptoms
            duration: How long symptoms have persisted
            intensity: Severity of symptoms
            timing: When symptoms occur
        
        Returns:
            Dict containing medical analysis
        """
        if not self.use_medgemma or not self.medgemma_service:
            return {
                "success": False,
                "error": "MedGemma Model Garden not available",
                "response": "MedGemma analysis is not available. Using standard medical knowledge base."
            }
        
        try:
            result = await self.medgemma_service.analyze_symptoms(symptoms, duration, intensity, timing)
            logger.info(" Symptoms analyzed with MedGemma Model Garden")
            return result
        except Exception as e:
            logger.error(f" MedGemma Model Garden symptom analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble analyzing your symptoms right now. Please consult with a healthcare professional."
            }
    
    def _get_vision_prompt(self, context: str) -> str:
        """Get appropriate prompt based on context"""
        prompts = {
            "medical": """
                As a medical AI assistant, analyze this image carefully and describe what you observe. 
                Focus on:
                - Visible symptoms or conditions
                - Signs of inflammation, swelling, discoloration, or injury
                - Any abnormalities you can detect
                - Areas that might require medical attention
                
                Be specific but avoid making definitive diagnoses. Instead, describe what you see and suggest that a healthcare professional should evaluate the condition.
                Keep your response concise and professional.
            """,
            "general": """
                Analyze this image and describe what you see. Focus on the main elements, 
                any notable features, and provide a clear, concise description.
            """
        }
        
        return prompts.get(context, prompts["general"]).strip()
    
    def _get_audio_extension(self, filename: str) -> str:
        """Determine appropriate file extension for audio"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
            return os.path.splitext(filename_lower)[1]
        else:
            return '.wav'  # Default to WAV if unknown
    
    def get_service_status(self) -> Dict[str, bool]:
        """Check the status of AI services"""
        status = {
            "vision_available": True,  # GPT-4 Vision
            "transcription_available": True,  # Whisper
            "text_generation_available": True,  # GPT-4
            "client_initialized": self.client is not None,
            "medgemma_available": self.use_medgemma and self.medgemma_service is not None,
            "service_type": "model-garden" if self.use_medgemma else "openai-only"
        }
        
        # Add MedGemma Model Garden specific info if available
        if self.medgemma_service:
            medgemma_info = self.medgemma_service.get_model_info()
            status.update({
                "medgemma_model_loaded": medgemma_info["client_initialized"],
                "medgemma_project_id": medgemma_info["project_id"],
                "medgemma_endpoint": medgemma_info["endpoint"],
                "medgemma_service": medgemma_info["service"]
            })
        
        return status 