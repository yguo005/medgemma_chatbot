"""
Optimized AI Service Manager
Prioritizes MedGemma (local/cloud) over OpenAI for appropriate tasks
"""

import os
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceMode(Enum):
    LOCAL_DEMO = "local_demo"        # Pure local, no cloud dependencies
    HYBRID = "hybrid"                # Cloud preferred, local fallback  
    CLOUD_FIRST = "cloud_first"      # Production mode with cloud priority

class OptimizedAIServiceManager:
    """
    Manages AI services with proper prioritization:
    - MedGemma (local/cloud) for medical text and images
    - OpenAI only for audio transcription and emergency fallbacks
    """
    
    def __init__(self, mode: ServiceMode = ServiceMode.HYBRID):
        self.mode = mode
        self.services = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize services based on mode and availability"""
        
        # Always try to initialize local MedGemma first
        try:
            from src.services.ai.medgemma.medgemma_service import MedGemmaService
            self.services['medgemma_local'] = MedGemmaService(
                model_name="google/medgemma-4b-it",
                multimodal=True,  # Enable image processing
                use_quantization=True  # Memory efficient
            )
            logger.info(" Local MedGemma initialized (multimodal)")
        except Exception as e:
            logger.warning(f" Local MedGemma failed: {e}")
            self.services['medgemma_local'] = None
        
        # Initialize Model Garden if not in local demo mode
        if self.mode != ServiceMode.LOCAL_DEMO:
            try:
                from src.services.ai.medgemma.model_garden import MedGemmaModelGarden
                gcp_project_id = os.getenv("GCP_PROJECT_ID")
                if gcp_project_id:
                    self.services['medgemma_cloud'] = MedGemmaModelGarden(
                        project_id=gcp_project_id
                    )
                    logger.info(" MedGemma Model Garden initialized")
                else:
                    self.services['medgemma_cloud'] = None
            except Exception as e:
                logger.warning(f" Model Garden failed: {e}")
                self.services['medgemma_cloud'] = None
        
        # Initialize OpenAI services (primarily for audio)
        try:
            from src.services.ai.openai_services import AIServices
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.services['openai'] = AIServices(
                    api_key=openai_key,
                    use_medgemma=False  # Don't let it override our logic
                )
                logger.info(" OpenAI services initialized (audio + fallback)")
            else:
                self.services['openai'] = None
                if self.mode == ServiceMode.LOCAL_DEMO:
                    logger.info("  Demo mode: OpenAI disabled (audio unavailable)")
                else:
                    logger.warning("  OpenAI API key missing (audio unavailable)")
        except Exception as e:
            logger.warning(f" OpenAI services failed: {e}")
            self.services['openai'] = None
    
    async def analyze_image(self, image_data: str, context: str = "medical") -> Dict[str, Any]:
        """
        Analyze image with proper service prioritization:
        1. MedGemma multimodal (local)
        2. MedGemma Model Garden (cloud) 
        3. OpenAI GPT-4 Vision (emergency fallback)
        """
        
        # Priority 1: Local MedGemma multimodal
        if self.services.get('medgemma_local'):
            try:
                # Convert base64 to temp file for MedGemma
                import tempfile
                import base64
                
                # Clean image data
                if image_data.startswith('data:'):
                    image_data = image_data.split(',', 1)[1]
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(base64.b64decode(image_data))
                    temp_path = temp_file.name
                
                result = await self.services['medgemma_local'].analyze_image_with_text(
                    image_path=temp_path,
                    text_prompt="Analyze this medical image. Describe any visible symptoms, conditions, or abnormalities."
                )
                
                # Cleanup
                os.unlink(temp_path)
                
                if result.get('success'):
                    logger.info(" Image analyzed with local MedGemma")
                    return {
                        "success": True,
                        "analysis": result['response'],
                        "service_used": "medgemma_local_multimodal"
                    }
            except Exception as e:
                logger.warning(f"  Local MedGemma image analysis failed: {e}")
        
        # Priority 2: MedGemma Model Garden multimodal
        if self.services.get('medgemma_cloud'):
            try:
                # Convert to proper format for Model Garden
                image_url = f"data:image/jpeg;base64,{image_data}"
                result = await self.services['medgemma_cloud'].analyze_symptoms_multimodal(
                    text_prompt="Analyze this medical image for any visible symptoms or conditions.",
                    image_url=image_url
                )
                
                if result.get('success'):
                    logger.info(" Image analyzed with MedGemma Model Garden")
                    return {
                        "success": True,
                        "analysis": result['response'],
                        "service_used": "medgemma_cloud_multimodal"
                    }
            except Exception as e:
                logger.warning(f"  Model Garden image analysis failed: {e}")
        
        # Priority 3: OpenAI GPT-4 Vision (emergency fallback)
        if self.services.get('openai'):
            try:
                result = await self.services['openai'].analyze_image(image_data, context)
                if result.get('success'):
                    logger.info(" Image analyzed with OpenAI (fallback)")
                    return {
                        "success": True,
                        "analysis": result['analysis'],
                        "service_used": "openai_vision_fallback"
                    }
            except Exception as e:
                logger.error(f" OpenAI image analysis failed: {e}")
        
        # All services failed
        return {
            "success": False,
            "analysis": "I'm unable to analyze images right now. Please describe what you see in the image.",
            "error": "All image analysis services unavailable"
        }
    
    async def transcribe_audio(self, audio_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Transcribe audio - only OpenAI Whisper available
        MedGemma doesn't support audio processing
        """
        if not self.services.get('openai'):
            return {
                "success": False,
                "transcription": "Audio transcription is unavailable. Please type your message.",
                "error": "OpenAI Whisper not available"
            }
        
        try:
            result = await self.services['openai'].transcribe_audio(audio_content, filename)
            if result.get('success'):
                logger.info(" Audio transcribed with OpenAI Whisper")
            return result
        except Exception as e:
            logger.error(f" Audio transcription failed: {e}")
            return {
                "success": False,
                "transcription": "Audio transcription failed. Please type your message.",
                "error": str(e)
            }
    
    async def generate_medical_response(self, query: str, context: str = "", **kwargs) -> Dict[str, Any]:
        """
        Generate medical text response with proper prioritization:
        1. MedGemma (local)
        2. MedGemma Model Garden (cloud)
        3. OpenAI GPT-4 (emergency fallback)
        
        Args:
            query: The medical query
            context: Additional context
            **kwargs: Additional parameters (max_length, temperature, etc.) for underlying services
        """
        
        # Priority 1: Local MedGemma
        if self.services.get('medgemma_local'):
            try:
                result = await self.services['medgemma_local'].generate_medical_response(
                    query=query, 
                    context=context,
                    **kwargs  # Forward all additional parameters
                )
                if result.get('success'):
                    logger.info(" Response generated with local MedGemma")
                    return result
            except Exception as e:
                logger.warning(f"  Local MedGemma text generation failed: {e}")
        
        # Priority 2: MedGemma Model Garden
        if self.services.get('medgemma_cloud'):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful medical AI assistant."},
                    {"role": "user", "content": f"{context}\n\n{query}" if context else query}
                ]
                result = await self.services['medgemma_cloud'].generate_medical_response(messages)
                if result.get('success'):
                    logger.info(" Response generated with MedGemma Model Garden")
                    return result
            except Exception as e:
                logger.warning(f"  Model Garden text generation failed: {e}")
        
        # Priority 3: OpenAI GPT-4 (emergency fallback)
        if self.services.get('openai'):
            try:
                # Use the RAG enhancement as text generation fallback
                result = await self.services['openai'].enhance_diagnosis_with_rag(query, context)
                logger.info(" Response generated with OpenAI (fallback)")
                return {
                    "success": True,
                    "response": result,
                    "service_used": "openai_gpt4_fallback"
                }
            except Exception as e:
                logger.error(f" OpenAI text generation failed: {e}")
        
        # All services failed
        return {
            "success": False,
            "response": "I'm having trouble processing your request. Please consult a healthcare professional.",
            "error": "All text generation services unavailable"
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "mode": self.mode.value,
            "services": {
                "medgemma_local": {
                    "available": self.services.get('medgemma_local') is not None,
                    "capabilities": ["text", "images", "multimodal"] if self.services.get('medgemma_local') else []
                },
                "medgemma_cloud": {
                    "available": self.services.get('medgemma_cloud') is not None,
                    "capabilities": ["text", "images", "multimodal"] if self.services.get('medgemma_cloud') else []
                },
                "openai": {
                    "available": self.services.get('openai') is not None,
                    "capabilities": ["audio", "images", "text"] if self.services.get('openai') else []
                }
            },
            "image_analysis_available": any([
                self.services.get('medgemma_local'),
                self.services.get('medgemma_cloud'), 
                self.services.get('openai')
            ]),
            "audio_transcription_available": self.services.get('openai') is not None,
            "text_generation_available": any([
                self.services.get('medgemma_local'),
                self.services.get('medgemma_cloud'),
                self.services.get('openai')
            ])
        }

# Factory function for easy initialization
def create_ai_service_manager(mode_str: str = "hybrid") -> OptimizedAIServiceManager:
    """Create AI service manager with specified mode"""
    mode_map = {
        "local_demo": ServiceMode.LOCAL_DEMO,
        "hybrid": ServiceMode.HYBRID,
        "cloud_first": ServiceMode.CLOUD_FIRST
    }
    
    mode = mode_map.get(mode_str.lower(), ServiceMode.HYBRID)
    return OptimizedAIServiceManager(mode)
