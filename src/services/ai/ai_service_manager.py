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
        
        logger.info(f"Initializing AI Service Manager in '{self.mode.value}' mode.")
        
        # Always try to initialize local MedGemma first
        try:
            from src.services.ai.medgemma.medgemma_service import MedGemmaService
            self.services['medgemma_local'] = MedGemmaService(
                model_name="google/medgemma-4b-it",
                use_quantization=None  # Auto-detect based on platform (Mac compatibility)
            )
            quant_status = "with auto-detected quantization" if self.services['medgemma_local'].use_quantization else "without quantization (Mac compatible)"
            logger.info(f" Local MedGemma initialized (multimodal 4B-IT {quant_status})")
        except Exception as e:
            logger.warning(f" Local MedGemma failed: {e}")
            self.services['medgemma_local'] = None
        
        # Initialize Model Garden if not in local demo mode
        if self.mode != ServiceMode.LOCAL_DEMO:
            try:
                from src.services.ai.medgemma.model_garden import MedGemmaModelGarden
                gcp_project_id = os.getenv("GCP_PROJECT_ID")
                medgemma_endpoint_id = os.getenv("MEDGEMMA_ENDPOINT_ID")
                if gcp_project_id and medgemma_endpoint_id:
                    self.services['medgemma_cloud'] = MedGemmaModelGarden(
                        project_id=gcp_project_id,
                        endpoint_id=medgemma_endpoint_id
                    )
                    logger.info(" MedGemma Model Garden initialized")
                else:
                    self.services['medgemma_cloud'] = None
                    if not gcp_project_id:
                        logger.warning(" GCP_PROJECT_ID not set, Model Garden disabled.")
                    if not medgemma_endpoint_id:
                        logger.warning(" MEDGEMMA_ENDPOINT_ID not set, Model Garden disabled.")

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
        3. OpenAI GPT-4o (emergency fallback)
        """
        
        # Priority 1: Local MedGemma multimodal (should work on Colab T4 with quantization)
        medgemma_service = self.services.get('medgemma_local')
        if medgemma_service:
            try:
                # First ensure the model is actually loaded and ready
                await medgemma_service._ensure_model_loaded()
                
                # Check if service is actually ready
                if not medgemma_service.is_service_ready():
                    logger.warning("  Local MedGemma service exists but model failed to load")
                    # Continue to next service
                else:
                    # Convert base64 to PIL Image for MedGemma
                    import tempfile
                    import base64
                    from PIL import Image as PILImage
                    
                    # Clean image data
                    if image_data.startswith('data:'):
                        image_data = image_data.split(',', 1)[1]
                    
                    # Convert to PIL Image
                    image_bytes = base64.b64decode(image_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        temp_file.write(image_bytes)
                        temp_path = temp_file.name
                    
                    # Load as PIL Image
                    pil_image = PILImage.open(temp_path)
                    
                    result = await medgemma_service.analyze_image_with_text(
                        image=pil_image,  # Pass PIL Image object
                        text_prompt="Analyze this medical image. Describe any visible symptoms, conditions, or abnormalities."
                    )
                    
                    # Cleanup
                    os.unlink(temp_path)
                    
                    if result.get('success'):
                        logger.info(" Image analyzed with local MedGemma multimodal")
                        return {
                            "success": True,
                            "analysis": result['response'],
                            "service_used": "medgemma_local_multimodal"
                        }
                    else:
                        # Log the specific error for debugging
                        logger.warning(f"  Local MedGemma image analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.warning(f"  Local MedGemma image analysis failed: {e}")
                # Continue to fallback services
        
        # Priority 2: MedGemma Model Garden multimodal
        model_garden_service = self.services.get('medgemma_cloud')
        if model_garden_service:
            try:
                # Convert to proper format for Model Garden
                image_url = f"data:image/jpeg;base64,{image_data}"
                result = await model_garden_service.analyze_symptoms_multimodal(
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
        
        # Priority 3: OpenAI GPT-4o (emergency fallback)
        openai_service = self.services.get('openai')
        if openai_service:
            try:
                result = await openai_service.analyze_image(image_data, context)
                if result.get('success'):
                    logger.info(" Image analyzed with OpenAI (fallback)")
                    return {
                        "success": True,
                        "analysis": result['analysis'],
                        "service_used": "openai_4o_fallback"
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
        Generate comprehensive medical diagnostic response using all collected symptoms and answers.
        Prioritizes MedGemma (local/cloud) over OpenAI for medical accuracy.
        
        Args:
            query: The medical query (diagnostic request)
            context: Additional context including symptoms, duration, intensity, timing, etc.
            **kwargs: Additional parameters (max_length, temperature, etc.) for underlying services
        
        Returns:
            Dict containing the diagnostic response and metadata
        """
        
        # Define service priority based on the manager's mode
        if self.mode == ServiceMode.CLOUD_FIRST:
            service_priority = ['medgemma_cloud', 'medgemma_local', 'openai']
            logger.info("🎯 Cloud-first mode: Prioritizing MedGemma Model Garden for diagnostic summaries")
        else:  # Default to hybrid or local_demo priority
            service_priority = ['medgemma_local', 'medgemma_cloud', 'openai']
            logger.info("🏠 Hybrid mode: Prioritizing local MedGemma first")
            
        logger.info(f"Service priority for diagnostic generation: {service_priority}")

        # Iterate through services based on priority
        for service_name in service_priority:
            if service_name == 'medgemma_local':
                medgemma_local = self.services.get('medgemma_local')
                if medgemma_local:
                    try:
                        await medgemma_local._ensure_model_loaded()
                        if not medgemma_local.is_service_ready():
                            logger.warning("Local MedGemma service exists but model is not ready.")
                            continue
                        
                        result = await medgemma_local.generate_medical_response(query=query, context=context, **kwargs)
                        if result.get('success'):
                            logger.info("✅ Diagnostic response generated with local MedGemma")
                            return {**result, "service_used": "medgemma_local"}
                        else:
                            logger.warning(f"Local MedGemma failed: {result.get('error', 'Unknown')}")
                    except Exception as e:
                        logger.warning(f"Local MedGemma diagnostic generation failed with exception: {e}")

            elif service_name == 'medgemma_cloud':
                medgemma_cloud = self.services.get('medgemma_cloud')
                if medgemma_cloud:
                    try:
                        # Format the comprehensive medical context for Model Garden
                        system_instruction = "You are a medical AI assistant. Provide a comprehensive diagnostic analysis based on the patient's symptoms and answers."
                        
                        # Combine context and query into a medical consultation format
                        medical_prompt = f"""Patient Information and Symptoms:
{context}

Diagnostic Request:
{query}

Please provide a thorough medical analysis including:
1. Possible conditions to consider
2. Recommended next steps
3. When to seek immediate medical attention
4. Important disclaimers about professional medical consultation"""

                        messages = [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": medical_prompt}
                        ]
                        
                        result = await medgemma_cloud.generate_medical_response(messages)
                        if result.get('success'):
                            logger.info("✅ Diagnostic response generated with MedGemma Model Garden")
                            return {**result, "service_used": "medgemma_cloud"}
                        else:
                            logger.warning(f"Model Garden failed: {result.get('error', 'Unknown')}")
                    except Exception as e:
                        logger.warning(f"Model Garden diagnostic generation failed with exception: {e}")

            elif service_name == 'openai':
                openai_fallback = self.services.get('openai')
                if openai_fallback:
                    try:
                        # Use OpenAI as the final fallback for diagnostic generation
                        diagnostic_query = f"""Based on the following patient information, provide a comprehensive medical analysis:

{context}

Query: {query}

Please provide:
1. Possible medical conditions to consider
2. Recommended actions and next steps
3. Red flags that require immediate medical attention
4. Clear disclaimer about consulting healthcare professionals

Remember to be informative but not definitive in diagnosis."""

                        response_text = await openai_fallback.enhance_diagnosis_with_rag(diagnostic_query, "")
                        logger.info("⚠️  Diagnostic response generated with OpenAI (fallback)")
                        return {
                            "success": True,
                            "response": response_text,
                            "service_used": "openai_gpt4_diagnostic_fallback"
                        }
                    except Exception as e:
                        logger.error(f"OpenAI diagnostic generation failed with exception: {e}")

        # All services failed - return a safe fallback response
        logger.error("❌ All diagnostic generation services failed")
        return {
            "success": False,
            "response": """I'm having trouble processing your medical information right now. 

For your safety, please consider:
- Consulting with a healthcare professional about your symptoms
- Seeking immediate medical attention if you have severe or worsening symptoms
- Contacting emergency services if this is a medical emergency

*This system is not a substitute for professional medical advice, diagnosis, or treatment.*""",
            "error": "All diagnostic generation services unavailable"
        }

    async def generate_conversational_response(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Generates a conversational (non-diagnostic) response, always using OpenAI for speed and cost.
        Used for tasks like generating follow-up questions.
        """
        openai_service = self.services.get('openai')
        
        try:
            # Use the specialized simple question generator for clean, concise questions
            response_text = await openai_service.generate_simple_question(query)
            logger.info("✅ Conversational question generated with OpenAI.")
            return {
                "success": True,
                "response": response_text,
                "service_used": "openai_gpt4_simple_question"
            }
        except Exception as e:
            logger.error(f" OpenAI conversational response failed: {e}")
            # Fallback to a simple, hard-coded response
            return {
                "success": False,
                "response": "Could you please provide more details?",
                "error": str(e)
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
