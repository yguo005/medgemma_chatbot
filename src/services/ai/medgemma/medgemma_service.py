"""
Simplified MedGemma Service following official Google implementation
Based on: quick_start_with_hugging_face.ipynb
"""

import asyncio
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForImageTextToText, 
    AutoProcessor
)
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedGemmaService:
    """
    Simplified MedGemma service following official Google implementation
    """
    
    def __init__(
        self, 
        model_name: str = "google/medgemma-4b-it",
        device: str = "auto",
        use_quantization: bool = False,
        multimodal: Optional[bool] = None
    ):
        """
        Initialize MedGemma service following official Google notebook
        
        Args:
            model_name: HuggingFace model identifier (e.g., "google/medgemma-4b-it")
            device: Device setting (kept for compatibility, always uses "auto")
            use_quantization: Enable 4-bit quantization for memory efficiency
            multimodal: Multimodal setting (auto-detected from model name)
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        # Extract model variant (following official notebook)
        self.model_variant = model_name.split("/")[-1].replace("medgemma-", "") if "/" in model_name else "4b-it"
        self.is_text_only = "text" in self.model_variant
        
        # Set task based on variant (following official logic)
        if self.is_text_only:
            self.task = "text-generation"
        else:
            self.task = "image-text-to-text"
        
        # Initialize components
        self.pipeline = None
        self.model = None
        self.processor_or_tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load model following official implementation
        self._load_model()
    
    def _load_model(self):
        """Load model following official Google notebook implementation"""
        try:
            logger.info(f"🚀 Loading MedGemma model: {self.model_name}")
            logger.info(f"   Variant: {self.model_variant}")
            logger.info(f"   Task: {self.task}")
            logger.info(f"   Quantization: {self.use_quantization}")
            
            # Model kwargs following official notebook exactly
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
            
            # Add quantization if requested (following official pattern)
            if self.use_quantization:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            
            # Create pipeline (following official implementation)
            self.pipeline = pipeline(self.task, model=self.model_name, model_kwargs=model_kwargs)
            
            # Set generation config (following official notebook)
            self.pipeline.model.generation_config.do_sample = False
            
            # Load model and processor/tokenizer directly for advanced usage
            if self.is_text_only:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.processor_or_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
                self.processor_or_tokenizer = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info("✅ MedGemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MedGemma model: {e}")
            self.pipeline = None
            self.model = None
            self.processor_or_tokenizer = None
            raise
    
    async def generate_medical_response(
        self, 
        query: str, 
        context: str = "", 
        max_new_tokens: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate medical response following official implementation
        
        Args:
            query: User's medical question
            context: Additional context (optional)
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional parameters (ignored for simplicity)
        
        Returns:
            Dict containing the response and metadata
        """
        if not self.pipeline:
            return {
                "success": False,
                "response": "MedGemma model is not available.",
                "error": "Model not initialized"
            }
        
        try:
            # Create messages following official format
            system_instruction = "You are a helpful medical assistant."
            
            user_text = f"Context: {context}\n\nQuestion: {query}" if context else query
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]
                }
            ]
            
            # Adjust max_new_tokens based on variant (following official notebook)
            if "27b" in self.model_variant:
                max_new_tokens = min(max_new_tokens, 1500)
            else:
                max_new_tokens = min(max_new_tokens, 500)
            
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_with_pipeline,
                messages,
                max_new_tokens
            )
            
            # Extract response (following official format)
            response = result[0]["generated_text"][-1]["content"]
            
            logger.info("✅ MedGemma response generated successfully")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "model_variant": self.model_variant,
                "max_new_tokens": max_new_tokens
            }
            
        except Exception as e:
            logger.error(f"❌ MedGemma generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical query. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _generate_with_pipeline(self, messages: List[Dict[str, Any]], max_new_tokens: int):
        """Generate text using pipeline (following official implementation)"""
        return self.pipeline(messages, max_new_tokens=max_new_tokens)
    
    async def analyze_image_with_text(
        self, 
        image, 
        text_prompt: str = "Describe this medical image.",
        max_new_tokens: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze medical images following official multimodal implementation
        
        Args:
            image: PIL Image or image path
            text_prompt: Text prompt for analysis
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional parameters (for compatibility)
        
        Returns:
            Dict containing the multimodal analysis
        """
        if self.is_text_only:
            return {
                "success": False,
                "response": "This is a text-only model variant. Multimodal analysis not supported.",
                "error": "Text-only model"
            }
        
        if not self.pipeline:
            return {
                "success": False,
                "response": "MedGemma model is not available.",
                "error": "Model not initialized"
            }
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image)
            
            # Create messages following official multimodal format
            system_instruction = "You are an expert radiologist."
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_with_pipeline,
                messages,
                max_new_tokens
            )
            
            # Extract response
            response = result[0]["generated_text"][-1]["content"]
            
            logger.info("✅ MedGemma multimodal response generated successfully")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "mode": "multimodal"
            }
            
        except Exception as e:
            logger.error(f"❌ MedGemma multimodal generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble analyzing this image.",
                "error": str(e)
            }

    # Legacy method compatibility
    async def analyze_symptoms(self, symptoms: str, duration: str = "", intensity: str = "", timing: str = "") -> Dict[str, Any]:
        """Legacy method for symptom analysis"""
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
        """Legacy method for diagnosis enhancement"""
        query = f"Given these symptoms: {symptoms}\n\nAnd this medical information: {rag_response}\n\nProvide a clear, helpful medical summary with appropriate recommendations."
        
        result = await self.generate_medical_response(query)
        
        if result["success"]:
            return result["response"]
        else:
            return rag_response  # Fallback to original RAG response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_variant": self.model_variant,
            "text_only": self.is_text_only,
            "device": "auto",
            "task": self.task,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.processor_or_tokenizer is not None,
            "pipeline_ready": self.pipeline is not None,
            "quantization_enabled": self.use_quantization,
            "cuda_available": torch.cuda.is_available(),
            "official_implementation": True,
            "torch_dtype": "bfloat16"
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
