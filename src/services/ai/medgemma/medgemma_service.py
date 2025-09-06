"""
Simplified MedGemma Service following official Google implementation
Based on: quick_start_with_hugging_face.ipynb
"""

import asyncio
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import (
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
        use_quantization: bool = True,  # Default to True for Colab efficiency
        multimodal: Optional[bool] = None
    ):
        """
        Initialize MedGemma service with LAZY LOADING and optimized for Colab demos.
        - Defaults to 4-bit quantization for speed and memory.
        - Uses only the direct, memory-efficient inference method.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device setting ("auto").
            use_quantization: Enable 4-bit quantization.
            multimodal: Auto-detected from model name.
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        # Extract model variant (following official notebook)
        self.model_variant = model_name.split("/")[-1].replace("medgemma-", "") if "/" in model_name else "4b-it"
        self.is_text_only = "text" in self.model_variant
        
        # Set task based on variant (following official logic)
        self.task = "text-generation" if self.is_text_only else "image-text-to-text"
        
        # LAZY LOADING: Initialize components as None, load only on first request
        self.model = None
        self.processor_or_tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=1) # Single worker is enough for a demo
        
        # Lazy loading state management
        self.is_loaded = False
        self.load_lock = asyncio.Lock()
        
        logger.info(f" MedGemmaService initialized for LAZY LOADING with model {self.model_name}")
        logger.info(f"   Quantization: {self.use_quantization}. Model will be loaded on first request.")
    
    async def _ensure_model_loaded(self):
        """
        Asynchronously checks if the model is loaded and loads it if not.
        Uses a lock to prevent multiple concurrent loading attempts.
        """
        if self.is_loaded:
            return
        
        async with self.load_lock:
            # Double-check after acquiring the lock
            if self.is_loaded:
                return
            
            logger.info(" First request received. Lazily loading MedGemma model...")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._load_model_and_processor
                )
                self.is_loaded = True
                logger.info(" Model loaded successfully on first use.")
            except Exception as e:
                logger.error(f" Failed to lazily load MedGemma model: {e}")
                
                # Reset state to allow another attempt
                self.model = None
                self.processor_or_tokenizer = None
                self.is_loaded = False
                raise  # Re-raise the exception to the caller
    
    def _load_model_and_processor(self):
        """
        Load model following official Google notebook implementation exactly.
        Optimized for Colab demos with 4-bit quantization by default.
        """
        try:
            logger.info(f" Loading MedGemma model: {self.model_name}")
            logger.info(f"   Quantization: {self.use_quantization}")
            
            # Model kwargs following official notebook exactly
            model_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Add quantization if requested (official pattern)
            if self.use_quantization:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                    logger.info("    4-bit quantization enabled for Colab efficiency.")
                except Exception as e:
                    logger.warning(f"    Quantization failed, falling back to non-quantized: {e}")
                    self.use_quantization = False
            
            # Load model and processor/tokenizer directly (official implementation)
            
            if self.is_text_only:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.processor_or_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                from transformers import AutoModelForImageTextToText, AutoProcessor
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
                self.processor_or_tokenizer = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info(" MedGemma model loaded successfully following official notebook patterns")
            
        except Exception as e:
            logger.error(f" Failed to load MedGemma model: {e}")
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
        Generate medical response using the optimized direct method with LAZY LOADING.
        
        Args:
            query: User's medical question.
            context: Additional context (optional).
            max_new_tokens: Maximum new tokens to generate.
            **kwargs: Additional parameters (ignored).
        
        Returns:
            Dict containing the response and metadata.
        """
        # LAZY LOADING: Ensure model is loaded before generating response
        try:
            await self._ensure_model_loaded()
        except Exception as e:
            return {
                "success": False,
                "response": "Failed to load MedGemma model. Please try again later.",
                "error": f"Model loading failed: {str(e)}"
            }
        
        if not self.model or not self.processor_or_tokenizer:
            return {
                "success": False,
                "response": "MedGemma model is not available.",
                "error": "Model not initialized"
            }
        
        try:
            # Create messages following official notebook format exactly
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
            
            # Adjust max_new_tokens following official notebook (4b model: 500 for text)
            max_new_tokens = min(max_new_tokens, 500)
            
            # Run inference using the direct model method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_with_direct_model,
                messages,
                max_new_tokens
            )
            
            logger.info(f" MedGemma response generated successfully using direct method")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "model_variant": self.model_variant,
                "method": "direct_model",
                "max_new_tokens": max_new_tokens
            }
            
        except Exception as e:
            logger.error(f" MedGemma generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical query. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _generate_with_direct_model(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """
        Generate text using direct model method (exact official notebook implementation).
        This follows the official HuggingFace notebook pattern precisely.
        """
        # Apply chat template following official implementation exactly
        inputs = self.processor_or_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with inference mode (exact official pattern)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False  # Following official notebook default
            )
            generation = generation[0][input_len:]
        
        # Decode response following official implementation
        response = self.processor_or_tokenizer.decode(generation, skip_special_tokens=True)
        return response
    
    async def analyze_image_with_text(
        self, 
        image, 
        text_prompt: str = "Describe this medical image.",
        max_new_tokens: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze medical images using the optimized direct method with LAZY LOADING.
        
        Args:
            image: PIL Image or image path.
            text_prompt: Text prompt for analysis.
            max_new_tokens: Maximum new tokens to generate.
            **kwargs: Additional parameters (ignored).
        
        Returns:
            Dict containing the multimodal analysis.
        """
        if self.is_text_only:
            return {
                "success": False,
                "response": "This is a text-only model variant. Multimodal analysis not supported.",
                "error": "Text-only model"
            }
        
        # LAZY LOADING: Ensure model is loaded before analyzing image
        try:
            await self._ensure_model_loaded()
        except Exception as e:
            return {
                "success": False,
                "response": "Failed to load MedGemma model for image analysis. Please try again later.",
                "error": f"Model loading failed: {str(e)}"
            }
        
        if not self.model or not self.processor_or_tokenizer:
            return {
                "success": False,
                "response": "MedGemma model is not available.",
                "error": "Model not initialized"
            }
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                try:
                    from PIL import Image
                    image = Image.open(image)
                except ImportError:
                    return {
                        "success": False,
                        "response": "PIL library not available for image processing.",
                        "error": "PIL import failed"
                    }
            
            # Create messages following official notebook multimodal format exactly
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
            
            # Adjust max_new_tokens following official notebook (4b model: 300 for multimodal)
            max_new_tokens = min(max_new_tokens, 300)
            
            # Run inference using the direct model method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_multimodal_with_direct_model,
                messages,
                max_new_tokens
            )
            
            logger.info(f" MedGemma multimodal response generated successfully using direct method")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "method": "direct_model",
                "mode": "multimodal"
            }
            
        except Exception as e:
            logger.error(f" MedGemma multimodal generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical image query. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _generate_multimodal_with_direct_model(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """
        Generate multimodal response using direct model method (exact official notebook implementation).
        This follows the official HuggingFace notebook pattern precisely for multimodal tasks.
        """
        # Apply chat template for multimodal following official implementation exactly
        inputs = self.processor_or_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with inference mode (exact official pattern)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False  # Following official notebook default
            )
            generation = generation[0][input_len:]
        
        # Decode response following official implementation
        response = self.processor_or_tokenizer.decode(generation, skip_special_tokens=True)
        return response

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
            "quantization_enabled": self.use_quantization,
            "cuda_available": torch.cuda.is_available(),
            "official_implementation": True,
            "torch_dtype": "bfloat16"
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
