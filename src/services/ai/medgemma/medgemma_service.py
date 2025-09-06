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
        """Load model following official Google notebook implementation exactly"""
        try:
            logger.info(f"🚀 Loading MedGemma model: {self.model_name}")
            logger.info(f"   Variant: {self.model_variant}")
            logger.info(f"   Task: {self.task}")
            logger.info(f"   Quantization: {self.use_quantization}")
            
            # Check 27B variant requirements (following official notebook)
            if "27b" in self.model_variant and self.use_quantization:
                try:
                    import google.colab  # type: ignore
                    google_colab = True
                except ImportError:
                    google_colab = False
                
                if google_colab and self.use_quantization:
                    if not ("A100" in torch.cuda.get_device_name(0) and self.use_quantization):
                        logger.warning(
                            "Runtime may have insufficient memory to run a 27B variant. "
                            "A100 GPU and 4-bit quantization are recommended."
                        )
            
            # Model kwargs following official notebook exactly
            model_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            # Add quantization if requested (following official pattern exactly)
            if self.use_quantization:
                try:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                    logger.info("   ✅ Quantization config created successfully")
                except Exception as e:
                    logger.warning(f"   ⚠️  Quantization failed, falling back to non-quantized: {e}")
                    self.use_quantization = False  # Update the flag
            
            # Load model and processor/tokenizer directly (official implementation)
            try:
                if self.is_text_only:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                    self.processor_or_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                else:
                    self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
                    self.processor_or_tokenizer = AutoProcessor.from_pretrained(self.model_name)
                    
                logger.info("   ✅ Direct model loading successful")
                
            except Exception as e:
                if "bitsandbytes" in str(e) and "quantization_config" in model_kwargs:
                    logger.warning("   🔄 Direct model loading failed with quantization, retrying without...")
                    # Remove quantization and retry direct model loading
                    model_kwargs.pop("quantization_config", None)
                    self.use_quantization = False
                    if self.is_text_only:
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                        self.processor_or_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    else:
                        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
                        self.processor_or_tokenizer = AutoProcessor.from_pretrained(self.model_name)
                else:
                    raise
            
            # Create pipeline as backup method (following official implementation)
            try:
                self.pipeline = pipeline(self.task, model=self.model_name, model_kwargs=model_kwargs)
                # Set generation config (following official notebook)
                self.pipeline.model.generation_config.do_sample = False
                logger.info("   ✅ Pipeline created successfully")
            except Exception as e:
                logger.warning(f"   ⚠️  Pipeline creation failed: {e}")
                self.pipeline = None
            
            logger.info("✅ MedGemma model loaded successfully with official methods")
            
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
        use_direct_method: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate medical response using official implementation methods
        
        Args:
            query: User's medical question
            context: Additional context (optional)
            max_new_tokens: Maximum new tokens to generate
            use_direct_method: Use direct model generation (more memory efficient) vs pipeline
            **kwargs: Additional parameters (ignored for simplicity)
        
        Returns:
            Dict containing the response and metadata
        """
        if not self.model or not self.processor_or_tokenizer:
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
            
            # Use direct method (more memory efficient) or pipeline method
            if use_direct_method:
                # Run inference using direct model method (official implementation)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.executor,
                    self._generate_with_direct_model,
                    messages,
                    max_new_tokens
                )
                method_used = "direct_model"
            else:
                # Fallback to pipeline method
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._generate_with_pipeline,
                    messages,
                    max_new_tokens
                )
                response = result[0]["generated_text"][-1]["content"]
                method_used = "pipeline"
            
            logger.info(f"✅ MedGemma response generated successfully using {method_used}")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "model_variant": self.model_variant,
                "method": method_used,
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
    
    def _generate_with_direct_model(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """
        Generate text using direct model method (following official notebook)
        This is more memory efficient than the pipeline approach
        """
        # Apply chat template following official implementation
        inputs = self.processor_or_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with inference mode for memory efficiency (official pattern)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
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
        use_direct_method: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze medical images following official multimodal implementation
        
        Args:
            image: PIL Image or image path
            text_prompt: Text prompt for analysis
            max_new_tokens: Maximum new tokens to generate
            use_direct_method: Use direct model generation (more memory efficient) vs pipeline
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
            
            # Adjust max_new_tokens for multimodal (following official notebook)
            if "27b" in self.model_variant:
                max_new_tokens = min(max_new_tokens, 1300)
            else:
                max_new_tokens = min(max_new_tokens, 300)
            
            # Use direct method (more memory efficient) or pipeline method
            if use_direct_method:
                # Run inference using direct model method (official implementation)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.executor,
                    self._generate_multimodal_with_direct_model,
                    messages,
                    max_new_tokens
                )
                method_used = "direct_model"
            else:
                # Fallback to pipeline method
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._generate_with_pipeline,
                    messages,
                    max_new_tokens
                )
                response = result[0]["generated_text"][-1]["content"]
                method_used = "pipeline"
            
            logger.info(f"✅ MedGemma multimodal response generated successfully using {method_used}")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "method": method_used,
                "mode": "multimodal"
            }
            
        except Exception as e:
            logger.error(f"❌ MedGemma multimodal generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical image query. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _generate_multimodal_with_direct_model(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """
        Generate multimodal response using direct model method (following official notebook)
        This is more memory efficient than the pipeline approach for multimodal tasks
        """
        # Apply chat template for multimodal following official implementation
        inputs = self.processor_or_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with inference mode for memory efficiency (official pattern)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
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
