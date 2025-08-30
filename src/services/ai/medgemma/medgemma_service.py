import os
import logging
from typing import Dict, Any, Optional, Union, List
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoProcessor  # Added for multimodal support
)

# Import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    logging.warning(" PIL not available. Image processing features will be limited.")

# Import requests for URL image loading
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    logging.warning(" requests not available. URL image loading will be disabled.")

# Try to import pipeline (may be missing in corrupted installations)
try:
    from transformers import pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    pipeline = None
    PIPELINE_AVAILABLE = False
    logging.warning(" transformers.pipeline not available. MedGemma service will be disabled.")

# Try to import AutoModelForImageTextToText (available in transformers >= 4.42.0)
try:
    from transformers import AutoModelForImageTextToText
    MULTIMODAL_AVAILABLE = True
except ImportError:
    # Fallback for older transformers versions
    AutoModelForImageTextToText = None
    MULTIMODAL_AVAILABLE = False
    logging.warning(" AutoModelForImageTextToText not available. Multimodal features disabled. Upgrade transformers to >= 4.42.0 for full functionality.")
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedGemmaService:
    """
    MedGemma service for medical AI responses using Hugging Face Transformers
    Based on Google Health's MedGemma model
    """
    
    def __init__(
        self, 
        model_name: str = "google/medgemma-4b-it",  # Fixed: Use 4B instruction-tuned variant
        device: str = "auto",
        use_quantization: bool = False,
        multimodal: bool = False
    ):
        """
        Initialize MedGemma service
        
        Args:
            model_name: HuggingFace model identifier for MedGemma
            device: Device to run the model on ("auto", "cpu", "cuda")
            use_quantization: Enable 4-bit quantization for memory efficiency
            multimodal: Use multimodal variant for image-text-to-text tasks
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.use_quantization = use_quantization
        self.multimodal = multimodal
        self.model = None
        self.processor_or_tokenizer = None  # Use unified attribute for processor/tokenizer
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # For async operations
        
        # Determine the appropriate task and model class
        # Disable multimodal if not available in current transformers version
        if multimodal and not MULTIMODAL_AVAILABLE:
            logger.warning(" Multimodal requested but not available. Falling back to text-only mode.")
            multimodal = False
            self.multimodal = False
        
        self.task = "image-text-to-text" if multimodal else "text-generation"
        self.model_class = AutoModelForImageTextToText if (multimodal and MULTIMODAL_AVAILABLE) else AutoModelForCausalLM
        
        # Check if pipeline is available
        if not PIPELINE_AVAILABLE:
            logger.error(" transformers.pipeline not available. MedGemma service disabled.")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            return
        
        # Initialize the model
        self._initialize_model()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():  
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self):
        """Initialize the MedGemma model and tokenizer with improved configuration"""
        try:
            logger.info(f" Loading MedGemma model: {self.model_name}")
            logger.info(f" Using device: {self.device}")
            logger.info(f" Quantization: {'Enabled' if self.use_quantization else 'Disabled'}")
            logger.info(f" Multimodal: {'Enabled' if self.multimodal else 'Disabled'}")
            
            # Load processor for multimodal or tokenizer for text-only
            if self.multimodal and MULTIMODAL_AVAILABLE:
                logger.info(" Loading AutoProcessor for multimodal support...")
                self.processor_or_tokenizer = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            else:
                logger.info(" Loading AutoTokenizer for text-only support...")
                self.processor_or_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            
            # Configure model loading parameters
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            # Use torch.bfloat16 for better performance and memory efficiency
            if self.device != "cpu":
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32
            
            # Add quantization configuration if enabled
            if self.use_quantization and self.device == "cuda":
                logger.info("🔧 Enabling 4-bit quantization for memory efficiency")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.use_quantization and self.device != "cuda":
                logger.warning(" Quantization is only supported on CUDA devices. Disabling quantization.")
                self.use_quantization = False
            
            # Load the appropriate model class
            self.model = self.model_class.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map (for non-CUDA devices)
            if self.device != "cuda" and not self.use_quantization:
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier inference
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.processor_or_tokenizer,  # Use unified processor/tokenizer
            }
            
            if not self.use_quantization:
                pipeline_kwargs["device"] = 0 if self.device == "cuda" else -1
                if self.device != "cpu":
                    pipeline_kwargs["torch_dtype"] = torch.bfloat16
            
            self.pipeline = pipeline(self.task, **pipeline_kwargs)
            
            # Configure generation settings to match official implementation
            self.pipeline.model.generation_config.do_sample = False
            
            logger.info(" MedGemma model loaded successfully")
            
        except Exception as e:
            logger.error(f" Failed to load MedGemma model: {e}")
            self.model = None
            self.processor_or_tokenizer = None
            self.pipeline = None
    
    async def generate_medical_response(
        self, 
        query: str, 
        context: str = "", 
        max_length: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate medical response using MedGemma
        
        Args:
            query: User's medical question
            context: Additional context (symptoms, history, etc.)
            max_length: Maximum response length
            temperature: Sampling temperature (lower = more conservative)
        
        Returns:
            Dict containing the response and metadata
        """
        if not self.pipeline:
            return {
                "success": False,
                "response": "MedGemma model is not available. Please check the model loading.",
                "error": "Model not initialized"
            }
        
        try:
            # Construct chat messages using modern chat template format
            messages = self._construct_chat_messages(query, context)
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_with_chat_template,
                messages,
                max_length
            )
            
            # Extract the response from chat template output
            response = result[0]["generated_text"][-1]["content"]
            
            logger.info(" MedGemma response generated successfully")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "device": self.device,
                "prompt_length": len(str(messages)),  # Approximate length of messages
                "response_length": len(response)
            }
            
        except Exception as e:
            logger.error(f" MedGemma generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical query right now. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _construct_chat_messages(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """Construct structured message list for chat template (NEW METHOD)"""
        system_instruction = "You are a helpful medical assistant. Provide informative but not diagnostic advice, and always recommend consulting a healthcare professional."
        
        user_content = []
        if context:
            user_content.append({"type": "text", "text": f"Context: {context}\n\nQuestion: {query}"})
        else:
            user_content.append({"type": "text", "text": query})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": user_content}
        ]

    def _generate_with_chat_template(self, messages: List[Dict[str, Any]], max_length: int) -> list:
        """Generate text using chat template with pipeline (NEW METHOD)"""
        return self.pipeline(
            messages,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=self.processor_or_tokenizer.eos_token_id,
        )

    def _construct_medical_prompt(self, query: str, context: str = "") -> str:
        """Construct a proper medical prompt for MedGemma (LEGACY METHOD - kept for fallback)"""
        
        # MedGemma-specific prompt format
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
    
    def _generate_text(self, prompt: str, max_length: int, temperature: float) -> list:
        """Generate text using the pipeline (LEGACY METHOD - kept for fallback)"""
        # Use official MedGemma generation parameters
        return self.pipeline(
            prompt,
            max_new_tokens=300,  # Official notebook uses max_new_tokens instead of max_length
            do_sample=False,     # Official implementation uses deterministic generation
            pad_token_id=self.processor_or_tokenizer.eos_token_id,
            num_return_sequences=1,
            return_full_text=False  # Return only generated text, not full input
        )
    
    def _generate_multimodal_response(self, inputs: Dict[str, Any], max_length: int, temperature: float) -> Dict[str, Any]:
        """Generate multimodal response using the image-text-to-text pipeline (runs in thread pool)"""
        # Use official MedGemma generation parameters for multimodal
        return self.pipeline(
            inputs,
            max_new_tokens=300,  # Official notebook uses max_new_tokens
            do_sample=False,     # Official implementation uses deterministic generation
            pad_token_id=self.processor_or_tokenizer.eos_token_id,  # Updated to use processor_or_tokenizer
            num_return_sequences=1
        )
    
    def _construct_multimodal_chat_messages(self, text_prompt: str, image: Any) -> List[Dict[str, Any]]:
        """Construct structured message list for multimodal chat template (NEW METHOD)"""
        system_instruction = "You are a medical AI assistant analyzing medical images. Provide helpful, accurate medical observations while always emphasizing the importance of consulting healthcare professionals for proper diagnosis and treatment."
        
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt}
        ]

        return [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": user_content}
        ]
    
    def _generate_multimodal_with_chat_template(self, messages: List[Dict[str, Any]], max_length: int) -> list:
        """Generate multimodal text using chat template with pipeline (NEW METHOD)"""
        return self.pipeline(
            messages,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=self.processor_or_tokenizer.eos_token_id,
        )

    def _construct_multimodal_prompt(self, text_prompt: str) -> str:
        """Construct a proper multimodal medical prompt for MedGemma"""
        
        system_prompt = """You are a medical AI assistant analyzing medical images. Provide helpful, accurate medical observations while always emphasizing the importance of consulting healthcare professionals for proper diagnosis and treatment.

Important guidelines:
- Describe what you observe in the image objectively
- Mention relevant medical findings or patterns
- Suggest when professional medical evaluation is needed
- Do not provide definitive diagnoses
- Keep responses clear and informative"""
        
        return f"{system_prompt}\n\nTask: {text_prompt}\n\nObservation:"
    
    def _extract_multimodal_response(self, result: Dict[str, Any]) -> str:
        """Extract the actual response from multimodal generation result"""
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated_text = result.get("generated_text", "")
        else:
            generated_text = str(result)
        
        # Clean up the response
        response = generated_text.replace("</s>", "").strip()
        
        # Remove any remaining prompt artifacts
        if "Observation:" in response:
            response = response.split("Observation:")[-1].strip()
        
        return response
    
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
            if last_punct > len(response) * 0.7:  # If we found punctuation in the last 30%
                response = response[:last_punct + 1]
        
        return response
    
    async def analyze_symptoms(
        self, 
        symptoms: str, 
        duration: str = "", 
        intensity: str = "", 
        timing: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze symptoms using MedGemma for medical context
        
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
    
    async def analyze_image_with_text(
        self, 
        image_path: str, 
        text_prompt: str = "Describe what you see in this medical image.",
        max_length: int = 512,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze medical images with text prompts using multimodal MedGemma
        
        Args:
            image_path: Path to the image file or image URL
            text_prompt: Text prompt to guide the analysis
            max_length: Maximum response length
            temperature: Sampling temperature
        
        Returns:
            Dict containing the multimodal analysis
        """
        if not self.multimodal or not MULTIMODAL_AVAILABLE:
            error_msg = "Multimodal analysis is not available. " + (
                "Please upgrade transformers to >= 4.42.0 for multimodal support." 
                if not MULTIMODAL_AVAILABLE 
                else "Please initialize with multimodal=True."
            )
            return {
                "success": False,
                "response": error_msg,
                "error": "Multimodal not available"
            }
        
        if not self.pipeline:
            return {
                "success": False,
                "response": "MedGemma model is not available. Please check the model loading.",
                "error": "Model not initialized"
            }
        
        try:
            # Check if required dependencies are available
            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "response": "PIL (Pillow) is required for image processing. Please install with: pip install Pillow",
                    "error": "PIL not available"
                }
            
            # Load image using PIL (following official notebook best practices)
            if image_path.startswith(('http://', 'https://')):
                if not REQUESTS_AVAILABLE:
                    return {
                        "success": False,
                        "response": "requests library is required for URL image loading. Please install with: pip install requests",
                        "error": "requests not available"
                    }
                from io import BytesIO
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)
            
            # Construct multimodal chat messages using new chat template format
            messages = self._construct_multimodal_chat_messages(text_prompt, image)
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_multimodal_with_chat_template,
                messages,
                max_length
            )
            
            # Extract the response from chat template output
            response = result[0]["generated_text"][-1]["content"]
            
            logger.info(" MedGemma multimodal response generated successfully")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "device": self.device,
                "mode": "multimodal",
                "prompt_text": text_prompt
            }
            
        except Exception as e:
            logger.error(f" MedGemma multimodal generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble analyzing the image right now. Please try describing the symptoms in text.",
                "error": str(e)
            }
    
    async def enhance_diagnosis(self, symptoms: str, rag_response: str) -> str:
        """
        Enhance diagnosis by combining symptoms with RAG response using MedGemma
        
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model and its capabilities"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_ready": self.pipeline is not None,
            "task": self.task,
            "multimodal_enabled": self.multimodal,
            "quantization_enabled": self.use_quantization,
            "model_class": self.model_class.__name__ if self.model_class else None,
            "torch_dtype": "bfloat16" if self.device != "cpu" else "float32",
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "capabilities": {
                "text_generation": True,
                "image_analysis": self.multimodal,
                "memory_efficient": self.use_quantization,
                "async_processing": True
            }
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 