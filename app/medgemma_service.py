import os
import logging
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
    
    def __init__(self, model_name: str = "google/medgemma-7b", device: str = "auto"):
        """
        Initialize MedGemma service
        
        Args:
            model_name: HuggingFace model identifier for MedGemma
            device: Device to run the model on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # For async operations
        
        # Initialize the model
        self._initialize_model()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self):
        """Initialize the MedGemma model and tokenizer"""
        try:
            logger.info(f"🔄 Loading MedGemma model: {self.model_name}")
            logger.info(f"🖥️ Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            
            logger.info("✅ MedGemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MedGemma model: {e}")
            self.model = None
            self.tokenizer = None
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
            # Construct the medical prompt
            prompt = self._construct_medical_prompt(query, context)
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._generate_text,
                prompt,
                max_length,
                temperature
            )
            
            # Extract and clean the response
            generated_text = result[0]['generated_text']
            response = self._extract_response(generated_text, prompt)
            
            logger.info("✅ MedGemma response generated successfully")
            
            return {
                "success": True,
                "response": response,
                "model_used": self.model_name,
                "device": self.device,
                "prompt_length": len(prompt),
                "response_length": len(response)
            }
            
        except Exception as e:
            logger.error(f"❌ MedGemma generation failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your medical query right now. Please consult with a healthcare professional.",
                "error": str(e)
            }
    
    def _construct_medical_prompt(self, query: str, context: str = "") -> str:
        """Construct a proper medical prompt for MedGemma"""
        
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
        """Generate text using the pipeline (runs in thread pool)"""
        return self.pipeline(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            return_full_text=True
        )
    
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
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_ready": self.pipeline is not None,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 