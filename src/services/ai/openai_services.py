'''
 1. primary and exclusive role of openai_services.py is to handle audio-to-text transcription (def transcribe_audio), because MedGemma does not have that capability
 2. acts as the crucial "emergency fallback" if gedgemma fails for text and image analysis in ai_service_manager.py
'''



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
    def __init__(self, api_key: str):
        """Initialize AI services with OpenAI API key"""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(" OpenAI Services client initialized successfully")
    
    async def analyze_image(self, image_data: str, context: str = "medical") -> Dict[str, Any]:
        """
        Analyze an image using GPT-4o API
        
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
            
            # Make API call to GPT-4o (using the current model)
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Updated to current GPT-4 with vision model
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
                "model_used": "gpt-4o",  # Updated model name
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
    
    async def enhance_diagnosis_with_rag(self, query: str, context: str) -> str:
        """
        Enhances a response by combining a query with context using the GPT-4o model.
        This method acts as the text-generation engine for the OpenAI service,
        used for conversational responses, initial symptom extraction, and as a
        diagnostic fallback.

        Args:
            query: The primary question or text to be processed.
            context: Supporting information from the RAG system or conversation history.

        Returns:
            An enhanced text response from the GPT-4o model.
        """
        try:
            # The AIServiceManager has already determined this is the correct service to use.
            # This method's single responsibility is to execute the OpenAI call.
            response = self.client.chat.completions.create(
                model="gpt-4o", # Use the latest powerful and cost-effective model
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical AI assistant. Your role is to synthesize user queries with medical knowledge to provide helpful information. Always remind users to consult healthcare professionals for proper diagnosis and treatment. Structure your response clearly and use supportive language."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Based on the following information:
                        ---CONTEXT---
                        {context}
                        ---END CONTEXT---

                        Please address this query: "{query}"

                        Provide a clear, helpful summary that:
                        1. Directly answers the query.
                        2. Incorporates relevant information from the context.
                        3. Suggests appropriate next steps if applicable.
                        4. Includes a reminder to consult a healthcare professional for medical advice.
                        """
                    }
                ],
                max_tokens=400, # Increased token limit for more comprehensive summaries
                temperature=0.3
            )
            
            logger.info(" Response generated with OpenAI GPT-4o.")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f" OpenAI RAG enhancement failed: {str(e)}")
            # Fall back to the original context or a safe message if context is empty
            return context if context else "I am sorry, but I am unable to process your request at the moment."
    
    async def generate_simple_question(self, query: str) -> str:
        """
        Generate a simple, direct question without any additional context or explanation.
        This is called by ai_service_manager.py's generate_conversational_response method, to generate dynamic questions in the conversation flow.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical question generator. Your ONLY job is to return a single question - nothing else.

CRITICAL RULES:
- Return ONLY the question text
- NO "Please select:" text
- NO numbered lists (1. 2. 3. 4.)
- NO multiple choice options
- NO explanations or summaries
- NO additional text

Example input: "Generate a question about headache duration"
Example output: "How long have you been experiencing this headache?"

That's it. Just the question."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=30,  # Even shorter to prevent extra text
                temperature=0  # Most consistent responses
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Additional cleanup to remove any formatting that might slip through
            response_text = response_text.replace("Please select:", "").strip()
            response_text = response_text.split("\n")[0]  # Take only the first line
            
            return response_text
            
        except Exception as e:
            logger.error(f"Simple question generation failed: {str(e)}")
            return "Could you please provide more details?"
    
    async def extract_structured_data(self, extraction_prompt: str) -> str:
        """
        Extract structured data from text using a focused prompt.
        This is called by the ConversationManager in _extract_symptoms_with_ai as the primary method for the initial symptom extraction.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant. Follow the user's instructions exactly and return only the requested structured data format."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                max_tokens=200,  # Enough for JSON but not for explanations
                temperature=0  # Consistent, predictable responses
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {str(e)}")
            raise e
    
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
            "vision_available": True,
            "transcription_available": True,
            "text_generation_available": True,
            "client_initialized": self.client is not None
        }
        return status 