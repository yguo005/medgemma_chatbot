import os
import traceback
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import modules from new structure
from src.services.ai.rag.chatbot import Chatbot
from src.services.conversation.manager import ConversationManager
from services.ai.openai_services import AIServices
from src.services.safety.safety_guardrails import MedicalSafetyGuardrails

# Import optimized AI service manager
from src.services.ai.ai_service_manager import create_ai_service_manager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MEDGEMMA_GARDEN = os.getenv("USE_MEDGEMMA_GARDEN", "false").lower() == "true"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
MEDGEMMA_ENDPOINT_ID = os.getenv("MEDGEMMA_ENDPOINT_ID")

# New: AI Service Manager configuration
AI_SERVICE_MODE = os.getenv("AI_SERVICE_MODE", "hybrid")  # hybrid, local_demo, cloud_first

# Initialize safety guardrails first
safety_guardrails = MedicalSafetyGuardrails()

# Initialize all services
try:
    # Initialize chatbot with proper MedGemma integration
    chatbot = Chatbot(
        openai_api_key=OPENAI_API_KEY,
        use_medgemma_garden=USE_MEDGEMMA_GARDEN,
        gcp_project_id=GCP_PROJECT_ID,
        endpoint_id=MEDGEMMA_ENDPOINT_ID
    )
    
    ai_services = AIServices(
        api_key=OPENAI_API_KEY, 
        use_medgemma=USE_MEDGEMMA_GARDEN,
        gcp_project_id=GCP_PROJECT_ID
    )
    
    # Initialize optimized AI service manager (new approach)
    ai_service_manager = create_ai_service_manager(AI_SERVICE_MODE)
    
    # Initialize conversation manager with AI service for Phase 1: Symptom Understanding
    conversation_manager = ConversationManager(ai_service=ai_service_manager)
    
    if USE_MEDGEMMA_GARDEN:
        logger.info(f" MedGemma 4B Model Garden integration enabled (Project: {GCP_PROJECT_ID})")
    else:
        logger.info(" Using MedGemma 4B local + RAG architecture")
    
    logger.info(" All services initialized successfully with safety guardrails.")
    
except Exception as e:
    logger.error(f" Failed to initialize services: {e}")
    chatbot = None
    conversation_manager = None
    ai_services = None
    ai_service_manager = None

app = FastAPI(
    title="AI Health Consultant API", 
    description="A secure medical AI assistant with MedGemma + RAG architecture", 
    version="3.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Request models
class QueryRequest(BaseModel):
    query: str
    session_id: str
    is_choice: bool = False

class ImageAnalysisRequest(BaseModel):
    session_id: str
    image_data: str
    filename: str

@app.post("/chat")
async def chat(query_request: QueryRequest):
    """Enhanced chat endpoint with safety guardrails and MedGemma-RAG integration."""
    if not conversation_manager or not safety_guardrails:
        raise HTTPException(status_code=500, detail="Required services not initialized.")

    try:
        query_text = query_request.query.strip()
        session_id = query_request.session_id
        is_choice = query_request.is_choice

        logger.info(f" Session {session_id}: {query_text}")

        # Step 1: Safety check on user input
        safety_result = safety_guardrails.process_user_input(query_text)
        
        # Handle emergency situations
        if safety_result["should_block"]:
            logger.warning(f" Session {session_id}: Emergency situation detected")
            return {
                "response_type": "emergency",
                "response": safety_result["emergency_response"],
                "session_id": session_id
            }

        # Step 2: Process with conversation manager
        response = conversation_manager.process_message(session_id, query_text, is_choice)
        
        # Step 3: Enhanced diagnosis with MedGemma + RAG + Safety
        if response.get('response_type') == 'diagnostic' and chatbot:
            try:
                session = conversation_manager.get_session(session_id)
                collected_data = session['collected_data']
                
                # Construct comprehensive medical query
                medical_query = f"""
                Primary symptoms: {collected_data.get('symptoms', '')}
                Duration: {collected_data.get('duration', '')}
                Intensity: {collected_data.get('intensity', '')}
                Timing: {collected_data.get('timing', '')}
                Image analysis: {collected_data.get('image_analysis', '')}
                
                Please provide medical information and recommendations for these symptoms.
                """
                
                # Get enhanced response using MedGemma + RAG
                enhanced_response = await chatbot.get_response(medical_query)
                
                # Apply safety validation to the AI response
                safety_validation = safety_guardrails.validate_response(enhanced_response)
                
                if safety_validation["is_safe"]:
                    response['diagnosis_description'] = safety_validation["filtered_response"]
                    response['enhanced_with'] = 'MedGemma + RAG + Safety'
                    response['safety_validated'] = True
                else:
                    # Use fallback safe response
                    response['diagnosis_description'] = """
                    Based on your symptoms, I recommend consulting with a healthcare professional for proper evaluation and diagnosis. 
                    They will be able to assess your condition thoroughly and provide appropriate medical advice and treatment options.
                    
                    *This information is for educational purposes only and is not a substitute for professional medical advice.*
                    """
                    response['enhanced_with'] = 'Safety Fallback'
                    response['safety_validated'] = False
                    logger.warning(f" Session {session_id}: AI response failed safety validation")
                
                logger.info(f" Session {session_id}: Enhanced with {response['enhanced_with']}")
                
            except Exception as enhancement_error:
                logger.error(f" Enhancement failed: {enhancement_error}")
                # Fallback to safe default response
                response['diagnosis_description'] = """
                I recommend consulting with a healthcare professional about your symptoms. 
                They can provide proper evaluation, diagnosis, and treatment recommendations based on a thorough assessment.
                
                *Always seek professional medical advice for health concerns.*
                """
                response['enhanced_with'] = 'Default Safe Response'
        
        # Step 4: Final safety check on complete response
        if 'diagnosis_description' in response:
            final_safety_check = safety_guardrails.validate_response(response['diagnosis_description'])
            if not final_safety_check["is_safe"]:
                response['diagnosis_description'] = final_safety_check["filtered_response"]
                logger.warning(f" Session {session_id}: Final response required safety filtering")

        logger.info(f" Session {session_id}: Response sent successfully")
        return response

    except Exception as e:
        error_message = f" Internal Server Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/analyze_image")
async def analyze_image(request: ImageAnalysisRequest):
    """Enhanced image analysis with safety guardrails using optimized service manager."""
    if not conversation_manager or not safety_guardrails:
        raise HTTPException(status_code=500, detail="Required services not initialized.")

    try:
        session_id = request.session_id
        image_data = request.image_data
        filename = request.filename

        logger.info(f" Session {session_id}: Analyzing image {filename}")

        # Use optimized service manager for image analysis (tries MedGemma first, then OpenAI)
        if ai_service_manager:
            analysis_result = await ai_service_manager.analyze_image(image_data, context="medical")
            service_used = analysis_result.get("service_used", "unknown")
            logger.info(f" Image analysis service used: {service_used}")
        else:
            # Fallback to old openai_services if ai_service_manager failed to initialize
            logger.warning(" Using fallback openai_services for image analysis")
            analysis_result = await ai_services.analyze_image(image_data, context="medical")
        
        if analysis_result["success"]:
            analysis_text = analysis_result["analysis"]
            
            # Safety check on image analysis result
            safety_result = safety_guardrails.process_user_input(analysis_text)
            
            if safety_result["should_block"]:
                # Handle emergency situations detected in image
                return {
                    "response_type": "emergency",
                    "response": safety_result["emergency_response"],
                    "session_id": session_id
                }
            
            # Apply response safety validation
            safety_validation = safety_guardrails.validate_response(analysis_text)
            final_analysis = safety_validation["filtered_response"]
            
            logger.info(f" Session {session_id}: Image analysis successful and validated")
        else:
            final_analysis = analysis_result["analysis"]  # Fallback message
            logger.warning(f" Session {session_id}: Image analysis failed - {analysis_result.get('error', 'Unknown error')}")
        
        response = conversation_manager.process_image_analysis(session_id, final_analysis)
        
        logger.info(f" Session {session_id}: Image analysis response sent")
        return response

    except Exception as e:
        error_message = f" Image Analysis Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        try:
            fallback_response = conversation_manager.process_image_analysis(
                request.session_id, 
                "I'm sorry, I couldn't analyze the image at the moment. Please try describing your symptoms in text or consult with a healthcare professional."
            )
            return fallback_response
        except:
            raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/transcribe")
async def transcribe_audio(session_id: str = Form(...), audio: UploadFile = File(...)):
    """Enhanced audio transcription with safety guardrails using optimized service manager."""
    if not conversation_manager or not safety_guardrails:
        raise HTTPException(status_code=500, detail="Required services not initialized.")

    try:
        logger.info(f" Session {session_id}: Transcribing audio file: {audio.filename}")

        audio_content = await audio.read()
        
        # Use optimized service manager for audio transcription (OpenAI Whisper only)
        if ai_service_manager:
            transcription_result = await ai_service_manager.transcribe_audio(audio_content, audio.filename or "audio.wav")
        else:
            # Fallback to old ai_services if ai_service_manager failed to initialize
            logger.warning(" Using fallback openai_services for audio transcription")
            transcription_result = await ai_services.transcribe_audio(audio_content, audio.filename or "audio.wav")
        
        if transcription_result["success"]:
            transcription_text = transcription_result["transcription"]
            
            # Safety check on transcription
            safety_result = safety_guardrails.process_user_input(transcription_text)
            
            if safety_result["should_block"]:
                return {
                    "response_type": "emergency", 
                    "response": safety_result["emergency_response"],
                    "session_id": session_id
                }
            
            logger.info(f" Session {session_id}: Audio transcription successful and validated")
        else:
            transcription_text = transcription_result["transcription"]
            logger.warning(f" Session {session_id}: Audio transcription failed")
        
        response = conversation_manager.process_voice_transcription(session_id, transcription_text)
        
        logger.info(f"📤 Session {session_id}: Audio transcription response sent")
        return response

    except Exception as e:
        error_message = f" Transcription Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        try:
            fallback_response = conversation_manager.process_voice_transcription(
                session_id, 
                "I'm sorry, I couldn't transcribe the audio. Please try typing your message instead or speak with a healthcare professional directly."
            )
            return fallback_response
        except:
            raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health")
async def health_check():
    """Enhanced health check including safety systems and optimized service manager."""
    status = {
        "api_status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "conversation_manager_initialized": conversation_manager is not None,
        "ai_services_initialized": ai_services is not None,
        "ai_service_manager_initialized": ai_service_manager is not None,
        "safety_guardrails_initialized": safety_guardrails is not None,
        "medgemma_garden_enabled": USE_MEDGEMMA_GARDEN,
        "ai_service_mode": AI_SERVICE_MODE,
        "architecture": "MedGemma + RAG + Safety Guardrails + Optimized Service Manager"
    }
    
    # Add detailed service manager status if available
    if ai_service_manager:
        service_status = ai_service_manager.get_service_status()
        status.update({
            "service_manager_details": service_status,
            "image_analysis_available": service_status["image_analysis_available"],
            "audio_transcription_available": service_status["audio_transcription_available"],
            "text_generation_available": service_status["text_generation_available"]
        })
    
    if ai_services:
        status.update(ai_services.get_service_status())
    
    if chatbot:
        status.update(chatbot.get_service_info())
    
    return status

@app.get("/safety-info")
async def get_safety_info():
    """Get information about safety guardrails."""
    return {
        "emergency_keywords_count": len(safety_guardrails.emergency_keywords),
        "forbidden_phrases_count": len(safety_guardrails.forbidden_phrases),
        "safety_features": [
            "Emergency keyword detection",
            "Diagnostic language filtering", 
            "Automatic disclaimer injection",
            "Response severity scoring",
            "Medical system prompt enforcement"
        ],
        "disclaimer": "This system includes comprehensive safety measures to ensure responsible AI behavior in medical contexts."
    }

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/mobile")
async def read_mobile():
    return FileResponse('mobile.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)