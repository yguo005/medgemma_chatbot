import os
import traceback
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot import Chatbot
from app.conversation_manager import ConversationManager
from app.ai_services import AIServices
from fastapi.staticfiles import StaticFiles

# Load configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MEDGEMMA = os.getenv("USE_MEDGEMMA", "false").lower() == "true"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# Initialize all services
try:
    chatbot = Chatbot(openai_api_key=OPENAI_API_KEY)
    conversation_manager = ConversationManager()
    ai_services = AIServices(
        api_key=OPENAI_API_KEY, 
        use_medgemma=USE_MEDGEMMA,
        gcp_project_id=GCP_PROJECT_ID
    )
    
    if USE_MEDGEMMA:
        if GCP_PROJECT_ID:
            print(f"🧠 MedGemma Model Garden integration enabled (Project: {GCP_PROJECT_ID})")
        else:
            print("⚠️ MedGemma enabled but GCP_PROJECT_ID not set")
    else:
        print("🤖 Using OpenAI models only")
    
    print("✅ All services initialized successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize services: {e}")
    chatbot = None
    conversation_manager = None
    ai_services = None

app = FastAPI(
    title="AI Health Consultant API", 
    description="A mobile health consultant API with optional MedGemma integration", 
    version="2.1.0"
)

# Enable CORS for frontend access
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
    """Handles user queries with session-based conversation management."""
    if not conversation_manager:
        raise HTTPException(status_code=500, detail="Conversation manager not initialized.")

    try:
        query_text = query_request.query.strip()
        session_id = query_request.session_id
        is_choice = query_request.is_choice

        print(f"📩 Session {session_id}: {query_text}")

        # Use conversation manager for guided flow
        response = conversation_manager.process_message(session_id, query_text, is_choice)
        
        # Enhanced diagnosis with MedGemma or GPT-4
        if response.get('response_type') == 'diagnostic' and chatbot and ai_services:
            try:
                # Get session data for MedGemma analysis
                session = conversation_manager.get_session(session_id)
                collected_data = session['collected_data']
                
                # Try MedGemma analysis first if available
                if USE_MEDGEMMA:
                    medgemma_result = await ai_services.analyze_symptoms_with_medgemma(
                        symptoms=collected_data.get('symptoms', ''),
                        duration=collected_data.get('duration', ''),
                        intensity=collected_data.get('intensity', ''),
                        timing=collected_data.get('timing', '')
                    )
                    
                    if medgemma_result["success"]:
                        response['diagnosis_description'] = medgemma_result["response"]
                        response['enhanced_with'] = 'MedGemma'
                        print(f"✅ Session {session_id}: Enhanced with MedGemma")
                    else:
                        print(f"⚠️ Session {session_id}: MedGemma failed, using standard diagnosis")
                else:
                    # Use RAG + GPT-4 enhancement
                    rag_response = chatbot.get_response(f"Based on these symptoms: {query_text}, provide medical information.")
                    enhanced_diagnosis = await ai_services.enhance_diagnosis_with_rag(query_text, rag_response)
                    response['diagnosis_description'] = enhanced_diagnosis
                    response['enhanced_with'] = 'RAG + GPT-4'
                    print(f"✅ Session {session_id}: Enhanced with RAG + GPT-4")
                    
            except Exception as rag_error:
                print(f"⚠️ Enhancement failed: {rag_error}")
                # Continue with original response if enhancement fails
        
        print(f"📤 Session {session_id}: {response}")
        return response

    except Exception as e:
        error_message = f"❌ Internal Server Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/analyze_image")
async def analyze_image(request: ImageAnalysisRequest):
    """Analyzes uploaded images using GPT-4 Vision."""
    if not conversation_manager or not ai_services:
        raise HTTPException(status_code=500, detail="Required services not initialized.")

    try:
        session_id = request.session_id
        image_data = request.image_data
        filename = request.filename

        print(f"📷 Session {session_id}: Analyzing image {filename}")

        # Use real GPT-4 Vision API
        analysis_result = await ai_services.analyze_image(image_data, context="medical")
        
        if analysis_result["success"]:
            analysis_text = analysis_result["analysis"]
            print(f"✅ Session {session_id}: Image analysis successful")
        else:
            analysis_text = analysis_result["analysis"]  # This contains the fallback message
            print(f"⚠️ Session {session_id}: Image analysis failed - {analysis_result.get('error', 'Unknown error')}")
        
        response = conversation_manager.process_image_analysis(session_id, analysis_text)
        
        print(f"📤 Session {session_id}: Image analysis response sent")
        return response

    except Exception as e:
        error_message = f"❌ Image Analysis Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        # Fallback to conversation manager with error message
        try:
            fallback_response = conversation_manager.process_image_analysis(
                request.session_id, 
                "I'm sorry, I couldn't analyze the image at the moment. Please try describing your symptoms in text."
            )
            return fallback_response
        except:
            raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/transcribe")
async def transcribe_audio(session_id: str = Form(...), audio: UploadFile = File(...)):
    """Transcribes audio using OpenAI Whisper."""
    if not conversation_manager or not ai_services:
        raise HTTPException(status_code=500, detail="Required services not initialized.")

    try:
        print(f"🎤 Session {session_id}: Transcribing audio file: {audio.filename}")

        # Read audio file
        audio_content = await audio.read()
        
        # Use real Whisper API
        transcription_result = await ai_services.transcribe_audio(audio_content, audio.filename or "audio.wav")
        
        if transcription_result["success"]:
            transcription_text = transcription_result["transcription"]
            print(f"✅ Session {session_id}: Audio transcription successful: '{transcription_text}'")
        else:
            transcription_text = transcription_result["transcription"]  # This contains the fallback message
            print(f"⚠️ Session {session_id}: Audio transcription failed - {transcription_result.get('error', 'Unknown error')}")
        
        response = conversation_manager.process_voice_transcription(session_id, transcription_text)
        
        print(f"📤 Session {session_id}: Audio transcription response sent")
        return response

    except Exception as e:
        error_message = f"❌ Transcription Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        # Fallback to conversation manager with error message
        try:
            fallback_response = conversation_manager.process_voice_transcription(
                session_id, 
                "I'm sorry, I couldn't transcribe the audio. Please try typing your message instead."
            )
            return fallback_response
        except:
            raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health")
async def health_check():
    """Check the health status of all services."""
    status = {
        "api_status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "conversation_manager_initialized": conversation_manager is not None,
        "ai_services_initialized": ai_services is not None,
        "medgemma_enabled": USE_MEDGEMMA,
    }
    
    if ai_services:
        status.update(ai_services.get_service_status())
    
    return status

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/mobile")
async def read_mobile():
    return FileResponse('mobile.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


