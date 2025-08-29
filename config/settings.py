import os
from typing import Optional

# Path settings
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ==========================================
# PHASE 1: RAG Configuration (OpenAI Embeddings)
# ==========================================
# Using OpenAI for embeddings (retrieval step)
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 dimension

# Text splitter settings for RAG
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==========================================
# PHASE 1: MedGemma Configuration
# ==========================================

# Environment-based configuration
USE_MEDGEMMA_GARDEN = os.getenv("USE_MEDGEMMA_GARDEN", "false").lower() == "true"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
MEDGEMMA_ENDPOINT_ID = os.getenv("MEDGEMMA_ENDPOINT_ID")

# Model Garden Configuration (Production)
MEDGEMMA_MODEL_GARDEN_CONFIG = {
    "project_id": GCP_PROJECT_ID,
    "location": "us-central1",
    "endpoint_id": MEDGEMMA_ENDPOINT_ID,
    "model_name": "google/medgemma-4b",  
    "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
}

# Local MedGemma Configuration (Development)
MEDGEMMA_LOCAL_CONFIG = {
    "model_name": "google/medgemma-4b",  
    "device": "auto",
    "use_quantization": True,  # Memory efficiency
    "multimodal": False,  # Enable if needed
    "max_length": 512,
    "temperature": 0.3  # Conservative for medical responses
}

# ==========================================
# Safety and Compliance Configuration
# ==========================================

# Safety guardrails settings
SAFETY_CONFIG = {
    "enable_emergency_detection": True,
    "enable_response_filtering": True,
    "enable_severity_scoring": True,
    "mandatory_disclaimer": True,
    "max_severity_threshold": 0.7,
    "enable_logging": True
}

# HIPAA Compliance settings (for future implementation)
HIPAA_CONFIG = {
    "enable_audit_logging": True,
    "encrypt_session_data": True,
    "data_retention_days": 30,
    "anonymize_logs": True
}

# ==========================================
# API Configuration
# ==========================================

# OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "vision_model": "gpt-4-vision-preview",
    "transcription_model": "whisper-1", 
    "fallback_model": "gpt-4",  # For enhancement fallback
    "max_tokens": 512,
    "temperature": 0.3
}

# ==========================================
# Deployment Configuration
# ==========================================

# Development vs Production settings
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "development").lower()

if DEPLOYMENT_MODE == "production":
    # Production settings
    LOG_LEVEL = "INFO"
    DEBUG_MODE = False
    FORCE_HTTPS = True
    ENABLE_RATE_LIMITING = True
    
    # Prefer Model Garden in production
    if not USE_MEDGEMMA_GARDEN and GCP_PROJECT_ID:
        print(" WARNING: Production mode detected but Model Garden not enabled. Consider using Model Garden for production.")
        
elif DEPLOYMENT_MODE == "development":
    # Development settings
    LOG_LEVEL = "DEBUG"
    DEBUG_MODE = True
    FORCE_HTTPS = False
    ENABLE_RATE_LIMITING = False
    
    # Local MedGemma is fine for development
    if USE_MEDGEMMA_GARDEN:
        print(" INFO: Using Model Garden in development mode.")

# ==========================================
# Feature Flags
# ==========================================

FEATURE_FLAGS = {
    "enable_image_analysis": True,
    "enable_voice_transcription": True,
    "enable_multimodal_medgemma": False,  # Enable when multimodal is needed
    "enable_conversation_memory": True,
    "enable_appointment_booking": True,  # Future feature
    "enable_medication_suggestions": False,  # Requires additional safety measures
}

# ==========================================
# Validation Functions
# ==========================================

def validate_configuration() -> dict:
    """Validate the current configuration and return status."""
    issues = []
    warnings = []
    
    # Check OpenAI API key
    if not OPENAI_CONFIG["api_key"]:
        issues.append("Missing OpenAI API key")
    
    # Check MedGemma configuration
    if USE_MEDGEMMA_GARDEN:
        if not GCP_PROJECT_ID:
            issues.append("Model Garden enabled but GCP_PROJECT_ID not set")
        if not MEDGEMMA_ENDPOINT_ID:
            warnings.append("MEDGEMMA_ENDPOINT_ID not set, will use default")
    
    # Check data paths
    if not os.path.exists(DATA_PATH):
        warnings.append(f"Data directory not found: {DATA_PATH}")
    
    if not os.path.exists(DB_FAISS_PATH):
        warnings.append(f"FAISS vector store not found: {DB_FAISS_PATH} - Run create_memory_for_llm.py first")
    
    # Production readiness checks
    if DEPLOYMENT_MODE == "production":
        if not FORCE_HTTPS:
            warnings.append("HTTPS not enforced in production mode")
        if not SAFETY_CONFIG["enable_emergency_detection"]:
            issues.append("Emergency detection disabled in production - this is unsafe")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "deployment_mode": DEPLOYMENT_MODE,
        "using_model_garden": USE_MEDGEMMA_GARDEN,
        "architecture": "MedGemma + RAG + Safety"
    }

def get_active_config() -> dict:
    """Get the current active configuration."""
    return {
        "deployment_mode": DEPLOYMENT_MODE,
        "medgemma_mode": "Model Garden" if USE_MEDGEMMA_GARDEN else "Local Hugging Face",
        "embedding_model": EMBEDDING_MODEL,
        "safety_enabled": SAFETY_CONFIG["enable_emergency_detection"],
        "feature_flags": FEATURE_FLAGS,
        "data_paths": {
            "data": DATA_PATH,
            "vectorstore": DB_FAISS_PATH
        },
        "models": {
            "embedding": EMBEDDING_MODEL,
            "medgemma_4b": "google/medgemma-4b-it",
            "vision": OPENAI_CONFIG["vision_model"],
            "transcription": OPENAI_CONFIG["transcription_model"]
        }
    }

# Print configuration status on import (only in development)
if DEPLOYMENT_MODE == "development":
    validation = validate_configuration()
    print(f"\n Configuration Status:")
    print(f"   Mode: {DEPLOYMENT_MODE}")
    print(f"   MedGemma: {'Model Garden' if USE_MEDGEMMA_GARDEN else 'Local HF'}")
    print(f"   Valid: {validation['valid']}")
    if validation['issues']:
        print(f"   Issues: {', '.join(validation['issues'])}")
    if validation['warnings']:
        print(f"   Warnings: {', '.join(validation['warnings'])}")
    print()