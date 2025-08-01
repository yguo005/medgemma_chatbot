#!/usr/bin/env python3
"""
Test script for AI Services - GPT-4 Vision and Whisper API integration
This script demonstrates how to use the real API calls without hardcoding.
"""

import os
import asyncio
import base64
from app.ai_services import AIServices

async def test_ai_services():
    """Test the AI services with real API calls"""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize AI services
    try:
        ai_services = AIServices(api_key=api_key)
        print("✅ AI Services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI services: {e}")
        return
    
    # Test service status
    print("\n📊 Service Status:")
    status = ai_services.get_service_status()
    for service, available in status.items():
        print(f"  - {service}: {'✅' if available else '❌'}")
    
    # Test 1: Image Analysis (with a sample base64 image)
    print("\n🔍 Testing Image Analysis...")
    try:
        # This is a small test image (1x1 pixel red dot in base64)
        # In real usage, this would come from the frontend
        sample_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        result = await ai_services.analyze_image(sample_image_b64, context="medical")
        
        if result["success"]:
            print(f"✅ Image analysis successful!")
            print(f"📝 Analysis: {result['analysis']}")
            print(f"🤖 Model: {result['model_used']}")
            if result.get('tokens_used'):
                print(f"🔢 Tokens used: {result['tokens_used']}")
        else:
            print(f"⚠️ Image analysis failed: {result['error']}")
            print(f"📝 Fallback message: {result['analysis']}")
            
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
    
    # Test 2: Audio Transcription (simulated with empty bytes)
    print("\n🎤 Testing Audio Transcription...")
    try:
        # In real usage, this would be actual audio file bytes from the frontend
        # For testing, we'll use empty bytes which will trigger the error handling
        sample_audio_bytes = b""  # Empty bytes to test error handling
        
        result = await ai_services.transcribe_audio(sample_audio_bytes, "test_audio.wav")
        
        if result["success"]:
            print(f"✅ Audio transcription successful!")
            print(f"📝 Transcription: {result['transcription']}")
            print(f"🤖 Model: {result['model_used']}")
            if result.get('language'):
                print(f"🌐 Language: {result['language']}")
        else:
            print(f"⚠️ Audio transcription failed: {result['error']}")
            print(f"📝 Fallback message: {result['transcription']}")
            
    except Exception as e:
        print(f"❌ Audio transcription test failed: {e}")
    
    # Test 3: RAG Enhancement
    print("\n🧠 Testing RAG Enhancement...")
    try:
        sample_symptoms = "I have knee pain that started a week ago"
        sample_rag_response = "Knee pain can be caused by various factors including injury, arthritis, or overuse. Common treatments include rest, ice, compression, and elevation (RICE protocol)."
        
        enhanced_diagnosis = await ai_services.enhance_diagnosis_with_rag(sample_symptoms, sample_rag_response)
        
        print(f"✅ RAG enhancement successful!")
        print(f"📝 Enhanced diagnosis: {enhanced_diagnosis}")
        
    except Exception as e:
        print(f"❌ RAG enhancement test failed: {e}")
    
    print("\n🎉 AI Services testing completed!")

def main():
    """Main function to run the tests"""
    print("🚀 Starting AI Services Tests...")
    print("=" * 50)
    
    # Run the async tests
    asyncio.run(test_ai_services())
    
    print("\n💡 Usage Notes:")
    print("1. For real image analysis, pass actual base64 image data from frontend")
    print("2. For real audio transcription, pass actual audio file bytes")
    print("3. The services include proper error handling and fallback messages")
    print("4. All API calls are non-hardcoded and use your OpenAI API key")

if __name__ == "__main__":
    main() 