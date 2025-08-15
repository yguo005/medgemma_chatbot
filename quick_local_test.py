#!/usr/bin/env python3
"""
Quick local test for the AI Health Consultant
Run this to test basic functionality without starting the full server
"""

import asyncio
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_basic_functionality():
    """Test basic chatbot functionality"""
    print("🧪 Quick Local Test - AI Health Consultant")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print(f"✅ OpenAI API key found (ending with: ...{api_key[-4:]})")
    
    try:
        # Test the chatbot
        from chatbot import Chatbot
        
        print("\n🤖 Testing Basic RAG Chatbot...")
        chatbot = Chatbot(openai_api_key=api_key)
        
        response = await chatbot.get_response("What are the common symptoms of pneumonia?")
        print("✅ Chatbot Response:")
        print(response[:300] + "..." if len(response) > 300 else response)
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
    
    try:
        # Test AI Services
        from ai_services import AIServices
        
        print("\n🔬 Testing AI Services...")
        ai_services = AIServices(api_key=api_key, use_medgemma=False)
        
        # Test with a simple text (simulating image analysis)
        result = await ai_services.analyze_image(
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
            context="medical"
        )
        
        print("✅ AI Services working:")
        print(f"Success: {result['success']}")
        print(f"Analysis: {result['analysis'][:100]}...")
        
    except Exception as e:
        print(f"❌ AI Services test failed: {e}")
    
    try:
        # Test Conversation Manager
        from conversation_manager import ConversationManager
        
        print("\n💬 Testing Conversation Manager...")
        conv_manager = ConversationManager()
        
        response = conv_manager.process_message("test123", "I have a headache", False)
        print("✅ Conversation Manager working:")
        print(f"Response Type: {response['response_type']}")
        print(f"Message: {response['response_text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Conversation Manager test failed: {e}")
    
    print("\n🎉 Local test completed!")
    print("\n📝 Next Steps:")
    print("1. Run full server: uvicorn main:app --reload")
    print("2. Open browser: http://localhost:8000")
    print("3. Try mobile interface: http://localhost:8000/mobile.html")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())