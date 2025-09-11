#!/usr/bin/env python3
"""
Comprehensive Colab Chat Debugging Script
Debug the "sorry something went wrong" error in Colab
"""

import os
import sys
import requests
import json
import traceback
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def test_server_health():
    """Test if the server is running and healthy"""
    print_section("Server Health Check")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running and healthy")
            health_data = response.json()
            print(f"   AI Service Mode: {health_data.get('ai_service_mode', 'Unknown')}")
            print(f"   Services Initialized: {health_data.get('conversation_manager_initialized', False)}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint with detailed debugging"""
    print_section("Chat Endpoint Test")
    
    test_message = "I have headache"
    test_session_id = "debug_test_" + str(int(datetime.now().timestamp()))
    
    payload = {
        "query": test_message,
        "session_id": test_session_id
    }
    
    print(f"Testing with message: '{test_message}'")
    print(f"Session ID: {test_session_id}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print("\n🔄 Sending request...")
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout for model loading
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("✅ Chat endpoint working!")
                print(f"Response keys: {list(response_data.keys())}")
                
                # Check response format
                if 'response_text' in response_data:
                    print(f"✅ Response text: {response_data['response_text'][:100]}...")
                elif 'response' in response_data:
                    print(f"✅ Response: {response_data['response'][:100]}...")
                else:
                    print("⚠️  No response_text or response field found")
                    print(f"Full response: {json.dumps(response_data, indent=2)}")
                
                return True
            except json.JSONDecodeError:
                print("❌ Response is not valid JSON")
                print(f"Raw response: {response.text[:500]}...")
                return False
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (120s)")
        print("   This might indicate model loading issues")
        return False
    except Exception as e:
        print(f"❌ Chat request failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test the Python environment and dependencies"""
    print_section("Environment Check")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check key environment variables
    env_vars = ['OPENAI_API_KEY', 'AI_SERVICE_MODE', 'HF_TOKEN', 'NGROK_AUTHTOKEN']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var or 'TOKEN' in var:
                print(f"✅ {var}: Set (length: {len(value)})")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        
        # Check Colab secrets
        try:
            from google.colab import userdata
            print("✅ Colab userdata available")
            
            for var in ['OPENAI_API_KEY', 'HF_TOKEN', 'NGROK_AUTHTOKEN']:
                try:
                    secret_value = userdata.get(var)
                    if secret_value:
                        print(f"✅ Colab secret {var}: Available (length: {len(secret_value)})")
                    else:
                        print(f"⚠️  Colab secret {var}: Empty or not set")
                except Exception as e:
                    print(f"❌ Colab secret {var}: Error - {e}")
                    
        except ImportError:
            print("❌ Colab userdata not available")
    except ImportError:
        print("⚠️  Not running in Google Colab")
    
    # Check critical imports
    critical_imports = [
        'fastapi', 'uvicorn', 'transformers', 'torch', 
        'openai', 'langchain', 'faiss'
    ]
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}: Available")
        except ImportError as e:
            print(f"❌ {module}: Missing - {e}")

def check_server_logs():
    """Check server logs for errors"""
    print_section("Server Log Analysis")
    
    # In Colab, we can't easily access server logs, but we can check for common issues
    print("💡 Common issues in Colab:")
    print("   1. Model loading failures (CUDA/memory issues)")
    print("   2. Missing dependencies (pillow, bitsandbytes)")
    print("   3. API key not properly loaded")
    print("   4. Network/firewall issues")
    print("   5. Timeout during model initialization")

def test_openai_connection():
    """Test OpenAI API connection"""
    print_section("OpenAI API Test")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10
        )
        print("✅ OpenAI API connection working")
        return True
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def run_comprehensive_debug():
    """Run all debugging tests"""
    print("🏥 AI Doctor - Colab Chat Debugging")
    print(f"Debug started at: {datetime.now()}")
    
    results = {
        'environment': test_environment(),
        'openai': test_openai_connection(),
        'server_health': test_server_health(),
        'chat_endpoint': test_chat_endpoint()
    }
    
    print_section("Debug Summary")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The chat should work.")
    else:
        print("\n🔧 Issues found. Check the details above.")
        print("\n💡 Quick fixes to try:")
        print("   1. Restart the Colab runtime")
        print("   2. Re-run the start_colab.py script")
        print("   3. Check that all API keys are in Colab secrets")
        print("   4. Wait longer for model loading (first request can take 2-3 minutes)")
    
    return all_passed

if __name__ == "__main__":
    run_comprehensive_debug()
