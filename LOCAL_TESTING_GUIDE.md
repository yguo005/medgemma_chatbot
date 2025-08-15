# 🚀 Local Testing Guide for AI Health Consultant

## **Prerequisites**

### 1. Install Dependencies
```bash
# Install pipenv if you haven't already
pip install pipenv

# Install dependencies from Pipfile
pipenv install

# Activate the virtual environment
pipenv shell
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory:

```bash
# Required for basic functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For MedGemma integration
USE_MEDGEMMA=false
GCP_PROJECT_ID=your_gcp_project_id

# Optional: For Google Cloud authentication
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

## **🖥️ Running Locally**

### Method 1: FastAPI Server (Full Features)
```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or with pipenv
pipenv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access the chatbot:**
- **Web Interface**: http://localhost:8000/
- **Mobile Interface**: http://localhost:8000/mobile.html
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Method 2: Direct Python Testing
```python
import asyncio
from app.chatbot import Chatbot

async def test_chatbot():
    chatbot = Chatbot(openai_api_key="your_key_here")
    
    response = await chatbot.get_response("What are the symptoms of fever?")
    print(response)

# Run the test
asyncio.run(test_chatbot())
```

## **🧪 Testing Different Components**

### 1. Test Basic RAG Chatbot
```bash
python3 -c "
import asyncio
from app.chatbot import Chatbot
import os

async def test():
    chatbot = Chatbot(openai_api_key=os.getenv('OPENAI_API_KEY'))
    response = await chatbot.get_response('What is pneumonia?')
    print('Response:', response)

asyncio.run(test())
"
```

### 2. Test AI Services (Image + Voice)
```bash
python3 test_ai_services.py
```

### 3. Test Conversation Flow
```bash
# Start the server and visit http://localhost:8000/mobile.html
uvicorn main:app --reload
```

## **📱 Mobile Interface Testing**

1. Start the server: `uvicorn main:app --reload`
2. Open browser: http://localhost:8000/mobile.html
3. Test features:
   - Text input
   - Photo upload
   - Voice recording (requires HTTPS for production)
   - Conversation flow

## **🔧 Troubleshooting**

### Common Issues:

1. **Port already in use:**
   ```bash
   uvicorn main:app --reload --port 8001
   ```

2. **Missing dependencies:**
   ```bash
   pipenv install fastapi uvicorn python-multipart
   ```

3. **OpenAI API key not set:**
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

4. **CORS issues:**
   - The app is configured to allow all origins in development
   - For production, update CORS settings in main.py

### Performance Tips:
- Use `--workers 1` for development
- Set `--reload` for auto-restart on file changes
- Use `--log-level debug` for detailed logging

## **📊 Monitoring**

- Check logs in the terminal where you ran uvicorn
- Visit `/health` endpoint for service status
- Use `/docs` for interactive API testing

## **🎯 Quick Test Commands**

```bash
# Test if server is running
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, what are the symptoms of flu?"}'

# Test conversation endpoint
curl -X POST "http://localhost:8000/conversation" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test123", "message": "I have a headache", "is_choice": false}'
```