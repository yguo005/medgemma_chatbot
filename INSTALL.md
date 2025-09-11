# AI Doctor Installation Guide

## Quick Start

### 1. Create Virtual Environment
```bash
python3 -m venv ai_doctor_env
source ai_doctor_env/bin/activate  # On macOS/Linux
# OR
ai_doctor_env\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
# Full installation (recommended)
pip install -r requirements.txt

# OR minimal installation
pip install -r requirements-minimal.txt
```

### 3. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 4. Start the Server
```bash
# Using the startup script (recommended)
./start_mac_simple.sh

# OR manually
python main.py

# OR with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application
- Web Interface: http://localhost:8000/
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Required API Keys

### OpenAI (Required)
1. Get your API key from: https://platform.openai.com/api-keys
2. Add to `.env`: `OPENAI_API_KEY=your_key_here`

### HuggingFace (Optional - for local models)
1. Get your token from: https://huggingface.co/settings/tokens
2. Add to `.env`: `HF_TOKEN=your_token_here`

### Ngrok (Optional - for public access)
1. Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken
2. Add to `.env`: `NGROK_AUTHTOKEN=your_token_here`

## Troubleshooting

### Memory Issues (Mac)
If you see MPS memory errors, add to `.env`:
```
MEDGEMMA_ENABLED=false
AI_SERVICE_MODE=openai
```

### Port Already in Use
```bash
# Kill existing uvicorn processes
pkill -f uvicorn

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install specific missing packages
pip install fastapi uvicorn openai transformers
```
