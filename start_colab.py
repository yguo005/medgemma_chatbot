#!/usr/bin/env python3
"""
Google Colab Startup Script for AI Doctor with MedGemma
Optimized for T4 GPU with proper environment setup
"""

import os
import sys
import subprocess
import platform

def setup_colab_environment():
    """Setup environment for Google Colab with T4 GPU"""
    
    print("🚀 Setting up AI Doctor for Google Colab with T4 GPU...")
    
    # Set environment variables for Colab
    os.environ["AI_SERVICE_MODE"] = "hybrid"  # Use both MedGemma and OpenAI
    os.environ["DEPLOYMENT_MODE"] = "development"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP issues
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        IN_COLAB = True
    except ImportError:
        print("❌ Not in Google Colab - using local settings")
        IN_COLAB = False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Enable quantization for GPU
            os.environ["MEDGEMMA_USE_QUANTIZATION"] = "true"
        else:
            print("⚠️  No GPU detected - using CPU mode")
            os.environ["MEDGEMMA_USE_QUANTIZATION"] = "false"
    except ImportError:
        print("⚠️  PyTorch not available")
    
    return IN_COLAB

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    deps = [
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "accelerate>=0.24.0",
        "bitsandbytes>=0.43.1",  # Updated version for CPU support
        "pillow>=9.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0"
    ]
    
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {dep}")

def setup_ngrok_tunnel():
    """Setup ngrok tunnel for public access"""
    print("\n🌐 Setting up ngrok tunnel...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "--quiet"])
        
        # You'll need to set your ngrok token
        ngrok_token = os.getenv("NGROK_AUTHTOKEN")
        if not ngrok_token:
            print("⚠️  NGROK_AUTHTOKEN not set. Set it with:")
            print("     os.environ['NGROK_AUTHTOKEN'] = 'your-token-here'")
            return False
            
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        
        # Start tunnel
        public_url = ngrok.connect(8000)
        print(f"✅ Public URL: {public_url}")
        print(f"   Web interface: {public_url}")
        print(f"   Mobile: {public_url}/mobile")
        print(f"   API docs: {public_url}/docs")
        
        return True
        
    except Exception as e:
        print(f"❌ Ngrok setup failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("\n🏥 Starting AI Doctor server...")
    
    try:
        # Import and start the app
        from main import app
        import uvicorn
        
        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # Disable reload in Colab
        )
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 AI Doctor - Google Colab Setup")
    print("   MedGemma 4B + RAG + Safety Guardrails")
    print("=" * 60)
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Setup ngrok if in Colab
    if in_colab:
        setup_ngrok_tunnel()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
