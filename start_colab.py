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
    
    # Load secrets from Colab userdata (preferred) or .env file
    try:
        # First try to load from Google Colab secrets
        from google.colab import userdata
        print("✅ Google Colab userdata available")
        
        # Load API keys from Colab secrets
        try:
            openai_key = userdata.get('OPENAI_API_KEY')
            os.environ['OPENAI_API_KEY'] = openai_key
            print("✅ OpenAI API key loaded from Colab secrets")
        except Exception as e:
            print("⚠️  OpenAI API key not found in Colab secrets")
            print("   Add 'OPENAI_API_KEY' to Colab secrets panel")
        
        try:
            hf_token = userdata.get('HF_TOKEN')
            os.environ['HF_TOKEN'] = hf_token
            print("✅ Hugging Face token loaded from Colab secrets")
        except Exception as e:
            print("⚠️  HF_TOKEN not found in Colab secrets (optional)")
        
        try:
            ngrok_token = userdata.get('NGROK_AUTHTOKEN')
            os.environ['NGROK_AUTHTOKEN'] = ngrok_token
            print("✅ Ngrok token loaded from Colab secrets")
        except Exception as e:
            print("⚠️  NGROK_AUTHTOKEN not found in Colab secrets (optional)")
            
    except ImportError:
        print("📝 Not in Google Colab, trying .env file...")
        # Fallback to .env file for local development
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env file loaded")
            
            # Check if OpenAI key is available
            if os.getenv("OPENAI_API_KEY"):
                print("✅ OpenAI API key found in .env")
            else:
                print("⚠️  OpenAI API key not found in .env")
                print("   Please set: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
        except ImportError:
            print("⚠️  python-dotenv not available")
    
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
        "python-dotenv>=1.0.0",
        "nest_asyncio>=1.5.0"  # For Jupyter/Colab compatibility
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
        
        # Check if ngrok token is available
        ngrok_token = os.getenv("NGROK_AUTHTOKEN")
        if not ngrok_token:
            print("⚠️  NGROK_AUTHTOKEN not found")
            print("   Add 'NGROK_AUTHTOKEN' to Colab secrets panel")
            print("   Or set manually: os.environ['NGROK_AUTHTOKEN'] = 'your-token-here'")
            print("   Skipping public tunnel setup...")
            return False
            
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = ngrok_token
        
        # Start tunnel
        public_url = ngrok.connect(8000)
        print(f"✅ Public URL: {public_url}")
        print(f"   🖥️  Web interface: {public_url}")
        print(f"   📱 Mobile: {public_url}/mobile")
        print(f"   📚 API docs: {public_url}/docs")
        print(f"   🔍 Health check: {public_url}/health")
        
        # Store the public URL for easy access
        os.environ['PUBLIC_URL'] = str(public_url)
        
        return True
        
    except Exception as e:
        print(f"❌ Ngrok setup failed: {e}")
        return False

def start_server():
    """Start the FastAPI server - Colab compatible"""
    print("\n🏥 Starting AI Doctor server...")
    
    try:
        # Import the app
        from main import app
        import uvicorn
        import asyncio
        
        # Check if we're in a Jupyter/Colab environment
        try:
            import IPython
            in_jupyter = True
            print("📓 Jupyter/Colab environment detected")
        except ImportError:
            in_jupyter = False
        
        if in_jupyter:
            # Use nest_asyncio for Jupyter compatibility
            try:
                import nest_asyncio
                nest_asyncio.apply()
                print("✅ nest_asyncio applied for Jupyter compatibility")
            except ImportError:
                print("⚠️  Installing nest_asyncio...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
                import nest_asyncio
                nest_asyncio.apply()
        
        # Create server config
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
        # Start server
        server = uvicorn.Server(config)
        
        if in_jupyter:
            # Run in background for Jupyter
            print("🚀 Starting server in background...")
            print("   Server will be available at: http://localhost:8000")
            print("   Use Ctrl+C to stop")
            
            # Create a task that runs the server
            import threading
            server_thread = threading.Thread(target=lambda: asyncio.run(server.serve()))
            server_thread.daemon = True
            server_thread.start()
            
            print("✅ Server started successfully!")
            print("   Access your AI Doctor at: http://localhost:8000")
            print("   API documentation: http://localhost:8000/docs")
            
            return server_thread
        else:
            # Regular startup for non-Jupyter environments
            uvicorn.run(
                app,
                host="0.0.0.0", 
                port=8000,
                log_level="info",
                reload=False
            )
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

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
