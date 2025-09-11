#!/usr/bin/env python3
"""
Mac Startup Script for AI Doctor with MedGemma
Optimized for Mac Silicon (M1/M2/M3) without CUDA
"""

import os
import sys
import subprocess
import platform

def setup_mac_environment():
    """Setup environment for Mac"""
    
    print("🍎 Setting up AI Doctor for Mac...")
    
    # Set environment variables for Mac
    os.environ["AI_SERVICE_MODE"] = "hybrid"  # Use both MedGemma and OpenAI
    os.environ["DEPLOYMENT_MODE"] = "development"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP issues
    
    # Check Mac architecture
    arch = platform.machine()
    print(f"✅ Mac detected: {platform.system()} {platform.release()}")
    print(f"   Architecture: {arch}")
    
    if arch == "arm64":
        print("   🔥 Apple Silicon detected - optimized for M1/M2/M3")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
    else:
        print("   💻 Intel Mac detected")
    
    # Check PyTorch MPS availability
    try:
        import torch
        if torch.backends.mps.is_available():
            print("   ✅ MPS (Metal Performance Shaders) available")
            print("   🚀 Using Apple Silicon GPU acceleration")
        else:
            print("   ⚠️  MPS not available - using CPU")
    except ImportError:
        print("   ⚠️  PyTorch not available")

def check_dependencies():
    """Check if dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required = [
        "torch", "transformers", "accelerate", "pillow", "fastapi", 
        "uvicorn", "langchain", "langchain-openai", "faiss-cpu", "openai"
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (missing)")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_openai_key():
    """Check OpenAI API key"""
    print("\n🔑 Checking OpenAI API key...")
    
    # Check .env file
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content and "sk-" in content:
                print("  ✅ OpenAI API key found in .env")
                return True
    
    # Check environment variable
    if os.getenv("OPENAI_API_KEY"):
        print("  ✅ OpenAI API key found in environment")
        return True
    
    print("  ❌ OpenAI API key not found")
    print("     Add to .env file: OPENAI_API_KEY=your-key-here")
    return False

def check_vector_store():
    """Check if FAISS vector store exists"""
    print("\n🗃️  Checking vector store...")
    
    faiss_path = "data/vectorstore/db_faiss"
    if os.path.exists(faiss_path):
        files = os.listdir(faiss_path)
        if "index.faiss" in files and "index.pkl" in files:
            print("  ✅ FAISS vector store ready")
            return True
    
    print("  ❌ FAISS vector store not found")
    print("     Run: python src/services/ai/rag/create_memory_for_llm.py")
    return False

def start_server():
    """Start the FastAPI server with Mac optimizations"""
    print("\n🏥 Starting AI Doctor server...")
    
    try:
        # Set Mac-specific optimizations
        os.environ["OMP_NUM_THREADS"] = "1"  # Prevent OpenMP issues
        os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL optimization
        
        # Start server using uvicorn CLI for better Mac compatibility
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        print(f"   URL: http://127.0.0.1:8000")
        print(f"   Mobile: http://127.0.0.1:8000/mobile")
        print(f"   API docs: http://127.0.0.1:8000/docs")
        print("\n   Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 AI Doctor - Mac Setup")
    print("   MedGemma 4B + RAG + Safety Guardrails")
    print("   Optimized for Mac Silicon (No CUDA required)")
    print("=" * 60)
    
    # Setup environment
    setup_mac_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check OpenAI key
    if not check_openai_key():
        print("\n⚠️  OpenAI API key recommended for full functionality")
    
    # Check vector store
    if not check_vector_store():
        print("\n⚠️  Vector store missing - some features may be limited")
    
    print("\n✅ All checks complete - starting server...")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
