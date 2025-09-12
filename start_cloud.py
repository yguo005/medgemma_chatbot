#!/usr/bin/env python3
"""
Mac Startup Script for AI Doctor (Cloud Mode)
Connects to Google Cloud's Vertex AI Model Garden for MedGemma
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

def setup_cloud_environment():
    """Setup environment for Model Garden"""
    
    print("🚀 Setting up AI Doctor for Mac (Cloud Mode)...")

    # Load .env file
    load_dotenv()
    print("✅ .env file loaded")
    
    # Set environment variables for cloud mode
    os.environ["AI_SERVICE_MODE"] = "cloud_first"
    os.environ["USE_MEDGEMMA_GARDEN"] = "true"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    print("✅ Environment set to 'cloud_first' mode")
    print("   - Using MedGemma via Google Cloud Model Garden")

def check_gcloud_setup():
    """Check for Google Cloud SDK and authentication"""
    print("\n🔍 Checking Google Cloud setup...")
    
    # 1. Check if gcloud CLI is installed
    try:
        subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
        print("  ✅ Google Cloud SDK (gcloud) is installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ Google Cloud SDK (gcloud) not found.")
        print("     Please install it to use Model Garden.")
        print("     Installation command for Mac:")
        print("     curl -sSL https://sdk.cloud.google.com | bash")
        print("     Then, restart your terminal and run:")
        print("     gcloud init")
        return False

    # 2. Check for Application Default Credentials (ADC)
    try:
        # This command checks if ADC are available
        subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            check=True,
            capture_output=True
        )
        print("  ✅ Google Cloud authentication found.")
    except subprocess.CalledProcessError:
        print("  ❌ Google Cloud authentication not found.")
        print("     Please run this command to log in:")
        print("     gcloud auth application-default login")
        return False
        
    # 3. Check for GCP_PROJECT_ID
    if not os.getenv("GCP_PROJECT_ID"):
        print("  ❌ GCP_PROJECT_ID environment variable not set.")
        print("     Please add it to your .env file:")
        print("     GCP_PROJECT_ID=your-gcp-project-id-here")
        return False
    else:
        print(f"  ✅ GCP Project ID: {os.getenv('GCP_PROJECT_ID')}")

    return True

def start_server():
    """Start the FastAPI server"""
    print("\n🏥 Starting AI Doctor server...")
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        print(f"   URL: http://127.0.0.1:8000")
        print("\n   Press Ctrl+C to stop")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 AI Doctor - Mac Cloud Demo Setup")
    print("=" * 60)
    
    setup_cloud_environment()
    
    if not check_gcloud_setup():
        print("\n❌ Google Cloud setup is incomplete. Please follow the instructions above.")
        return
        
    print("\n✅ All checks complete - starting server...")
    start_server()

if __name__ == "__main__":
    main()
