#!/usr/bin/env python3
"""
Google Colab Startup Script for AI Doctor with MedGemma
Optimized for T4 GPU with proper environment setup
"""

import os
import sys
import subprocess
import platform

# --- FIX: Add Colab-specific authentication ---
try:
    import google.colab
    from google.colab import auth
    print("Authenticating Google Colab user...")
    auth.authenticate_user()
    print("✅ Colab user authenticated.")
    
    # Set the project ID for gcloud commands
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    if gcp_project_id:
        subprocess.run(['gcloud', 'config', 'set', 'project', gcp_project_id], capture_output=True)
        print(f"✅ Google Cloud project set to: {gcp_project_id}")
    else:
        print("⚠️ GCP_PROJECT_ID not found in environment for gcloud config.")

except ImportError:
    print("Not in a Colab environment, skipping Colab-specific authentication.")
except Exception as e:
    print(f"⚠️ An error occurred during Colab authentication: {e}")
# ----------------------------------------------

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
            if openai_key and openai_key.strip():
                os.environ['OPENAI_API_KEY'] = openai_key.strip()
                print("✅ OpenAI API key loaded from Colab secrets")
            else:
                print("⚠️  OpenAI API key is empty in Colab secrets")
                print("   Add 'OPENAI_API_KEY' to Colab secrets panel")
        except Exception as e:
            print(f"⚠️  OpenAI API key not found in Colab secrets: {e}")
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
            
        # --- FIX: Load Google Cloud secrets for Model Garden ---
        try:
            gcp_project_id = userdata.get('GCP_PROJECT_ID')
            if gcp_project_id and gcp_project_id.strip():
                os.environ['GCP_PROJECT_ID'] = gcp_project_id.strip()
                print("✅ GCP_PROJECT_ID loaded from Colab secrets")
            else:
                print("⚠️  GCP_PROJECT_ID is empty in Colab secrets")
        except Exception as e:
            print("⚠️  GCP_PROJECT_ID not found in Colab secrets. Model Garden will not work.")

        try:
            medgemma_endpoint_id = userdata.get('MEDGEMMA_ENDPOINT_ID')
            if medgemma_endpoint_id and medgemma_endpoint_id.strip():
                os.environ['MEDGEMMA_ENDPOINT_ID'] = medgemma_endpoint_id.strip()
                print("✅ MEDGEMMA_ENDPOINT_ID loaded from Colab secrets")
            else:
                print("⚠️  MEDGEMMA_ENDPOINT_ID is empty in Colab secrets")
        except Exception as e:
            print("⚠️  MEDGEMMA_ENDPOINT_ID not found in Colab secrets. Will use default if available.")
        # ---------------------------------------------------------
            
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
    os.environ["AI_SERVICE_MODE"] = "cloud_first"  # Prioritize Model Garden
    os.environ["USE_MEDGEMMA_GARDEN"] = "true"  # Explicitly enable Model Garden
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
    
    # Check GPU availability and set quantization (following official notebook logic)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU memory: {gpu_memory_gb:.1f} GB")
            
            # Following official notebook logic: T4 can handle 4B with quantization
            if "T4" in gpu_name or gpu_memory_gb >= 15:
                os.environ["MEDGEMMA_USE_QUANTIZATION"] = "true"
                print("   🚀 Quantization enabled (T4 GPU or sufficient memory)")
            elif "A100" in gpu_name:
                os.environ["MEDGEMMA_USE_QUANTIZATION"] = "true"
                print("   🚀 Quantization enabled (A100 GPU)")
            else:
                # For other GPUs, be conservative
                if gpu_memory_gb >= 8:
                    os.environ["MEDGEMMA_USE_QUANTIZATION"] = "true"
                    print("   🚀 Quantization enabled (sufficient GPU memory)")
                else:
                    os.environ["MEDGEMMA_USE_QUANTIZATION"] = "false"
                    print("   ⚠️  Quantization disabled (limited GPU memory)")
        else:
            print("⚠️  No GPU detected - using CPU mode")
            os.environ["MEDGEMMA_USE_QUANTIZATION"] = "false"
    except ImportError:
        print("⚠️  PyTorch not available")
    
    
    
    return IN_COLAB

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies - matching official notebook exactly
    deps = [
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "accelerate>=0.24.0",
        "bitsandbytes>=0.43.1",  # Updated version for better compatibility
        "pillow>=9.0.0",  # Required for image processing
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "nest_asyncio>=1.5.0",  # For Jupyter/Colab compatibility
        "huggingface_hub>=0.16.0"  # For proper HF authentication
    ]
    
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {dep}")

def setup_ngrok_tunnel():
    """Setup ngrok tunnel for public access with Colab integration"""
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
        
        # Setup Colab keep-alive for ngrok tunnel
        setup_colab_keepalive()
        
        return True
        
    except Exception as e:
        print(f"❌ Ngrok setup failed: {e}")
        return False

def setup_colab_keepalive():
    """Setup Colab built-in keep-alive to maintain ngrok tunnel"""
    try:
        # Check if we're in Colab
        import google.colab
        from google.colab import output
        import threading
        import time
        import requests
        from datetime import datetime
        
        print("🔄 Setting up Colab keep-alive for ngrok tunnel...")
        
        def keepalive_worker():
            """Advanced background worker to keep Colab session active"""
            last_activity = datetime.now()
            ping_count = 0
            
            while True:
                try:
                    current_time = datetime.now()
                    
                    # Ping every 5 minutes to prevent timeout
                    time.sleep(300)
                    ping_count += 1
                    
                    # Health check ping to keep server active
                    try:
                        response = requests.get("http://localhost:8000/health", timeout=5)
                        if response.status_code == 200:
                            last_activity = current_time
                            
                            # Periodic status update (every hour)
                            if ping_count % 12 == 0:  # Every 12 pings = 1 hour
                                print(f"🔄 Keep-alive active - Runtime uptime: {(current_time - last_activity).total_seconds() / 3600:.1f}h")
                                
                    except requests.exceptions.RequestException:
                        # Server might be down, but keep session alive anyway
                        pass
                    
                    # Colab-specific keep-alive mechanisms
                    try:
                        # Simulate minimal activity to prevent runtime timeout
                        # This is more effective than just pinging
                        import IPython
                        if hasattr(IPython, 'get_ipython'):
                            kernel = IPython.get_ipython()
                            if kernel:
                                # Execute a minimal command to show activity
                                kernel.run_cell_magic('javascript', '', 'console.log("keep-alive");')
                    except:
                        pass
                    
                    # Memory management - prevent accumulation of logs
                    if ping_count % 24 == 0:  # Every 2 hours
                        try:
                            if hasattr(output, 'clear'):
                                # Clear old output but preserve important messages
                                pass  # Don't clear to avoid disrupting user experience
                        except:
                            pass
                        
                except Exception as e:
                    print(f"⚠️  Keep-alive error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        def setup_colab_widgets():
            """Setup Colab widgets for interactive keep-alive"""
            try:
                from IPython.display import display, HTML, Javascript
                
                # Display a small status widget
                status_html = """
                <div id="ai-doctor-status" style="
                    position: fixed; 
                    top: 10px; 
                    right: 10px; 
                    background: #4CAF50; 
                    color: white; 
                    padding: 5px 10px; 
                    border-radius: 5px; 
                    font-size: 12px; 
                    z-index: 1000;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                ">
                    🏥 AI Doctor Active
                </div>
                <script>
                    // Keep-alive JavaScript that runs in browser
                    setInterval(function() {
                        // Prevent browser from going idle
                        console.log('AI Doctor keep-alive ping');
                        
                        // Update status indicator
                        var status = document.getElementById('ai-doctor-status');
                        if (status) {
                            var now = new Date();
                            status.innerHTML = '🏥 AI Doctor Active - ' + now.toLocaleTimeString();
                        }
                    }, 60000); // Every minute
                </script>
                """
                
                display(HTML(status_html))
                print("✅ Colab status widget activated")
                
            except Exception as e:
                print(f"⚠️  Widget setup failed: {e}")
        
        # Start keep-alive in daemon thread
        keepalive_thread = threading.Thread(target=keepalive_worker, daemon=True)
        keepalive_thread.start()
        
        # Setup interactive widgets
        setup_colab_widgets()
        
        print("✅ Advanced Colab keep-alive activated")
        print("   • Server health monitoring every 5 minutes")
        print("   • Runtime timeout prevention active")
        print("   • Status widget displayed in top-right corner")
        print("   • Ngrok tunnel will remain available")
        
        return True
        
    except ImportError:
        # Not in Colab, use regular keep-alive
        print("📝 Not in Colab - using standard keep-alive")
        setup_standard_keepalive()
        return True
    except Exception as e:
        print(f"⚠️  Keep-alive setup failed: {e}")
        setup_standard_keepalive()  # Fallback to standard
        return False

def setup_standard_keepalive():
    """Standard keep-alive for non-Colab environments"""
    import threading
    import time
    
    def standard_keepalive():
        """Standard background keep-alive"""
        while True:
            time.sleep(600)  # 10 minutes
            try:
                import requests
                requests.get("http://localhost:8000/health", timeout=5)
                print("🔄 Keep-alive ping sent")
            except:
                pass
    
    threading.Thread(target=standard_keepalive, daemon=True).start()

def kill_existing_server():
    """Kill any existing server on port 8000"""
    try:
        import subprocess
        # Find and kill process on port 8000
        result = subprocess.run(['lsof', '-ti:8000'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"🔄 Killing existing server process {pid}")
                    subprocess.run(['kill', '-9', pid], capture_output=True)
            import time
            time.sleep(2)  # Wait for cleanup
    except Exception as e:
        print(f"⚠️  Could not check/kill existing server: {e}")

def start_server():
    """Start the FastAPI server - Colab compatible with error handling"""
    print("\n🏥 Starting AI Doctor server...")
    
    # Kill any existing server first
    kill_existing_server()
    
    try:
        # Import required modules first
        import sys
        import os
        
        # Set environment variables for Colab compatibility
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # Add current directory to Python path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"✅ Added {current_dir} to Python path")
        
        # Import the app
        try:
            from main import app
            print("✅ Successfully imported main app")
        except ImportError as e:
            print(f"❌ Failed to import main app: {e}")
            print("   Make sure you're in the correct directory with main.py")
            print("   Try: %cd /path/to/your/AI_Doctor-main")
            return None
            
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
        
        # Test app import and configuration
        print("🔧 Testing application configuration...")
        try:
            from config.settings import validate_configuration
            config_result = validate_configuration()
            if config_result.get('valid'):
                print("✅ Configuration is valid")
            else:
                issues = config_result.get('issues', [])
                print(f"⚠️  Configuration issues: {issues}")
                
                # Check if OpenAI key is the issue
                if any('OpenAI' in issue for issue in issues):
                    print("🔑 Attempting to reload OpenAI API key...")
                    
                    # Try to get it from Colab secrets again
                    try:
                        from google.colab import userdata
                        openai_key = userdata.get('OPENAI_API_KEY')
                        if openai_key and openai_key.strip():
                            os.environ['OPENAI_API_KEY'] = openai_key.strip()
                            print("✅ OpenAI API key reloaded from Colab secrets")
                            
                            # Re-validate
                            config_result = validate_configuration()
                            if config_result.get('valid'):
                                print("✅ Configuration now valid after key reload")
                            else:
                                print(f"⚠️  Still have issues: {config_result.get('issues', [])}")
                        else:
                            print("❌ OpenAI API key still empty or missing")
                            print("   Please check that 'OPENAI_API_KEY' is properly set in Colab secrets")
                            print("   Go to 🔑 (secrets) in left sidebar and add your OpenAI API key")
                    except Exception as e:
                        print(f"❌ Failed to reload OpenAI key: {e}")
                        
        except Exception as e:
            print(f"⚠️  Configuration validation failed: {e}")
        
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
            print("   Use Ctrl+C to stop")
            
            # Create a task that runs the server
            import threading
            
            def run_server():
                try:
                    asyncio.run(server.serve())
                except Exception as e:
                    print(f"❌ Server thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait a moment and test the server
            import time
            print("⏳ Waiting for server to fully initialize...")
            time.sleep(5)  # Give server more time to start
            
            # Test server health
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    
                    # Test chat endpoint with longer timeout for model loading
                    try:
                        print("🔄 Testing chat endpoint (may take 2-3 minutes for first model loading)...")
                        chat_response = requests.post(
                            "http://localhost:8000/chat",
                            json={"query": "test", "session_id": "colab_test"},
                            timeout=180  # 3 minute timeout for MedGemma loading (first time)
                        )
                        if chat_response.status_code == 200:
                            print("✅ Chat endpoint is working!")
                            try:
                                response_data = chat_response.json()
                                if 'response' in response_data or 'response_text' in response_data:
                                    print("✅ Chat response format is correct")
                                else:
                                    print("⚠️  Unexpected response format")
                            except:
                                print("⚠️  Response is not valid JSON")
                        else:
                            print(f"⚠️  Chat endpoint returned status {chat_response.status_code}")
                            print(f"   Response: {chat_response.text[:200]}...")
                    except requests.exceptions.Timeout:
                        print("⚠️  Chat endpoint test timed out (3 minutes)")
                        print("   This may be normal for first request (MedGemma downloading/loading)")
                        print("   The server should still work - try the web interface manually")
                        print("   First chat in browser may take 2-3 minutes, then it'll be fast")
                    except Exception as e:
                        print(f"⚠️  Chat endpoint test failed: {e}")
                        print("   Server may still be functional - try accessing manually")
                        print("   Run this to debug further:")
                        print("   exec(open('debug_colab_chat.py').read())")
                        
                else:
                    print(f"❌ Server health check failed: {response.status_code}")
                    print(f"   Response: {response.text}")
            except Exception as e:
                print(f"❌ Server health check failed: {e}")
                print("   The server may still be starting up...")
            
            # Provide final instructions
            print("\n" + "="*60)
            print("🎉 AI Doctor Setup Complete!")
            print("="*60)
            
            # Show ngrok URL prominently if available
            ngrok_url = os.getenv('PUBLIC_URL')
            if ngrok_url:
                print("🌐 ** USE THESE PUBLIC URLs (NOT localhost) **")
                print("="*60)
                print(f"   🖥️  Desktop: {ngrok_url}")
                print(f"   📱 Mobile: {ngrok_url}/mobile.html")
                print(f"   📚 API Docs: {ngrok_url}/docs")
                print(f"   🔍 Health: {ngrok_url}/health")
                print("="*60)
                print("⚠️  IMPORTANT: Don't use localhost:8000 in Colab!")
                print("   Use the ngrok URLs above instead.")
            else:
                print("⚠️  Ngrok tunnel not available")
                print("📱 Local Access (won't work in Colab):")
                print("   🖥️  Desktop: http://localhost:8000")
                print("   📱 Mobile: http://localhost:8000/mobile.html") 
                print("   📚 API Docs: http://localhost:8000/docs")
            
            print("\n💡 Usage Tips:")
            print("   • First chat may take 2-3 minutes (MedGemma downloading/loading)")
            print("   • Be patient! The model is 4GB+ and needs to load into memory")
            print("   • Subsequent chats will be much faster (seconds)")
            print("   • Server runs in background - keep this cell running")
            print("   • Keep-alive is active - session won't timeout automatically")
            print("   • If chat times out, just wait and try again")
            print("   • Use Ctrl+C to stop server if needed")
            print("="*60)
            
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
        
        # Provide debugging information
        print("\n🔍 Debugging Information:")
        print("1. Check if all dependencies are installed")
        print("2. Verify API keys are set in Colab secrets")
        print("3. Check if GPU is available and properly configured")
        print("4. Try restarting the runtime if issues persist")
        
        return None

def check_ngrok_status():
    """Check and display current ngrok tunnel status"""
    try:
        from pyngrok import ngrok
        tunnels = ngrok.get_tunnels()
        
        if tunnels:
            print("\n🌐 Active Ngrok Tunnels:")
            for tunnel in tunnels:
                print(f"   ✅ {tunnel.public_url} -> {tunnel.config['addr']}")
            return True
        else:
            print("\n⚠️  No active ngrok tunnels found")
            return False
    except Exception as e:
        print(f"\n❌ Could not check ngrok status: {e}")
        return False

def manual_api_key_setup():
    """Allow manual API key setup if secrets don't work"""
    print("\n🔑 Manual API Key Setup (if secrets don't work)")
    print("   You can manually set your API keys by running:")
    print("   import os")
    print("   os.environ['OPENAI_API_KEY'] = 'sk-proj-your-key-here'")
    print("   os.environ['HF_TOKEN'] = 'hf_your-token-here'")
    print("   os.environ['NGROK_AUTHTOKEN'] = 'your-ngrok-token-here'")
    print("   Then re-run: exec(open('start_colab.py').read())")

def get_colab_keep_alive_status():
    """Get a simple function to check keep-alive status in Colab"""
    try:
        import google.colab
        
        def status():
            """Check current AI Doctor status in Colab"""
            print("🏥 AI Doctor Status Check")
            print("=" * 40)
            
            # Check server health
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Server: Running")
                else:
                    print("⚠️  Server: Issues detected")
            except:
                print("❌ Server: Not responding")
            
            # Check ngrok tunnel
            ngrok_active = check_ngrok_status()
            
            # Show public URL
            public_url = os.getenv('PUBLIC_URL')
            if public_url:
                print(f"🌐 Public URL: {public_url}")
                print("   Use this URL to access your AI Doctor!")
            else:
                print("⚠️  No public URL available")
            
            print("=" * 40)
            
        return status
        
    except ImportError:
        return None

def check_and_setup_repository():
    """Check if we're in the right directory and setup if needed"""
    print("\n📁 Checking repository setup...")
    
    # Check if main.py exists
    if os.path.exists('main.py'):
        print("✅ Found main.py - you're in the right directory")
        return True
    
    print("⚠️  main.py not found in current directory")
    print(f"   Current directory: {os.getcwd()}")
    
    # Check if we need to clone the repository
    print("\n🔄 Setting up AI Doctor repository...")
    
    try:
        import subprocess
        
        # Clone the repository
        print("📥 Cloning AI Doctor repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/yguo005/medgemma_chatbot.git',
            'AI_Doctor'
        ], check=True)
        
        # Change to the directory
        os.chdir('AI_Doctor')
        print(f"✅ Changed to directory: {os.getcwd()}")
        
        # Verify main.py exists now
        if os.path.exists('main.py'):
            print("✅ Repository setup complete!")
            return True
        else:
            print("❌ main.py still not found after cloning")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        print("\n📝 Manual setup instructions:")
        print("1. Run: !git clone https://github.com/yguo005/medgemma_chatbot.git")
        print("2. Run: %cd medgemma_chatbot")
        print("3. Re-run this script")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 AI Doctor - Google Colab Setup")
    print("   MedGemma 4B + RAG + Safety Guardrails")
    print("=" * 60)
    
    # Check repository setup first
    if not check_and_setup_repository():
        print("\n❌ Repository setup failed. Please set up manually.")
        return
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Check if OpenAI key was loaded
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ OpenAI API Key is still missing!")
        manual_api_key_setup()
        return
    
    # Install dependencies
    install_dependencies()
    
    # Setup ngrok if in Colab (mandatory for access)
    if in_colab:
        print("\n🌐 Setting up public access via ngrok...")
        print("   ⚠️  NOTE: localhost:8000 will NOT work in Colab!")
        print("   You MUST use the ngrok public URL to access the interface.")
        ngrok_success = setup_ngrok_tunnel()
        if not ngrok_success:
            print("\n❌ CRITICAL: Ngrok setup failed!")
            print("   Without ngrok, you won't be able to access the web interface in Colab.")
            print("   Please add 'NGROK_AUTHTOKEN' to Colab secrets and try again.")
            return
    
    # Start server
    server_result = start_server()
    
    # Provide status checker function for Colab users
    if in_colab and server_result:
        print("\n🔧 Utility Functions:")
        print("   To check status anytime, run: status()")
        print("   To check ngrok tunnels, run: check_ngrok_status()")
        
        # Make status function available globally
        status_func = get_colab_keep_alive_status()
        if status_func:
            globals()['status'] = status_func
            globals()['check_ngrok_status'] = check_ngrok_status

if __name__ == "__main__":
    main()
