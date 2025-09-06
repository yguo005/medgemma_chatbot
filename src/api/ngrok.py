# Run the FastAPI Server using ngrok
import os
import sys
import asyncio
from pyngrok import ngrok, conf

sys.path.insert(0, os.path.abspath('src'))

# Validate environment variables
NGROK_TOKEN = os.environ.get("NGROK_AUTHTOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not NGROK_TOKEN:
    print(" ERROR: NGROK_AUTHTOKEN not set!")
    print("Set it with: os.environ['NGROK_AUTHTOKEN'] = 'your-token-here'")
    sys.exit(1)

if not OPENAI_KEY:
    print(" ERROR: OPENAI_API_KEY not set!")
    print("Set it with: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
    sys.exit(1)



# Set the ngrok auth token
conf.get_default().auth_token = NGROK_TOKEN

async def run_fastapi():
    try:
        # Use nest_asyncio to allow uvicorn to run in a notebook
        import nest_asyncio
        nest_asyncio.apply()
        
        # Import uvicorn
        import uvicorn
        
        print(" Starting FastAPI server...")
        print(f" Working directory: {os.getcwd()}")
        
        # Check if main.py exists
        if not os.path.exists("main.py"):
            print(" ERROR: main.py not found in current directory!")
            print("Make sure you're in the correct directory with your FastAPI app.")
            return
        
        # Configure uvicorn server
        config = uvicorn.Config(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            reload=False  # Disable reload in Colab
        )
        server = uvicorn.Server(config)
        
        # Open a tunnel to the uvicorn server
        print(" Opening ngrok tunnel...")
        public_url = ngrok.connect(8000)
        print(f" FastAPI server is live at: {public_url}")
        print(f" Mobile interface: {public_url}/mobile.html")
        print(f" Desktop interface: {public_url}/")
        print(f" API docs: {public_url}/docs")
        print("\n To stop the server, interrupt this cell (Runtime > Interrupt execution)")
        
        # Run the server
        await server.serve()
        
    except Exception as e:
        print(f" Error starting server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up ngrok tunnels
        try:
            ngrok.disconnect(8000)
            print(" Cleaned up ngrok tunnel")
        except:
            pass

# Run the server asynchronously
await run_fastapi()