# AI Doctor: MedGemma & RAG Medical Chatbot

This project is a sophisticated medical chatbot that leverages Google's MedGemma model and a Retrieval-Augmented Generation (RAG) architecture to provide a safe and informative conversational AI experience. It is built with FastAPI and designed for both local development and cloud deployment.

## Features

*   **Advanced Medical AI**: Utilizes the specialized `google/medgemma-4b-it` model for detailed diagnostic analysis.
*   **Retrieval-Augmented Generation (RAG)**: Enhances AI responses with factual context from a medical encyclopedia, improving the quality of conversational questions.
*   **Intelligent Conversation Flow**: A robust state machine guides the user through a series of clinically relevant questions to gather symptoms effectively.
*   **Multi-Modal Input**: Supports text, image analysis, and audio transcription (via OpenAI Whisper).
*   **Comprehensive Safety Guardrails**: Includes a multi-layered safety system to detect emergencies, filter AI responses, and prevent the AI from giving unsafe or definitive medical advice.
*   **Flexible Service Architecture**: The `AIServiceManager` intelligently delegates tasks, using OpenAI for speed (symptom extraction, conversational questions) and MedGemma for accuracy (final diagnosis).
*   **Multiple Deployment Modes**: Supports `local_demo`, `hybrid`, and `cloud_first` modes for flexible development and deployment.

## Required Keys

To run this application, you will need to create a file named `.env` in the root directory of the project. This file stores your secret keys and configuration settings.

```
# .env file

# --- Core Keys (Required for most functionality) ---
# Your OpenAI API Key for embeddings, symptom extraction, and conversational AI
OPENAI_API_KEY="sk-..."

# --- Google Cloud & Model Garden Keys (Required for 'cloud_first' mode) ---
# Your Google Cloud Project ID
GCP_PROJECT_ID="your-gcp-project-id"

# The Vertex AI Endpoint ID for your deployed MedGemma model
MEDGEMMA_ENDPOINT_ID="1234567890123456789"



# --- Configuration Modes ---
# Sets the overall behavior of the application (development or production)
# DEPLOYMENT_MODE="development"

# Sets the AI service priority ('hybrid', 'local_demo', 'cloud_first')
# AI_SERVICE_MODE="hybrid"
```

## How to Run the Demo (Local)

Follow these steps to set up and run the AI Doctor chatbot on your local machine.

### 1. Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Navigate to the project directory
cd /path/to/AI_Doctor-main

# Create a virtual environment
python3 -m venv ai_doctor_env

# Activate the virtual environment
source ai_doctor_env/bin/activate
```

### 2. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*Note: If you encounter issues with `torch`, you may need to install a version specific to your CPU/GPU setup. Please refer to the official PyTorch installation guide.*

### 3. Create and Populate the RAG Vector Store

The chatbot uses a FAISS vector store built from a medical encyclopedia for its RAG capabilities. You need to run a script to create this vector store before starting the main application.

*Note: This process can take a few minutes as it involves downloading models and processing the document.*

```bash
python src/services/ai/rag/create_memory_for_llm.py
```

You should see output indicating that the vector store has been created and saved successfully in the `data/vectorstore/db_faiss` directory.

### 4. Configure Your `.env` File

Create a file named `.env` in the root of the project directory. Copy the content from the "Required Keys" section above and fill in your `OPENAI_API_KEY`.

If you plan to test the Google Cloud `cloud_first` mode, you must also provide your `GCP_PROJECT_ID` and `MEDGEMMA_ENDPOINT_ID`.

### 5. Run the demo

python start_cloud.py

### 6. Open the Chatbot in Your Browser

Once the server is running, open your web browser and navigate to:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

You can now start interacting with the AI Doctor!
