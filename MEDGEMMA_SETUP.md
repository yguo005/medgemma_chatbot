# MedGemma Integration Guide

This guide shows you how to integrate [Google Health's MedGemma](https://github.com/Google-Health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb) with your AI Health Consultant for specialized medical AI responses.

## What is MedGemma?

MedGemma is Google Health's specialized medical AI model built on the Gemma architecture. It's specifically trained on medical data and provides more accurate medical information compared to general-purpose AI models.

## Benefits of Using MedGemma

✅ **Medical Specialization**: Trained specifically on medical data  
✅ **Privacy**: Runs locally on your server (no external API calls)  
✅ **Cost Effective**: No per-token charges like OpenAI  
✅ **Offline Capability**: Works without internet connectivity  
✅ **Faster Response**: No network latency  

## Prerequisites

### System Requirements

- **RAM**: Minimum 16GB (32GB recommended for better performance)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 15GB+ free space for model files
- **Python**: 3.8 or higher

### Hardware Recommendations

| Setup | RAM | GPU | Performance |
|-------|-----|-----|-------------|
| **Minimum** | 16GB | CPU only | Slow but functional |
| **Recommended** | 32GB | RTX 3080/4070+ | Good performance |
| **Optimal** | 64GB+ | RTX 4080/4090+ | Excellent performance |

## Installation Steps

### 1. Install Dependencies

```bash
# Install PyTorch (choose based on your system)
# For CUDA (NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For Apple Silicon (M1/M2):
pip install torch torchvision torchaudio

# Install Transformers and other dependencies
pip install transformers accelerate bitsandbytes sentencepiece
```

### 2. Configure Environment Variables

Create or update your `.env` file:

```bash
# Required
OPENAI_API_KEY="your-openai-api-key-here"

# Enable MedGemma
USE_MEDGEMMA=true
```

### 3. Test MedGemma Installation

```bash
python test_ai_services.py
```

This will test both OpenAI services and MedGemma integration.

## Configuration Options

### Model Selection

You can choose different MedGemma model sizes in `app/medgemma_service.py`:

```python
# Available models (choose based on your hardware):
model_options = {
    "small": "google/medgemma-2b",      # Faster, less accurate
    "medium": "google/medgemma-7b",     # Balanced (default)
    "large": "google/medgemma-13b"      # More accurate, slower
}
```

### Device Configuration

The system automatically detects the best device:

- **CUDA**: NVIDIA GPU (fastest)
- **MPS**: Apple Silicon GPU (M1/M2)
- **CPU**: Fallback option (slowest)

## Usage Examples

### 1. Enable MedGemma

```bash
export USE_MEDGEMMA=true
python main.py
```

### 2. API Endpoints

Your existing endpoints now use MedGemma when enabled:

```bash
# Check service status (includes MedGemma info)
curl http://localhost:8000/health

# Chat with MedGemma enhancement
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I have knee pain",
    "session_id": "test_session",
    "is_choice": false
  }'
```

### 3. Mobile Interface

Access your mobile interface at `http://localhost:8000/mobile` - it will automatically use MedGemma when enabled.

## Performance Optimization

### Memory Management

For better performance with limited RAM:

```python
# In app/medgemma_service.py, modify model loading:
model_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": torch.float16,  # Use half precision
    "low_cpu_mem_usage": True,
    "load_in_8bit": True,          # 8-bit quantization
    # "load_in_4bit": True,        # Even more aggressive (experimental)
}
```

### GPU Optimization

For NVIDIA GPUs, ensure CUDA is properly configured:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use smaller model: `google/medgemma-2b`
- Enable quantization: `load_in_8bit=True`
- Reduce batch size or max_length

#### 2. Model Download Fails

```
OSError: Can't load tokenizer for 'google/medgemma-7b'
```

**Solutions:**
- Check internet connection
- Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`
- Try different model variant

#### 3. Slow Performance

**Solutions:**
- Use GPU instead of CPU
- Enable half precision: `torch_dtype=torch.float16`
- Reduce `max_length` parameter

### Debug Commands

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check model loading
python -c "from app.medgemma_service import MedGemmaService; service = MedGemmaService()"
```

## Comparison: MedGemma vs OpenAI

| Feature | MedGemma | OpenAI GPT-4 |
|---------|----------|--------------|
| **Medical Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Privacy** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## Hybrid Approach (Recommended)

You can use both MedGemma and OpenAI together:

1. **MedGemma**: For medical diagnosis and symptom analysis
2. **GPT-4 Vision**: For image analysis (medical images)
3. **Whisper**: For audio transcription
4. **Your RAG System**: For specific medical knowledge from your PDF

This gives you the best of all worlds!

## Production Deployment

### Docker Configuration

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch transformers fastapi uvicorn

# Copy your application
COPY . /app
WORKDIR /app

# Run with MedGemma enabled
ENV USE_MEDGEMMA=true
CMD ["python3", "main.py"]
```

### Resource Monitoring

Monitor GPU/CPU usage:

```bash
# GPU monitoring
nvidia-smi -l 1

# CPU/Memory monitoring  
htop
```

## Next Steps

1. **Test with your medical data**: Upload actual medical images and voice recordings
2. **Fine-tune prompts**: Customize medical prompts in `app/medgemma_service.py`
3. **Scale horizontally**: Deploy multiple instances with load balancing
4. **Monitor performance**: Set up logging and metrics

## Support

- **MedGemma Issues**: [GitHub Issues](https://github.com/Google-Health/medgemma/issues)
- **Transformers Library**: [Hugging Face Docs](https://huggingface.co/docs/transformers)
- **PyTorch**: [PyTorch Docs](https://pytorch.org/docs/)

---

🎉 **Congratulations!** You now have a state-of-the-art medical AI system combining the best of specialized medical models with general-purpose AI capabilities. 