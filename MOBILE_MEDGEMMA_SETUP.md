# MedGemma Model Garden Setup for Mobile Apps

This guide shows you how to integrate MedGemma using **Google Cloud Model Garden** for your mobile app backend - the optimal choice for mobile applications.

## Why Model Garden for Mobile Apps?

### ✅ **Model Garden Advantages**

| Feature | Model Garden | Hugging Face Local |
|---------|--------------|-------------------|
| **Mobile Battery Life** | ⭐⭐⭐⭐⭐ No local processing | ⭐⭐ Heavy local processing |
| **App Size** | ⭐⭐⭐⭐⭐ Small (~50MB) | ⭐ Large (15GB+ models) |
| **Performance** | ⭐⭐⭐⭐⭐ Cloud infrastructure | ⭐⭐ Limited by device |
| **Scalability** | ⭐⭐⭐⭐⭐ Handles thousands of users | ⭐ Single device only |
| **Updates** | ⭐⭐⭐⭐⭐ Automatic model updates | ⭐⭐ Manual updates required |
| **Device Compatibility** | ⭐⭐⭐⭐⭐ Works on all devices | ⭐⭐ Requires powerful hardware |

### 📱 **Mobile App Architecture**

```
📱 Mobile App (React Native/Flutter/Native)
    ↓ HTTP API calls
🖥️ Your Backend Server (FastAPI)
    ↓ API calls
☁️ Google Cloud Model Garden (MedGemma)
```

## Prerequisites

### Google Cloud Setup

1. **Google Cloud Account**: [Create account](https://cloud.google.com/)
2. **Project**: Create a new GCP project
3. **Billing**: Enable billing (Model Garden requires it)
4. **APIs**: Enable AI Platform API

### System Requirements

- **Backend Server**: Any server (even small VPS works)
- **RAM**: 4GB+ (much less than local models)
- **Storage**: Minimal (no model files to store)
- **Network**: Stable internet connection

## Installation Steps

### 1. Google Cloud Setup

```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
```

### 2. Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create medgemma-service \
    --description="Service account for MedGemma Model Garden" \
    --display-name="MedGemma Service"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:medgemma-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create medgemma-key.json \
    --iam-account=medgemma-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3. Deploy MedGemma to Model Garden

```bash
# Deploy MedGemma model (this may take 10-15 minutes)
gcloud ai models upload \
    --region=us-central1 \
    --display-name=medgemma-7b \
    --container-image-uri=gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-9:latest \
    --artifact-uri=gs://medgemma-models/medgemma-7b

# Create endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=medgemma-endpoint
```

### 4. Install Dependencies

```bash
# Install Google Cloud AI Platform
pip install google-cloud-aiplatform

# Install other dependencies (already in your project)
pip install fastapi uvicorn openai
```

### 5. Configure Environment Variables

Create or update your `.env` file:

```bash
# Required
OPENAI_API_KEY="your-openai-api-key"

# Enable MedGemma Model Garden
USE_MEDGEMMA=true

# Google Cloud Configuration
GCP_PROJECT_ID="your-gcp-project-id"
GOOGLE_APPLICATION_CREDENTIALS="path/to/medgemma-key.json"
```

## Configuration

### Update Your Backend

Your existing code is already updated! Just set the environment variables:

```bash
export USE_MEDGEMMA=true
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="./medgemma-key.json"

python main.py
```

### Test the Integration

```bash
# Check service status
curl http://localhost:8000/health

# Test MedGemma endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I have knee pain for 2 weeks",
    "session_id": "test_mobile_session",
    "is_choice": false
  }'
```

## Mobile App Integration

### React Native Example

```javascript
// MedicalConsultant.js
class MedicalConsultant {
  constructor(baseUrl = 'https://your-backend.com') {
    this.baseUrl = baseUrl;
  }

  async sendMessage(message, sessionId) {
    try {
      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          session_id: sessionId,
          is_choice: false
        })
      });

      return await response.json();
    } catch (error) {
      console.error('Medical consultation error:', error);
      throw error;
    }
  }

  async uploadImage(imageUri, sessionId) {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('image_data', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'symptom_photo.jpg'
    });

    const response = await fetch(`${this.baseUrl}/analyze_image`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });

    return await response.json();
  }
}

export default MedicalConsultant;
```

### Flutter Example

```dart
// medical_consultant.dart
class MedicalConsultant {
  final String baseUrl;
  final http.Client client = http.Client();

  MedicalConsultant({this.baseUrl = 'https://your-backend.com'});

  Future<Map<String, dynamic>> sendMessage(String message, String sessionId) async {
    final response = await client.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'query': message,
        'session_id': sessionId,
        'is_choice': false,
      }),
    );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to send message');
    }
  }

  Future<Map<String, dynamic>> uploadImage(File imageFile, String sessionId) async {
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/analyze_image'));
    request.fields['session_id'] = sessionId;
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    var response = await request.send();
    var responseBody = await response.stream.bytesToString();
    
    return json.decode(responseBody);
  }
}
```

## Cost Optimization

### Model Garden Pricing

- **Model hosting**: ~$0.10-0.50 per hour (when active)
- **Predictions**: ~$0.001-0.01 per request
- **Auto-scaling**: Scales to zero when not in use

### Cost-Saving Tips

1. **Auto-scaling**: Configure endpoints to scale down when idle
2. **Regional deployment**: Use closest region to reduce latency costs
3. **Batch requests**: Group multiple queries when possible
4. **Caching**: Cache common responses on your backend

## Performance Optimization

### Response Time Optimization

```python
# In app/medgemma_model_garden.py
async def generate_medical_response(self, query: str, **kwargs):
    # Use connection pooling
    # Set appropriate timeouts
    # Implement retry logic
    pass
```

### Mobile-Specific Optimizations

1. **Request Compression**: Enable gzip compression
2. **Response Caching**: Cache responses on mobile device
3. **Offline Fallback**: Provide offline responses for common queries
4. **Progressive Loading**: Show partial responses as they arrive

## Monitoring and Analytics

### Health Monitoring

```bash
# Check Model Garden endpoint health
curl http://localhost:8000/health

# Monitor response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/chat
```

### Usage Analytics

```python
# Add to your backend
import logging
from google.cloud import monitoring_v3

# Track usage metrics
def log_medical_query(session_id, query_type, response_time):
    logger.info(f"Medical query: {session_id}, {query_type}, {response_time}ms")
```

## Security Considerations

### Data Privacy

1. **HIPAA Compliance**: Ensure your implementation meets healthcare data requirements
2. **Data Encryption**: All data encrypted in transit and at rest
3. **Access Control**: Implement proper authentication and authorization
4. **Audit Logging**: Log all medical consultations for compliance

### API Security

```python
# Add authentication to your endpoints
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/chat")
async def chat(query_request: QueryRequest, token: str = Depends(security)):
    # Validate token
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    # ... rest of your code
```

## Deployment

### Production Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set environment variables
ENV USE_MEDGEMMA=true
ENV GCP_PROJECT_ID=your-project-id

CMD ["python", "main.py"]
```

### Cloud Deployment Options

1. **Google Cloud Run**: Serverless, auto-scaling
2. **Google Kubernetes Engine**: Full container orchestration  
3. **AWS/Azure**: Cross-cloud deployment
4. **DigitalOcean/Heroku**: Simple VPS deployment

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

```
google.auth.exceptions.DefaultCredentialsError
```

**Solution:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

#### 2. Model Garden Endpoint Not Found

```
404 Endpoint not found
```

**Solution:**
- Verify endpoint is deployed: `gcloud ai endpoints list`
- Check project ID and region in configuration

#### 3. High Latency

**Solutions:**
- Use regional endpoints closer to users
- Implement response caching
- Use connection pooling

## Next Steps

1. **Deploy to production**: Use Cloud Run or Kubernetes
2. **Add monitoring**: Set up alerts and dashboards
3. **Scale testing**: Test with multiple concurrent users
4. **Mobile app integration**: Connect your React Native/Flutter app
5. **Compliance**: Ensure HIPAA/medical compliance if needed

## Support Resources

- **Google Cloud Model Garden**: [Documentation](https://cloud.google.com/vertex-ai/docs/model-garden)
- **MedGemma**: [GitHub Repository](https://github.com/Google-Health/medgemma)
- **AI Platform**: [API Reference](https://cloud.google.com/ai-platform/docs)

---

🎉 **You're ready!** Your mobile app now has access to state-of-the-art medical AI through Google Cloud Model Garden, optimized for mobile performance and scalability. 