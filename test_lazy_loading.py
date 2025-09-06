#!/usr/bin/env python3
"""
Test script to demonstrate lazy loading in MedGemma service
This prevents server startup crashes by loading the model only on first request
"""

import asyncio
import sys
import os
import time

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.ai.medgemma.medgemma_service import MedGemmaService

async def test_lazy_loading():
    """Test the lazy loading functionality"""
    
    print("🧪 Testing MedGemma Service with LAZY LOADING")
    print("=" * 60)
    
    # Test instant initialization (should not load model)
    print("📋 Step 1: Initializing service (should be instant)")
    start_time = time.time()
    
    service = MedGemmaService(
        model_name="google/medgemma-4b-it",
        use_quantization=False
    )
    
    init_time = time.time() - start_time
    print(f"✅ Service initialized in {init_time:.2f} seconds")
    print(f"🔧 Model loaded: {service.is_loaded}")
    print(f"📊 Model object: {service.model is not None}")
    print(f"🔧 Pipeline object: {service.pipeline is not None}")
    
    # Test first request (should load model)
    print(f"\n📋 Step 2: Making first request (should load model now)")
    start_time = time.time()
    
    response = await service.generate_medical_response(
        query="What are the main symptoms of pneumonia?",
        max_new_tokens=100,
        use_direct_method=True
    )
    
    first_request_time = time.time() - start_time
    print(f"✅ First request completed in {first_request_time:.2f} seconds")
    print(f"🔧 Model loaded: {service.is_loaded}")
    print(f"📊 Success: {response['success']}")
    print(f"💬 Response preview: {response['response'][:100]}...")
    
    # Test second request (should be fast)
    print(f"\n📋 Step 3: Making second request (should be fast)")
    start_time = time.time()
    
    response2 = await service.generate_medical_response(
        query="How is diabetes diagnosed?",
        max_new_tokens=100,
        use_direct_method=True
    )
    
    second_request_time = time.time() - start_time
    print(f"✅ Second request completed in {second_request_time:.2f} seconds")
    print(f"📊 Success: {response2['success']}")
    print(f"💬 Response preview: {response2['response'][:100]}...")
    
    # Summary
    print(f"\n🎯 LAZY LOADING PERFORMANCE SUMMARY:")
    print(f"   📊 Service initialization: {init_time:.2f}s (instant!)")
    print(f"   📊 First request (with model loading): {first_request_time:.2f}s")
    print(f"   📊 Second request (model already loaded): {second_request_time:.2f}s")
    print(f"   🚀 Speed improvement after loading: {first_request_time/second_request_time:.1f}x faster")
    
    print(f"\n✅ LAZY LOADING BENEFITS FOR SERVER DEPLOYMENT:")
    print(f"   🚀 FastAPI server starts instantly (no model loading delay)")
    print(f"   🔗 ngrok can connect immediately (no connection refused errors)")
    print(f"   💾 Model loads only when needed (saves memory until first request)")
    print(f"   🔄 Graceful error handling if model loading fails")

if __name__ == "__main__":
    asyncio.run(test_lazy_loading())
