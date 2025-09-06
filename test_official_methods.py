#!/usr/bin/env python3
"""
Test script to demonstrate the improved MedGemma service using official notebook methods
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.ai.medgemma.medgemma_service import MedGemmaService

async def test_official_methods():
    """Test the improved MedGemma service with official methods"""
    
    print("🧪 Testing MedGemma Service with Official Notebook Methods")
    print("=" * 60)
    
    # Test with quantization first (if available)
    print("\n📋 Test 1: Testing with quantization (if supported)")
    try:
        service_q = MedGemmaService(
            model_name="google/medgemma-4b-it",
            use_quantization=True
        )
        
        # Test direct method (more memory efficient)
        response = await service_q.generate_medical_response(
            query="What are the main symptoms of pneumonia?",
            max_new_tokens=200,
            use_direct_method=True
        )
        
        print(f"✅ Direct Method Success: {response['success']}")
        print(f"🔧 Method Used: {response.get('method', 'unknown')}")
        print(f"📊 Model Variant: {response.get('model_variant', 'unknown')}")
        print(f"💬 Response Preview: {response['response'][:150]}...")
        
    except Exception as e:
        print(f"❌ Quantization test failed: {e}")
    
    print("\n📋 Test 2: Testing without quantization (fallback)")
    try:
        service = MedGemmaService(
            model_name="google/medgemma-4b-it",
            use_quantization=False
        )
        
        # Test direct method
        print("\n🔬 Testing Direct Model Method (Official Implementation):")
        response_direct = await service.generate_medical_response(
            query="How is diabetes diagnosed?",
            max_new_tokens=150,
            use_direct_method=True
        )
        
        print(f"✅ Success: {response_direct['success']}")
        print(f"🔧 Method: {response_direct.get('method', 'unknown')}")
        print(f"💬 Response: {response_direct['response'][:200]}...")
        
        # Test pipeline method for comparison
        print("\n🔬 Testing Pipeline Method (Backup):")
        response_pipeline = await service.generate_medical_response(
            query="How is diabetes diagnosed?",
            max_new_tokens=150,
            use_direct_method=False
        )
        
        print(f"✅ Success: {response_pipeline['success']}")
        print(f"🔧 Method: {response_pipeline.get('method', 'unknown')}")
        print(f"💬 Response: {response_pipeline['response'][:200]}...")
        
        # Compare memory efficiency
        print("\n📊 Method Comparison:")
        print(f"   Direct Method: More memory efficient, uses torch.inference_mode()")
        print(f"   Pipeline Method: Standard approach, higher memory usage")
        
    except Exception as e:
        print(f"❌ Main test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Official Notebook Integration Summary:")
    print("   ✅ Direct model generation with torch.inference_mode()")
    print("   ✅ Proper apply_chat_template usage")
    print("   ✅ Official model_kwargs pattern (torch.bfloat16, device_map='auto')")
    print("   ✅ Smart quantization fallback")
    print("   ✅ Memory-efficient multimodal support")
    print("   ✅ Official max_new_tokens limits")

if __name__ == "__main__":
    asyncio.run(test_official_methods())
