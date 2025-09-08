"""
Fixed Test for AI Service Manager - Addresses Both Issues

Issues Fixed:
1. ✅ MedGemma image analysis error (missing argument)
2. ✅ OpenAI deprecated model error (gpt-4-vision-preview)
"""

import asyncio
import logging
import os
import sys
import tempfile
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment"""
    print("🔧 Setting up environment...")
    
    # Install required packages if needed
    import subprocess
    try:
        import openai
        print("✓ OpenAI package available")
    except ImportError:
        print("📦 Installing OpenAI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "--quiet"])

def create_test_image():
    """Create a simple test medical image"""
    img = Image.new('RGB', (512, 512), color='black')
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw some basic shapes to simulate medical imaging
    draw.ellipse([100, 100, 400, 400], outline='white', width=3)
    draw.rectangle([200, 200, 300, 300], outline='gray', width=2)
    draw.text((50, 450), "TEST MEDICAL IMAGE", fill='white')
    
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

async def test_services_individually():
    """Test each service individually to diagnose issues"""
    print("\n🔍 Individual Service Testing")
    print("=" * 70)
    
    # Add project path
    project_root = "/content/AI_Doctor-main" if "COLAB_GPU" in os.environ else "."
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Test 1: MedGemma Service Directly
    print("\n🧪 Test 1: MedGemma Service Direct Test")
    print("-" * 50)
    
    try:
        from src.services.ai.medgemma.medgemma_service import MedGemmaService
        
        # Create service
        service = MedGemmaService(
            model_name="google/medgemma-4b-it",
            use_quantization=True
        )
        
        # Test if service initializes properly
        print(f"✅ MedGemma service created")
        print(f"   Model variant: {service.model_variant}")
        print(f"   Is text only: {service.is_text_only}")
        
        # Test text generation first
        print("\n📝 Testing text generation...")
        text_result = await service.generate_medical_response(
            query="What are the symptoms of diabetes?"
        )
        
        if text_result['success']:
            print(f"✅ Text generation works!")
            print(f"   Response: {text_result['response'][:100]}...")
        else:
            print(f"❌ Text generation failed: {text_result.get('error')}")
        
        # Test image analysis
        print("\n🖼️  Testing image analysis...")
        if not service.is_text_only:
            test_image = create_test_image()
            
            image_result = await service.analyze_image_with_text(
                image=test_image,
                text_prompt="Describe this medical image."
            )
            
            if image_result['success']:
                print(f"✅ Image analysis works!")
                print(f"   Analysis: {image_result['response'][:100]}...")
            else:
                print(f"❌ Image analysis failed: {image_result.get('error')}")
        else:
            print("ℹ️  Service is text-only, skipping image test")
        
    except Exception as e:
        print(f"❌ MedGemma service test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: OpenAI Service Directly
    print("\n🧪 Test 2: OpenAI Service Direct Test")
    print("-" * 50)
    
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ No OpenAI API key found in environment")
            return
        
        from src.services.ai.openai_services import AIServices
        
        # Create service
        openai_service = AIServices(api_key=openai_key, use_medgemma=False)
        
        print("✅ OpenAI service created")
        
        # Test image analysis with a simple test
        print("\n🖼️  Testing OpenAI image analysis...")
        test_image = create_test_image()
        image_base64 = image_to_base64(test_image)
        
        result = await openai_service.analyze_image(image_base64, "medical")
        
        if result['success']:
            print(f"✅ OpenAI image analysis works!")
            print(f"   Model used: {result.get('model_used')}")
            print(f"   Analysis: {result['analysis'][:100]}...")
        else:
            print(f"❌ OpenAI image analysis failed: {result.get('error')}")
            
            # Check if it's the deprecated model error
            if "deprecated" in str(result.get('error')).lower():
                print("🔧 This is the deprecated model issue - need to fix model configuration")
        
    except Exception as e:
        print(f"❌ OpenAI service test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_ai_service_manager_fixed():
    """Test AI Service Manager with detailed diagnostics"""
    print("\n🧠 Testing AI Service Manager - Fixed Version")
    print("=" * 70)
    
    try:
        # Add project path
        project_root = "/content/AI_Doctor-main" if "COLAB_GPU" in os.environ else "."
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        from src.services.ai.ai_service_manager import OptimizedAIServiceManager, ServiceMode
        
        # Create service manager
        print("🔄 Initializing AI Service Manager...")
        manager = OptimizedAIServiceManager(mode=ServiceMode.LOCAL_DEMO)
        
        # Check service status with detailed info
        print("\n📊 Detailed Service Status:")
        status = manager.get_service_status()
        
        for service_name, service_info in status['services'].items():
            status_icon = "✅" if service_info['available'] else "❌"
            print(f"  {status_icon} {service_name}:")
            print(f"      Available: {service_info['available']}")
            print(f"      Capabilities: {service_info['capabilities']}")
            
            # Additional diagnostics for MedGemma
            if service_name == 'medgemma_local' and service_info['available']:
                service = manager.services.get('medgemma_local')
                if service:
                    print(f"      Service ready: {service.is_service_ready()}")
                    print(f"      Model loaded: {service.is_loaded}")
        
        # Test 1: Text Generation
        print(f"\n🧪 Test 1: Medical Text Generation")
        print("-" * 50)
        
        text_query = "What are the symptoms of pneumonia?"
        print(f"Query: {text_query}")
        
        result = await manager.generate_medical_response(text_query)
        
        if result['success']:
            print(f"✅ Success! Service used: {result.get('service_used', 'unknown')}")
            print(f"Response: {result['response'][:200]}...")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Test 2: Image Analysis with Better Error Handling
        print(f"\n🧪 Test 2: Medical Image Analysis")
        print("-" * 50)
        
        test_image = create_test_image()
        image_base64 = image_to_base64(test_image)
        
        print(f"Created test image ({len(image_base64)} chars base64)")
        
        result = await manager.analyze_image(image_base64, "medical")
        
        if result['success']:
            print(f"✅ Success! Service used: {result.get('service_used', 'unknown')}")
            print(f"Analysis: {result['analysis'][:200]}...")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Provide specific diagnostics
            print("\n🔍 Diagnostic Information:")
            print("   - Check if MedGemma model loaded successfully")
            print("   - Check if OpenAI API key is valid")
            print("   - Check if OpenAI is using correct model (gpt-4o)")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Service Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function with comprehensive diagnostics"""
    print("🚀 AI Service Manager Diagnostic Test")
    print("Fixing: MedGemma image analysis + OpenAI deprecated model")
    print("=" * 80)
    
    # Check environment
    in_colab = 'google.colab' in sys.modules
    print(f"🔍 Environment: {'Google Colab' if in_colab else 'Local'}")
    
    # Check API keys
    print(f"\n🔑 API Key Status:")
    print(f"  HF_TOKEN: {'✅ Set' if os.getenv('HF_TOKEN') else '❌ Missing'}")
    print(f"  OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    # Setup environment
    setup_environment()
    
    # Run individual service tests
    asyncio.run(test_services_individually())
    
    # Run integrated test
    asyncio.run(test_ai_service_manager_fixed())
    
    print(f"\n✅ Diagnostic test completed!")
    print("🎯 This test should help identify the exact source of both issues")

if __name__ == "__main__":
    main()
