#!/usr/bin/env python3
"""
Test MedGemma using the exact same approach as the official notebook
This helps debug why our implementation has loading issues
"""

import os
import sys
import torch
from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor

def test_official_medgemma_loading():
    """Test MedGemma loading using the exact official notebook approach"""
    print("🧪 Testing MedGemma loading using official notebook approach")
    print("="*60)
    
    # Set HF token if available
    if os.getenv('HF_TOKEN'):
        print("✅ HF_TOKEN found")
    else:
        print("⚠️  HF_TOKEN not set - you may need to authenticate")
    
    # Model configuration exactly like official notebook
    model_variant = "4b-it"
    model_id = f"google/medgemma-{model_variant}"
    use_quantization = True
    
    print(f"Model ID: {model_id}")
    print(f"Use quantization: {use_quantization}")
    
    # Check if we're in Colab
    google_colab = "google.colab" in sys.modules
    print(f"Google Colab: {google_colab}")
    
    # GPU check exactly like official notebook
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        
        # Official notebook check for 27B variants (we're using 4B so this is informational)
        if "27b" in model_variant and google_colab:
            if not ("A100" in gpu_name and use_quantization):
                print("⚠️  Would need A100 + quantization for 27B")
    else:
        print("⚠️  No GPU detected")
        use_quantization = False  # Disable quantization without GPU
    
    # Model kwargs exactly like official notebook
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        print("✅ 4-bit quantization enabled")
    
    print("\n🔄 Loading model (this may take a while)...")
    print(f"Model kwargs: {model_kwargs}")
    
    try:
        # Load exactly like official notebook
        if "text" in model_variant:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
            processor_or_tokenizer = AutoProcessor.from_pretrained(model_id)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Processor type: {type(processor_or_tokenizer)}")
        
        # Test a simple inference
        print("\n🔄 Testing inference...")
        
        system_instruction = "You are a helpful medical assistant."
        user_prompt = "What is a headache?"
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        inputs = processor_or_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generation = generation[0][input_len:]
        
        response = processor_or_tokenizer.decode(generation, skip_special_tokens=True)
        
        print("✅ Inference successful!")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set environment variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    success = test_official_medgemma_loading()
    
    if success:
        print("\n🎉 Official approach works! The issue is in our service wrapper.")
    else:
        print("\n❌ Official approach also fails. This is a deeper system issue.")
        print("\n💡 Suggestions:")
        print("1. Check GPU memory availability")
        print("2. Try without quantization") 
        print("3. Check PyTorch/CUDA compatibility")
        print("4. Ensure HF_TOKEN is set correctly")
