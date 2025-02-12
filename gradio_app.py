import argparse
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from peft import PeftModel
from PIL import Image

# Parse arguments
parser = argparse.ArgumentParser(description="Gradio Interface for Fine-Tuned LLaVA Model")
parser.add_argument("--base_model_path", type=str, default="../../llava-v1.6-mistral-7b", help="Path to the base model")
parser.add_argument("--lora_weights_path", type=str, default="./llava-v1.6-mistral-7b-RLAIF-V-ORPO/checkpoint-99000", help="Path to LoRA weights")
parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing")
args = parser.parse_args()

# Load LLaVA model with vision encoder
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.base_model_path,
    model_base=None,  # If not using a separate base model
    model_name="llava-1.6-mistral-7b"
)

# Load LoRA weights and merge
model = PeftModel.from_pretrained(model, args.lora_weights_path)
model = model.merge_and_unload()  # Merge LoRA weights for efficient inference

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def multimodal_response(image, prompt):
    # Process image using LLaVA's image encoder
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(device)
    
    # Process text
    inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    inputs = tokenizer_image_token(inp, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    
    input_ids = inputs.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)  # ✅ Create attention mask

    # Generate response
    with torch.no_grad():
        output = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,  # ✅ Explicitly pass attention mask
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id  # ✅ Prevents unexpected behavior
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Gradio Interface
iface = gr.Interface(
    fn=multimodal_response,
    inputs=[
        gr.Image(label="Upload an image"),  # Image input
        gr.Textbox(label="Enter your prompt")  # Text prompt input
    ],
    outputs=gr.Textbox(label="Response"),
    title="LLaVA 1.6 Mistral 7B - RLAIF ORPO LoRA Multimodal Demo",
    description="Test the fine-tuned LLaVA 1.6 Mistral 7B model with both images and text inputs on the RLAIF dataset."
)

iface.launch(share=args.share)
