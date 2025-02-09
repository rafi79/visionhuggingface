import streamlit as st
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    MultiModalityCausalLM,
    pipeline
)
from PIL import Image
import os
from typing import Optional

# Set environment variable for HuggingFace token
os.environ["HUGGINGFACE_TOKEN"] = "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"

class ModelManager:
    def __init__(self):
        self.pali_processor = None
        self.pali_model = None
        self.deepseek_model = None
        
    def load_pali(self):
        if not self.pali_model:
            with st.spinner("Loading PaLI-GEMMA model..."):
                self.pali_processor = AutoProcessor.from_pretrained(
                    "google/paligemma-3b-pt-224",
                    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
                )
                self.pali_model = AutoModelForImageTextToText.from_pretrained(
                    "google/paligemma-3b-pt-224",
                    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
                )
    
    def load_deepseek(self):
        if not self.deepseek_model:
            with st.spinner("Loading DeepSeek-VL model..."):
                self.deepseek_model = MultiModalityCausalLM.from_pretrained(
                    "deepseek-ai/deepseek-vl-1.3b-base",
                    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
                )

    def process_with_pali(self, image: Image, prompt: str) -> str:
        inputs = self.pali_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        outputs = self.pali_model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
        
        return self.pali_processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def process_with_deepseek(self, image: Image, prompt: str) -> str:
        # Convert PIL Image to format expected by DeepSeek
        # Add implementation based on DeepSeek's specific requirements
        outputs = self.deepseek_model.generate(
            image=image,
            prompt=prompt,
            max_length=100
        )
        return outputs[0]  # Adjust based on actual output format

def main():
    st.title("Multimodal Chat Interface")
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Text input for prompt
        prompt = st.text_input("Enter your prompt:", "Describe this image")
        
        if st.button("Process"):
            # Load models
            st.session_state.model_manager.load_pali()
            st.session_state.model_manager.load_deepseek()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PaLI-GEMMA Output")
                pali_output = st.session_state.model_manager.process_with_pali(image, prompt)
                st.write(pali_output)
            
            with col2:
                st.subheader("DeepSeek-VL Output")
                deepseek_output = st.session_state.model_manager.process_with_deepseek(image, prompt)
                st.write(deepseek_output)
            
            # Combined output
            st.subheader("Combined Analysis")
            st.write("Combined insights from both models:")
            combined_output = f"""
            Model Comparison:
            - PaLI-GEMMA sees: {pali_output}
            - DeepSeek-VL sees: {deepseek_output}
            """
            st.write(combined_output)

if __name__ == "__main__":
    main()
