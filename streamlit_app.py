import streamlit as st
import torch
from PIL import Image
import os

# Wrap transformer imports in try-except for better error handling
try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
    )
except Exception as e:
    st.error(f"Error importing transformers: {str(e)}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Image Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        # Initialize models with error handling
        processor = AutoProcessor.from_pretrained(
            "google/paligemma-3b-pt-224",
            trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            "google/paligemma-3b-pt-224",
            trust_remote_code=True,
            device_map="auto"
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_image(image, processor, model, prompt):
    """Process image and generate response"""
    try:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("🤖 AI Image Chatbot")
    st.write("Upload an image and ask questions about it!")

    # Model loading with progress indicator
    with st.spinner("Loading AI models... This might take a minute."):
        processor, model = load_models()
        
    if processor is None or model is None:
        st.error("Failed to load models. Please try again later.")
        st.stop()
    else:
        st.success("Models loaded successfully! ✅")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Initialize chat history if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask something about the image"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing image..."):
                    response = process_image(image, processor, model, prompt)
                    if response:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        st.write(response)

if __name__ == "__main__":
    main()
