import streamlit as st
import torch
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        pipeline,
        AutoTokenizer,
        AutoModel
    )
except Exception as e:
    st.error(f"Error importing transformers: {str(e)}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Dual Model Image Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

@st.cache_resource
def load_pali_pipeline():
    """Load PaLI-GEMMA pipeline with caching"""
    try:
        pipe = pipeline(
            "image-to-text",
            model="google/paligemma-3b-pt-224",
            device_map="auto"
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading PaLI-GEMMA model: {str(e)}")
        return None

@st.cache_resource
def load_deepseek_pipeline():
    """Load DeepSeek pipeline with caching"""
    try:
        pipe = pipeline(
            "image-to-text",
            model="deepseek-ai/deepseek-vl-1.3b-base",
            device_map="auto"
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading DeepSeek model: {str(e)}")
        return None

def analyze_with_model(image, pipe, prompt, model_name):
    """Analyze image with specified model"""
    try:
        outputs = pipe(
            images=image,
            prompt=prompt,
            max_new_tokens=100,
            num_beams=4
        )
        return {
            "model": model_name,
            "response": outputs[0]['generated_text'],
            "status": "success"
        }
    except Exception as e:
        return {
            "model": model_name,
            "response": f"Error with {model_name}: {str(e)}",
            "status": "error"
        }

def analyze_image_parallel(image, pali_pipe, deepseek_pipe, prompt):
    """Analyze image with both models in parallel"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        pali_future = executor.submit(
            analyze_with_model, 
            image, 
            pali_pipe, 
            prompt, 
            "PaLI-GEMMA"
        )
        deepseek_future = executor.submit(
            analyze_with_model, 
            image, 
            deepseek_pipe, 
            prompt, 
            "DeepSeek"
        )
        
        results = {
            "PaLI-GEMMA": pali_future.result(),
            "DeepSeek": deepseek_future.result()
        }
        
    return results

def main():
    st.title("🤖 Dual AI Image Analysis Bot")
    st.markdown("""
    ### Compare responses from two different AI models!
    This bot uses both PaLI-GEMMA and DeepSeek models to analyze images and respond to your questions.
    """)

    # Initialize session state
    initialize_session_state()

    # Load models
    with st.spinner("Loading AI models... This might take a minute."):
        pali_pipe = load_pali_pipeline()
        deepseek_pipe = load_deepseek_pipeline()
        
        if pali_pipe is None or deepseek_pipe is None:
            st.error("Failed to load models. Please try again later.")
            st.stop()
        else:
            st.success("✅ Both models loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"],
        help="Upload an image you want to analyze"
    )

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display chat interface
        st.markdown("### Chat with both AIs about your image")
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("PaLI-GEMMA Response")
                    st.write(message["content"]["PaLI-GEMMA"]["response"])
                
                with col2:
                    st.subheader("DeepSeek Response")
                    st.write(message["content"]["DeepSeek"]["response"])

        # Chat input
        if prompt := st.chat_input("What would you like to know about the image?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display responses
            with st.spinner("Both models are analyzing the image..."):
                results = analyze_image_parallel(
                    image, 
                    pali_pipe, 
                    deepseek_pipe, 
                    prompt
                )
                
                # Add responses to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": results
                })
                
                # Display responses in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("PaLI-GEMMA Response")
                    st.write(results["PaLI-GEMMA"]["response"])
                
                with col2:
                    st.subheader("DeepSeek Response")
                    st.write(results["DeepSeek"]["response"])

    else:
        # Display placeholder or instructions
        st.info("👆 Upload an image to get started!")

if __name__ == "__main__":
    main()
