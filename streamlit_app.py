import streamlit as st
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    MultiModalityCausalLM,
    AutoTokenizer,
    pipeline
)
from huggingface_hub import login, HfFolder
from PIL import Image
import torch
from gtts import gTTS
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
from pathlib import Path
import sys

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.model_utils import (
    verify_token,
    get_device,
    load_model_config,
    prepare_image,
    format_model_output,
    handle_model_error,
    cleanup_temp_files
)

# Suppress warnings
warnings.filterwarnings('ignore')

class DualModelImageChatbot:
    def __init__(self):
        """Initialize the chatbot with both models"""
        try:
            # Setup Hugging Face token
            self.hf_token = os.getenv("HF_TOKEN", "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")
            if not verify_token(self.hf_token):
                raise ValueError("Invalid Hugging Face token")

            # Load configuration
            self.config = load_model_config()
            
            # Initialize device
            self.device = get_device()
            st.sidebar.info(f"Using device: {self.device}")

            # Initialize PaLI-GEMMA model
            st.sidebar.text("Loading PaLI-GEMMA model...")
            self.pali_processor = AutoProcessor.from_pretrained(
                self.config["pali_model"],
                token=self.hf_token
            )
            self.pali_model = AutoModelForImageTextToText.from_pretrained(
                self.config["pali_model"],
                token=self.hf_token
            ).to(self.device)

            # Initialize DeepSeek model
            st.sidebar.text("Loading DeepSeek model...")
            self.deepseek_model = MultiModalityCausalLM.from_pretrained(
                self.config["deepseek_model"],
                token=self.hf_token
            ).to(self.device)
            self.deepseek_tokenizer = AutoTokenizer.from_pretrained(
                self.config["deepseek_model"],
                token=self.hf_token
            )

            # Initialize session state
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            st.sidebar.success("✅ Models loaded successfully!")
        
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.stop()

    def generate_pali_response(self, image: Image.Image, prompt: str) -> str:
        """Generate response using PaLI-GEMMA model"""
        try:
            # Prepare image
            image = prepare_image(image)
            
            # Generate response
            inputs = self.pali_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.pali_model.generate(
                **inputs,
                max_length=self.config["max_length"],
                num_beams=self.config["num_beams"]
            )
            
            response = self.pali_processor.decode(outputs[0], skip_special_tokens=True)
            return format_model_output(response)
        
        except Exception as e:
            return handle_model_error(e, "PaLI-GEMMA")

    def generate_deepseek_response(self, image: Image.Image, prompt: str) -> str:
        """Generate response using DeepSeek model"""
        try:
            # Prepare image and input
            image = prepare_image(image)
            inputs = self.deepseek_model.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            outputs = self.deepseek_model.generate(
                **inputs,
                max_length=self.config["max_length"],
                num_beams=self.config["num_beams"]
            )
            
            response = self.deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return format_model_output(response)
        
        except Exception as e:
            return handle_model_error(e, "DeepSeek")

    def generate_parallel_responses(self, image: Image.Image, prompt: str) -> dict:
        """Generate responses from both models in parallel"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            pali_future = executor.submit(self.generate_pali_response, image, prompt)
            deepseek_future = executor.submit(self.generate_deepseek_response, image, prompt)
            
            responses = {
                "PaLI-GEMMA": pali_future.result(),
                "DeepSeek": deepseek_future.result()
            }
            
        return responses

    def text_to_speech(self, text: str) -> str:
        """Convert text response to speech"""
        try:
            if not os.getenv("ENABLE_TTS", "true").lower() == "true":
                return None
                
            tts = gTTS(
                text=text,
                lang=os.getenv("TTS_LANGUAGE", "en")
            )
            audio_file = f"response_{hash(text)}.mp3"
            tts.save(audio_file)
            return audio_file
        
        except Exception as e:
            st.warning(f"Error generating audio: {str(e)}")
            return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Dual AI Image Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Display header
    st.title("🤖 Dual AI Image Chatbot")
    st.write("Compare responses from PaLI-GEMMA and DeepSeek models!")

    # Model information in sidebar
    with st.sidebar:
        st.header("Model Information")
        st.markdown("""
        ### Models Used:
        1. **PaLI-GEMMA (3B)**
           - Vision-language model from Google
           - Optimized for image understanding
        
        2. **DeepSeek VL (1.3B)**
           - Efficient vision-language model
           - Balanced performance
        """)

    # Initialize chatbot
    try:
        chatbot = DualModelImageChatbot()
    except Exception as e:
        st.error("Failed to initialize chatbot. Please check your Hugging Face token.")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"],
        help="Upload an image to analyze"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("PaLI-GEMMA Response")
                    st.write(message["content"]["PaLI-GEMMA"])
                    if "audio" in message and message["audio"].get("PaLI-GEMMA"):
                        st.audio(message["audio"]["PaLI-GEMMA"])
                
                with col2:
                    st.subheader("DeepSeek Response")
                    st.write(message["content"]["DeepSeek"])
                    if "audio" in message and message["audio"].get("DeepSeek"):
                        st.audio(message["audio"]["DeepSeek"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the image?"):
        if not uploaded_file:
            st.error("Please upload an image first!")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate responses
        with st.spinner("🤖 Analyzing image and generating responses..."):
            responses = chatbot.generate_parallel_responses(image, prompt)
            
            # Generate audio for responses
            audio_files = {
                "PaLI-GEMMA": chatbot.text_to_speech(responses["PaLI-GEMMA"]),
                "DeepSeek": chatbot.text_to_speech(responses["DeepSeek"])
            }
            
            # Add responses to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": responses,
                "audio": audio_files
            })
        
        # Cleanup old audio files
        cleanup_temp_files("response_*.mp3")
        
        # Force refresh
        st.rerun()

if __name__ == "__main__":
    main()
