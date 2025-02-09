# Dual Model AI Image Chatbot

A Streamlit-based chatbot that processes images using two state-of-the-art AI models (PaLI-GEMMA and DeepSeek) and provides comparative responses with text-to-speech capabilities.

## Features

- 🖼️ Image understanding using two different models
- 💬 Natural language interaction
- 🔊 Text-to-speech response generation
- 📊 Side-by-side model comparison
- 🚀 GPU acceleration support
- 🔒 Secure token handling

## Models Used

1. **PaLI-GEMMA (3B)** - `google/paligemma-3b-pt-224`
   - Advanced vision-language model from Google
   - Optimized for detailed image understanding

2. **DeepSeek VL (1.3B)** - `deepseek-ai/deepseek-vl-1.3b-base`
   - Efficient vision-language model
   - Balanced performance and resource usage

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dual-model-chatbot.git
cd dual-model-chatbot
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

Or manually set up the environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/huggingface_setup.py
```

3. Create a `.env` file:
```bash
cp .env.example .env
```
Edit `.env` and add your Hugging Face token:
```
HF_TOKEN=your_token_here
```

## Usage

1. Start the application:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Upload an image and start chatting!

## Project Structure

```
.
├── src/
│   ├── app.py                # Main Streamlit application
│   ├── huggingface_setup.py  # Hugging Face authentication setup
│   └── utils/
│       └── model_utils.py    # Model utility functions
├── requirements.txt          # Python dependencies
├── setup.sh                 # Setup script
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for providing the model hosting and APIs
- Streamlit for the wonderful web framework
- Google and DeepSeek for the amazing models

## Support

For support, please open an issue in the GitHub repository or contact [your-email].
