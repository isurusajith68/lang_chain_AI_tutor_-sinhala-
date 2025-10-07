# Sinhala AI Tutor 🤖

A Retrieval-Augmented Generation (RAG) chatbot designed to teach Artificial Intelligence concepts in Sinhala language. This project combines modern NLP techniques with educational content to provide an interactive learning experience.

## 🌟 Features

- **Sinhala Language Support**: Full support for Sinhala text processing and responses
- **Advanced RAG System**: Uses FAISS vectorstore with MMR retrieval and cross-encoder re-ranking
- **Token-Aware Chunking**: Intelligent text splitting optimized for multilingual content
- **Web Interface**: Clean, responsive HTML/CSS/JavaScript client
- **REST API**: Flask-based API for easy integration
- **Educational Content**: Structured lessons covering AI, Machine Learning, and Deep Learning

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GROQ API key
- HuggingFace API token

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/isurusajith68/lang_chain_AI_tutor_-sinhala-.git
   cd lang_chain_AI_tutor_-sinhala-
   ```

2. **Create virtual environment**

   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   # source myenv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirments.txt
   ```

4. **Set up environment variables**

   - Copy `.env.example` to `.env` (if exists) or create `.env`
   - Add your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
     ```

5. **Build the vectorstore** (optional - runs automatically on first API call)
   ```bash
   python -c "from rag-tutor-chatbot import *; build_vectorstore()"
   ```
   Or run the Jupyter notebook `rag-tutor-chatbot.ipynb`

### Usage

#### Web Interface

1. Start the Flask API:
   ```bash
   python app.py
   ```
2. Open `client/index.html` in your web browser
3. Start chatting with the AI tutor!

#### API Usage

The Flask app runs on `http://localhost:5000` with CORS enabled.

**Endpoint:** `POST /ask`
**Request:**

```json
{
  "question": "AI යනු කුමක්ද?"
}
```

**Response:**

```json
{
  "answer": "කෘතිම බුද්ධිකත්වය (AI) යනු මිනිස් මනසෙහි බුද්ධික ක්‍රියාකාරකම් අනුකරණය කිරීමයි...",
  "sources": ["lesson1.txt", "lesson2.txt"]
}
```

#### Jupyter Notebook

Run `rag-tutor-chatbot.ipynb` for interactive development and testing of the RAG system.

## 📚 Content Structure

- `docs/`: Educational content in Sinhala
  - `lesson1.txt`: Introduction to AI
  - `lesson2.txt`: Machine Learning basics
  - `lesson3.txt`: Deep Learning concepts
- `client/`: Web interface files
- `vectorstore/`: FAISS index and embeddings
- `app.py`: Flask API server
- `rag-tutor-chatbot.ipynb`: Development notebook

## 🛠️ Technologies Used

- **LangChain**: RAG framework and LLM integration
- **Groq API**: Gemma2-9b-it model for Sinhala text generation
- **HuggingFace Transformers**: Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **Sentence Transformers**: Cross-encoder for re-ranking (ms-marco-MiniLM-L-6-v2)
- **FAISS**: Efficient vector similarity search
- **Flask**: REST API with CORS support
- **Tiktoken**: Token counting for chunking

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token

### Chunking Parameters

- Max tokens per chunk: 750
- Overlap tokens: 120
- MMR lambda: 0.5

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- Built with LangChain and modern NLP techniques
- Sinhala language support for educational accessibility
- Inspired by the need for localized AI education tools

---

**Note:** This is an educational project demonstrating RAG techniques. For production use, consider additional security measures and error handling.
