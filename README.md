# DeepSeek Document Chatbot

A document-based chatbot system using DeepSeek LLM for intelligent question answering over your knowledge base.

## Features

- 📄 **Document Processing**: Automatically process and index your document collection
- 🧠 **DeepSeek LLM Integration**: Leverage powerful language models for high-quality responses
- 💬 **Conversational Memory**: Maintain context across multiple user interactions
- 🎯 **Customizable Prompts**: Tailor the system to different use cases and domains
- 🌐 **Web Interface**: User-friendly interface for easy interaction
- 📊 **Monitoring & Evaluation**: Comprehensive tools for performance tracking

## Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/Egberun/deepseek-document-chatbot.git
cd deepseek-document-chatbot

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. Add your documents to the `documents` directory (text files are recommended)

2. Run the application in CLI mode:

```bash
python main.py --cli
```

3. Or use the web interface:

```bash
python advanced_web_ui.py
```

## Configuration

Create a `config.json` file to customize settings:

```json
{
  "document_dir": "./documents",
  "vector_db_path": "./chroma_db",
  "chunk_size": 1000,
  "chunk_overlap": 100,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "deepseek-ai/deepseek-llm-7b-chat",
  "use_gpu": true,
  "max_new_tokens": 512,
  "temperature": 0.7
}
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- LangChain
- Sentence Transformers
- Chroma Vector Database
- Gradio (for web UI)

## System Architecture

The DeepSeek Document Chatbot consists of three main components:

1. **Document Processor**: Handles document loading, chunking, and vector embedding
2. **DeepSeek LLM Integration**: Manages the language model for generating responses
3. **Retrieval Chain**: Connects document retrieval with the language model

## Advanced Features

- **Document Upload**: Add new documents through the web interface
- **Conversation Export**: Save conversations for later reference
- **Customizable Prompts**: Choose different prompt templates for various use cases
- **Performance Monitoring**: Track usage and response times

## License

MIT