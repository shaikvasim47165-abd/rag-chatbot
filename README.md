# RAG  Chatbot with LangChain

A Streamlit-based retrieval-augmented generation chatbot that lets you upload documents, build a Chroma vector store, and chat with your data using multiple LLM providers.

## Features

- Upload PDF, TXT, CSV, and DOCX files
- Build and persist a Chroma vector store locally
- Chat with your documents using:
  - OpenAI
  - Google Generative AI
  - HuggingFace
- Choose between retriever strategies:
  - Cohere reranker
  - Contextual compression
  - Vectorstore-backed retrieval
- Keep conversational context with LangChain memory

## Tech Stack

- Streamlit
- LangChain
- Chroma
- OpenAI / Google Generative AI / HuggingFace integrations
- Cohere reranking

## Project Structure

- `run_app.py` - Streamlit application entry point
- `rag_chatbot/app.py` - application startup and page configuration
- `rag_chatbot/ui.py` - Streamlit layout, controls, and chat interface
- `rag_chatbot/retrieval.py` - document loading, chunking, embeddings, vector stores, and retrievers
- `rag_chatbot/chains.py` - prompts, conversation memory, and LLM chains
- `rag_chatbot/integrations.py` - LangChain provider and optional integration imports
- `rag_chatbot/config.py` - application paths, provider options, and UI metadata
- `rag_chatbot/dependencies.py` - dependency availability checks
- `requirements.txt` - Python dependencies
- `data/docs/` - project assets and screenshots
- `data/vector_stores/` - locally saved Chroma stores

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys

Set the provider keys in the Streamlit sidebar when you run the app.

### 4. Run the app

```bash
streamlit run run_app.py
```


## How It Works

1. Upload documents.
2. Split the content into chunks.
3. Create embeddings and store them in Chroma.
4. Use the selected retriever to fetch relevant context.
5. Ask questions in the chat interface.

## Notes

- The app stores vector data locally under `data/vector_stores/`.
- Generated cache files and compiled Python files are intentionally ignored from version control.
- Model options and provider metadata are managed inside `rag_chatbot/config.py`.

## Screenshots

You can find example screenshots in `data/docs/screenshots/`.

## License

No license has been specified yet.
