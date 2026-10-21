"""Application configuration and provider metadata."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TMP_DIR = DATA_DIR / "tmp"
LOCAL_VECTOR_STORE_DIR = DATA_DIR / "vector_stores"

LIST_LLM_PROVIDERS = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    "**HuggingFace**",
]

LIST_RETRIEVER_TYPES = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

WELCOME_MESSAGE = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

OPENAI_API_KEYS_URL = "https://platform.openai.com/docs/quickstart/make-your-first-api-request"
GOOGLE_API_KEYS_URL = "https://aistudio.google.com/"
OPENAI_MODEL_OPTIONS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
GOOGLE_MODEL_OPTIONS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
HF_MODEL_OPTIONS = ["mistralai/Mistral-7B-Instruct-v0.3"]
COHERE_RERANK_MODEL = "rerank-v4.0-fast"

MODEL_OPTIONS_BY_PROVIDER = {
    "OpenAI": OPENAI_MODEL_OPTIONS,
    "Google": GOOGLE_MODEL_OPTIONS,
    "HuggingFace": HF_MODEL_OPTIONS,
}

API_KEY_PROMPTS_BY_PROVIDER = {
    "OpenAI": f"OpenAI API Key - [Get an API key]({OPENAI_API_KEYS_URL})",
    "Google": f"Google API Key - [Get an API key]({GOOGLE_API_KEYS_URL})",
    "HuggingFace": "HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
}
