"""Optional LangChain integrations used by the application."""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
except ModuleNotFoundError:
    ConversationalRetrievalChain = None
    ConversationBufferMemory = None
    ConversationSummaryBufferMemory = None

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, CSVLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_classic.retrievers.document_compressors import (
        DocumentCompressorPipeline,
        EmbeddingsFilter,
        LLMChainExtractor,
    )
    from langchain_cohere import CohereRerank
    from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from langchain_community.llms import HuggingFaceHub
except ModuleNotFoundError:
    PyPDFLoader = TextLoader = DirectoryLoader = CSVLoader = Docx2txtLoader = None
    RecursiveCharacterTextSplitter = CharacterTextSplitter = None
    Chroma = DocumentCompressorPipeline = EmbeddingsFilter = CohereRerank = None
    LLMChainExtractor = None
    EmbeddingsRedundantFilter = LongContextReorder = None
    ContextualCompressionRetriever = None
    HuggingFaceInferenceAPIEmbeddings = HuggingFaceHub = None
