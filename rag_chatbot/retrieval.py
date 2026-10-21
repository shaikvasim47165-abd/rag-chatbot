"""Document ingestion, embeddings, vector stores, and retrievers."""

import glob
import os
import streamlit as st

from .config import COHERE_RERANK_MODEL, LOCAL_VECTOR_STORE_DIR, TMP_DIR
from .dependencies import require_rag_dependencies
from .integrations import (
    Chroma, ChatGoogleGenerativeAI, ChatOpenAI, CharacterTextSplitter,
    CohereRerank, ContextualCompressionRetriever, DirectoryLoader,
    Docx2txtLoader, DocumentCompressorPipeline, EmbeddingsFilter,
    EmbeddingsRedundantFilter, GoogleGenerativeAIEmbeddings, HuggingFaceHub,
    HuggingFaceInferenceAPIEmbeddings, LLMChainExtractor, LongContextReorder,
    OpenAIEmbeddings, PyPDFLoader, RecursiveCharacterTextSplitter, TextLoader,
    CSVLoader,
)


def delete_temp_files():
    """Delete files from the temporary upload folder."""
    for file_path in glob.glob(TMP_DIR.as_posix() + "/*"):
        try:
            os.remove(file_path)
        except Exception:
            pass


# Keep the original public spelling available to callers during migration.
delte_temp_files = delete_temp_files


def langchain_document_loader():
    documents = []
    loaders = [
        ("**/*.txt", TextLoader, {}),
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.csv", CSVLoader, {"loader_kwargs": {"encoding": "utf8"}}),
        ("**/*.docx", Docx2txtLoader, {}),
    ]
    for pattern, loader_cls, kwargs in loaders:
        loader = DirectoryLoader(
            TMP_DIR.as_posix(),
            glob=pattern,
            loader_cls=loader_cls,
            show_progress=True,
            **kwargs,
        )
        documents.extend(loader.load())
    return documents


def split_documents_to_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def select_embeddings_model():
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
    if st.session_state.LLM_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=st.session_state.google_api_key)
    if st.session_state.LLM_provider == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=st.session_state.hf_api_key, model_name="thenlper/gte-large")
    return embeddings


def Vectorstore_backed_retriever(vectorstore, search_type="similarity", k=4, score_threshold=None):
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    """Create a contextual compressor that extracts query-relevant content with an LLM."""
    if st.session_state.LLM_provider == "OpenAI":
        compression_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
    if st.session_state.LLM_provider == "Google":
        compression_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
    if st.session_state.LLM_provider == "HuggingFace":
        compression_llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )

    compressor = LLMChainExtractor.from_llm(compression_llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def CohereRerank_retriever(base_retriever, cohere_api_key, cohere_model=COHERE_RERANK_MODEL, top_n=10):
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


def create_retriever(vector_store, embeddings, retriever_type="Contextual compression", base_retriever_search_type="semilarity", base_retriever_k=16, compression_retriever_k=20, cohere_api_key="", cohere_model=COHERE_RERANK_MODEL, cohere_top_n=10):
    base_retriever = Vectorstore_backed_retriever(vector_store, base_retriever_search_type, base_retriever_k, None)
    if retriever_type == "Vectorstore backed retriever":
        return base_retriever
    if retriever_type == "Contextual compression":
        return create_compression_retriever(embeddings, base_retriever, k=compression_retriever_k)
    if retriever_type == "Cohere reranker":
        return CohereRerank_retriever(base_retriever, cohere_api_key, cohere_model, cohere_top_n)


def initialize_retrieval_chain(embeddings):
    from .chains import clear_chat_history, create_ConversationalRetrievalChain

    st.session_state.retriever = create_retriever(
        st.session_state.vector_store,
        embeddings,
        st.session_state.retriever_type,
        "similarity",
        16,
        20,
        st.session_state.cohere_api_key,
        COHERE_RERANK_MODEL,
        10,
    )
    st.session_state.chain, st.session_state.memory = (
        create_ConversationalRetrievalChain(
            st.session_state.retriever,
            "stuff",
            st.session_state.assistant_language,
        )
    )
    clear_chat_history()


def chain_RAG_blocks():
    from .config import LIST_RETRIEVER_TYPES

    with st.spinner("Creating vectorstore..."):
        try:
            require_rag_dependencies()
        except RuntimeError as error:
            st.error(str(error))
            return
        errors = []
        if not st.session_state.openai_api_key and not st.session_state.google_api_key and not st.session_state.hf_api_key:
            errors.append(f"insert your {st.session_state.LLM_provider} API key")
        if st.session_state.retriever_type == LIST_RETRIEVER_TYPES[0] and not st.session_state.cohere_api_key:
            errors.append("insert your Cohere API key")
        if not st.session_state.uploaded_file_list:
            errors.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            errors.append("provide a Vectorstore name")
        if len(errors) == 1:
            st.session_state.error_message = "Please " + errors[0] + "."
        elif len(errors) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(errors[:-1])
                + ", and "
                + errors[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                delete_temp_files()
                for uploaded_file in st.session_state.uploaded_file_list:
                    error_message = ""
                    try:
                        with open(os.path.join(TMP_DIR.as_posix(), uploaded_file.name), "wb") as temp_file:
                            temp_file.write(uploaded_file.read())
                    except Exception as error:
                        error_message += error
                if error_message != "":
                    st.warning(f"Errors: {error_message}")
                chunks = split_documents_to_chunks(langchain_document_loader())
                embeddings = select_embeddings_model()
                persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/" + st.session_state.vector_store_name
                try:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=persist_directory,
                    )
                    st.info(f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully.")
                    initialize_retrieval_chain(embeddings)
                except Exception as error:
                    st.error(error)
            except Exception as error:
                st.error(f"An error occurred: {error}")
