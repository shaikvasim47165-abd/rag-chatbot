"""Streamlit controls and chatbot presentation."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import streamlit as st

from .config import (
    API_KEY_PROMPTS_BY_PROVIDER,
    LIST_LLM_PROVIDERS,
    LIST_RETRIEVER_TYPES,
    MODEL_OPTIONS_BY_PROVIDER,
    WELCOME_MESSAGE,
)
from .chains import clear_chat_history, get_response_from_LLM
from .retrieval import initialize_retrieval_chain, select_embeddings_model, chain_RAG_blocks
from .integrations import Chroma


def expander_model_parameters(LLM_provider="OpenAI", text_input_API_key=None, list_models=None):
    st.session_state.LLM_provider = LLM_provider

    text_input_API_key = text_input_API_key or API_KEY_PROMPTS_BY_PROVIDER[LLM_provider]

    list_models = list_models or MODEL_OPTIONS_BY_PROVIDER[LLM_provider]

    key_name = {
        "OpenAI": "openai_api_key",
        "Google": "google_api_key",
        "HuggingFace": "hf_api_key",
    }[LLM_provider]

    st.session_state[key_name] = st.text_input(text_input_API_key, type="password", placeholder="insert your API key")

    for other_key in ("openai_api_key", "google_api_key", "hf_api_key"):
        if other_key != key_name:
            st.session_state[other_key] = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(f"Choose {LLM_provider} model", list_models)
        st.session_state.temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        st.session_state.top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.05)


def sidebar_and_documentChooser():
    with st.sidebar:
        llm_chooser = st.radio("Select provider", LIST_LLM_PROVIDERS)

        st.divider()

        expander_model_parameters(["OpenAI", "Google", "HuggingFace"][LIST_LLM_PROVIDERS.index(llm_chooser)])

        st.write("")

        st.session_state.assistant_language = st.selectbox("Assistant language", list(WELCOME_MESSAGE.keys()))

        st.divider()
        st.subheader("Retrievers")

        retrievers = LIST_RETRIEVER_TYPES

        if (
            st.session_state.LLM_provider == "OpenAI"
            and st.session_state.selected_model.startswith("gpt-3.5")
        ):
            retrievers = LIST_RETRIEVER_TYPES[:-1]
        st.session_state.retriever_type = st.selectbox("Select retriever type", retrievers)

        st.write("")

        if st.session_state.retriever_type == LIST_RETRIEVER_TYPES[0]:
            st.session_state.cohere_api_key = st.text_input("Coher API Key - [Get an API key](https://dashboard.cohere.com/api-keys)", type="password", placeholder="insert your API key")

        st.write("\n\n")

        st.write(f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._")

    tab_new, tab_open = st.tabs(["Create a new Vectorstore", "Open a saved Vectorstore"])

    with tab_new:
        st.session_state.uploaded_file_list = st.file_uploader("**Select documents**", accept_multiple_files=True, type=["pdf", "txt", "docx", "csv"])
        st.session_state.vector_store_name = st.text_input("**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**", placeholder="Vectorstore name")
        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "": st.warning(st.session_state.error_message)
        except Exception: pass

    with tab_open:
        st.write("Please select a Vectorstore:")

        clicked = st.button("Vectorstore chooser")
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        st.session_state.selected_vectorstore_name = ""
        if clicked:
            errors = []
            if not st.session_state.openai_api_key and not st.session_state.google_api_key and not st.session_state.hf_api_key: errors.append(f"insert your {st.session_state.LLM_provider} API key")
            if st.session_state.retriever_type == LIST_RETRIEVER_TYPES[0] and not st.session_state.cohere_api_key: errors.append("insert your Cohere API key")
            if errors:
                st.session_state.error_message = "Please " + (errors[0] + "." if len(errors) == 1 else ", ".join(errors[:-1]) + ", and " + errors[-1] + ".")
                st.warning(st.session_state.error_message)
            else:
                selected_path = filedialog.askdirectory(master=root)
                if selected_path == "":
                    st.info("Please select a valid path.")
                else:
                    with st.spinner("Loading vectorstore..."):
                        try:
                            embeddings = select_embeddings_model()
                            st.session_state.vector_store = Chroma(embedding_function=embeddings, persist_directory=selected_path)
                            st.session_state.selected_vectorstore_name = Path(selected_path).name
                            initialize_retrieval_chain(embeddings)
                            st.info(f"**{st.session_state.selected_vectorstore_name}** is loaded successfully.")
                        except Exception as error:
                            st.error(error)


def chatbot():

    sidebar_and_documentChooser()
    st.divider()

    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE[st.session_state.assistant_language]}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not st.session_state.openai_api_key and not st.session_state.google_api_key and not st.session_state.hf_api_key:
            st.info(f"Please insert your {st.session_state.LLM_provider} API key to continue.")
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)
