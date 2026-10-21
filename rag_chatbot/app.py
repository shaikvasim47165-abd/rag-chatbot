"""Streamlit entry point for the organized package."""

import streamlit as st

from .ui import chatbot


def run() -> None:
    """Start the organized chatbot implementation."""
    st.set_page_config(page_title="Chat With Your Data")
    st.title("🤖 RAG chatbot")
    st.session_state.openai_api_key = ""
    st.session_state.google_api_key = ""
    st.session_state.cohere_api_key = ""
    st.session_state.hf_api_key = ""
    chatbot()


if __name__ == "__main__":
    run()
