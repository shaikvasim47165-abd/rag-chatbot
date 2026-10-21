"""Organized entry points for the RAG chatbot."""

__all__ = ["run"]


def run() -> None:
    """Run the Streamlit chatbot."""
    from .app import run

    run()
