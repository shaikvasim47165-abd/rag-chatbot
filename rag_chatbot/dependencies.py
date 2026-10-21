"""Optional dependency loading and actionable dependency checks."""

from importlib.util import find_spec


REQUIRED_RAG_PACKAGES = (
    "langchain_community",
    "langchain_text_splitters",
    "langchain_classic",
    "chromadb",
)


def missing_rag_packages() -> list[str]:
    """Return installed-package names that are required for RAG operations."""
    return [package for package in REQUIRED_RAG_PACKAGES if find_spec(package) is None]


def require_rag_dependencies() -> None:
    """Raise an actionable error if the RAG integrations are unavailable."""
    missing = missing_rag_packages()
    if missing:
        raise RuntimeError(
            "RAG dependencies are missing: "
            + ", ".join(missing)
            + ". Install the project dependencies, then restart Streamlit."
        )
