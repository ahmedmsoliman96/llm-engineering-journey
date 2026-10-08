import json
import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings

from .retriever import get_retriever

DEFAULT_CONFIG = {
    "use_hybrid": True,
    "rerank_method": "cross_encoder",
    "rewrite_enabled": True,
    "candidate_k": 20,
    "final_k": 10,
}


def load_active_config(active_config_path: Path) -> dict:
    """Loads the active runtime configuration from a JSON file.

    Reads the configuration file, falling back to DEFAULT_CONFIG if the file
    does not exist yet.

    Args:
        active_config_path (Path): The file path to the active configuration JSON.

    Returns:
        dict: The merged configuration dictionary combining defaults and active overrides.
    """
    active_config_path = Path(active_config_path)
    if active_config_path.exists():
        with open(active_config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {**DEFAULT_CONFIG, **payload.get("config", {})}
    return dict(DEFAULT_CONFIG)


@lru_cache(maxsize=2)
def get_cached_vectorstore(persist_dir: Path) -> Chroma:
    """Loads and caches a Chroma vector store instance from disk.

    Utilizes an LRU cache to prevent reloading embedding models and database
    connections on repetitive queries or evaluation passes.

    Args:
        persist_dir (Path): The directory path where the Chroma vector store is persisted.

    Returns:
        Chroma: An initialized Chroma vector store instance.

    Raises:
        FileNotFoundError: If the specified persistence directory does not exist.
        ValueError: If the current config embedding model does not match embedding model that built the vector store.
    """
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"No vectorstore found at {persist_dir}. Run ingestor.py first.")

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    manifest_path = persist_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        built_with = manifest.get("embedding_model")
        if built_with != embedding_model:
            raise ValueError(
                f"Embedding model mismatch! Store was built with {built_with!r}, "
                f"but current EMBEDDING_MODEL={embedding_model!r}. Re-run ingestor.py."
            )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, encode_kwargs={"normalize_embeddings": True})

    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def build_retriever(persist_dir: Path, config: dict | None = None) -> Runnable[str, list[Document]]:
    """Builds a configured retriever instance from a vector store and settings.

    Fetches the cached vector store, reconstructs LangChain Document chunks required
    for sparse retrieval (BM25), and passes them along with configuration parameters
    into the core retriever builder.

    Args:
        persist_dir (Path): The directory path to the Chroma persistence layer.
        config (dict | None): Custom configuration overrides.
            Defaults to None.

    Returns:
        Runnable: A configured retriever or runnable pipeline capable of fetching and refining documents.
    """
    persist_dir = Path(persist_dir)
    config = {**DEFAULT_CONFIG, **(config or {})}
    config["candidate_k"] = max(config["candidate_k"], config["final_k"])

    vectorstore = get_cached_vectorstore(persist_dir)
    raw_data = vectorstore.get()

    chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]

    return get_retriever(
        chunks, vectorstore,
        use_hybrid=config["use_hybrid"],
        rerank_method=config["rerank_method"],
        candidate_k=config["candidate_k"],
        final_k=config["final_k"]
    )