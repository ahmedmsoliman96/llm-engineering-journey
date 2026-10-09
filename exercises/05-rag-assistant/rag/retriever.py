import os
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str) -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=model_name)


def _build_bm25(chunks: list[Document], k: int) -> BM25Retriever:
    """Builds and configures a BM25 sparse retriever from document chunks.

    Args:
        chunks (list[Document]): The collection of document chunks to index.
        k (int): The number of top documents to retrieve during searches.

    Returns:
        BM25Retriever: An initialized BM25 retriever instance set with the given `k` value.
    """
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
    bm25.k = k
    return bm25


def get_retriever(chunks: list[Document], vectorstore: Chroma, use_hybrid: bool = True,
    rerank_method: str = "cross_encoder",  candidate_k: int = 20, final_k: int = 10) -> Runnable[str, list[Document]]:
    """Builds a flexible hybrid or dense retriever configuration with optional reranking.

    This function constructs a base retrieval pipeline (either dense-only or an ensemble
    combining BM25 sparse retrieval and dense vector search via Reciprocal Rank Fusion),
    and applies an optional post-processing step like cross-encoder reranking.

    Args:
        chunks (list[Document]): The raw document chunks required to build the BM25 sparse index.
        vectorstore (Chroma): The vector database used to instantiate the dense retriever.
        use_hybrid (bool): Whether to combine dense retrieval with BM25 sparse retrieval.
            Defaults to True.
        rerank_method (str): The ranking strategy for processing final candidates.
            Options include:
            - `"none"`: Truncates the fused or dense candidate list to `final_k`.
            - `"cross_encoder"`: Applies a cross-encoder model to rerank and filter down to `final_k`.
            Defaults to `"cross_encoder"`.
        candidate_k (int): The number of initial candidates fetched by the dense/bm25 retriever(s).
            Defaults to 20.
        final_k (int): The target number of documents returned.
            Defaults to 10.

    Returns:
        Runnable: A configured retriever or runnable pipeline capable of fetching and refining documents.

    Raises:
        ValueError: If an unsupported `rerank_method` string is provided.
    """
    valid_methods = {"none", "cross_encoder"}
    if rerank_method not in valid_methods:
        raise ValueError(
            f"Unknown rerank_method: {rerank_method!r}. Expected one of {valid_methods}"
        )

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": candidate_k})

    if use_hybrid:
        bm25_retriever = _build_bm25(chunks, k=candidate_k)
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5],
        )
    else:
        base_retriever = dense_retriever

    if rerank_method == "none":
        return base_retriever | RunnableLambda(lambda docs: docs[:final_k])

    cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    cross_encoder = _get_cross_encoder(model_name=cross_encoder_model)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=final_k)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

