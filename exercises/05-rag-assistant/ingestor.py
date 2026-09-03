import argparse
import glob
import logging
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def len_token(text: str, tokenizer: PreTrainedTokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_documents(source_dir: str) -> list:
    """Load every .md file under source_dir, tagging each with doc_type
    from its parent folder name — mirrors the tutor's day1.ipynb loader."""
    documents = []
    subfolders = glob.glob(f"{source_dir}/*")

    for folder in subfolders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
        documents.extend(folder_docs)

    logger.info(
        f"Loaded {len(documents)} documents across {len(subfolders)} doc_types: "
        f"{[os.path.basename(f) for f in subfolders]}"
    )

    return documents


def chunk_documents(documents: list, chunk_size: int, chunk_overlap: int, tokenizer: PreTrainedTokenizer) -> list:
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len_token(text, tokenizer)
    )
    chunks = splitter.split_documents(documents)

    logger.info(
        f"Split documents into {len(chunks)} chunks (size={chunk_size} tokens, overlap={chunk_overlap} tokens)"
    )

    return chunks


def build_vectorstore(chunks: list, persist_directory: str, embedding_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Wipe any existing collection at this path so we don't mix embedding spaces.
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        logger.info(f"Removed existing vector store at {persist_directory}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    logger.info(f"Persisted {len(chunks)} chunks to '{persist_directory}' using {embedding_model}")

    count = vectorstore._collection.count()
    if count > 0:
        sample_data = vectorstore._collection.get(limit=1, include=["embeddings"])
        embeddings_data = sample_data.get("embeddings") if sample_data else None

        if embeddings_data is not None and len(embeddings_data) > 0:
            dimensions = len(embeddings_data[0])
            logger.info(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
        else:
            logger.info(f"There are {count:,} vectors, but could not retrieve sample embeddings")
    else:
        logger.info("vector store is empty")

    return vectorstore


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=os.path.join(script_dir, "knowledge-base"),
        help="Path to the knowledge-base folder (subfolders = doc_type)"
    )
    parser.add_argument(
        "--persist",
        default=os.path.join(script_dir, "vector_db"),
        help="Where to write the Chroma collection"
    )

    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size = 450  # token
    chunk_overlap = 100  # token

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(embedding_model)

    documents = load_documents(args.source)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer
    )
    build_vectorstore(chunks, args.persist, embedding_model)


if __name__ == "__main__":
    main()
