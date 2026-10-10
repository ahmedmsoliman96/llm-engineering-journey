from pathlib import Path

import gradio as gr
import litellm
from dotenv import load_dotenv
from langchain_core.documents import Document
from rag.answer import ask
from rag.factory import build_retriever, load_active_config
from rag.logging_config import setup_logging

load_dotenv()

script_dir = Path(__file__).resolve().parent
config = load_active_config(script_dir / "active_config.json")
retriever = build_retriever(script_dir / "vector_db", config)

NO_CONTEXT = "*No context retrieved yet.*"
_last_context = NO_CONTEXT


def format_context(chunks: list[Document]) -> str:
    """Formats retrieved chunks as Markdown for the side panel.

    Args:
        chunks (list): Retrieved LangChain Document chunks, in rank order.

    Returns:
        str: One Markdown block per chunk, labeled with its rank and doc_type.
    """
    if not chunks:
        return NO_CONTEXT
    return "\n\n---\n\n".join(
        f"**#{i} · Source: {c.metadata.get('doc_type', 'unknown')}**\n\n{c.page_content}"
        for i, c in enumerate(chunks, start=1)
    )


def show_context(history: list[dict]) -> str:
    """Chooses what the context panel shows whenever the chat changes.

    Args:
        history (list[dict]): The current chat history turns.

    Returns:
        str: The Markdown-formatted context string, loading message, or default state.
    """
    if not history:
        return NO_CONTEXT                 # chat was cleared
    if history[-1].get("role") == "user":
        return "*Retrieving…*"            # answer still being generated
    return _last_context


def chat(message: str, history: list[dict]) -> str:
    """Runs one RAG turn for ChatInterface.

    The retrieved chunks are stored in `_last_context` for the side panel.

    Args:
        message (str): The user's latest message.
        history (list[dict]): Previous turns as {"role", "content"} dicts.

    Returns:
        str: The assistant's answer.

    Raises:
        gr.Error: With a readable message if the LLM call or retrieval fails.
    """
    global _last_context
    clean_history = [{"role": t["role"], "content":t["content"][0].get("text")} for t in history]

    try:
        result = ask(retriever, message, history=clean_history, rewrite_enabled=config["rewrite_enabled"])
    except litellm.APIConnectionError as e:
        raise gr.Error("Can't reach the model server.") from e
    except litellm.RateLimitError as e:
        raise gr.Error("Rate limit hit. Wait a moment and try again.") from e
    except litellm.NotFoundError as e:
        raise gr.Error("Model not found. Check GENERATION_MODEL / REWRITE_MODEL in .env.") from e
    except Exception as e:
        raise gr.Error(f"Something went wrong ({type(e).__name__}). Check the console logs.") from e

    _last_context = format_context(result.retrieved_chunks)

    return result.answer


def main():
    setup_logging()

    with gr.Blocks(title="Knowledge Worker") as UI:
        gr.Markdown("# Knowledge Worker")
        gr.Markdown(
            "Ask questions about the knowledge base. "
            f"Retrieval: {'hybrid (BM25 + dense)' if config['use_hybrid'] else 'dense only'}, "
            f"reranker: {config['rerank_method']}."
        )
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, render=False)
                gr.ChatInterface(fn=chat, chatbot=chatbot)
            with gr.Column(scale=1):
                gr.Markdown("### Retrieved context (testing)")
                context_md = gr.Markdown(NO_CONTEXT)

        chatbot.change(show_context, inputs=chatbot, outputs=context_md)

    UI.launch(inbrowser=True)

if __name__ == "__main__":
    main()