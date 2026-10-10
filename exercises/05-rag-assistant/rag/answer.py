import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import litellm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

load_dotenv()

logger = logging.getLogger(__name__)

REWRITE_MODEL = os.getenv("REWRITE_MODEL", "ollama_chat/qwen3.5:4b")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "ollama_chat/qwen3.5:4b")

RAG_SYSTEM_PROMPT = """You are a precise, direct factual assistant. Answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. If the answer is not in the context, output ONLY: "I do not have that information"
    Do not guess, apologize, or use outside knowledge.
2. Give a direct answer immediately in a complete sentence. NEVER use conversational filler, meta-commentary,
    or phrases like "Based on the context...", "According to the text...", or "Here is the answer:".
3. Exclude background details or extra information from the context unless it directly answers the question.
4. Be concise and fully answer the question without padding.

Context:
{context}
"""


REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the
follow-up into a standalone refined short question that contains all necessary context.
If the follow-up question is already standalone, return it unchanged. Return ONLY the
rewritten question that will be used to search the knowledge base.

Conversation history:
{history}

Follow-up question: {question}
"""


@dataclass(frozen=True, slots=True)
class CallLog:
    """Performance and usage metrics for an LLM API call.

    Attributes:
        stage (str): Pipeline stage that made the call, e.g. ``"rewrite"`` or ``"generate"``.
        model (str): LiteLLM model identifier used for the call.
        seconds (float): Wall-clock duration of the call, including retries.
        cost_usd (float | None): Estimated cost in USD, or ``None`` when LiteLLM
            has no pricing data for the model (e.g. local Ollama models).
        prompt_tokens (int | None): Input tokens, or ``None`` if usage wasn't reported.
        completion_tokens (int | None): Output tokens, or ``None`` if usage wasn't reported.
    """
    stage: str
    model: str
    seconds: float
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class AnswerResult:
    """Final output of one RAG turn, with telemetry.

    Attributes:
        answer (str): The generated answer (or the abstention message).
        rewritten_question (str): The standalone query used for retrieval, or
            the original question when rewriting was skipped.
        retrieved_chunks (list[Document]): Chunks given to the generator, in rank order.
        call_logs (list[CallLog]): One entry per LLM call (rewrite and/or generate).
    """
    answer: str
    rewritten_question: str
    retrieved_chunks: list[Document]
    call_logs: list[CallLog] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        """Float: Total LLM time across all calls."""
        return sum(c.seconds for c in self.call_logs)

    @property
    def total_tokens(self) -> int:
        """Float: Total LLM (input/output) tokens across all calls."""
        return sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in self.call_logs)

    @property
    def total_cost_usd(self) -> float:
        """Float: Total LLM cost in USD across all calls."""
        return sum(c.cost_usd or 0.0 for c in self.call_logs)


def preview(text: str, limit: int = 80) -> str:
    """Collapse whitespace and truncate text so it fits on one log line.

    Args:
        text (str): The input text to format.
        limit (int): Maximum character limit before truncation. Defaults to 80.

    Returns:
        str: The collapsed and truncated preview string.
    """
    text = " ".join(text.split())

    return text if len(text) <= limit else text[:limit - 1] + "…"


def _log_call(log: CallLog) -> None:
    """Log a CallLog instance as a single formatted console line.

    Args:
        log (CallLog): The call performance record to log.
    """
    cost = "n/a" if log.cost_usd is None else f"${log.cost_usd:.5f}"
    logger.info(
        "[%s] done | model=%s | %.2fs | tokens in=%s out=%s | cost=%s",
        log.stage, log.model, log.seconds, log.prompt_tokens, log.completion_tokens, cost,
    )

def _timed_completion(model: str, messages: list[dict], stage: str) -> tuple:
    """Executes a litellm completion call while measuring elapsed time, token usage, and cost.

    Args:
        model (str): The LLM model to call.
        messages (list[dict]): The chat message payload.
        stage (str): Label identifying the pipeline stage (e.g., 'rewrite', 'generate').

    Returns:
        tuple: A tuple containing (raw completion response object, populated CallLog).

    Raises:
        litellm.APIConnectionError: If the model server is unreachable.
        litellm.RateLimitError: If the rate limit persists after ``num_retries`` retries.
        litellm.AuthenticationError: If the API key is missing or invalid.
        litellm.BadRequestError: If the request is rejected (e.g. context window exceeded).
        litellm.NotFoundError: If the model name doesn't exist.
        Exception: Any other error from ``litellm.completion`` is logged and re-raised unchanged.
    """
    extra = {"num_ctx": 8192} if model.startswith("ollama") else {}
    logger.info("[%s] start | model=%s", stage, model)

    start = time.perf_counter()
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            num_retries=4,
            **extra,
        )
    except Exception as e:
        logger.error(
            "[%s] FAILED after %.2fs (includes retries) | model=%s | %s: %s",
            stage, time.perf_counter() - start, model, type(e).__name__, e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        raise

    elapsed = time.perf_counter() - start

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None  # not every provider/model has cost data in litellm's map

    usage = getattr(response, "usage", None)
    log = CallLog(
        stage=stage,
        model=model,
        seconds=elapsed,
        cost_usd=cost,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
    _log_call(log)

    return response, log


def rewrite_query(question: str, history: list[dict], rewrite_model: str = REWRITE_MODEL, enabled: bool = True) -> tuple:
    """Rewrites a conversational follow-up question into a self-contained search query.

    Args:
        question (str): The latest user follow-up question.
        history (list[dict]): Prior conversation turns.
        rewrite_model (str): The LLM model used for query reformulation.
            Defaults to REWRITE_MODEL.
        enabled (bool): Whether rewriting is active.
            Defaults to True.

    Returns:
        tuple: A tuple containing (rewritten question, CallLog) or (original question, None).
    """
    if not enabled or not history:
        logger.info("[rewrite] skipped | enabled=%s | history_turns=%d", enabled, len(history))
        return question, None
    history_str = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

    prompt = REWRITE_PROMPT.format(history=history_str, question=question)

    response, log = _timed_completion(
        rewrite_model, [{"role": "user", "content": prompt}], stage="rewrite"
    )

    rewritten = (response.choices[0].message.content or "").strip()
    if not rewritten:
        logger.warning("[rewrite] model returned empty output — using the original question")
        rewritten = question

    logger.info("[rewrite] %r -> %r", preview(question), preview(rewritten))
    return rewritten, log



def generate_answer(question: str, context_chunks: list[Document]) -> tuple:
    """Generates an answer using retrieved knowledge base context.

    Args:
        question (str): The standalone question (or rewritten query).
        context_chunks (list[Document]): Document chunks retrieved from the vector store.

    Returns:
        tuple: A tuple containing (generated answer, CallLog).
    """
    context = "\n\n---\n\n".join(
        f"[{c.metadata.get('doc_type', 'unknown')}] {c.page_content}" for c in context_chunks
    )
    logger.info("[generate] context | chunks=%d | chars=%d", len(context_chunks), len(context))
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": f"Question: {question}"},
    ]
    response, log = _timed_completion(GENERATION_MODEL, messages, stage="generate")
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        logger.warning("[generate] empty answer from %s",GENERATION_MODEL)

    logger.info("[generate] answer | answer=%s | chars=%d", preview(answer,300) , len(answer))

    return answer, log


def ask(retriever: Runnable[str,list[Document]], question: str, history: list[dict] | None = None,
        rewrite_enabled: bool = True) -> AnswerResult:
    """Executes the full end-to-end rag pipeline turn: rewrite -> retrieve -> generate.

    Args:
        retriever (Runnable[str, list[Document]]): The configured document retriever instance.
        question (str): The user's input question.
        history (list[dict] | None): Prior chat history.
            Defaults to None.
        rewrite_enabled (bool): Flag to toggle conversational query rewriting.
            Defaults to True.

    Returns:
        AnswerResult: Comprehensive result object containing the answer, rewritten question, retrieved chunks, and metrics.
    """
    history = history or []
    call_logs = []
    ask_start = time.perf_counter()

    logger.info("[ask] start | question=%r | history_turns=%d", preview(question), len(history))

    rewritten, rewrite_log = rewrite_query(question, history, enabled=rewrite_enabled)
    if rewrite_log:
        call_logs.append(rewrite_log)

    retrieve_start = time.perf_counter()
    try:
        retrieved = retriever.invoke(rewritten)
    except Exception as e:
        logger.error(
            "[retrieve] FAILED after %.2fs | %s: %s",
            time.perf_counter() - retrieve_start, type(e).__name__, e,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        raise

    if not retrieved:
        logger.warning("[retrieve] no chunks retrieved for this query")

    logger.info(
        "[retrieve] done | %d chunks | %.2fs | doc_types=%s",
        len(retrieved),
        time.perf_counter() - retrieve_start,
        dict(Counter(c.metadata.get("doc_type", "unknown") for c in retrieved)),
    )

    answer_text, gen_log = generate_answer(rewritten, retrieved)

    call_logs.append(gen_log)

    result = AnswerResult(
        answer=answer_text,
        rewritten_question=rewritten,
        retrieved_chunks=retrieved,
        call_logs=call_logs,
    )

    logger.info(
        "[ask] done | wall=%.2fs | tokens=%d | cost=$%.5f",
        time.perf_counter() - ask_start, result.total_tokens, result.total_cost_usd,
    )

    return result
