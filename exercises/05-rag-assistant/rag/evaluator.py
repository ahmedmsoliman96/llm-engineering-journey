import argparse
import json
import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import litellm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from .answer import GENERATION_MODEL, ask, preview
from .factory import build_retriever, load_active_config
from .logging_config import setup_logging
from .questions_loader import EvalQuestion, load_questions

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-3.5-flash-lite")

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of answers.
Evaluate the generated answer by comparing it to the reference answer.
"""
JUDGE_PROMPT = """You are grading a rag system's answer.

Question: {question}
Reference answer: {reference}
System's answer: {answer}
Is this question answerable from the knowledge base? {answerable}

Score the system's answer from 1-5 on each of:
- accuracy (does it match the reference / correctly abstain if unanswerable?)
- completeness (does it cover what the reference covers?)
- relevance (does it actually address the question, no padding/hallucination?)

If the question is unanswerable:
- If the system correctly declined to answer, set "accuracy" to 5, "abstained_correctly" to true,
and leave "completeness" and "relevance" as null.
- If the system fabricated an answer instead, set "accuracy", "completeness", and "relevance" to 1,
and "abstained_correctly" to false.

If the question is answerable:
- Score "accuracy", "completeness", and "relevance" normally from 1 to 5.
- Set "abstained_correctly" to false (since it should have answered).
"""

class AnswerEvaluation(BaseModel):
    """Model represents the judge's evaluation of a rag-generated answer."""
    accuracy: int = Field(
        ge=1,
        le=5,
        description=("How factually correct is the answer compared to the reference answer? "
                     "1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). "
                     "An acceptable answer would score 3. correct abstention would score 5")
    )
    completeness: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) "
                    "to 5 (ideal - all the information from the reference answer is provided completely). Only "
                    "answer 5 if ALL information from the reference answer is included. Leave null for correctly-abstained questions."
    )
    relevance: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) "
                    "to 5 (ideal - directly addresses question and gives no additional information). Only "
                    "answer 5 if the answer is completely relevant to the question and gives no additional information. "
                    "Leave null for correctly-abstained questions."
    )
    abstained_correctly: bool = Field(description="True if an unanswerable question was correctly declined. "
                                                  "False if an answerable question was not answered, or if "
                                                  "an answer for an unanswerable question was fabricated")


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Compiles a case-insensitive pattern that matches a keyword as a whole token.

    Lookarounds are used so keywords that start or end with symbols
    (e.g. ``$5,000``, ``10%``) still match.

    Args:
        keyword (str): The keyword to compile into a pattern.

    Returns:
        re.Pattern: The compiled regular expression pattern.
    """
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


def mrr_per_keyword(keyword: str, texts: list[str]) -> float:
    """Calculates reciprocal rank for a single keyword (case-insensitive).

    Args:
        keyword (str): The keyword to check.
        texts (list[str]): A list of retrieved text chunks.

    Returns:
        float: Reciprocal rank score or 0.0 if not found.
    """
    pattern = _keyword_pattern(keyword)
    for rank, text in enumerate(texts, start=1):
        if pattern.search(text):
            return 1.0 / rank

    return 0.0


def _dcg(relevances: list[int]) -> float:
    """Calculates Discounted Cumulative Gain for a binary relevance vector.

    Args:
        relevances (list[int]): List of binary relevance scores (0 or 1).

    Returns:
        float: The computed DCG score.
    """
    return sum(rel/math.log2(i+2) for i, rel in enumerate(relevances))


def ndcg_per_keyword(keyword: str, texts: list[str], k: int | None = None) -> float:
    """Calculates normalized Discounted Cumulative Gain (nDCG) for a single keyword.

    Args:
        keyword (str): The keyword to check.
        texts (list[str]): A list of retrieved text chunks.
        k (int | None): Top-k threshold for metrics.

    Returns:
        float: nDCG score between 0.0 and 1.0.
    """
    window = texts[:k] if k is not None else texts
    pattern = _keyword_pattern(keyword)
    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [1 if pattern.search(text) else 0 for text in window]
    dcg = _dcg(relevances)
    # Ideal DCG (best case: keyword in first position)
    idcg = _dcg(sorted(relevances, reverse=True))

    return dcg / idcg if idcg > 0 else 0.0


def retrieval_proxy_scores(retrieved_texts: list[str], keywords: list[str], k: int | None = None) -> dict:
    """Computes retrieval proxy metrics (MRR, nDCG, and keyword coverage) for a set of retrieved texts.

    Args:
        retrieved_texts (list[str]): A list of retrieved text chunks.
        keywords (list[str]): Keywords associated with the test question.
        k (int | None): Top-k threshold for metrics.

    Returns:
        dict: A dictionary containing the following keys:
            - mrr (float | None): Mean Reciprocal Rank score.
            - ndcg (float | None): Normalized Discounted Cumulative Gain score.
            - keyword_coverage (float | None): Percentage of keywords found.
            - keywords_found (int | None): Count of unique keywords found.
            - total_keywords (int | None): Total number of keywords checked.
    """
    if not keywords:
        return {
            "mrr": None,
            "ndcg": None,
            "keyword_coverage": None,
            "keywords_found": None,
            "total_keywords": None,
        }

    mrr_scores = [mrr_per_keyword(kw, retrieved_texts) for kw in keywords]
    mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    ndcg_scores = [ndcg_per_keyword(kw, retrieved_texts, k=k) for kw in keywords]
    ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    keywords_found = sum(1 for s in mrr_scores if s > 0)
    total_keywords = len(keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return {
        "mrr": mrr,
        "ndcg": ndcg,
        "keyword_coverage": keyword_coverage,
        "keywords_found": keywords_found,
        "total_keywords": total_keywords,
    }


def judge_answer(question: str, reference: str, answer_text: str,
                 answerable: bool, retries: int=1) -> AnswerEvaluation:
    """Evaluates a RAG-generated answer using an LLM judge and return structured scores.

    Args:
        question (str): The user's query.
        reference (str): Reference answer for the test question.
        answer_text (str): Answer produced by the RAG system.
        answerable (bool): Flag indicating if the question can be answered from the KB.
        retries (int): Number of parsing retry attempts if model validation fails.
            Defaults to 1.

    Returns:
        AnswerEvaluation: A Pydantic object containing structured evaluation scores.

    Raises:
        RuntimeError: If the judge model fails to return a valid AnswerEvaluation after all retry attempts.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference or "(no reference — this question is unanswerable from the KB)",
        answer=answer_text,
        answerable=answerable,
    )
    extra = {"num_ctx": 8192} if JUDGE_MODEL.startswith("ollama") else {}

    last_error = None
    for attempt in range(retries + 1):
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            response_format=AnswerEvaluation,
            num_retries=4,
            **extra,
        )
        try:
            return AnswerEvaluation.model_validate_json(response.choices[0].message.content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
    raise RuntimeError(
        f"Judge failed to return valid AnswerEval after {retries + 1} attempt(s): {last_error}"
    )


def _select_tests(tests: list[EvalQuestion], tests_limit: int | None,
                  seed: int | None = DEFAULT_SEED) -> list[EvalQuestion]:
    """Selects which test questions to run.

    With no limit, or a limit at least as large as the test set, every question
    is used. Otherwise, a random sample of `limit` questions is drawn and kept in
    the file's original order. The same seed always gives the same sample, so
    two runs can be compared on identical questions.

    Args:
        tests (list[EvalQuestion]): All loaded test questions.
        tests_limit (int | None): Number of questions to run. None or 0 means all.
        seed (int | None): Random seed. None draws a different sample each time.
            Defaults to DEFAULT_SEED.

    Returns:
        list[EvalQuestion]: The selected questions, in their original order.
    """
    if not tests_limit or tests_limit < 0 or tests_limit >= len(tests):
        return tests
    picked = sorted(random.Random(seed).sample(range(len(tests)), tests_limit))
    logger.info("[eval] sampling | %d of %d questions | seed=%s", tests_limit, len(tests), seed)

    return [tests[i] for i in picked]


def _retrieval_summary(scores: dict) -> str:
    """Formats retrieval proxy scores as a short string for log lines.

    Args:
        scores (dict): The dictionary returned by retrieval_proxy_scores.

    Returns:
        str: The scores on one line, or a note if the question has no keywords.
    """
    if scores.get("mrr") is None:
        return "no keywords (not scored)"

    return (f"mrr={scores['mrr']:.2f} ndcg={scores['ndcg']:.2f} "
            f"keyword_coverage={scores['keyword_coverage']:.0f}% "
            f"({scores['keywords_found']}/{scores['total_keywords']} keywords)")


def retrieval_eval(retriever: Runnable[str,list[Document]], tests_path: Path, tests_limit: int | None = None,
                   k: int | None = None, seed: int | None = DEFAULT_SEED):
    """Runs retrieval-only evaluation across test cases, yielding results and progress.

    Args:
        retriever (Runnable[str,list[Document]]): The retrieval engine instance.
        tests_path (Path): Path to the test cases JSONL file.
        tests_limit (int | None): Optional limit on number of test cases to process.
            Defaults to None.
        k (int | None): Top-k threshold for retrieval metrics.
            Defaults to None.
        seed (int | None): Random seed for choosing the sample.
            Defaults to DEFAULT_SEED.

    Yields:
        tuple: A tuple containing the result dict and progress float (0.0 to 1.0).
    """
    tests = _select_tests(load_questions(tests_path), tests_limit, seed)

    total = len(tests)
    for i, t in enumerate(tests):
        logger.info("[retrieval_eval] %d/%d start | question=%r", i + 1, total, preview(t.question))
        start = time.perf_counter()
        try:
            retrieved = retriever.invoke(t.question)
        except Exception as e:
            logger.error(
                "[retrieval_eval] %d/%d FAILED after %.2fs | %s: %s",
                i + 1, total, time.perf_counter() - start, type(e).__name__, e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise

        retrieved_texts = [c.page_content for c in retrieved]
        retrieval_scores = retrieval_proxy_scores(retrieved_texts, t.keywords , k=k)

        elapsed = time.perf_counter() - start

        logger.info("[retrieval_eval] %d/%d done | %d chunks | %.2fs | %s",
                    i + 1, total, len(retrieved), elapsed, _retrieval_summary(retrieval_scores))
        result = {
            "question": t.question,
            "category": t.category,
            "retrieval_scores": retrieval_scores,
            "wall_seconds": elapsed,
        }

        yield result, (i + 1) / total


def _judge_summary(evaluation: AnswerEvaluation) -> str:
    """Formats the judge's scores as a short string for log lines.

    Args:
        evaluation (AnswerEvaluation): The judge's structured evaluation.

    Returns:
        str: The scores on one line.
    """
    return (f"accuracy={evaluation.accuracy} completeness={evaluation.completeness} "
            f"relevance={evaluation.relevance} abstained_correctly={evaluation.abstained_correctly}")


def _evaluate_single_question(t: EvalQuestion, retriever: Runnable[str,list[Document]],
                              k: int | None = None, retries: int = 1) -> dict:
    """Evaluates a single test question by executing the RAG pipeline and grading the output.

    This function never raises exceptions; any unexpected failure during execution
    or grading is caught, logged, and reported within the returned dictionary's
    'error' field.

    Args:
        t (EvalQuestion): The test question to evaluate.
        retriever (Runnable[str,list[Document]]): The retrieval engine instance.
        k (int | None): Top-k threshold for retrieval metrics.
            Defaults to None.
        retries (int): Number of parsing retry attempts for the judge model.
            Defaults to 1.

    Returns:
        dict: A dictionary containing evaluation details with the following keys:
            - question (str): The test question text.
            - category (str): The category of the question.
            - answerable (bool): Whether the question can be answered from the KB.
            - answer (str | None): The RAG-generated answer.
            - rewritten_question (str | None): The query-rewritten text.
            - retrieval_scores (dict | None): Proxy scores
            - judge_result (dict | None): Structured evaluation scores from the judge.
            - wall_seconds (float | None): Execution time in seconds.
            - tokens (int | None): Total token usage count.
            - error (str | None): Error message if evaluation failed, otherwise None.
    """
    logger.info("[eval] start | category=%s | answerable=%s | question=%r",
                t.category, t.answerable, preview(t.question))
    try:
        start = time.perf_counter()
        result = ask(retriever, t.question, rewrite_enabled=False)

        retrieved_texts = [d.page_content for d in result.retrieved_chunks]
        retrieval_scores = retrieval_proxy_scores(retrieved_texts, t.keywords, k=k)

        logger.info("[eval] retrieval scores | %s | question=%r",
                    _retrieval_summary(retrieval_scores), preview(t.question, 50))
        logger.info("[judge] start | model=%s | answerable=%s", JUDGE_MODEL, t.answerable)

        judge_start = time.perf_counter()
        try:
            judge_result = judge_answer(t.question, t.reference_answer,
                                        result.answer, t.answerable, retries)
        except Exception as e:
            logger.error(
                "[judge] FAILED after %.2fs | %s: %s",
                time.perf_counter() - judge_start, type(e).__name__, e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            raise

        elapsed = time.perf_counter() - start

        logger.info("[judge] done | %.2fs | %s | question=%r",
                    time.perf_counter() - judge_start, _judge_summary(judge_result),
                    preview(t.question, 50))
        logger.info("[eval] done | wall=%.2fs | tokens=%d", elapsed, result.total_tokens)

        return {
            "question": t.question,
            "category": t.category,
            "answerable": t.answerable,
            "answer": result.answer,
            "rewritten_question": result.rewritten_question,
            "retrieval_scores": retrieval_scores,
            "judge_result": judge_result.model_dump(),
            "wall_seconds": elapsed,
            "tokens": result.total_tokens,
            "error": None,
        }
    except Exception as e:
        logger.error("[eval] question FAILED | %r | %s: %s", preview(t.question),
                     type(e).__name__, e, exc_info=logger.isEnabledFor(logging.DEBUG))

        return {
            "question": t.question,
            "category": t.category,
            "answerable": t.answerable,
            "answer": None,
            "rewritten_question": None,
            "retrieval_scores": None,
            "judge_result": None,
            "wall_seconds": None,
            "tokens": None,
            "error": f"{type(e).__name__}: {e}",
        }


def run_eval(retriever: Runnable[str,list[Document]], tests_path: Path, tests_limit: int | None = None,
             k: int | None = None, retries: int = 1, max_workers: int = 1,
             seed: int | None = DEFAULT_SEED):
    """Runs the full RAG and evaluation pipeline, sequentially or in a thread pool.

    Args:
        retriever (Runnable[str,list[Document]]): The retrieval engine instance.
        tests_path (Path): Path to the test cases JSONL file.
        tests_limit (int | None): Optional limit on test cases.
        k (int | None): Top-k threshold for metrics.
            Defaults to None.
        retries (int): Retries for judge parsing.
            Defaults to 1.
        max_workers (int): Maximum number of concurrent worker threads. Values <= 1 run sequentially.
            Defaults to 1.
        seed (int | None): Random seed for choosing the sample.
            Defaults to DEFAULT_SEED.

    Yields:
        tuple: A tuple containing the evaluation result dictionary and progress (0.0 to 1.0).
    """
    tests = _select_tests(load_questions(tests_path), tests_limit, seed)
    total = len(tests)
    if total == 0:
        return

    # Sequential execution path
    if max_workers <= 1:
        for i, t in enumerate(tests, start=1):
            yield _evaluate_single_question(t, retriever, k, retries), i / total

        return

    # Concurrent streaming execution path
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda q: _evaluate_single_question(q, retriever, k, retries), tests
        )

        for completed, result in enumerate(results, start=1):
            yield result, completed / total


def summarize(evaluations: list[dict]) -> dict:
    """Generates overall and category-specific aggregated metrics from evaluations.

    Args:
        evaluations (list[dict]): List of individual test evaluation dictionaries.

    Returns:
        dict: Aggregated summary statistics of the following structure:
            - overall (dict): Overall aggregated metrics with keys:
                - n (int): Total number of test questions evaluated.
                - n_errors (int): Number of failed evaluations.
                - avg_mrr (float | None): Average Mean Reciprocal Rank.
                - avg_ndcg (float | None): Average nDCG score.
                - avg_keyword_coverage (float | None): Average keyword coverage percentage.
                - avg_accuracy (float | None): Average judge accuracy score (1-5).
                - avg_completeness (float | None): Average judge completeness score (1-5).
                - avg_relevance (float | None): Average judge relevance score (1-5).
                - avg_wall_seconds (float | None): Average execution time per question.
                - avg_tokens (float | None): Average token count per question.
                - abstention_rate (float | None): Percentage of unanswerable questions correctly declined.
            - by_category (dict): Dictionary mapping each category name to its own aggregate dict
            (with the exact same metric keys as overall).
    """
    by_category = {}
    for r in evaluations:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r)

    summary = {"overall": _aggregate(evaluations), "by_category": {}}
    for cat, rows in by_category.items():
        summary["by_category"][cat] = _aggregate(rows)

    return summary


def _aggregate(rows: list[dict]) -> dict:
    """Helper function to calculate average metrics for a subset of evaluation rows.

    Args:
        rows (list[dict]): Rows corresponding to a category or overall run.

    Returns:
        dict: A dictionary of computed averages and counts.
    """
    def avg(key_path: list) -> float | None:
        """Calculates the average of numeric values extracted via a nested dictionary key path.

        Args:
            key_path (list): A sequence of keys representing the path to the target value
                within each row dictionary.

        Returns:
            float | None: The computed average, or None if no valid numeric values were found.
        """
        vals = []
        for r in rows:
            v = r
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    abstain_rows = [r for r in rows if r.get("answerable") is False]
    abstain_correct = [
        r for r in abstain_rows
        if (r.get("judge_result") or {}).get("abstained_correctly") is True
    ]
    abstention_rate_pct = (len(abstain_correct) / len(abstain_rows)) * 100 if abstain_rows else None

    errors = [r for r in rows if r.get("error")]

    return {
        "n": len(rows),
        "n_errors": len(errors),
        "avg_mrr": avg(["retrieval_scores", "mrr"]),
        "avg_ndcg": avg(["retrieval_scores", "ndcg"]),
        "avg_keyword_coverage": avg(["retrieval_scores", "keyword_coverage"]),
        "avg_accuracy": avg(["judge_result", "accuracy"]),
        "avg_completeness": avg(["judge_result", "completeness"]),
        "avg_relevance": avg(["judge_result", "relevance"]),
        "avg_wall_seconds": avg(["wall_seconds"]),
        "avg_tokens": avg(["tokens"]),
        "abstention_rate": abstention_rate_pct,
    }


def run_meta(kind: str, persist_dir: str | Path) -> dict:
    """Builds the extra details saved alongside a run, so runs can be told apart later.

    Args:
        kind (str): Run type; "retrieval" for retrieval-only runs or "full" for full-pipeline runs.
        persist_dir (str | Path): The vector store the run used.

    Returns:
        dict: The metadata to pass to save_run.
    """
    is_full = kind == "full"

    return {
        "type": kind,
        "persist_dir": str(persist_dir),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "cross_encoder_model": os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "generation_model": GENERATION_MODEL if is_full else None,
        "judge_model": JUDGE_MODEL if is_full else None,
    }


def save_run(config: dict, evaluations: list[dict], summary: dict, run_dir:Path,
             sampling: dict | None = None, meta: dict | None = None) -> Path:
    """Saves the run configuration, summary statistics, and detailed evaluations to a JSON file.

    Args:
        config (dict): Active configuration dictionary.
        evaluations (list[dict]): List of individual test evaluation dictionaries.
        summary (dict): Aggregated summary metrics.
        run_dir (Path): Directory path where run files are stored.
        sampling (dict | None): Dictionary tracking limit and seed used for tests selection.
            Defaults to None.
        meta (dict | None): Metadata for the run.
            Defaults to None.

    Returns:
        Path: File path to the saved JSON run report.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}"
    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": config_to_save,
        "meta": meta or {},
        "sampling": sampling,
        "summary": summary,
        "evaluations": evaluations,
    }
    path = run_dir / f"{run_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def parse_args():
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests",
        type=Path,
        default=script_dir / "tests.jsonl",
        help="Path to the test cases file"
    )
    parser.add_argument(
        "--tests-limit",
        type=int,
        default=None,
        help="Limit the number of test cases to run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for choosing which questions --tests_limit runs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent worker threads for evaluation",
    )
    parser.add_argument(
        "--persist",
        type=Path,
        default=script_dir.parent / "vector_db",
        help="Path to the persistence vector store directory"
    )
    parser.add_argument(
        "--use-hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="combine dense retrieval with BM25 sparse retrieval"
    )
    parser.add_argument(
        "--rerank-method",
        default=None,
        choices=["none", "cross_encoder"],
        help="The ranking strategy for document reranking"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=script_dir.parent / "runs",
        help="Path to the directory where runs are stored"
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=script_dir.parent / "active_config.json",
        help="Path to the active configuration file"
    )

    return parser.parse_args()


def _fmt(value: float | None, spec: str = ".2f", suffix: str = "") -> str:
    """Formats a number for console output, or 'N/A' when it is missing.

    Args:
        value (float | None): The numeric value to format, or None.
        spec (str): The format specification string.
            Defaults to ".2f".
        suffix (str): An optional string to append to the formatted result.
            Defaults to "".

    Returns:
        str: The formatted string, or "N/A" if the value is None.
    """
    return "N/A" if value is None else f"{value:{spec}}{suffix}"


def print_summary(summary: dict):
    """Prints the overall and per-category results to the console.

    Args:
        summary (dict): The dictionary returned by summarize().
    """
    o = summary["overall"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (OVERALL)")
    print("=" * 60)
    print(f"Questions           : {o['n']}")
    print(f"Errors              : {o['n_errors']}")
    print(f"Accuracy            : {_fmt(o['avg_accuracy'])} / 5")
    print(f"Completeness        : {_fmt(o['avg_completeness'])} / 5")
    print(f"Relevance           : {_fmt(o['avg_relevance'])} / 5")
    print(f"MRR                 : {_fmt(o['avg_mrr'], '.4f')}")
    print(f"nDCG                : {_fmt(o['avg_ndcg'], '.4f')}")
    print(f"Keyword coverage    : {_fmt(o['avg_keyword_coverage'], '.1f', '%')}")
    print(f"Abstention rate     : {_fmt(o['abstention_rate'], '.1f', '%')}")
    print(f"Avg total time / q  : {_fmt(o['avg_wall_seconds'], '.1f', 's')}")
    print(f"Avg tokens / q      : {_fmt(o['avg_tokens'], '.0f')}")
    print("-" * 60)
    for cat, a in summary["by_category"].items():
        print(f"{cat} (n={a['n']}, errors={a['n_errors']})")
        print(f"  judge    : acc {_fmt(a['avg_accuracy'])} | comp {_fmt(a['avg_completeness'])}"
              f" | rel {_fmt(a['avg_relevance'])}")
        print(f"  retrieval: MRR {_fmt(a['avg_mrr'])} | nDCG {_fmt(a['avg_ndcg'])}"
              f" | coverage {_fmt(a['avg_keyword_coverage'], '.1f', '%')}")


def main():
    setup_logging()

    args = parse_args()
    test_path = args.tests
    tests_limit = args.tests_limit
    persist_dir = args.persist
    runs_dir = args.runs
    active_config_path = args.configs
    seed = args.seed
    max_workers = args.max_workers

    config = load_active_config(active_config_path)
    if args.use_hybrid is not None:
        config["use_hybrid"] = args.use_hybrid
    if args.rerank_method is not None:
        config["rerank_method"] = args.rerank_method
    config["rewrite_enabled"] = False

    retriever = build_retriever(persist_dir, config)

    evaluations = []
    for result, progress in run_eval(retriever=retriever, tests_path=test_path, tests_limit=tests_limit,
                                     k=config["final_k"], retries=1, max_workers=max_workers, seed=seed):
        evaluations.append(result)
        print(
            f"[{progress * 100:5.1f}%] Completed evaluation for: {preview(result['question'], 40)}"
        )

    summary = summarize(evaluations)
    path = save_run(config, evaluations, summary, runs_dir,
                    sampling={"tests_limit": tests_limit, "seed": seed},
                    meta=run_meta("full", persist_dir))

    print_summary(summary)
    print(f"\nSaved run to {path}")

    failed = [e["question"] for e in evaluations if e.get("error")]
    if failed:
        print(f"\n{len(failed)} question(s) failed:\n")
        for q in failed:
            print(f"{q}\n")


if __name__ == "__main__":
    main()


