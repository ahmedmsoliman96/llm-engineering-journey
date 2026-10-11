import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from rag.evaluator import (
    retrieval_eval,
    run_eval,
    run_meta,
    save_run,
    summarize,
)
from rag.factory import DEFAULT_CONFIG, build_retriever
from rag.logging_config import setup_logging

script_dir = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIR = script_dir / "vector_db"
DEFAULT_TESTS_PATH = script_dir / "rag" / "tests.jsonl"
RUNS_DIR = script_dir / "runs"
RETRIEVAL_RUNS_DIR = script_dir / "retrieval_runs"


MRR_GREEN, MRR_AMBER = 0.9, 0.75
NDCG_GREEN, NDCG_AMBER = 0.9, 0.75
COVERAGE_GREEN, COVERAGE_AMBER = 90.0, 75.0          # 0-100 scale
ABSTENTION_GREEN, ABSTENTION_AMBER = 90.0, 75.0      # 0-100 scale
ANSWER_GREEN, ANSWER_AMBER = 4.5, 4.0                # 1-5 scale

PASS_THRESHOLDS = {"accuracy": 3.5, "completeness": 3.5, "relevance": 3.5}

RETRIEVAL_QUALITY = {
    "MRR": "avg_mrr",
    "nDCG": "avg_ndcg",
    "Keyword coverage %": "avg_keyword_coverage",
}
RETRIEVAL_METRICS = {
    **RETRIEVAL_QUALITY,
    "Retrieval seconds / question": "avg_wall_seconds",
}
JUDGE_METRICS = {
    "Accuracy": "avg_accuracy",
    "Completeness": "avg_completeness",
    "Relevance": "avg_relevance",
    "Correct abstention %": "abstention_rate",
}
FULL_RUN_METRICS = {
    **JUDGE_METRICS,
    **RETRIEVAL_QUALITY,
    "Total seconds / question (retrieve + generate + judge)": "avg_wall_seconds",
    "Tokens / question": "avg_tokens",
}
OVERALL_EXTRA = {"Questions": "n", "Errors": "n_errors"}  # overall table only

_BANDS = {
    "mrr": (MRR_GREEN, MRR_AMBER),
    "ndcg": (NDCG_GREEN, NDCG_AMBER),
    "coverage": (COVERAGE_GREEN, COVERAGE_AMBER),
    "abstention": (ABSTENTION_GREEN, ABSTENTION_AMBER),
    "accuracy": (ANSWER_GREEN, ANSWER_AMBER),
    "completeness": (ANSWER_GREEN, ANSWER_AMBER),
    "relevance": (ANSWER_GREEN, ANSWER_AMBER),
}


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type.

    Args:
        value (float): The numeric value of the metric.
        metric_type (str): The classification type of the metric.

    Returns:
        str: Color string ('green', 'orange', 'red', or 'black').
    """
    green, amber = _BANDS.get(metric_type, (None, None))
    if green is None:
        return "black"
    if value >= green:
        return "green"
    elif value >= amber:
        return "orange"

    return "red"


def _format_metric_html(label: str, value: float | None, metric_type: str,
                        is_percentage: bool = False, score_format: bool = False) -> str:
    """Format a metric with color coding. value=None renders a gray N/A card.

    Args:
        label (str): Name of the metric card.
        value (float | None): Metric value to format.
        metric_type (str): Metric category for color banding.
        is_percentage (bool): Whether to format as a percentage.
            Defaults to False.
        score_format (bool): Whether to format as a score out of 5.
            Defaults to False.

    Returns:
        str: HTML string representing the metric card.
    """
    if value is None:
        return f"""
        <div style="margin: 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid #ccc;">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 20px; color: #999;">N/A</div>
        </div>
        """

    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"

    return f"""
    <div style="margin: 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def _plain_card(label: str, value_str: str) -> str:
    """Non-color-coded info card for auxiliary execution metadata.

    Args:
        label (str): Title of the metric.
        value_str (str): Formatted string representation of the value.

    Returns:
        str: HTML string for the plain info card.
    """
    return f"""
    <div style="margin: 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px;">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: #333">{value_str}</div>
    </div>
    """


def _build_answer_metric_cards(overall: dict) -> str:
    """Colored summary cards for a full (retrieval + judge) run.

    Args:
        overall (dict): Aggregated overall dictionary metrics.

    Returns:
        str: Combined HTML string containing metric cards and error banners.
    """
    cards = "".join([
        _format_metric_html("Accuracy", overall["avg_accuracy"], "accuracy", score_format=True),
        _format_metric_html("Completeness", overall["avg_completeness"], "completeness", score_format=True),
        _format_metric_html("Relevance", overall["avg_relevance"], "relevance", score_format=True),
        _format_metric_html("MRR", overall["avg_mrr"], "mrr"),
        _format_metric_html("nDCG", overall["avg_ndcg"], "ndcg"),
        _format_metric_html("Keyword coverage", overall["avg_keyword_coverage"], "coverage", is_percentage=True),
        _format_metric_html("Abstention rate", overall["abstention_rate"], "abstention", is_percentage=True),
    ])

    extras = ""
    if overall["avg_wall_seconds"] is not None:
        extras += _plain_card(
            "Avg total time / question (retrieve + generate + judge)",
            f"{overall['avg_wall_seconds']:.2f}s"
        )
    if overall["avg_tokens"] is not None:
        extras += _plain_card("Avg tokens / question", f"{overall['avg_tokens']:.0f}")

    if overall["n_errors"]:
        banner = (f"<div style='margin-top:10px;padding:10px;background:#f8d7da;border-radius:5px;"
                  f"color:#721c24;'>⚠️ {overall['n_errors']} question(s) errored out — "
                  f"see the saved run file for details.</div>")
    else:
        banner = ("<div style='margin-top:10px;padding:10px;background:#d4edda;border-radius:5px;"
                   "color:#155724;'>✓ No errors.</div>")

    return (
        f"<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:10px;'>"
        f"{cards}</div>"
        f"<div style='display:flex; gap:15px; margin-top:5px;'>{extras}</div>"
        f"{banner}"
    )


def _build_retrieval_metric_cards(avg_mrr: float, avg_ndcg: float, avg_coverage: float, n: int) -> str:
    """Colored summary cards for a retrieval-only run.

    Args:
        avg_mrr (float): Average Mean Reciprocal Rank score.
        avg_ndcg (float): Average Normalized Discounted Cumulative Gain score.
        avg_coverage (float): Average keyword coverage percentage.
        n (int): Total number of questions checked.

    Returns:
        str: Combined HTML string containing metric cards and a status note.
    """
    cards = "".join([
        _format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr"),
        _format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg"),
        _format_metric_html("Keyword coverage", avg_coverage, "coverage", is_percentage=True),
    ])
    note = f"<div style='margin-top:10px;color:#666;font-size:13px;'>{n} question(s) checked — no judge/LLM calls made.</div>"

    return f"<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:10px;'>{cards}</div>{note}"


def do_run(persist_dir: Path, use_hybrid: bool, rerank_method: str, final_k: int,
           tests_path: Path, tests_limit: int | None, max_workers: int, seed: int | None,
           progress: gr.Progress = gr.Progress()) -> tuple:
    """Executes a full RAG evaluation run (retrieval + generation + judge).

    Args:
        persist_dir (Path): Vector store directory path.
        use_hybrid (bool): Whether to use hybrid search (BM25 + dense).
        rerank_method (str): Ranking strategy ("none" or "cross_encoder").
        final_k (int): Top-k final chunk count.
        tests_path (Path): Path to test dataset file.
        tests_limit (int | None): Limit on test cases to evaluate.
        max_workers (int): Concurrent worker threads count.
        seed (int | None): Random sampling seed.
        progress (gr.Progress): Gradio progress tracker utility.
            Defaults to gr.Progress().

    Returns:
        tuple: A tuple containing overall HTML cards, accuracy chart dataframe,
            results dataframe, status message string, and saved file Path.

    Raises:
        gr.Error: With a readable message if vector store was not found.
    """
    config = {
        **DEFAULT_CONFIG,
        "rewrite_enabled": False,
        "use_hybrid": use_hybrid,
        "rerank_method": rerank_method,
        "final_k": int(final_k),
    }
    try:
        retriever = build_retriever(persist_dir, config)
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e

    limit_val = int(tests_limit) if tests_limit else None
    seed_val = int(seed) if seed is not None else None

    evaluations = []
    count = 0
    for result, prog in run_eval(retriever, tests_path=tests_path, tests_limit=limit_val,
                                 k=final_k, max_workers=max_workers, seed=seed_val):
        evaluations.append(result)
        count += 1
        progress(prog, desc=f"Evaluating RAG pipeline — question {count}")

    summary = summarize(evaluations)
    path = save_run(config, evaluations, summary, RUNS_DIR,
                    sampling={"tests_limit": limit_val, "seed": seed_val},
                    meta=run_meta("full", persist_dir))

    overall_html = _build_answer_metric_cards(summary["overall"])
    
    rows = []
    for cat, agg in summary["by_category"].items():
        rows.append(
            {
                "category": cat,
                "n": agg["n"],
                "Errors": agg["n_errors"],
                "MRR": round(agg["avg_mrr"], 3) if agg["avg_mrr"] is not None else None,
                "nDCG": round(agg["avg_ndcg"], 3) if agg["avg_ndcg"] is not None else None,
                "Keyword coverage": round(agg["avg_keyword_coverage"], 2) if agg["avg_keyword_coverage"] is not None else None,
                "Accuracy": round(agg["avg_accuracy"], 2) if agg["avg_accuracy"] is not None else None,
                "Completeness": round(agg["avg_completeness"], 2) if agg["avg_completeness"] is not None else None,
                "Relevance": round(agg["avg_relevance"], 2) if agg["avg_relevance"] is not None else None,
                "Abstention rate": round(agg["abstention_rate"], 2) if agg["abstention_rate"] is not None else None,
                "Avg total seconds": round(agg["avg_wall_seconds"], 2) if agg["avg_wall_seconds"] is not None else None,
            }
        )
    df = pd.DataFrame(rows)

    accuracy_chart_df = pd.DataFrame(
        [
            {"Category": cat, "Average Accuracy": agg["avg_accuracy"]}
            for cat, agg in summary["by_category"].items() if agg["avg_accuracy"] is not None
        ]
    )

    status = f"Saved run: {path.name}"

    return overall_html, accuracy_chart_df, df, status, path


def run_retrieval_only(persist_dir: Path, use_hybrid: bool, rerank_method: str, final_k: int,
                       tests_path: Path, tests_limit: int | None, seed: int | None,
                       progress: gr.Progress = gr.Progress()) -> tuple:
    """Executes a fast retrieval and keyword-proxy evaluation run.

    Args:
        persist_dir (Path): Vector store directory path.
        use_hybrid (bool): Whether to use hybrid search.
        rerank_method (str): Ranking strategy.
        final_k (int): Top-k final chunk count.
        tests_path (Path): Path to test dataset file.
        tests_limit (int | None): Limit on test cases to evaluate.
        seed (int | None): Random sampling seed.
        progress (gr.Progress): Gradio progress tracker.
            Defaults to gr.Progress().

    Returns:
        tuple: Formatted HTML summary string and category chart dataframe.

    Raises:
        gr.Error: With a readable message if vector store was not found.
    """
    config = {
        **DEFAULT_CONFIG,
        "rewrite_enabled": False,
        "use_hybrid": use_hybrid,
        "rerank_method": rerank_method,
        "final_k": int(final_k),
    }

    try:
        retriever = build_retriever(persist_dir, config)
    except FileNotFoundError as e:
        raise gr.Error(str(e)) from e

    limit_val = int(tests_limit) if tests_limit else None
    seed_val = int(seed) if seed is not None else None
    total_mrr = total_ndcg = total_coverage = 0.0
    scored = 0
    count = 0
    category_mrr = defaultdict(list)

    results = []
    for result, prog in retrieval_eval(retriever, tests_path, tests_limit=limit_val, k=final_k, seed=seed_val):
        count += 1
        results.append(result)
        r = result["retrieval_scores"]
        if r["mrr"] is not None:  # None when a test has no keywords to score against
            total_mrr += r["mrr"]
            total_ndcg += r["ndcg"]
            total_coverage += r["keyword_coverage"]
            scored += 1
            category_mrr[result["category"]].append(r["mrr"])
        progress(prog, desc=f"Checking retrieval — question {count}")

    if scored == 0:
        return "<p>No scoreable tests (missing keywords?).</p>", pd.DataFrame()

    html = _build_retrieval_metric_cards(total_mrr / scored, total_ndcg / scored, total_coverage / scored, count)
    chart_df = pd.DataFrame([
        {"Category": cat, "Average MRR": sum(v) / len(v)} for cat, v in category_mrr.items()
    ])

    summary = summarize(results)
    path = save_run(config, results, summary, RETRIEVAL_RUNS_DIR,
                    sampling={"tests_limit": limit_val, "seed": seed_val},
                    meta=run_meta("retrieval", persist_dir))

    html += (f"<div style='margin-top:6px;color:#666;font-size:13px;'>"
             f"Saved to {RETRIEVAL_RUNS_DIR.name}/{path.name}</div>")
    
    return html, chart_df


def _list_runs(runs_dir: Path = RUNS_DIR) -> list[str]:
    """Lists saved run files in a directory, newest first.

    Args:
        runs_dir (Path): Directory path to search for saved JSON runs.
            Defaults to RUNS_DIR.

    Returns:
        list[str]: Sorted list of run filenames in descending order.
    """
    runs_dir = Path(runs_dir)
    
    return [f.name for f in sorted(runs_dir.glob("*.json"), reverse=True)]


def _load_run(filename: str, runs_dir: Path = RUNS_DIR) -> dict:
    """Loads one saved run JSON from a directory.

    Args:
        filename (str): Name of the run JSON file to load.
        runs_dir (Path): Directory containing the run file.
            Defaults to RUNS_DIR.

    Returns:
        dict: Parsed dictionary data from the run JSON file.
    """
    runs_dir = Path(runs_dir)
    
    with open(runs_dir / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def _pass_by_metric(row: dict) -> dict:
    """Evaluates whether metric scores meet pass thresholds.

    Args:
        row (dict): A single evaluation row containing judge results.

    Returns:
        dict: Mapping of metric names to boolean pass/fail status,
            or None if the metric value is missing.
    """
    judge_results = row.get("judge_result") or {}
    
    return {
        c: (judge_results.get(c) >= t) if isinstance(judge_results.get(c), (int, float)) else None
        for c, t in PASS_THRESHOLDS.items()
    }


def _r(value: Any, nd: int = 4):
    """Rounds numeric values safely.

    Args:
        value: The numeric value to round.
        nd (int): Number of decimal places. Defaults to 4.

    Returns:
        float | None: The rounded value, or None if the input is not numeric.
    """
    return round(value, nd) if isinstance(value, (int, float)) else None


def _delta(a: Any, b: Any, nd: int = 4) -> float | None:
    """Computes numerical difference between run B and run A.

    Args:
        a: The baseline numeric value (Run A).
        b: The comparison numeric value (Run B).
        nd (int): Number of decimal places to round the result.
            Defaults to 4.

    Returns:
        float | None: The computed difference (b - a), or None if either value is non-numeric.
    """
    both = isinstance(a, (int, float)) and isinstance(b, (int, float))

    return round(b - a, nd) if both else None


def _metric_rows(a: dict[str,float], b: dict, metrics: dict) -> list[dict]:
    """Generates comparison rows for metrics.

    Args:
        a (dict): Metrics dictionary for Run A.
        b (dict): Metrics dictionary for Run B.
        metrics (dict): Mapping of display labels to dictionary keys.

    Returns:
        list[dict]: A list of comparison dictionaries containing metric labels,
            Run A values, Run B values, and their calculated deltas.
    """
    return [
        {"metric": label, "A": _r(a.get(key)), "B": _r(b.get(key)),
         "Δ (B − A)": _delta(a.get(key), b.get(key))}
        for label, key in metrics.items()
    ]


def _overall_table(run_a: dict, run_b: dict, metrics: dict) -> pd.DataFrame:
    """Builds overall comparison DataFrame.

    Args:
        run_a (dict): Full dictionary data for Run A.
        run_b (dict): Full dictionary data for Run B.
        metrics (dict): Metrics dictionary to include in the comparison.

    Returns:
        pd.DataFrame: DataFrame containing overall metric comparisons between Run A and Run B.
    """
    a, b = run_a["summary"]["overall"], run_b["summary"]["overall"]

    return pd.DataFrame(_metric_rows(a, b, {**OVERALL_EXTRA, **metrics}))


def _category_table(run_a: dict, run_b: dict, metrics: dict) -> pd.DataFrame:
    """Builds per-category comparison DataFrame.

    Args:
        run_a (dict): Full dictionary data for Run A.
        run_b (dict): Full dictionary data for Run B.
        metrics (dict): Metrics dictionary to include in the category-wise comparison.

    Returns:
        pd.DataFrame: DataFrame containing per-category comparisons across both runs.
    """
    cats_a, cats_b = run_a["summary"]["by_category"], run_b["summary"]["by_category"]
    rows = []
    for cat in sorted(set(cats_a) | set(cats_b)):
        for row in _metric_rows(cats_a.get(cat, {}), cats_b.get(cat, {}), metrics):
            rows.append({"category": cat, **row})
            
    return pd.DataFrame(rows)


def _describe_runs(name_a: str, run_a: dict, name_b: str, run_b: dict, summary_line: str) -> str:
    """Describes comparison run metadata and warnings.

    Args:
        name_a (str): Display name or filename of Run A.
        run_a (dict): Full dictionary data for Run A.
        name_b (str): Display name or filename of Run B.
        run_b (dict): Full dictionary data for Run B.
        summary_line (str): A summary statement explaining key performance changes.

    Returns:
        str: Formatted Markdown text describing configuration blocks, sampling notes,
            and any mismatch warnings.
    """
    def block(tag: str, name: str, run: dict) -> str:
        """Generates a Markdown block detailing a single run's metadata.

        Args:
            tag (str): Run identifier tag (e.g., 'A' or 'B').
            name (str): Name or filename of the run.
            run (dict): Full dictionary data for the run.

        Returns:
            str: Formatted Markdown string containing config, sampling, and meta details.
        """
        return (f"### Run {tag}: {name}\n"
                f"- **Config:** `{run['config']}`\n"
                f"- **Sampling:** `{json.dumps(run.get('sampling', {}))}`\n"
                f"- **Meta:** `{json.dumps(run.get('meta', {}))}`\n")

    warning = ""
    if run_a.get("sampling") != run_b.get("sampling"):
        warning = ("\n\n**Warning:** the runs used different sampling (limit/seed), "
                   "so they were not scored on identical questions.")

    return (f"{block('A', name_a, run_a)}\n{block('B', name_b, run_b)}\n"
            f"**Summary:** {summary_line} Δ is B − A "
            f"{warning}")


def do_compare(run_a_name: str, run_b_name: str) -> tuple:
    """Compares two saved full-pipeline runs side-by-side.

    Args:
        run_a_name (str): Filename of Run A json report.
        run_b_name (str): Filename of Run B json report.

    Returns:
        tuple: Overall metrics dataframe, per-category dataframe, pass/fail flips dataframe,
            and Markdown description string.
    """
    if not run_a_name or not run_b_name:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Pick two runs to compare."

    run_a, run_b = _load_run(run_a_name), _load_run(run_b_name)

    a_by_q = {r["question"]: r for r in run_a["evaluations"]}
    b_by_q = {r["question"]: r for r in run_b["evaluations"]}
    flips = []
    for question, evaluation in a_by_q.items():
        if question not in b_by_q:
            continue
        pa = _pass_by_metric(evaluation)
        pb = _pass_by_metric(b_by_q[question])
        changed = [m for m in PASS_THRESHOLDS
                   if pa[m] is not None and pb[m] is not None and pa[m] != pb[m]]
        if changed:
            flips.append({"question": question, "category": evaluation["category"],
                          "changed metric(s)": ", ".join(changed)})

    return (
        _overall_table(run_a, run_b, FULL_RUN_METRICS),
        _category_table(run_a, run_b, FULL_RUN_METRICS),
        pd.DataFrame(flips),
        _describe_runs(run_a_name, run_a, run_b_name, run_b,
                       f"{len(flips)} question(s) changed pass/fail on at least one metric."),
    )


def do_compare_retrieval(run_a_name: str, run_b_name: str) -> tuple:
    """Compares two saved retrieval-only runs side-by-side.

    Args:
        run_a_name (str): Filename of Run A retrieval report.
        run_b_name (str): Filename of Run B retrieval report.

    Returns:
        tuple: Overall metrics dataframe, per-category dataframe, MRR changes dataframe,
            and Markdown description string.
    """
    if not run_a_name or not run_b_name:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Pick two runs to compare."

    run_a = _load_run(run_a_name, RETRIEVAL_RUNS_DIR)
    run_b = _load_run(run_b_name, RETRIEVAL_RUNS_DIR)

    a_by_q = {r["question"]: r for r in run_a["evaluations"]}
    b_by_q = {r["question"]: r for r in run_b["evaluations"]}
    changed = []
    for question, evaluation in a_by_q.items():
        if question not in b_by_q:
            continue
        ma = (evaluation.get("retrieval_scores") or {}).get("mrr")
        mb = (b_by_q[question].get("retrieval_scores") or {}).get("mrr")
        if ma is None or mb is None or abs(mb - ma) < 1e-9:
            continue
        changed.append({"question": question, "category": evaluation["category"],
                        "MRR A": _r(ma, 3), "MRR B": _r(mb, 3),
                        "Δ MRR": _delta(ma, mb, 3)})
    changed.sort(key=lambda r: r["Δ MRR"])

    return (
        _overall_table(run_a, run_b, RETRIEVAL_METRICS),
        _category_table(run_a, run_b, RETRIEVAL_METRICS),
        pd.DataFrame(changed),
        _describe_runs(run_a_name, run_a, run_b_name, run_b,
                       f"MRR changed on {len(changed)} question(s)."),
    )


def main():
    setup_logging()

    with gr.Blocks(title="rag Pipeline Lab") as demo:
        gr.Markdown(
            "# rag Pipeline Lab\nToggle retrieval config, run the eval suite, compare results."
        )

        with gr.Tab("Full Run"):
            with gr.Row():
                persist_dir = gr.Textbox(label="Vector store path", value=str(DEFAULT_PERSIST_DIR))
                tests_path = gr.Textbox(label="Tests file", value=str(DEFAULT_TESTS_PATH))
            with gr.Row():
                use_hybrid = gr.Checkbox(label="Hybrid retrieval (BM25 + dense)", value=True)
                rerank_method = gr.Radio(["none", "cross_encoder"], label="Reranker", value="cross_encoder")
            with gr.Row():
                final_k = gr.Number(label="final_k", value=10, precision=0)
                tests_limit = gr.Number(label="Limit # of questions (blank = all)", value=None)
                seed = gr.Number(
                    label="Random seed (blank = new sample each run)",
                    value=42,
                    precision=0,
                )
                max_workers = gr.Slider(label="Concurrent workers", minimum=1, maximum=4, value=1, step=1)

            gr.Markdown(
                 "Free-tier rate limits are per-minute, not per-connection — raising workers speeds "
                 "things up only up to a point, past which more requests will be retired."
            )
            run_btn = gr.Button("Run evaluation", variant="primary")
            status = gr.Markdown()
            run_metrics = gr.HTML(label="Summary")
            with gr.Row():
                run_accuracy_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by category",
                    y_lim=[1, 5],
                    height=350,
                )
            results_table = gr.Dataframe(label="Results by category")
            last_run_path = gr.State()

            run_btn.click(
                do_run,
                inputs=[
                    persist_dir,
                    use_hybrid,
                    rerank_method,
                    final_k,
                    tests_path,
                    tests_limit,
                    max_workers,
                    seed,
                ],
                outputs=[
                    run_metrics,
                    run_accuracy_chart,
                    results_table,
                    status,
                    last_run_path,
                ],
                show_progress_on=run_accuracy_chart,
            )

        with gr.Tab("Retrieval Check"):
            gr.Markdown(
                "Retrieval + keyword scoring only — no LLM judge calls, so this is free and fast. "
                "Good for tuning retrieval before spending judge budget on a full run."
            )
            with gr.Row():
                rc_persist_dir = gr.Textbox(label="Vector store path", value=str(DEFAULT_PERSIST_DIR))
                rc_tests_path = gr.Textbox(label="Tests file", value=str(DEFAULT_TESTS_PATH))
            with gr.Row():
                rc_use_hybrid = gr.Checkbox(label="Hybrid retrieval (BM25 + dense)", value=True)
                rc_rerank_method = gr.Radio(["none", "cross_encoder"], label="Reranker", value="cross_encoder")
                rc_final_k = gr.Number(label="final_k", value=10, precision=0)
                rc_tests_limit = gr.Number(label="Limit # of questions (blank = all)", value=None)
                rc_seed = gr.Number(
                    label="Random seed (blank = new sample each run)",
                    value=42,
                    precision=0,
                )

            rc_btn = gr.Button("Check retrieval", variant="primary")
            rc_metrics = gr.HTML(label="Summary")
            rc_chart = gr.BarPlot(
                x="Category",
                y="Average MRR",
                title="Average MRR by category",
                y_lim=[0, 1],
                height=350,
            )

            rc_btn.click(
                run_retrieval_only,
                inputs=[
                    rc_persist_dir,
                    rc_use_hybrid,
                    rc_rerank_method,
                    rc_final_k,
                    rc_tests_path,
                    rc_tests_limit,
                    rc_seed,
                ],
                outputs=[rc_metrics, rc_chart],
                show_progress_on=rc_chart,
            )

        with gr.Tab("Compare Full Runs"):
            cmp_refresh_btn  = gr.Button("Refresh run list")
            with gr.Row():
                cmp_run_a  = gr.Dropdown(label="Run A", choices=_list_runs())
                cmp_run_b  = gr.Dropdown(label="Run B", choices=_list_runs())
            cmp_compare_btn  = gr.Button("Compare", variant="primary")
            cmp_label  = gr.Markdown()
            cmp_overall_table = gr.Dataframe(label="Overall (Δ = B − A)")
            cmp_category_table = gr.Dataframe(label="Per-category (Δ = B − A)")
            cmp_questions_table = gr.Dataframe(label="Per-question changes (pass/fail flips)")

            cmp_refresh_btn.click(
                lambda: (gr.update(choices=_list_runs()), gr.update(choices=_list_runs())),
                outputs=[cmp_run_a, cmp_run_b],
            )
            cmp_compare_btn.click(
                do_compare,
                inputs=[cmp_run_a, cmp_run_b],
                outputs=[cmp_overall_table, cmp_category_table, cmp_questions_table, cmp_label],
            )

        with gr.Tab("Compare Retrieval Runs"):
            rcmp_refresh_btn = gr.Button("Refresh run list")
            with gr.Row():
                rcmp_run_a = gr.Dropdown(label="Run A", choices=_list_runs(RETRIEVAL_RUNS_DIR))
                rcmp_run_b = gr.Dropdown(label="Run B", choices=_list_runs(RETRIEVAL_RUNS_DIR))
            rcmp_compare_btn = gr.Button("Compare", variant="primary")
            rcmp_label = gr.Markdown()
            rcmp_overall_table = gr.Dataframe(label="Overall (Δ = B − A)")
            rcmp_category_table = gr.Dataframe(label="Per-category (Δ = B − A)")
            rcmp_questions_table = gr.Dataframe(label="Per-question changes (MRR, worst for B first)")

            rcmp_refresh_btn.click(
                lambda: (
                    gr.update(choices=_list_runs(RETRIEVAL_RUNS_DIR)),
                    gr.update(choices=_list_runs(RETRIEVAL_RUNS_DIR)),
                ),
                outputs=[rcmp_run_a, rcmp_run_b],
            )
            rcmp_compare_btn.click(
                do_compare_retrieval,
                inputs=[rcmp_run_a, rcmp_run_b],
                outputs=[rcmp_overall_table, rcmp_category_table, rcmp_questions_table, rcmp_label],
            )

    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()