import os
import logging
from dotenv import load_dotenv

import gradio as gr
from openai import APIConnectionError, APIStatusError, OpenAI
from utility import run_python_code, validate_code_non_empty, clean_llm_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def model_message(code: str, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_prompt+"\n\n"+code},
    ]


def optimize(code: str, client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    try:
        validate_code_non_empty(code)
    except RuntimeError as e:
        raise gr.Error(str(e))

    try:
        logger.info(f"sending python code to LLM model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=model_message(code, system_prompt, user_prompt)
        )
                
    except APIConnectionError as e:
        logger.error(f"failed to connect to LLM backend: {str(e)}")
        raise gr.Error(f"Failed to generate optimized python code via LLM:\n"
                       f"*Could not connect to LLM backend at the configured base URL: {str(e)}*")

    except APIStatusError as e:
        logger.error(f"LLM backend returned an error: {str(e)}")
        raise gr.Error(f"Failed to generate optimized python code via LLM:\n *LLM backend returned an error: {str(e)}*")

    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise gr.Error(f"Failed to generate optimized python code via LLM:\n *{str(e)}*")

    if not response.choices:
        raise gr.Error("LLM returned no response choices")

    result = response.choices[0].message.content
    if not result:
        raise gr.Error("LLM returned an empty response")

    return clean_llm_response(result)


def format_run_result(result: dict) -> str:
    status = (
        "✅ exit code 0"
        if result["returncode"] == 0
        else f"❌ exit code {result['returncode']}"
    )
    parts = [status]
    stdout = result.get("stdout","").strip()
    if stdout:
        parts.append(f"**output**\n{stdout}")
    stderr = result.get("stderr","").strip()
    if stderr:
        parts.append(f"**stderr**\n{stderr}")
    return "\n".join(parts)


def build_comparison(original: dict, optimized: dict) -> str:
    if original["returncode"] != 0:
        return "⚠️ The **original** code didn't run successfully — fix that before comparing performance."
    if optimized["returncode"] != 0:
        return "⚠️ The **optimized** code didn't run successfully — check its output above."

    speedup = (
        original["elapsed"] / optimized["elapsed"]
        if optimized["elapsed"] > 0
        else float("inf")
    )
    correctness = (
        "✅ Outputs match"
        if original["stdout"].strip() == optimized["stdout"].strip()
        else "⚠️ Outputs differ — verify correctness before trusting this speedup!"
    )
    return (
        f"⏱️ {original['elapsed']:.3f}s → {optimized['elapsed']:.3f}s - "
        f"🚀 {speedup:.2f}x faster\n\n{correctness}"
    )


def run_original_wrapper(code: str, results: dict):
    try:
        validate_code_non_empty(code)
        result = run_python_code(code)
    except RuntimeError as e:
        raise gr.Error(str(e))

    results["original"] = result
    return format_run_result(result), f"{result['elapsed']:.3f}s", results


def run_optimized_wrapper(code: str, results: dict):
    try:
        validate_code_non_empty(code)
        result = run_python_code(code)
    except RuntimeError as e:
        raise gr.Error(str(e))

    results["optimized"] = result
    comparison = (
        build_comparison(results["original"], results["optimized"])
        if "original" in results
        else "Run the original code first to see a comparison."
    )
    return format_run_result(result), f"{result['elapsed']:.3f}s", comparison, results


def clear_optimized_state(results: dict):
    results.pop("optimized", None)
    return results, "", "", ""


def clear_original_state(results: dict):
    results.pop("original", None)
    return results, "", "", ""


def display_code(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    def optimize_wrapper(code: str):
        return optimize(code, client, model, system_prompt, user_prompt)

    with gr.Blocks(
        title="🖥️ Python code optimization",
        css="""
        #original_code .cm-editor,
        #optimized_code .cm-editor {
            min-height: 400px !important;
            height: 400px !important;
        }
        #original_code .cm-scroller,
        #optimized_code .cm-scroller {
            min-height: 400px !important;
        }
        """
    ) as UI:
        results_state = gr.State({})

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                original_python = gr.Code(
                    value="",
                    label="original Python code",
                    language="python",
                    lines=26,
                    interactive=True,
                    elem_id="original_code"
                )
            with gr.Column(scale=6):
                optimized_python = gr.Code(
                    value="",
                    label="optimized Python code",
                    language="python",
                    lines=26,
                    interactive=True,
                    elem_id="optimized_code"
                )

        with gr.Row(equal_height=True):
            run_python = gr.Button("▶️ Run original code")
            optimize_code = gr.Button("✨ Optimize code")
            run_optimized_python = gr.Button("▶️ Run optimized code")

        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                python_output = gr.TextArea(label="output (original)", lines=5, interactive=False)
                original_time = gr.TextArea(label="⏱️ Execution time", lines=1, interactive=False)
            with gr.Column(scale=6):
                optimized_python_output = gr.TextArea(label="output (optimized)", lines=5, interactive=False)
                optimized_time = gr.TextArea(label="⏱️ Execution time", lines=1, interactive=False)

        with gr.Row():
            comparison_output = gr.TextArea(label="output comparison", lines=3,interactive=False)

        original_python.change(
            fn=clear_original_state,
            inputs=[results_state],
            outputs=[results_state, original_time, comparison_output, python_output]
        )

        run_python.click(
            fn=run_original_wrapper,
            inputs=[original_python, results_state],
            outputs=[python_output, original_time, results_state]
        )

        optimize_code.click(
            fn=clear_optimized_state,
            inputs=[results_state],
            outputs=[results_state, optimized_time, comparison_output, optimized_python_output]
        ).then(
            fn=optimize_wrapper,
            inputs=[original_python],
            outputs=[optimized_python]
        )

        run_optimized_python.click(
            fn=run_optimized_wrapper,
            inputs=[optimized_python, results_state],
            outputs=[optimized_python_output, optimized_time, comparison_output, results_state]
        )

    UI.launch(inbrowser=True)


if __name__ == "__main__":

    load_dotenv()

    DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    
    if not LLM_API_KEY:
        logger.error("LLM_API_KEY environment variable not set")
        raise SystemExit("Missing LLM_API_KEY: set it in your environment variable before running this script")

    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=LLM_API_KEY)

    system_prompt = (
        "You optimize python code to run faster while preserving the output. Your requirements are:\n"
        "- Return valid python code.\n"
        "- Do not include explanations or markdown.\n"
        "- preserve the functionality and output.\n"
        "- Improve the performance wherever possible.\n"
        "- Remove redundant variables, imports, loops, and computations.\n"
        "- Fix syntax, formatting and style issues.\n"
        "- Make the code readable and maintainable.\n"
        "- If no optimization is possible, return clean and formatted version of code. "
    )

    user_prompt = (
        "Optimize the following Python code.\n"
        "Return only the complete optimized source code.\n"
        "code:"
    )

    display_code(client,DEFAULT_MODEL,system_prompt,user_prompt)