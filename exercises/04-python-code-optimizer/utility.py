import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)
THOUGHT_RE = re.compile(r"<thought\b[^>]*>.*?</thought>\s*",re.DOTALL | re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```",re.DOTALL | re.IGNORECASE)

def clean_llm_response(text: str) -> str:
    """
    Cleans the raw response returned by an LLM.

    Removes internal reasoning blocks (e.g. <thought>...</thought>) and, if
    present, extracts the contents of the first Markdown code block. If no code
    block is found, the cleaned text is returned.

    Args:
        text (str): Raw text returned by the language model.
    Returns:
        str: Cleaned code or text with LLM-specific artifacts removed.
    """
    text = THOUGHT_RE.sub("", text).strip()
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).rstrip() if match else text


def run_python_code(code: str, timeout: int = 60) -> dict:
    """
    Executes a snippet of Python code in an isolated subprocess and measures wall-clock time.

    The code is written to a temporary .py file and run with the same interpreter that is
    running this script (via sys.executable).

    Args:
        code (str): The Python source code to execute.
        timeout (int): Max seconds to allow the code to run before killing it. Defaults to 60.
    Returns:
        dict: {
            "stdout": str,      # captured standard output
            "stderr": str,      # captured standard error (e.g. tracebacks)
            "returncode": int,  # process exit code, 0 means success
            "elapsed": float,   # wall-clock seconds the subprocess took to run
        }
    Raises:
        RuntimeError: If the code times out or the interpreter subprocess cannot be started.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "script.py"
        tmp_path.write_text(code, encoding="utf-8")

        try:
            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            elapsed = time.perf_counter() - start

        except subprocess.TimeoutExpired as e:
            logger.error(f"execution timed out after {timeout}s")
            raise RuntimeError(f"execution timed out after {timeout}s") from e

        except OSError as e:
            logger.error(f"failed to launch python subprocess: {str(e)}")
            raise RuntimeError(f"could not launch the python interpreter: {str(e)}") from e

        except Exception as e:
            logger.error(f"unexpected error while running the code: {str(e)}")
            raise RuntimeError(f"unexpected error while running the code: {str(e)}") from e

    logger.info(f"execution finished in {elapsed:.3f}s with exit code {result.returncode}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "elapsed": elapsed
    }


def validate_code_non_empty(code: str) -> None:
    """
    Validates that a non-empty code string was provided.
    Args:
        code (str): The code to validate.
    Raises:
        RuntimeError: If code is empty or contains only whitespace.
    """
    if not code or not code.strip():
        raise RuntimeError("No python code was provided.")