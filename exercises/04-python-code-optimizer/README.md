# Exercise 04: Python Code Optimizer ⚡

A Gradio web app that runs your Python code, asks an LLM to produce a faster equivalent, and lets you benchmark the two side by side — execution time and output correctness for both versions.

---

## 🏗️ Technical Architecture

The application is split across two core modules:

* **`utility.py` (Execution & Parsing Engine):** Executes arbitrary Python snippets in an isolated subprocess launched with the same interpreter (`sys.executable`), capturing stdout, stderr, exit code, and wall-clock execution time. Also cleans raw LLM responses and validates that submitted code isn't empty.
* **`code_optimizer.py` (Orchestrator & UI Layer):** Manages the LLM using the `openai` API client, pointed at any OpenAI-compatible endpoint — Google's API by default, or any other OpenAI-compatible provider. Prompts the LLM to return an optimized, cleaned-up version of the submitted code with the same functionality and output. Builds the Gradio Blocks UI: side-by-side code editors for the original and optimized versions, run/optimize buttons, per-side output and timing panels, and a live comparison panel reporting speedup and output correctness.

---

## 🚀 How to Run It

Ensure you have synchronized the root environment via `uv sync` from the main project root folder first.

#### 📦 Core Libraries Used
*   **`openai`** — Orchestrates the LLM prompt and routes requests to any OpenAI-compatible backend (Google's API by default).
*   **`gradio`** — Provides the browser-based UI: side-by-side code editors, run/optimize controls, and comparison output.
*   **`python-dotenv`** — Loads environment variables from a local `.env` file.

#### ⚙️ Configuration
The LLM backend is fixed entirely at startup via environment variables (a `.env` file in the project root is loaded automatically).

| Setting              | Env Var        | Editable in UI? | Default                                                    |
|----------------------|----------------|-----------------|------------------------------------------------------------|
| LLM model name       | `LLM_MODEL`    | ❌ startup only | `gemma-4-31b-it`                                           |
| LLM backend base URL | `LLM_BASE_URL` | ❌ startup only | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| LLM API key          | `LLM_API_KEY`  | ❌ startup only | *(required — no default)*                                  |

The Python code to optimize is entered directly in the browser.

### 1. Get an API key for your LLM backend
The default configuration points at Google's API (via its OpenAI-compatible endpoint), so grab a key from [Google AI Studio](https://aistudio.google.com/app/apikey). Put it in a `.env` file in the project root:
```
LLM_API_KEY=your_key_here
```
For a different provider, set `LLM_BASE_URL` and `LLM_MODEL` to point at that endpoint instead and use its respective key.

### 2. Launch the app
Run the script using `uv run`. It starts a local Gradio server and opens a browser tab automatically:
```bash
uv run exercises/04-python-code-optimizer/code_optimizer.py
```
Paste your Python code into the **original Python code** editor, then:
* **▶️ Run original code** — executes it in an isolated subprocess and shows output, exit code, and timing.
* **✨ Optimize code** — sends it to the LLM and drops the optimized version into the second editor.
* **▶️ Run optimized code** — executes the optimized version; once both sides have been run, the comparison panel reports the speedup and whether stdout matches.

Blank code is caught immediately with a clear in-UI error before anything is sent to the LLM or executed. Editing the original code after running it automatically clears any existing comparison, so you're never shown a stale speedup measured against code that's no longer in the box.

---

## ⚠️ A Note on Code Execution

Both the original and optimized code run as real subprocesses on your machine, using the same Python interpreter and environment as the app itself — there's no sandboxing, resource limits (beyond a 60-second timeout), or restricted permissions. That's fine for a local, single-user tool like this one, but don't expose it (e.g. via Gradio's `share=True`) to anyone you wouldn't hand a terminal to.

---

## 🧪 Trying It Out
A few snippets that reliably give the LLM something real to optimize:
* A Fibonacci function using naive recursion (no memoization) — a classic case for the model to fix.
```python
def nth_fibonacci(n):
    if n <= 1:
        return n
    return nth_fibonacci(n - 1) + nth_fibonacci(n - 2)

if __name__ == "__main__":
    print(nth_fibonacci(40))
```

* Maximum Subarray Sum using 2 loops instead of Kadane's algorithm.
```python
import random

def max_subarray_sum(arr):
    res = arr[0]
    for i in range(len(arr)):
        curr_sum = 0
        for j in range(i, len(arr)):
            curr_sum = curr_sum + arr[j]
            res = max(res, curr_sum)
    return res

if __name__ == "__main__":
    random.seed(42)
    random_numbers = [random.randint(-1000, 1000) for _ in range(10000)]
    print(max_subarray_sum(random_numbers))
```