# Exercise 01: Intelligent Website Summarizer 🌐

A CLI application that extracts the core content of any webpage and passes it to a Large Language Model (LLM) — local (Ollama), or any OpenAI-compatible cloud provider — to generate a structured, clean Markdown summary.

---

## 🏗️ Technical Architecture

The application is split across two core modules:

* **`utility.py` (Scraper Engine):** Leverages `playwright` to run a headless Chromium instance, handling dynamic rendering, paired with `beautifulsoup4` for raw text extraction. Validates the target URL before scraping, blocks non-essential resources (images, fonts, stylesheets) for faster page loads, and strips noise so only meaningful content reaches the LLM.
* **`summarizer.py` (Orchestrator):** Manages the LLM using the `openai` API Client Library, pointed at any OpenAI-compatible endpoint — local (Ollama) or cloud. Parses CLI arguments, distinguishes connection errors, API errors, and scraping timeouts for clearer failure messages, and outputs the summary via `rich` terminal markdown formatting.

---

## 🚀 How to Run It

Ensure you have synchronized the root environment via `uv sync` from the main project root folder first.

#### 📦 Core Libraries Used
*   **`openai`**  — Orchestrates the LLM prompts and routes requests to any OpenAI-compatible backend (local Ollama or a cloud provider).
*   **`playwright`** — Handles dynamic webpage rendering.
*   **`beautifulsoup4`** — Parses HTML and extracts clean, raw text from web pages.
*   **`rich`** — Renders the final structured Markdown beautifully directly within the CLI terminal.

#### ⚙️ Configuration
Settings can be overridden via environment variables or command-line flags. CLI flags take priority over environment variables, which take priority over the built-in defaults.

| Setting | Env Var | CLI Flag | Default |
|---|---|---|---|
| Model name | `LLM_MODEL` | `--model` | `llama3.2` |
| Backend base URL | `LLM_BASE_URL` | `--base-url` | `http://localhost:11434/v1` |
| API key | `LLM_API_KEY` | — | `ollama` |

The target URL is a positional argument (no flag needed) and defaults to `https://news.ycombinator.com` if omitted.

### 1. Set up your LLM backend
**Using local Ollama (the default):** make sure your instance is active and you have the model downloaded:
```bash
ollama run llama3.2
```
**Using a cloud provider instead:** set `LLM_BASE_URL` and `LLM_API_KEY` (via env vars or `--base-url`) to point at that provider's endpoint and key — no Ollama installation needed in that case.

### 2. Execute the pipeline
Run the script using `uv run`, passing any target URL as an argument or using the default fallback URL:
```bash
# Option A: Run with the default target URL (news.ycombinator.com)
uv run exercises/01-website-summarizer/summarizer.py

# Option B: Run with a custom target URL and model
uv run exercises/01-website-summarizer/summarizer.py https://ollama.com --model mistral

# Option C: Point at a different LLM backend for this run only
uv run exercises/01-website-summarizer/summarizer.py https://ollama.com --base-url https://api.example.com/v1
```
Malformed URLs (missing `http://`/`https://`, or missing a domain) are caught immediately with a clear error before any scraping begins.
