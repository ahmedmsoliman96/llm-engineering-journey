# Exercise 01: Intelligent Website Summarizer 🌐

A CLI application that extracts the core content of any webpage and passes it to a localized Large Language Model (LLM) to generate a structured, clean Markdown summary.

---

## 🏗️ Technical Architecture

The application is split across two core modules:

* **`utility.py` (Scraper Engine):** Leverages `playwright` to run a headless Chromium instance, handling dynamic rendering, paired with `beautifulsoup4` for raw text extraction.
* **`summarizer.py` (Orchestrator):** Manages the local LLM using the `openai` API Client Library mapped to an Ollama server backend, and outputs the summary via `rich` terminal markdown formatting.

---

## 🚀 How to Run It

Ensure you have synchronized the root environment via `uv sync` from the main project root folder first.

#### 📦 Core Libraries Used
*   **`openai`**  — Orchestrates the LLM prompts and routes requests to the local Ollama backend.
*   **`playwright`** — Handles dynamic webpage rendering.
*   **`beautifulsoup4`** — Parses HTML and extracts clean, raw text from web pages.
*   **`rich`** — Renders the final structured Markdown beautifully directly within the CLI terminal.

#### ⚙️ Optional Configuration
The application checks for environment variables if you wish to override the defaults:

* `LLM_MODEL` (Defaults to `llama3.2`)

* `OLLAMA_BASE_URL` (Defaults to `http://localhost:11434/v1`)

### 1. Start your local LLM
Make sure your Ollama instance is active and you have the model downloaded:
```bash
ollama run llama3.2
```
### 2. Execute the pipeline
Run the script using `uv run`, passing any target URL as an argument or using the default fallback URL:
```bash
# Option A: Run with the default target URL (ollama.com)
uv run exercises/01-website-summarizer/summarizer.py

# Option B: Run with a custom target URL of your choice
uv run exercises/01-website-summarizer/summarizer.py https://news.ycombinator.com
```

