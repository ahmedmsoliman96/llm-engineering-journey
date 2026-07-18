# Exercise 02: Company Brochure Generator 🏢

A Gradio web app that scrapes a company's homepage and turns it into a structured Markdown brochure using a Large Language Model (LLM) — local (Ollama), or any OpenAI-compatible cloud provider.

---

## 🏗️ Technical Architecture

The application is split across two core modules:

* **`utility.py` (Scraper Engine):** leverages `playwright` to run a headless Chromium instance, handling dynamic rendering, paired with `beautifulsoup4` for raw text extraction. Validates the target URL, blocks non-essential resources (images, fonts, stylesheets) for faster page loads, and strips noise so only meaningful content reaches the LLM.
* **`generator.py` (Orchestrator & UI Layer):** Manages the LLM using the `openai` API Client Library, pointed at any OpenAI-compatible endpoint — local (Ollama) or cloud. Prompts the LLM to write brochure-style sections (Company Overview, Products/Services, Culture, Careers). collects the URL and model name through a `gradio.Interface`, and streams back clean Markdown brochure.

---

## 🚀 How to Run It

Ensure you have synchronized the root environment via `uv sync` from the main project root folder first.

#### 📦 Core Libraries Used
*   **`openai`** — Orchestrates the LLM prompts and routes requests to any OpenAI-compatible backend (local Ollama or a cloud provider).
*   **`playwright`** — Handles dynamic webpage rendering.
*   **`beautifulsoup4`** — Parses HTML and extracts clean, raw text from web pages.
*   **`gradio`** — Provides the browser-based UI and streams the generated brochure.

#### ⚙️ Configuration
The model name can be changed per request directly in the browser. The backend base URL and API key are fixed at startup via environment variables only.

| Setting | Env Var | Editable in UI? | Default |
|---|---|---|---|
| Model name | `LLM_MODEL` | ✅ yes, per request | `llama3.2` |
| Backend base URL | `LLM_BASE_URL` | ❌ startup only | `http://localhost:11434/v1` |
| API key | `LLM_API_KEY` | ❌ startup only | `ollama` |

The company URL is entered directly in the browser form.

### 1. Set up your LLM backend
**Using local Ollama (the default):** make sure your instance is active and you have the model downloaded:
```bash
ollama run llama3.2
```
**Using a cloud provider instead:** set `LLM_BASE_URL` and `LLM_API_KEY` (via env vars) to point at that provider's endpoint and key before launching — no Ollama installation needed in that case.

### 2. Launch the app
Run the script using `uv run`. It starts a local Gradio server and opens a browser tab automatically:
```bash
uv run exercises/02-brochure-generator/generator.py
```
Once the tab opens, enter a company's homepage URL (including `http://` or `https://`) and, optionally, a different model name than the pre-filled default. Malformed URLs and blank fields are caught immediately with a clear in-UI error before any scraping begins.
