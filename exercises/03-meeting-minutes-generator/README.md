# Exercise 03: Meeting Minutes Generator 📝

A Gradio web app that transcribes a meeting audio recording locally using a Whisper model through a Hugging Face pipeline, then turns the transcript into structured Markdown meeting minutes using an LLM served through the Hugging Face Inference API.

---

## 🏗️ Technical Architecture

The application handles two distinct stages within a single module `minutes_generator.py` which collects the audio file and model name through Gradio UI, and streams back clean Markdown.

* **Local transcription (Hugging Face `transformers`):** Loads a Whisper model via the `automatic-speech-recognition` pipeline, running on CUDA if a GPU is available or falling back to CPU automatically (with a smaller default model to keep it practical).
* **Minutes generation (LLM via Hugging Face Inference API):** Sends the transcript to an OpenAI-compatible chat completion endpoint provided by Hugging Face's Inference router. Prompts the model to write structured minutes (summary, key discussion points, takeaways, action items). 

---

## 🚀 How to Run It

Ensure you have synchronized the root environment via `uv sync` from the main project root folder first.

#### 📦 Core Libraries Used
*   **`transformers`** — Runs the Whisper speech-to-text pipeline locally.
*   **`torch`** — Backend tensor library for Whisper; used with a CUDA build if available.
*   **`accelerate`** — Supports efficient model loading and dtype handling for `transformers`.
*   **`openai`** — Orchestrates the LLM prompt and routes requests to any OpenAI-compatible backend (Hugging Face Inference API by default).
*   **`gradio`** — Provides the browser-based UI, taking the audio file as input and streaming back the generated minutes.
*   **`python-dotenv`** — Loads environment variables from a local `.env` file.

#### 🎬 System Requirement: ffmpeg
Whisper's audio decoding step shells out to `ffmpeg` directly, so it must be installed and available on your system PATH (this is a system tool, not a Python package):
* **Windows:** `winget install ffmpeg`
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt install ffmpeg`

#### ⚙️ Configuration
The audio file and model name are provided directly in the browser. Everything else is fixed at startup via environment variables (a `.env` file in the project root is loaded automatically).

| Setting                | Env Var          | Editable in UI?     | Default                                                                 |
|------------------------|------------------|---------------------|-------------------------------------------------------------------------|
| LLM model name         | `HF_MODEL`       | ✅ yes, per request | `meta-llama/Llama-3.1-8B-Instruct`                                      |
| LLM backend base URL   | `HF_BASE_URL`    | ❌ startup only     | `https://router.huggingface.co/v1`                                      |
| Hugging Face API token | `HF_TOKEN`       | ❌ startup only     | *(required — no default)*                                               |
| Whisper model          | `WHISPER_MODEL`  | ❌ startup only     | `openai/whisper-large-v3-turbo` (GPU) / `openai/whisper-small.en` (CPU) |
| Whisper device         | `WHISPER_DEVICE` | ❌ startup only     | auto-detected: `cuda` if available, else `cpu`                          |

GPU/CPU selection and the Whisper model size are both auto-detected at startup and only need overriding if you want to force a specific setup.

### 1. Install ffmpeg
See the System Requirement section above — do this once, before the first run.

### 2. Get a Hugging Face API token
Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with the appropriate permissions enabled. If you're using a gated model like Llama 3.1 (the default here), accept Meta's license on [the model's page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) first and make sure you have been granted access — this is required before your token can call it. Put the token in a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

### 3. Launch the app
Run the script using `uv run`. It starts a local Gradio server and opens a browser tab automatically:
```bash
uv run exercises/03-meeting-minutes-generator/minutes_generator.py
```
Once the tab opens, upload a meeting audio file and, optionally, enter a different model name than the pre-filled default. Missing audio files and blank model names are caught immediately with a clear in-UI error before transcription begins.

---

## 🎧 Testing with Sample Audio

If you don't have a meeting recording handy, [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank) is a public research benchmark of real city council meetings from 6 U.S. cities (Alameda, Boston, Denver, King County, Long Beach, Seattle), with matching audio hosted at [huuuyeah/MeetingBank_Audio](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main). Browse into any city's folder and download the meeting file of your choice.

A few things worth knowing before testing with these:
* Meetings average 2.6 hours and ~28k transcript tokens — comfortably within Llama 3.1's context window, but expect transcription itself to take noticeably longer than a short clip. Try a short clip first to get a feel for timing before running a full meeting.
* The paired [meetingbank](https://huggingface.co/datasets/huuuyeah/meetingbank) dataset includes a human-written reference summary for each meeting — a useful way to informally check the quality of this app's generated minutes against a real one.
* Licensed `cc-by-nc-sa-4.0` (non-commercial, attribution required) — fine for personal testing; check the license before other use, and cite [the MeetingBank paper](https://arxiv.org/abs/2305.17529) if you build on it further.
