import os
import logging
from dotenv import load_dotenv

import gradio as gr
from openai import APIConnectionError, APIStatusError, OpenAI
import torch
from transformers import pipeline, Pipeline
from utility import get_audio_duration, format_duration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def model_messages(transcript: str, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\nTranscript:\n{transcript}"}
    ]

def transcribe_audio(audio_path: str, asr_pipeline: Pipeline) -> str:
    try:
        logger.info(f"transcribing audio file: {audio_path}")
        result = asr_pipeline(audio_path)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"failed to transcribe the audio file: {str(e)}")
        raise RuntimeError(f"could not transcribe the audio file: {str(e)}")


def generate(audio_path: str, asr_pipeline: Pipeline, client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    model = model.strip()

    if not audio_path:
        raise gr.Error("please upload an audio file")
    if not model:
        raise gr.Error("please enter a model name")

    duration = get_audio_duration(audio_path)
    if duration is not None:
        yield f"⏳ Transcribing a {format_duration(duration)} recording, this may take a few minutes for longer files..."
    else:
        yield "⏳ Transcribing audio, this may take a few minutes for longer files..."

    try:
        transcript = transcribe_audio(audio_path, asr_pipeline)
    except RuntimeError as e:
        raise gr.Error(str(e))
    
    if not transcript:
        raise gr.Error("transcription return empty text, check the audio file")

    try:
        logger.info(f"sending transcript to LLM model: {model}")
        stream = client.chat.completions.create(
            model=model,
            messages=model_messages(transcript, system_prompt, user_prompt),
            stream=True
        )
        result = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
                yield result

    except APIConnectionError as e:
        logger.error(f"failed to connect to LLM backend: {str(e)}")
        raise gr.Error(f"Failed to generate meeting minutes via LLM:\n"
                       f"*Could not connect to LLM backend at the configured base URL: {str(e)}*")

    except APIStatusError as e:
        logger.error(f"LLM backend returned an error: {str(e)}")
        raise gr.Error(f"Failed to generate meeting minutes via LLM:\n *LLM backend returned an error: {str(e)}*")

    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise gr.Error(f"Failed to generate meeting minutes via LLM:\n *{str(e)}*")

    if not result:
        raise gr.Error("LLM returned an empty response")


def display_minutes(asr_pipeline: Pipeline, client: OpenAI, default_model: str, system_prompt: str, user_prompt: str):
    def generate_wrapper(audio_path, model):
        yield from generate(audio_path, asr_pipeline, client, model, system_prompt, user_prompt)

    audio_input = gr.Audio(
        label="Meeting audio",
        type="filepath",
        sources=["upload"],
        waveform_options=gr.WaveformOptions(show_recording_waveform=False)
    )
    model_name = gr.Textbox(label="Model name (e.g. meta-llama/Llama-3.1-8B-Instruct)", value=default_model)
    minutes_output = gr.Markdown(label="Meeting minutes:")

    view = gr.Interface(
        fn=generate_wrapper,
        title="📝 Meeting Minutes Generator",
        inputs=[audio_input, model_name],
        outputs=[minutes_output],
        flagging_mode="never"
    )
    view.launch(inbrowser=True)


if __name__ == "__main__":

    load_dotenv()

    DEFAULT_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    DEFAULT_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        logger.error("HF_TOKEN environment variable not set")
        raise SystemExit("Missing HF_TOKEN: set it in your environment variable before running this script")

    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=HF_TOKEN)

    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    WHISPER_DTYPE = torch.float16 if WHISPER_DEVICE == "cuda" else torch.float32

    if WHISPER_DEVICE == "cpu":
        logger.warning("no CUDA GPU found, using CPU instead. Transcription will be slower. "
                       "Override WHISPER_MODEL with a smaller one (e.g. openai/whisper-small.en)"
                       )

    DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo" if WHISPER_DEVICE == "cuda" else "openai/whisper-small.en"
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)

    logger.info(f"loading Hugging Face ASR pipeline: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        dtype=WHISPER_DTYPE,
        device=WHISPER_DEVICE,
        return_timestamps=True,
    )

    system_prompt = (
        "You are an AI assistant that writes clear, structured meeting minutes in Markdown "
        "based on a raw meeting transcript. Include sections like summary, key discussion points, "
        "takeaways and action items with owners if relevant information is present. If a section "
        "has no supporting content, omit it rather than inventing details. respond in clear markdown "
        "with no surrounding code block"
    )

    user_prompt = (
        "please generate well structured meeting "
        "minutes from the meeting transcript down below "
    )

    display_minutes(asr_pipeline, client, DEFAULT_MODEL, system_prompt, user_prompt)
