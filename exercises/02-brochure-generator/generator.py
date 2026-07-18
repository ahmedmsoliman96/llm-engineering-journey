import os
import logging

import gradio as gr
from openai import APIConnectionError, APIStatusError, OpenAI
from utility import fetch_website_contents, validate_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def messages_for(website_content:str, system_prompt:str, user_prompt:str) -> list[dict[str, str]]:

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\nWeb content:\n{website_content}"}
    ]

def generate(url:str, client:OpenAI, model:str, system_prompt:str, user_prompt:str):

    url = url.strip()
    model = model.strip()

    if not url:
        raise gr.Error("please enter a URL")
    if not model:
        raise gr.Error("please enter a model name")

    try:
        validate_url(url)
    except RuntimeError as e:
        raise gr.Error(str(e))

    try:
        website_content = fetch_website_contents(url)
    except RuntimeError as e:
        logger.error(f"failed to fetch website content:\n *{str(e)}*")
        raise gr.Error(f"could not fetch website content:\n *{str(e)}*")

    try:
        logger.info(f"sending website content to LLM model: {model}")
        stream = client.chat.completions.create(
            model = model,
            messages = messages_for(website_content, system_prompt, user_prompt),
            stream = True
        )
        result = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
                yield result

    except APIConnectionError as e:
        logger.error(f"Could not connect to LLM backend: {str(e)}")
        raise gr.Error(f"Failed to generate a brochure via LLM:\n"
                       f"*Could not connect to LLM backend at the configured base URL: {str(e)}*")
    except APIStatusError as e:
        logger.error(f"LLM backend returned an error: {str(e)}")
        raise gr.Error(f"Failed to generate a brochure via LLM:\n *LLM backend returned an error: {str(e)}*")
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise gr.Error(f"Failed to generate a brochure via LLM:\n *{str(e)}*")

    if not result:
        raise gr.Error("LLM returned an empty response")

def display_brochure(client:OpenAI, default_model:str, system_prompt:str, user_prompt:str):

    def generate_wrapper(url, model):
        yield from generate(url, client, model, system_prompt, user_prompt)

    company_url = gr.Textbox(label="Company website URL including http:// or https://")
    model_name = gr.Textbox(label="Model name (e.g. llama3.2, gpt-4o-mini)", value=default_model)
    brochure_output = gr.Markdown(label="brochure:")
    view = gr.Interface(
        fn=generate_wrapper,
        title="🏢 Company Brochure Generator",
        inputs=[company_url,model_name],
        outputs=[brochure_output],
        flagging_mode="never"
    )
    view.launch(inbrowser = True)

if __name__ == "__main__":

    DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.2")
    DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=LLM_API_KEY)

    system_prompt = (
        "You are an AI assistant that writes a company brochure in Markdown "
        "based on the content of a company's webpage. Include sections such as "
        "Company Overview, Products/Services, Culture, and Careers if relevant "
        "information is present. If a section has no supporting content, omit it "
        "rather than inventing details. Respond strictly in clean markdown syntax. "
        "Do not wrap the markdown in a code block."
    )

    user_prompt = "Please generate a well-structured company brochure from the following webpage content."

    display_brochure(client, DEFAULT_MODEL, system_prompt, user_prompt)