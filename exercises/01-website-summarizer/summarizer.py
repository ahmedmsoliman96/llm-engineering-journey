import os
import logging
import sys
from openai import OpenAI
from utility import fetch_website_contents
from rich.console import Console
from rich.markdown import Markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

def messages_for(website_content:str, system_prompt:str, user_prompt:str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\nWeb content:\n{website_content}"}
    ]

def summarize(url:str, client:OpenAI, model:str, system_prompt:str, user_prompt:str) -> str:
    try:
        website_content = fetch_website_contents(url)
    except RuntimeError as e:
        return f"Summarization aborted:\n *{str(e)}*"

    try:
        logger.info(f"sending website content to LLM model: {model}")
        response = client.chat.completions.create(
            model = model,
            messages = messages_for(website_content, system_prompt, user_prompt)
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        return f"Failed to generate summary via LLM:\n *{str(e)}*"

def display_summary(url:str, client:OpenAI, model:str, system_prompt:str, user_prompt:str):
    with console.status("[bold green]Scraping and summarizing website...", spinner="dots"):
        summary = summarize(url, client, model, system_prompt, user_prompt)

    console.print("\n[bold cyan]=== AI Generated Summary ===[/bold cyan]\n")
    console.print(Markdown(summary))
    console.print("\n[bold cyan]============================[/bold cyan]")

if __name__ == "__main__":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://ollama.com"

    client = OpenAI(base_url=OLLAMA_BASE_URL,api_key=OLLAMA_API_KEY)

    system_prompt = (
        "You are a helpful AI assistant that analyzes the contents of a website, "
        "and delivers a structured, high-quality summary. Ignore navigation headers. "
        "Respond strictly in clean markdown syntax. Do not wrap the markdown in a code block"
    )

    user_prompt = "please provide a concise summary emphasizing key features of the following website content."

    display_summary(target_url,client,LLM_MODEL,system_prompt,user_prompt)