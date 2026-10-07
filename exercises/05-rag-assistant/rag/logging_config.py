import logging
import os
import sys

import litellm


def setup_logging(level: str | None = None):
    """Configures root logging to print to the console.

    Args:
        level (str | None): Log level name.
            Defaults to the LOG_LEVEL environment variable, or "INFO" if that is not set.
    """
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s | %(levelname)-7s | %(threadName)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "LiteLLM", "litellm",
                  "chromadb", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    litellm.suppress_debug_info = True