import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

NOISE_TAGS = ["script", "style", "img", "input", "nav", "footer", "header", "svg", "noscript", "form"]

def fetch_website_contents(url: str, max_chars=20000) -> str:
    """
    Launches a headless browser to extract clean text from a website.
    Handles dynamic content via Playwright and cleans noise via BeautifulSoup.
    Args:
        url (str): The target website address to scrape.
        max_chars (int): The maximum number of returned string. Defaults to 20,000.
    Returns:
        str: text content from the page, truncated max_chars.
    Raises:
        RuntimeError: If the browser cannot connect to or scrape the URL.
    """
    #1. Start an isolated Playwright browser session
    with sync_playwright() as p:
        logger.info(f"Launching browser to fetch: {url}")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font", "stylesheet")
            else route.continue_(),
        )

        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)
            html_content = page.content()
            logger.info("Successfully fetched HTML content.")
        except Exception as e:
            logger.error(f"Failed to scrape website {url}: {str(e)}")
            raise RuntimeError(f"Error capturing page content: {str(e)}")
        finally:
            browser.close()

    # 2. Parse and prune the rendered HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "No title found"

    if soup.body:
        for irrelevant in soup.body(NOISE_TAGS):
            irrelevant.decompose()

        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""

    return f"TITLE: {title}\n\n{text}"[:max_chars]


def validate_url(url: str) -> None:
    """
    Validates that a URL is well-formed and uses http/https.
    Args:
        url (str): The URL to validate.
    Raises:
        RuntimeError: If the URL has no scheme, an unsupported scheme, or no domain.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise RuntimeError(f"Invalid URL '{url}': must start with http:// or https://")
    if not parsed_url.netloc:
        raise RuntimeError(f"Invalid URL '{url}': missing a domain")