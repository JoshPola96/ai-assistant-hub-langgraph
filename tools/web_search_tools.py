# tools/web_search_tools.py

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

# Constants
MAX_CONTENT_LENGTH = 15000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# DuckDuckGo Tool Initialization
duckduckgo_search_tool = DuckDuckGoSearchResults()


@tool
def duckduckgo_search(query: str, chat_history: str = "") -> str:
    """
    Search the web using DuckDuckGo. Returns snippets with titles and URLs.
    """
    print(f"\n[DuckDuckGo Search] Received query: '{query}'")

    if not query.strip():
        return "Please provide a valid search query."

    try:
        results = duckduckgo_search_tool.run(query)
        print(f"[DuckDuckGo Search] Success. First 200 chars:\n{results[:200]}...")
        return results
    except Exception as e:
        print(f"[DuckDuckGo Search] Error: {e}")
        return f"Web search failed: {e}"


@tool
def fetch_web_page_content(url: str) -> str:
    """
    Fetch and return readable text content from a web page.
    Strips scripts, navigation, and excess formatting.
    """
    print(f"\n[Fetch Web Page] Received URL: {url}")

    # Validate the URL
    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc):
        return "Invalid URL. Please include http:// or https://."

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content tags
        for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH] + "\n... [Truncated due to length limits]"

        print(f"[Fetch Web Page] Content preview: {text[:200]}...")
        return text

    except requests.exceptions.RequestException as e:
        print(f"[Fetch Web Page] Network error: {e}")
        return f"Unable to fetch content from {url}: {e}"

    except Exception as e:
        print(f"[Fetch Web Page] Parsing error: {e}")
        return f"Failed to extract readable content from {url}: {e}"


# Optional CLI tests
if __name__ == "__main__":
    print("🔍 Testing Web Search Tools")

    # Test DuckDuckGo Search
    test_query = "latest news on renewable energy"
    print("\n[TEST] DuckDuckGo Search:")
    print(duckduckgo_search.invoke({"query": test_query}))

    # Test Web Page Fetch
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print("\n[TEST] Fetch Web Page Content:")
    print(fetch_web_page_content.invoke({"url": test_url})[:500])

    # Test Invalid URL
    invalid_url = "not-a-real-url"
    print("\n[TEST] Invalid URL:")
    print(fetch_web_page_content.invoke({"url": invalid_url}))
