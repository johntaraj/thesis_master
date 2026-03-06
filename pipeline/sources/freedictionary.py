"""FreeDictionary medical dictionary scraper."""

import time
import requests
from bs4 import BeautifulSoup
from pipeline.sources.base import BaseSource, SourceResult
from config import FREEDICTIONARY_BASE_URL, FREEDICTIONARY_RATE_LIMIT_SEC, MAX_RESULTS_PER_SOURCE


class FreeDictionarySource(BaseSource):
    """Scrape medical-dictionary.thefreedictionary.com for definitions."""

    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, groq_client=None):
        self._groq = groq_client
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self._USER_AGENT})

    @property
    def name(self) -> str:
        return "FreeDictionary"

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < FREEDICTIONARY_RATE_LIMIT_SEC:
            time.sleep(FREEDICTIONARY_RATE_LIMIT_SEC - elapsed)
        self._last_request_time = time.time()

    def search(self, term: str) -> list[SourceResult]:
        results: list[SourceResult] = []
        try:
            self._rate_limit()
            url = f"{FREEDICTIONARY_BASE_URL}/{requests.utils.quote(term)}"
            resp = self._session.get(url, timeout=10)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            # The main definition section
            definitions = []

            # Try the main definition div (section with id 'Definition')
            main_div = soup.find("div", {"id": "Definition"})
            if main_div:
                # Get paragraphs with definitions
                for p in main_div.find_all(["p", "div"], limit=5):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:
                        definitions.append(text)

            # Fallback: try the main content section
            if not definitions:
                content = soup.find("section", {"data-src": "hm"})
                if content:
                    for p in content.find_all("p", limit=5):
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:
                            definitions.append(text)

            # Fallback: broader search
            if not definitions:
                for div in soup.find_all("div", class_=lambda c: c and "ds-list" in str(c)):
                    text = div.get_text(strip=True)
                    if text and len(text) > 20:
                        definitions.append(text)

            # Take up to MAX_RESULTS_PER_SOURCE definitions
            for defn in definitions[:MAX_RESULTS_PER_SOURCE]:
                simplified = defn
                if self._groq:
                    simplified = self._groq.simplify_definition(term, defn)

                results.append(SourceResult(
                    source_name=self.name,
                    term=term,
                    definition=defn,
                    simplified_text=simplified,
                    url=url,
                    passage=defn[:300],
                ))

        except Exception:
            pass  # Graceful failure — source unavailable

        return results
