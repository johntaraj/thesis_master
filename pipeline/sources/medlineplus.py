"""MedlinePlus web service API client."""

import requests
from lxml import etree
from pipeline.sources.base import BaseSource, SourceResult
from config import MEDLINEPLUS_BASE_URL, MAX_RESULTS_PER_SOURCE


class MedlinePlusSource(BaseSource):
    """Query the MedlinePlus health topics web service."""

    def __init__(self, groq_client=None):
        self._groq = groq_client

    @property
    def name(self) -> str:
        return "MedlinePlus"

    def search(self, term: str) -> list[SourceResult]:
        results: list[SourceResult] = []
        try:
            params = {
                "db": "healthTopics",
                "term": term,
                "retmax": MAX_RESULTS_PER_SOURCE,
                "rettype": "brief",
                "tool": "ClinicalNoteSimplifier",
                "email": "thesis-research@example.com",
            }
            resp = requests.get(MEDLINEPLUS_BASE_URL, params=params, timeout=10)
            if resp.status_code != 200:
                return []

            root = etree.fromstring(resp.content)

            for doc in root.findall(".//document"):
                url = doc.get("url", "")

                # Extract title
                title = ""
                summary = ""
                snippet = ""

                for content in doc.findall("content"):
                    name = content.get("name", "")
                    # Get text content, stripping HTML tags
                    text = etree.tostring(content, method="text", encoding="unicode").strip()

                    if name == "title":
                        title = text
                    elif name == "FullSummary":
                        summary = text
                    elif name == "snippet":
                        snippet = text

                # Use the best available text
                passage = summary or snippet or title
                if not passage or len(passage) < 10:
                    continue

                # Truncate very long summaries
                if len(passage) > 500:
                    passage = passage[:500] + "..."

                simplified = passage
                if self._groq:
                    simplified = self._groq.simplify_definition(term, passage)

                results.append(SourceResult(
                    source_name=self.name,
                    term=term,
                    definition=passage,
                    simplified_text=simplified,
                    url=url,
                    passage=passage[:300],
                ))

                if len(results) >= MAX_RESULTS_PER_SOURCE:
                    break

        except Exception:
            pass  # Graceful failure

        return results
