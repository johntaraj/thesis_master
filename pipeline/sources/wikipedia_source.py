"""Wikipedia API source for medical term definitions."""

import wikipedia
from pipeline.sources.base import BaseSource, SourceResult
from config import WIKIPEDIA_SENTENCES, MAX_RESULTS_PER_SOURCE


class WikipediaSource(BaseSource):
    """Retrieve term definitions from Wikipedia."""

    def __init__(self, groq_client=None):
        self._groq = groq_client

    @property
    def name(self) -> str:
        return "Wikipedia"

    def search(self, term: str) -> list[SourceResult]:
        results: list[SourceResult] = []
        try:
            # Try direct summary first
            try:
                summary = wikipedia.summary(term, sentences=WIKIPEDIA_SENTENCES)
                page = wikipedia.page(term, auto_suggest=False)
                url = page.url
            except wikipedia.DisambiguationError as e:
                # Take the first suggestion
                if e.options:
                    first_option = e.options[0]
                    summary = wikipedia.summary(first_option, sentences=WIKIPEDIA_SENTENCES)
                    page = wikipedia.page(first_option, auto_suggest=False)
                    url = page.url
                else:
                    return []
            except wikipedia.PageError:
                # Try search instead
                search_results = wikipedia.search(term, results=2)
                if not search_results:
                    return []
                summary = wikipedia.summary(search_results[0], sentences=WIKIPEDIA_SENTENCES)
                page = wikipedia.page(search_results[0], auto_suggest=False)
                url = page.url

            if not summary or len(summary) < 10:
                return []

            passage = summary[:500]

            simplified = passage
            if self._groq:
                simplified = self._groq.simplify_definition(term, passage)

            results.append(SourceResult(
                source_name=self.name,
                term=term,
                definition=summary,
                simplified_text=simplified,
                url=url,
                passage=passage[:300],
            ))

            # Try to get a second result from search if available
            if len(results) < MAX_RESULTS_PER_SOURCE:
                search_results = wikipedia.search(term + " medical", results=2)
                for sr in search_results:
                    if sr.lower() != term.lower() and len(results) < MAX_RESULTS_PER_SOURCE:
                        try:
                            extra_summary = wikipedia.summary(sr, sentences=WIKIPEDIA_SENTENCES)
                            extra_page = wikipedia.page(sr, auto_suggest=False)
                            if extra_summary and len(extra_summary) > 10:
                                extra_simplified = extra_summary
                                if self._groq:
                                    extra_simplified = self._groq.simplify_definition(term, extra_summary[:500])
                                results.append(SourceResult(
                                    source_name=self.name,
                                    term=term,
                                    definition=extra_summary,
                                    simplified_text=extra_simplified,
                                    url=extra_page.url,
                                    passage=extra_summary[:300],
                                ))
                        except Exception:
                            continue

        except Exception:
            pass  # Graceful failure

        return results
