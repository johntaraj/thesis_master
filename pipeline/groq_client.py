"""Groq API client wrapper with model switching."""

from groq import Groq
from config import GROQ_API_KEY, DEFAULT_MODEL


class GroqClient:
    """Wrapper around the Groq SDK supporting multiple models."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        raw_key = api_key or GROQ_API_KEY
        # Strip non-ASCII chars (e.g. \u2011 non-breaking hyphens from copy-paste)
        self.api_key = raw_key.encode("ascii", errors="ignore").decode("ascii").strip()
        self.model = model
        self._client = Groq(api_key=self.api_key)

    def set_model(self, model: str):
        self.model = model

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 256) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def simplify_word(self, word: str, sentence: str) -> str:
        """Use the LLM to simplify a single word/phrase in context."""
        system = "You are a medical text simplifier. Return ONLY the simpler replacement word or short phrase. No explanations."
        user = (
            f"Replace the word/phrase '{word}' in the following sentence with a "
            f"simpler word or phrase that a non-medical person (8th-grade reading "
            f"level) would understand. Keep the same meaning.\n\n"
            f"Sentence: {sentence}\n\n"
            f"Simpler replacement for '{word}':"
        )
        return self._chat(system, user)

    def simplify_definition(self, word: str, definition: str) -> str:
        """Given a medical definition, produce a simple replacement phrase."""
        system = "You are a medical text simplifier. Return ONLY a simple replacement phrase (max 5 words). No explanations."
        user = (
            f"Given this medical definition of '{word}':\n\n"
            f"{definition}\n\n"
            f"Provide a simple phrase (max 5 words) that a non-medical person "
            f"would understand as a replacement for '{word}':"
        )
        return self._chat(system, user)

    def simplify_text(self, text: str) -> str:
        """Simplify an entire clinical note (baseline mode)."""
        system = (
            "You are a medical text simplifier for patients with an 8th-grade reading level. "
            "Simplify the text while keeping all important medical information."
        )
        user = (
            "Simplify the following clinical note. Rules:\n"
            "1. Expand ALL medical abbreviations (e.g., BP → blood pressure).\n"
            "2. Replace ALL difficult medical and non-medical words with simpler alternatives.\n"
            "3. Keep the same meaning and all important medical information.\n"
            "4. Make every sentence easy for a non-medical person to understand.\n\n"
            f"Clinical note:\n{text}\n\n"
            "Return the simplified version of the note."
        )
        return self._chat(system, user, max_tokens=2048)

    def extract_word_map(self, original: str, simplified: str) -> str:
        """Ask LLM to produce a JSON mapping of original→simplified words."""
        system = "You are a text comparison assistant. Return ONLY valid JSON. No markdown, no explanations."
        user = (
            "Compare the original clinical note and simplified version below. "
            "List EVERY word or phrase that was changed.\n\n"
            f"Original:\n{original}\n\n"
            f"Simplified:\n{simplified}\n\n"
            'Return a JSON object where keys are original words/phrases and values are '
            'their simplified replacements. Example: {"hypertension": "high blood pressure", "PRN": "as needed"}\n'
            "Return ONLY the JSON object."
        )
        return self._chat(system, user, max_tokens=2048)

    def simplify_non_medical_words(self, sentence: str, words: list[str]) -> str:
        """Simplify non-medical hard words in a sentence after medical terms are already replaced."""
        system = "You are a text simplifier. Rewrite the sentence replacing only the specified hard words with simpler alternatives. Return ONLY the rewritten sentence."
        user = (
            f"Simplify the following sentence by replacing these hard words with "
            f"simpler alternatives: {', '.join(words)}\n\n"
            f"Sentence: {sentence}\n\n"
            f"Rewritten sentence:"
        )
        return self._chat(system, user, max_tokens=512)

    def rationalize(self, original: str, simplified: str, source_name: str, passage: str) -> str:
        """Generate a rationale for why a word was replaced."""
        system = "You are explaining medical text simplification choices. Return ONLY one clear sentence."
        user = (
            f"We replaced '{original}' with '{simplified}' in a clinical note "
            f"to make it easier for patients to understand.\n"
            f"Source: {source_name}\n"
            f"Source passage: {passage}\n\n"
            f"Explain in 1 sentence why this replacement makes the text easier to understand."
        )
        return self._chat(system, user)
