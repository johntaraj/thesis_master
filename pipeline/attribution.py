"""ContextCite-inspired attribution and rationalization."""

from pipeline.sources.base import Replacement
from pipeline.groq_client import GroqClient


def generate_rationale(
    groq_client: GroqClient,
    replacement: Replacement,
) -> str:
    """Generate a human-readable rationale for a word replacement."""
    if replacement.is_abbreviation:
        return (
            f"'{replacement.original}' is a medical abbreviation that stands for "
            f"'{replacement.simplified}'. We expanded it so patients can understand "
            f"the full meaning."
        )

    if replacement.source_name.startswith("LLM"):
        passage = "No external source found. The model generated a simpler alternative."
    else:
        passage = replacement.source_passage or "Definition from external source."

    return groq_client.rationalize(
        original=replacement.original,
        simplified=replacement.simplified,
        source_name=replacement.source_name,
        passage=passage,
    )


def add_rationales(
    groq_client: GroqClient,
    replacements: list[Replacement],
) -> list[Replacement]:
    """Add rationalization to each replacement.

    Generates reasons for non-abbreviation replacements via the LLM.
    Abbreviation replacements get a template-based reason.
    """
    seen_originals: dict[str, str] = {}

    for r in replacements:
        key = r.original.lower()
        if key in seen_originals:
            # Reuse the same rationale for duplicate words
            r.reason = seen_originals[key]
        else:
            r.reason = generate_rationale(groq_client, r)
            seen_originals[key] = r.reason

    return replacements
