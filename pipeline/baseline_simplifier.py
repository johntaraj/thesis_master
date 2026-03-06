"""Baseline simplifier — LLM-only, no external sources."""

from pipeline.groq_client import GroqClient
from pipeline.abbreviations import expand_abbreviations, load_abbreviations
from pipeline.difficulty import find_hard_words
from pipeline.medical_detector import classify_hard_words
from pipeline.sources.base import Replacement, SimplificationResult


def _get_sentence_for_word(text: str, start: int, end: int) -> str:
    """Extract the sentence containing the word at [start, end)."""
    sent_start = text.rfind(".", 0, start)
    sent_start = 0 if sent_start == -1 else sent_start + 1

    sent_end = text.find(".", end)
    sent_end = len(text) if sent_end == -1 else sent_end + 1

    return text[sent_start:sent_end].strip()


def run_baseline_pipeline(
    text: str,
    groq_client: GroqClient,
    abbreviations: dict[str, str] | None = None,
    progress_callback=None,
) -> SimplificationResult:
    """Simplify a clinical note using only the LLM (no external sources).

    Uses the same word-level approach as the RAG pipeline but relies
    solely on the LLM for simplification (no external source lookups).

    Steps:
    1. Expand abbreviations from the dictionary
    2. Find hard words in the expanded text
    3. Classify hard words as medical vs non-medical
    4. Use LLM to simplify each hard word individually
    5. Reconstruct simplified text with tracked replacements

    Args:
        text: Original clinical note.
        groq_client: Configured Groq client.
        abbreviations: Optional pre-loaded abbreviation dict.
        progress_callback: Optional callable(step_name, progress_pct) for UI updates.

    Returns:
        SimplificationResult with simplified text and replacements.
    """
    all_replacements: list[Replacement] = []

    # --- Step 1: Expand abbreviations ---
    if progress_callback:
        progress_callback("Expanding abbreviations...", 10)

    if abbreviations is None:
        abbreviations = load_abbreviations()
    expanded_text, abbr_matches, expansion_ranges = expand_abbreviations(text, abbreviations)

    for am in abbr_matches:
        all_replacements.append(Replacement(
            original=am.abbreviation,
            simplified=am.expansion,
            start=am.start,
            end=am.end,
            source_name="Abbreviation Dictionary",
            source_url="",
            source_passage=f"{am.abbreviation} = {am.expansion}",
            is_abbreviation=True,
            is_medical=True,
        ))

    # --- Step 2: Find hard words (excluding abbreviation expansions) ---
    if progress_callback:
        progress_callback("Identifying hard words...", 20)

    hard_words = find_hard_words(expanded_text)

    # Filter out words inside abbreviation expansion ranges
    def _overlaps_expansion(hw):
        for rng_start, rng_end in expansion_ranges:
            if hw.start < rng_end and hw.end > rng_start:
                return True
        return False

    hard_words = [hw for hw in hard_words if not _overlaps_expansion(hw)]

    # --- Step 3: Classify medical vs non-medical ---
    if progress_callback:
        progress_callback("Detecting medical terms...", 30)

    medical_hard, non_medical_hard = classify_hard_words(expanded_text, hard_words)

    # --- Step 4: LLM-simplify all hard words ---
    if progress_callback:
        progress_callback("Simplifying hard words with LLM...", 40)

    all_hard = medical_hard + non_medical_hard
    total = len(set(hw.word.lower() for hw in all_hard))
    seen: dict[str, str] = {}
    processed = 0

    for hw in all_hard:
        word_key = hw.word.lower()
        if word_key in seen:
            all_replacements.append(Replacement(
                original=hw.word,
                simplified=seen[word_key],
                start=hw.start,
                end=hw.end,
                source_name=f"LLM ({groq_client.model})",
                source_url="",
                source_passage="Simplified by the language model without external sources.",
                is_medical=hw.is_medical,
                is_abbreviation=False,
            ))
            continue

        processed += 1
        if progress_callback:
            pct = 40 + int(40 * processed / max(total, 1))
            progress_callback(f"Simplifying '{hw.word}'... ({processed}/{total})", pct)

        sentence = _get_sentence_for_word(expanded_text, hw.start, hw.end)
        simplified = groq_client.simplify_word(hw.word, sentence)
        seen[word_key] = simplified

        all_replacements.append(Replacement(
            original=hw.word,
            simplified=simplified,
            start=hw.start,
            end=hw.end,
            source_name=f"LLM ({groq_client.model})",
            source_url="",
            source_passage="Simplified by the language model without external sources.",
            is_medical=hw.is_medical,
            is_abbreviation=False,
        ))

    # --- Step 5: Reconstruct the simplified text ---
    if progress_callback:
        progress_callback("Reconstructing simplified text...", 85)

    non_abbr_replacements = [r for r in all_replacements if not r.is_abbreviation]
    non_abbr_replacements.sort(key=lambda r: r.start, reverse=True)

    simplified_text = expanded_text
    for r in non_abbr_replacements:
        current_word = simplified_text[r.start:r.end]
        if current_word.lower() == r.original.lower():
            replacement = r.simplified
            if current_word[0].isupper() and replacement[0].islower():
                replacement = replacement[0].upper() + replacement[1:]
            simplified_text = simplified_text[:r.start] + replacement + simplified_text[r.end:]

    if progress_callback:
        progress_callback("Done!", 100)

    return SimplificationResult(
        original_text=text,
        simplified_text=simplified_text,
        replacements=all_replacements,
        mode="baseline",
        model_used=groq_client.model,
    )
