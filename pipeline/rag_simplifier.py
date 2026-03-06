"""RAG simplifier — full pipeline orchestrating abbreviations, difficulty, medical NER, external sources, and LLM."""

import re
from pipeline.groq_client import GroqClient
from pipeline.abbreviations import expand_abbreviations, load_abbreviations
from pipeline.difficulty import find_hard_words
from pipeline.medical_detector import classify_hard_words
from pipeline.sources.base import BaseSource, Replacement, SimplificationResult
from pipeline.source_selector import select_best_simplification
from pipeline.attribution import add_rationales


def _get_sentence_for_word(text: str, start: int, end: int) -> str:
    """Extract the sentence containing the word at [start, end)."""
    # Find sentence boundaries
    sent_start = text.rfind(".", 0, start)
    sent_start = 0 if sent_start == -1 else sent_start + 1

    sent_end = text.find(".", end)
    sent_end = len(text) if sent_end == -1 else sent_end + 1

    return text[sent_start:sent_end].strip()


def run_rag_pipeline(
    text: str,
    groq_client: GroqClient,
    sources: list[BaseSource],
    abbreviations: dict[str, str] | None = None,
    progress_callback=None,
) -> SimplificationResult:
    """Execute the full RAG simplification pipeline.

    Steps:
    1. Detect and expand abbreviations
    2. Find all hard words
    3. Classify hard words as medical vs non-medical
    4. For hard medical terms → query external sources → select best
    5. For hard non-medical words → LLM simplification
    6. Reconstruct simplified text with tracked replacements

    Args:
        text: Original clinical note.
        groq_client: Configured Groq client.
        sources: List of enabled external sources.
        progress_callback: Optional callable(step_name, progress_pct) for UI updates.

    Returns:
        SimplificationResult with simplified text and all replacements.
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

    # --- Step 2: Find hard words in the expanded text ---
    if progress_callback:
        progress_callback("Identifying hard words...", 20)

    hard_words = find_hard_words(expanded_text)

    # Filter out hard words that fall inside abbreviation expansion ranges
    # (these words are already the simplified form of an abbreviation)
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

    # Deduplicate medical terms (same word appearing multiple times)
    seen_medical: dict[str, Replacement] = {}

    # --- Step 4: Query external sources for medical hard words ---
    if progress_callback:
        progress_callback("Searching external sources...", 40)

    total_medical = len(set(hw.word.lower() for hw in medical_hard))
    processed = 0

    for hw in medical_hard:
        word_key = hw.word.lower()
        if word_key in seen_medical:
            # Reuse the same simplification
            r = seen_medical[word_key]
            all_replacements.append(Replacement(
                original=hw.word,
                simplified=r.simplified,
                start=hw.start,
                end=hw.end,
                source_name=r.source_name,
                source_url=r.source_url,
                source_passage=r.source_passage,
                is_medical=True,
            ))
            continue

        processed += 1
        if progress_callback:
            pct = 40 + int(30 * processed / max(total_medical, 1))
            progress_callback(f"Looking up '{hw.word}'... ({processed}/{total_medical})", pct)

        sentence = _get_sentence_for_word(expanded_text, hw.start, hw.end)

        # Query all enabled sources
        all_candidates = []
        for source in sources:
            candidates = source.search(hw.word)
            all_candidates.extend(candidates)

        if all_candidates:
            best = select_best_simplification(hw.word, sentence, all_candidates)
            if best:
                r = Replacement(
                    original=hw.word,
                    simplified=best.simplified_text,
                    start=hw.start,
                    end=hw.end,
                    source_name=best.source_name,
                    source_url=best.url,
                    source_passage=best.passage,
                    is_medical=True,
                )
                all_replacements.append(r)
                seen_medical[word_key] = r
                continue

        # Fallback: use LLM directly
        llm_simplified = groq_client.simplify_word(hw.word, sentence)
        r = Replacement(
            original=hw.word,
            simplified=llm_simplified,
            start=hw.start,
            end=hw.end,
            source_name=f"LLM ({groq_client.model})",
            source_url="",
            source_passage="No external source found. Generated by the language model.",
            is_medical=True,
        )
        all_replacements.append(r)
        seen_medical[word_key] = r

    # --- Step 5: LLM-simplify non-medical hard words ---
    if progress_callback:
        progress_callback("Simplifying non-medical hard words...", 75)

    seen_non_medical: dict[str, str] = {}

    for hw in non_medical_hard:
        word_key = hw.word.lower()
        if word_key in seen_non_medical:
            all_replacements.append(Replacement(
                original=hw.word,
                simplified=seen_non_medical[word_key],
                start=hw.start,
                end=hw.end,
                source_name=f"LLM ({groq_client.model})",
                source_url="",
                source_passage="Non-medical word simplified by the language model.",
                is_medical=False,
            ))
            continue

        sentence = _get_sentence_for_word(expanded_text, hw.start, hw.end)
        simplified = groq_client.simplify_word(hw.word, sentence)
        seen_non_medical[word_key] = simplified

        all_replacements.append(Replacement(
            original=hw.word,
            simplified=simplified,
            start=hw.start,
            end=hw.end,
            source_name=f"LLM ({groq_client.model})",
            source_url="",
            source_passage="Non-medical word simplified by the language model.",
            is_medical=False,
        ))

    # --- Step 6: Reconstruct the simplified text ---
    if progress_callback:
        progress_callback("Reconstructing simplified text...", 85)

    # Sort non-abbreviation replacements by position descending for safe substitution
    # (abbreviation replacements were already applied in step 1)
    non_abbr_replacements = [r for r in all_replacements if not r.is_abbreviation]
    non_abbr_replacements.sort(key=lambda r: r.start, reverse=True)

    simplified_text = expanded_text
    for r in non_abbr_replacements:
        # Only replace if the word at this position still matches
        current_word = simplified_text[r.start:r.end]
        if current_word.lower() == r.original.lower():
            # Preserve capitalization
            replacement = r.simplified
            if current_word[0].isupper() and replacement[0].islower():
                replacement = replacement[0].upper() + replacement[1:]
            simplified_text = simplified_text[:r.start] + replacement + simplified_text[r.end:]

    # --- Step 7: Generate rationales ---
    if progress_callback:
        progress_callback("Generating explanations...", 90)

    all_replacements = add_rationales(groq_client, all_replacements)

    if progress_callback:
        progress_callback("Done!", 100)

    return SimplificationResult(
        original_text=text,
        simplified_text=simplified_text,
        replacements=all_replacements,
        mode="rag",
        model_used=groq_client.model,
    )
