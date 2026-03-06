"""Medical term/chunk detection using scispaCy."""

import spacy
from pipeline.difficulty import HardWord


# Lazy-load the model to avoid startup cost if not needed
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp


def detect_medical_terms(text: str) -> list[dict]:
    """Detect medical entities/chunks in text using scispaCy.

    Returns list of dicts: {term, start, end, label}.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    terms = []
    seen_spans: set[tuple[int, int]] = set()

    # Use both entities and noun chunks for broader coverage
    for ent in doc.ents:
        span = (ent.start_char, ent.end_char)
        if span not in seen_spans:
            seen_spans.add(span)
            terms.append({
                "term": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
            })

    return terms


def classify_hard_words(text: str, hard_words: list[HardWord]) -> tuple[list[HardWord], list[HardWord]]:
    """Split hard words into medical and non-medical categories.

    Returns (medical_hard_words, non_medical_hard_words).
    Medical terms are those that overlap with scispaCy-detected entities.
    """
    medical_terms = detect_medical_terms(text)

    # Build a set of character ranges that are medical
    medical_ranges: list[tuple[int, int]] = [(t["start"], t["end"]) for t in medical_terms]

    medical_hard: list[HardWord] = []
    non_medical_hard: list[HardWord] = []

    for hw in hard_words:
        is_medical = False
        for m_start, m_end in medical_ranges:
            # Check if the hard word overlaps with any medical entity
            if hw.start < m_end and hw.end > m_start:
                is_medical = True
                break

        hw.is_medical = is_medical
        if is_medical:
            medical_hard.append(hw)
        else:
            non_medical_hard.append(hw)

    return medical_hard, non_medical_hard
