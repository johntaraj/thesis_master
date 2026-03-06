"""Select the best simplification from multiple source results using FKGL scoring."""

import re
import textstat
from pipeline.sources.base import SourceResult


def select_best_simplification(
    term: str,
    sentence: str,
    candidates: list[SourceResult],
) -> SourceResult | None:
    """Pick the candidate simplification that produces the lowest FKGL when
    substituted into the original sentence.

    Tiebreaker: shortest simplification wins.
    Returns None if no candidates.
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    best: SourceResult | None = None
    best_score = float("inf")
    best_length = float("inf")

    for candidate in candidates:
        # Substitute the simplified text into the sentence
        test_sentence = re.sub(
            re.escape(term),
            candidate.simplified_text,
            sentence,
            flags=re.IGNORECASE,
            count=1,
        )

        # Compute FKGL for the modified sentence
        score = textstat.flesch_kincaid_grade(test_sentence)
        length = len(candidate.simplified_text)

        if score < best_score or (score == best_score and length < best_length):
            best = candidate
            best_score = score
            best_length = length

    return best
