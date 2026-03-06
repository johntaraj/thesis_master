"""Evaluation metrics for comparing simplification approaches."""

from dataclasses import dataclass
import textstat


@dataclass
class MetricResult:
    metric_name: str
    original_score: float
    simplified_score: float
    delta: float


def compute_fkgl(text: str) -> float:
    """Flesch-Kincaid Grade Level."""
    return textstat.flesch_kincaid_grade(text)


def compute_dale_chall(text: str) -> float:
    """Dale-Chall Readability Score."""
    return textstat.dale_chall_readability_score(text)


def compute_fre(text: str) -> float:
    """Flesch Reading Ease (higher = easier)."""
    return textstat.flesch_reading_ease(text)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def compute_sari(original: str, simplified: str, reference: str | None = None) -> float:
    """Compute a simplified SARI-like score.

    SARI measures the quality of simplification by comparing:
    - What was kept from the original (keep score)
    - What was deleted from the original (delete score)
    - What was added that wasn't in the original (add score)

    Without a gold reference, we use the original as a pseudo-reference
    and focus on measuring meaningful edit quality.

    Returns a score 0-100 (higher = better simplification).
    """
    orig_tokens = set(_tokenize(original))
    simp_tokens = set(_tokenize(simplified))

    if reference:
        ref_tokens = set(_tokenize(reference))
    else:
        ref_tokens = orig_tokens

    # Keep precision: of words kept from original, how many are in reference
    kept = orig_tokens & simp_tokens
    if kept:
        keep_p = len(kept & ref_tokens) / len(kept) if kept else 0
    else:
        keep_p = 0

    # Delete precision: of words deleted from original, how many are NOT in reference
    deleted = orig_tokens - simp_tokens
    if deleted:
        # Without reference, we want some deletions of complex words
        del_p = len(deleted - ref_tokens) / len(deleted) if deleted else 0
    else:
        del_p = 0

    # Add precision: of words added, how many are in reference
    added = simp_tokens - orig_tokens
    if added:
        add_p = len(added & ref_tokens) / len(added) if reference else min(len(added) / max(len(orig_tokens), 1), 1.0)
    else:
        add_p = 0

    sari = (keep_p + del_p + add_p) / 3 * 100
    return round(sari, 2)


def evaluate(original: str, simplified: str, reference: str | None = None) -> list[MetricResult]:
    """Compute all evaluation metrics."""
    fkgl_orig = compute_fkgl(original)
    fkgl_simp = compute_fkgl(simplified)

    dc_orig = compute_dale_chall(original)
    dc_simp = compute_dale_chall(simplified)

    fre_orig = compute_fre(original)
    fre_simp = compute_fre(simplified)

    sari = compute_sari(original, simplified, reference)

    return [
        MetricResult("FKGL (Grade Level)", fkgl_orig, fkgl_simp, fkgl_orig - fkgl_simp),
        MetricResult("Dale-Chall", dc_orig, dc_simp, dc_orig - dc_simp),
        MetricResult("Flesch Reading Ease", fre_orig, fre_simp, fre_simp - fre_orig),
        MetricResult("SARI", 0, sari, sari),
    ]
