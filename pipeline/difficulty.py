"""Word difficulty classifier — determines if a word is harder than 8th-grade level."""

import re
from dataclasses import dataclass
from config import (
    DALE_CHALL_PATH,
    ZIPF_FREQUENCY_THRESHOLD,
    SYLLABLE_THRESHOLD,
    DIFFICULTY_MIN_CHECKS_FAILED,
)
from wordfreq import zipf_frequency
import textstat


@dataclass
class HardWord:
    word: str
    start: int
    end: int
    is_medical: bool = False  # Set later by medical_detector
    checks_failed: int = 0


_dale_chall_set: set[str] | None = None


def _load_dale_chall(path: str = DALE_CHALL_PATH) -> set[str]:
    global _dale_chall_set
    if _dale_chall_set is None:
        with open(path, "r", encoding="utf-8") as f:
            _dale_chall_set = {line.strip().lower() for line in f if line.strip()}
    return _dale_chall_set


def _count_syllables(word: str) -> int:
    """Count syllables using textstat's internal method."""
    return textstat.syllable_count(word)


def is_hard_word(word: str) -> tuple[bool, int]:
    """Check if a word is harder than 8th-grade level.

    A word is 'hard' if it fails >= DIFFICULTY_MIN_CHECKS_FAILED of these 3 checks:
    1. Not on the Dale-Chall familiar word list
    2. zipf frequency < threshold (uncommon word)
    3. >= SYLLABLE_THRESHOLD syllables

    Returns (is_hard, checks_failed_count).
    """
    clean = word.lower().strip(".,;:!?\"'()[]{}—–-")
    if len(clean) <= 2:
        return False, 0

    dale_chall = _load_dale_chall()
    checks_failed = 0

    # Check 1: Not on Dale-Chall list
    if clean not in dale_chall:
        checks_failed += 1

    # Check 2: Low frequency
    freq = zipf_frequency(clean, "en")
    if freq < ZIPF_FREQUENCY_THRESHOLD:
        checks_failed += 1

    # Check 3: Many syllables
    if _count_syllables(clean) >= SYLLABLE_THRESHOLD:
        checks_failed += 1

    return checks_failed >= DIFFICULTY_MIN_CHECKS_FAILED, checks_failed


def find_hard_words(text: str) -> list[HardWord]:
    """Find all hard words in the text with their positions."""
    hard_words: list[HardWord] = []
    seen: set[str] = set()

    for m in re.finditer(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b", text):
        word = m.group()
        word_lower = word.lower()
        if word_lower in seen:
            # Still record duplicate positions but skip re-checking
            if any(hw.word.lower() == word_lower for hw in hard_words):
                hard_words.append(HardWord(
                    word=word,
                    start=m.start(),
                    end=m.end(),
                    checks_failed=next(
                        hw.checks_failed for hw in hard_words if hw.word.lower() == word_lower
                    ),
                ))
            continue

        seen.add(word_lower)
        hard, checks = is_hard_word(word)
        if hard:
            hard_words.append(HardWord(
                word=word,
                start=m.start(),
                end=m.end(),
                checks_failed=checks,
            ))

    return hard_words
