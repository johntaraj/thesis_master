"""Medical abbreviation detection and expansion."""

import csv
import json
import re
from dataclasses import dataclass
from config import ABBREVIATIONS_PATH, ABBREVIATIONS_CSV_PATH


@dataclass
class AbbreviationMatch:
    abbreviation: str
    expansion: str
    start: int
    end: int


def load_abbreviations(path: str = ABBREVIATIONS_PATH) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_abbreviations_csv(path: str = ABBREVIATIONS_CSV_PATH) -> dict[str, str]:
    """Load abbreviations from the CSV file (Abbreviation,Meaning)."""
    abbrevs: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = row["Abbreviation"].strip()
            meaning = row["Meaning"].strip().strip('"')
            if abbr and meaning:
                # Take only the first meaning if comma-separated
                first_meaning = meaning.split(",")[0].strip()
                abbrevs[abbr] = first_meaning
    return abbrevs


def _build_abbr_pattern(abbr: str) -> str:
    """Build a flexible regex pattern for an abbreviation.

    Makes periods optional so e.g. 'pt.' matches 'Pt', 'pt', 'pt.' etc.
    """
    parts = []
    for char in abbr:
        if char == '.':
            parts.append(r'\.?')
        else:
            parts.append(re.escape(char))
    return ''.join(parts)


def detect_abbreviations(text: str, abbreviations: dict[str, str] | None = None) -> list[AbbreviationMatch]:
    """Find medical abbreviations in text and return their expansions.

    Uses word-boundary regex matching (case-insensitive, periods optional).
    Abbreviations are checked in longest-first order so that e.g. 'NKDA'
    is matched before 'NKA'.
    """
    if abbreviations is None:
        abbreviations = load_abbreviations()

    matches: list[AbbreviationMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    # Sort by length descending so longer abbreviations match first
    sorted_abbrevs = sorted(abbreviations.keys(), key=len, reverse=True)

    for abbr in sorted_abbrevs:
        # Skip very short abbreviations (1 char) to avoid false positives
        stripped = abbr.replace('.', '').strip()
        if len(stripped) <= 1:
            continue

        # Build flexible pattern: case-insensitive, optional periods
        inner_pattern = _build_abbr_pattern(abbr)
        pattern = r"(?<!\w)" + inner_pattern + r"(?!\w)"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            span = (m.start(), m.end())
            # Skip if this span overlaps with an already-matched abbreviation
            if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                continue
            seen_spans.add(span)
            matches.append(AbbreviationMatch(
                abbreviation=m.group(),  # Use what was actually matched in text
                expansion=abbreviations[abbr],
                start=m.start(),
                end=m.end(),
            ))

    # Sort by position in text
    matches.sort(key=lambda m: m.start)
    return matches


def expand_abbreviations(
    text: str, abbreviations: dict[str, str] | None = None
) -> tuple[str, list[AbbreviationMatch], list[tuple[int, int]]]:
    """Replace abbreviations in text with their expansions.

    Returns (new_text, matches, expansion_ranges) where expansion_ranges
    are the character ranges in new_text that correspond to expanded abbreviations.
    """
    matches = detect_abbreviations(text, abbreviations)
    if not matches:
        return text, [], []

    # Build new text by replacing from right to left to preserve positions
    new_text = text
    for m in reversed(matches):
        new_text = new_text[:m.start] + m.expansion + new_text[m.end:]

    # Compute expansion ranges in the new text (left to right with offset tracking)
    expansion_ranges: list[tuple[int, int]] = []
    offset = 0
    for m in matches:
        new_start = m.start + offset
        new_end = new_start + len(m.expansion)
        expansion_ranges.append((new_start, new_end))
        offset += len(m.expansion) - (m.end - m.start)

    return new_text, matches, expansion_ranges
