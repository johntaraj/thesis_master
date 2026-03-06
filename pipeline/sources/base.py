"""Base class and data structures for external sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SourceResult:
    """A single result from an external source."""
    source_name: str
    term: str
    definition: str  # Raw definition/passage from the source
    simplified_text: str  # LLM-simplified version
    url: str
    passage: str  # The exact snippet used


@dataclass
class Replacement:
    """Tracks a single word/phrase replacement with full attribution."""
    original: str
    simplified: str
    start: int
    end: int
    source_name: str  # "FreeDictionary" | "MedlinePlus" | "Wikipedia" | "LLM (model)"
    source_url: str
    source_passage: str
    reason: str = ""
    confidence: float = 0.0  # FKGL improvement from this substitution
    is_abbreviation: bool = False
    is_medical: bool = False


@dataclass
class SimplificationResult:
    """Full result of a simplification pipeline run."""
    original_text: str
    simplified_text: str
    replacements: list[Replacement] = field(default_factory=list)
    mode: str = "rag"  # "rag" or "baseline"
    model_used: str = ""


class BaseSource(ABC):
    """Abstract base for external knowledge sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def search(self, term: str) -> list[SourceResult]:
        """Search for a medical term. Returns min 1, max 2 results."""
        ...
