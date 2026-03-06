"""Configuration constants for the clinical note simplification system."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODELS = {
    "LLaMA 3.3 70B": "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B": "llama-3.1-8b-instant",
    "Gemma 2 9B": "gemma2-9b-it",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --- External sources ---
MEDLINEPLUS_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
FREEDICTIONARY_BASE_URL = "https://medical-dictionary.thefreedictionary.com"
WIKIPEDIA_SENTENCES = 2

# --- Difficulty thresholds ---
ZIPF_FREQUENCY_THRESHOLD = 3.5  # Words below this are uncommon
SYLLABLE_THRESHOLD = 4           # Words with >= this many syllables are complex
DIFFICULTY_MIN_CHECKS_FAILED = 2 # Must fail >= N of 3 checks to be "hard"

# --- Source retrieval ---
MAX_RESULTS_PER_SOURCE = 2
MIN_RESULTS_PER_SOURCE = 1
FREEDICTIONARY_RATE_LIMIT_SEC = 1.0

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ABBREVIATIONS_PATH = os.path.join(DATA_DIR, "medical_abbreviations.json")
ABBREVIATIONS_CSV_PATH = os.path.join(os.path.dirname(__file__), "abr.csv")
DALE_CHALL_PATH = os.path.join(DATA_DIR, "dale_chall_words.txt")

# --- Prompts ---
SIMPLIFY_WORD_PROMPT = (
    "You are a medical text simplifier. Replace the word/phrase '{word}' "
    "in the following sentence with a simpler word or phrase that a non-medical "
    "person (8th-grade reading level) would understand. Keep the same meaning. "
    "Return ONLY the replacement word/phrase, nothing else.\n\n"
    "Sentence: {sentence}"
)

SIMPLIFY_FROM_DEFINITION_PROMPT = (
    "Given this medical definition of '{word}':\n\n"
    "{definition}\n\n"
    "Provide a simple 1-phrase explanation (max 5 words) that a non-medical "
    "person would understand. Return ONLY the simplified phrase, nothing else."
)

BASELINE_SIMPLIFY_PROMPT = (
    "You are a medical text simplifier for patients with an 8th-grade reading level.\n\n"
    "Simplify the following clinical note. Rules:\n"
    "1. Expand ALL medical abbreviations (e.g., BP → blood pressure).\n"
    "2. Replace ALL difficult medical and non-medical words with simpler alternatives.\n"
    "3. Keep the same meaning and all important medical information.\n"
    "4. Make every sentence easy for a non-medical person to understand.\n\n"
    "Clinical note:\n{text}\n\n"
    "Return the simplified version of the note."
)

BASELINE_WORD_MAP_PROMPT = (
    "Compare the original clinical note and simplified version below. "
    "List EVERY word or phrase that was changed.\n\n"
    "Original:\n{original}\n\n"
    "Simplified:\n{simplified}\n\n"
    "Return a JSON object where keys are original words/phrases and values are "
    "their simplified replacements. Example: {{\"hypertension\": \"high blood pressure\", \"PRN\": \"as needed\"}}\n"
    "Return ONLY the JSON object, nothing else."
)

RATIONALIZATION_PROMPT = (
    "We replaced '{original}' with '{simplified}' in a clinical note to make it "
    "easier for patients to understand.\n"
    "Source: {source_name}\n"
    "Source passage: {passage}\n\n"
    "Explain in 1 sentence why this replacement makes the text easier to understand. "
    "Return ONLY the explanation sentence."
)

NON_MEDICAL_SIMPLIFY_PROMPT = (
    "You are simplifying a clinical note for patients. The medical terms have "
    "already been simplified. Now simplify the remaining hard non-medical words "
    "while keeping the sentence natural and preserving meaning.\n\n"
    "Current sentence: {sentence}\n\n"
    "Words to simplify: {words}\n\n"
    "Return ONLY the rewritten sentence with those words replaced by simpler "
    "alternatives. Keep everything else the same."
)
