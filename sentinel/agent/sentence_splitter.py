"""
SENTINEL — Clinical Sentence Splitter

Splits clinical response text into atomic sentences.
Configured with custom clinical abbreviations to prevent incorrect splits
on abbreviations like 'b.i.d.', 'q.d.', 'p.r.n.', or 'ICD-10' (Finding #38 / #55).
"""

from __future__ import annotations

import logging

import nltk

logger = logging.getLogger(__name__)

# Bounded clinical abbreviations to add to the sentence tokenizer
CLINICAL_ABBREVIATIONS = {
    "b.i.d", "q.d", "p.r.n", "dr", "fig", "icd-10", "mg/kg", "ml/kg",
    "tab", "cap", "mcg", "inj", "i.m", "i.v", "p.o", "amp", "min"
}



def _init_nltk_tokenizer() -> nltk.tokenize.punkt.PunktSentenceTokenizer:
    """Initializes and configures the NLTK Punkt tokenizer with clinical abbreviations."""
    global _nltk_loaded
    if not _nltk_loaded:
        try:
            # Check/download NLTK resources
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    # Add clinical abbreviations to the set of abbreviation types
    tokenizer._params.abbrev_types.update(CLINICAL_ABBREVIATIONS)
    return tokenizer


# Global tokenizer cache
_nltk_tokenizer = None
_nltk_loaded = False


def split_clinical_sentences(text: str) -> list[str]:
    """
    Splits clinical response text into sentences.
    Attempts to use scispaCy if installed; falls back to NLTK configured with abbreviations.
    """
    if not text.strip():
        return []

    # 1. Attempt scispaCy
    try:
        import en_core_sci_sm
        nlp = en_core_sci_sm.load(disable=["tagger", "parser", "ner", "lemmatizer"])
        # Use simple sentence segmenter component for speed
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        if sentences:
            return sentences
    except ImportError:
        pass
        
    # 2. Fall back to NLTK configured with abbreviations (Finding #38)
    global _nltk_tokenizer, _nltk_loaded
    if _nltk_tokenizer is None:
        try:
            _nltk_tokenizer = _init_nltk_tokenizer()
            _nltk_loaded = True
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK clinical tokenizer: {e}. Using basic split.")
            
    if _nltk_tokenizer:
        sentences = _nltk_tokenizer.tokenize(text)
    else:
        # Absolute fallback: split by periods if all else fails
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        
    # Clean up and filter out non-factual transitions
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Filter out generic/conversational prompts that don't need grounding
        if any(intro in s_clean.lower() for intro in ["here is what", "according to", "verified by sentinel", "please note that"]):
            continue
        cleaned.append(s_clean)
        
    return cleaned
