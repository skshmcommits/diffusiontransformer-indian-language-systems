"""
Stage 1: Linguistic Pre-processing
-----------------------------------
Input:  Raw sentence (plain text string)
Output: Processed spaCy Doc with:
        - POS tags for each token
        - Morphological features (gender, number, tense, person)
        - Dependency tree (how words relate to each other)
        - Named entities

Uses spaCy's pre-trained neural model — no training needed here.
"""

import spacy
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TokenInfo:
    """Holds all linguistic info for a single word extracted from spaCy."""
    idx: int                    # position in sentence
    text: str                   # original word
    lemma: str                  # base form (e.g. "went" → "go")
    pos: str                    # coarse POS: NOUN, VERB, ADJ, ADV, DET, ...
    tag: str                    # fine-grained POS tag
    dep: str                    # dependency relation to head
    head_idx: int               # index of the head token
    morph: dict                 # morphological features dict
    is_stop: bool               # is it a stopword?
    ent_type: Optional[str]     # named entity type if any


@dataclass
class LinguisticOutput:
    """Full output from Stage 1 for one sentence."""
    sentence: str
    tokens: List[TokenInfo]

    def get_token(self, idx: int) -> Optional[TokenInfo]:
        for t in self.tokens:
            if t.idx == idx:
                return t
        return None

    def get_children(self, head_idx: int) -> List[TokenInfo]:
        return [t for t in self.tokens if t.head_idx == head_idx and t.idx != head_idx]

    def get_root(self) -> Optional[TokenInfo]:
        for t in self.tokens:
            if t.dep == "ROOT":
                return t
        return None

    def __repr__(self):
        lines = [f"Sentence: '{self.sentence}'", "Tokens:"]
        for t in self.tokens:
            lines.append(
                f"  [{t.idx}] {t.text!r:15s} pos={t.pos:6s} dep={t.dep:12s} "
                f"head={t.head_idx} morph={t.morph}"
            )
        return "\n".join(lines)


# ── Core processor ───────────────────────────────────────────────────────────

class LinguisticPreprocessor:
    """
    Wraps spaCy to extract all linguistic features needed by the OKM pipeline.
    
    Usage:
        processor = LinguisticPreprocessor()
        output = processor.process("Ram went to school by bus.")
    """

    # Accepted spaCy model names in order of preference
    _MODEL_PREFERENCES = [
        "en_core_web_trf",   # transformer-based, most accurate
        "en_core_web_lg",    # large
        "en_core_web_md",    # medium
        "en_core_web_sm",    # small (minimum viable)
    ]

    def __init__(self, model_name: Optional[str] = None):
        self.nlp = self._load_model(model_name)
        print(f"[Stage 1] Loaded spaCy model: {self.nlp.meta['name']}")

    def _load_model(self, model_name: Optional[str]):
        """Load spaCy model, falling back through preferences if needed."""
        candidates = [model_name] if model_name else self._MODEL_PREFERENCES
        for name in candidates:
            if name is None:
                continue
            try:
                return spacy.load(name)
            except OSError:
                continue
        # Last resort: try to download the small model
        print("[Stage 1] No spaCy model found. Attempting to download en_core_web_sm...")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        return spacy.load("en_core_web_sm")

    def _extract_morph(self, token) -> dict:
        """Extract morphological features as a clean dict."""
        morph = {}
        for feature in token.morph:
            key, val = str(feature).split("=")
            morph[key] = val
        return morph

    def process(self, sentence: str) -> LinguisticOutput:
        """
        Process a single sentence through spaCy.
        
        Args:
            sentence: Plain text sentence.
        Returns:
            LinguisticOutput with all tokens annotated.
        """
        doc = self.nlp(sentence.strip())
        tokens = []
        for token in doc:
            tokens.append(TokenInfo(
                idx=token.i,
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head_idx=token.head.i,
                morph=self._extract_morph(token),
                is_stop=token.is_stop,
                ent_type=token.ent_type_ if token.ent_type_ else None,
            ))
        return LinguisticOutput(sentence=sentence, tokens=tokens)

    def process_batch(self, sentences: List[str]) -> List[LinguisticOutput]:
        """Process multiple sentences efficiently using spaCy's pipe."""
        results = []
        for doc, sentence in zip(self.nlp.pipe(sentences), sentences):
            tokens = []
            for token in doc:
                tokens.append(TokenInfo(
                    idx=token.i,
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    head_idx=token.head.i,
                    morph=self._extract_morph(token),
                    is_stop=token.is_stop,
                    ent_type=token.ent_type_ if token.ent_type_ else None,
                ))
            results.append(LinguisticOutput(sentence=sentence, tokens=tokens))
        return results


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    proc = LinguisticPreprocessor()
    test_sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "Brother, a letter for you.",
    ]
    for sent in test_sentences:
        out = proc.process(sent)
        print(out)
        print("-" * 60)
