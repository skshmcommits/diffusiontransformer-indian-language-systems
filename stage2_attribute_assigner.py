"""
Stage 2: OKM Attribute Assigner (Rule-Based)
----------------------------------------------
Input:  LinguisticOutput from Stage 1
Output: List of OKMToken objects — each word attributed with OKM features:
        - Nouns  → (gender g, number n, case c)
        - Verbs  → (person p, number n, tense t)
        - Others → labeled as descriptor (adjective/adverb/article)

Rules come directly from the OKM grammar defined in the paper.
Zero neural components — pure if-else logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import IntEnum

# Import Stage 1 output types
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_linguistic import LinguisticOutput, TokenInfo


# ── OKM Attribute Enums (from paper) ─────────────────────────────────────────

class Gender(IntEnum):
    MALE    = 1
    FEMALE  = 2
    NEUTRAL = 3

class Number(IntEnum):
    SINGULAR = 1
    DUAL     = 2
    PLURAL   = 3

class Case(IntEnum):
    """Eight cases derived from Sanskrit grammar as per the OKM paper."""
    SUBJECTIVE   = 1   # nsubj — who does the action
    OBJECTIVE    = 2   # obj   — direct receiver of action
    INSTRUMENTAL = 3   # obl (by/with) — the means/instrument
    DATIVE       = 4   # iobj  — indirect object (for/to whom)
    ABLATIVE     = 5   # obl (from) — source/origin
    GENITIVE     = 6   # nmod / poss — possession
    LOCATIVE     = 7   # obl (at/in/on) — location
    VOCATIVE     = 8   # vocative — direct address

class Person(IntEnum):
    FIRST  = 1
    SECOND = 2
    THIRD  = 3

class Tense(IntEnum):
    PAST    = -1
    PRESENT =  0
    FUTURE   =  1

class NodeType(IntEnum):
    NOUN       = 0
    VERB       = 1
    DESCRIPTOR = 2   # adjective, adverb, article/determiner


# ── OKM Token (output of Stage 2) ────────────────────────────────────────────

@dataclass
class OKMToken:
    """
    One word annotated with full OKM attributes.
    Flags any ambiguities for Stage 3 to resolve.
    """
    # From Stage 1
    idx: int
    text: str
    lemma: str
    pos: str
    dep: str
    head_idx: int

    # OKM node type
    node_type: NodeType = NodeType.DESCRIPTOR

    # Noun attributes (set when node_type == NOUN)
    gender: Optional[Gender] = None
    number: Optional[Number] = None
    case: Optional[Case]     = None

    # Verb attributes (set when node_type == VERB)
    person: Optional[Person] = None
    v_number: Optional[Number] = None
    tense: Optional[Tense]   = None

    # Ambiguity flags — Stage 3 will resolve these
    case_ambiguous: bool = False      # multiple cases could apply
    sense_ambiguous: bool = False     # word has multiple senses (bank, etc.)
    ambiguity_candidates: List = field(default_factory=list)

    def okm_notation(self) -> str:
        """Return the OKM tuple notation from the paper e.g. Ram(1,1,1)"""
        if self.node_type == NodeType.NOUN:
            g = self.gender.value if self.gender else 999
            n = self.number.value if self.number else 999
            c = self.case.value   if self.case   else 999
            return f"{self.text}({g},{n},{c})"
        elif self.node_type == NodeType.VERB:
            p  = self.person.value   if self.person   else 999
            n  = self.v_number.value if self.v_number else 999
            t  = self.tense.value    if self.tense    else 999
            return f"{self.text}({p},{n},{t})"
        else:
            return f"[d:{self.text}]"

    def __repr__(self):
        return self.okm_notation()


# ── Rule Tables ───────────────────────────────────────────────────────────────

# Dependency label → OKM Case (main mapping)
# Some labels can map to multiple cases depending on the preposition used
DEP_TO_CASE: dict = {
    "nsubj":    Case.SUBJECTIVE,
    "nsubjpass":Case.SUBJECTIVE,
    "obj":      Case.OBJECTIVE,
    "dobj":     Case.OBJECTIVE,
    "iobj":     Case.DATIVE,
    "nmod:poss":Case.GENITIVE,
    "poss":     Case.GENITIVE,
    "vocative": Case.VOCATIVE,
    "appos":    Case.GENITIVE,
    "attr":     Case.OBJECTIVE,
    "expl":     Case.SUBJECTIVE,
}

# For "obl" relations, the preposition child refines the case
PREP_TO_CASE: dict = {
    # Instrumental (by/with/using)
    "by":    Case.INSTRUMENTAL,
    "with":  Case.INSTRUMENTAL,
    "using": Case.INSTRUMENTAL,
    "via":   Case.INSTRUMENTAL,
    # Dative (to/for)
    "to":    Case.DATIVE,
    "for":   Case.DATIVE,
    # Ablative (from/out of/off)
    "from":  Case.ABLATIVE,
    "out":   Case.ABLATIVE,
    "off":   Case.ABLATIVE,
    # Locative (at/in/on/into/upon)
    "at":    Case.LOCATIVE,
    "in":    Case.LOCATIVE,
    "on":    Case.LOCATIVE,
    "into":  Case.LOCATIVE,
    "upon":  Case.LOCATIVE,
    "inside":Case.LOCATIVE,
    "near":  Case.LOCATIVE,
    # Genitive (of)
    "of":    Case.GENITIVE,
}

# Morphology tag → Gender
GENDER_MAP: dict = {
    "Masc": Gender.MALE,
    "Fem":  Gender.FEMALE,
    "Neut": Gender.NEUTRAL,
}

# Morphology tag → Number
NUMBER_MAP: dict = {
    "Sing": Number.SINGULAR,
    "Dual": Number.DUAL,
    "Plur": Number.PLURAL,
}

# Morphology tag → Tense
TENSE_MAP: dict = {
    "Past": Tense.PAST,
    "Pres": Tense.PRESENT,
    "Fut":  Tense.FUTURE,
}

# Morphology tag → Person
PERSON_MAP: dict = {
    "1": Person.FIRST,
    "2": Person.SECOND,
    "3": Person.THIRD,
}

# POS tags that count as descriptors in OKM
DESCRIPTOR_POS = {"ADJ", "ADV", "DET", "ADP", "PART", "SCONJ", "CCONJ", "INTJ"}

# POS tags for nouns
NOUN_POS = {"NOUN", "PROPN", "PRON"}

# POS tags for verbs
VERB_POS = {"VERB", "AUX"}

# Pronouns that suggest gender
PRONOUN_GENDER_MAP: dict = {
    "he":    Gender.MALE,
    "him":   Gender.MALE,
    "his":   Gender.MALE,
    "she":   Gender.FEMALE,
    "her":   Gender.FEMALE,
    "hers":  Gender.FEMALE,
    "they":  Gender.NEUTRAL,
    "them":  Gender.NEUTRAL,
    "their": Gender.NEUTRAL,
    "it":    Gender.NEUTRAL,
    "its":   Gender.NEUTRAL,
    "i":     Gender.NEUTRAL,
    "we":    Gender.NEUTRAL,
    "you":   Gender.NEUTRAL,
}

# Words known to be ambiguous (flagged for Stage 3)
AMBIGUOUS_WORDS = {
    "bank", "bat", "bark", "bear", "bow", "crane", "date",
    "fair", "fan", "flat", "fly", "light", "match", "mine",
    "mole", "net", "park", "plant", "ring", "rock", "rose",
    "saw", "scale", "spring", "suit", "tick", "tire", "watch", "well"
}


# ── Core Rule Engine ──────────────────────────────────────────────────────────

class OKMAttributeAssigner:
    """
    Applies OKM grammar rules to linguistically annotated tokens.
    
    Usage:
        assigner = OKMAttributeAssigner()
        okm_tokens = assigner.assign(linguistic_output)
    """

    def assign(self, linguistic_output: LinguisticOutput) -> List[OKMToken]:
        """
        Main entry point. Process all tokens in the sentence.
        
        Args:
            linguistic_output: Output from Stage 1.
        Returns:
            List of OKMToken with all OKM attributes filled.
        """
        tokens = linguistic_output.tokens
        okm_tokens = []

        for token_info in tokens:
            # Skip punctuation and spaces
            if token_info.pos in ("PUNCT", "SPACE", "SYM", "X"):
                continue

            okm_token = self._classify_and_attribute(token_info, tokens)
            okm_tokens.append(okm_token)

        return okm_tokens

    def _classify_and_attribute(
        self,
        token: TokenInfo,
        all_tokens: List[TokenInfo]
    ) -> OKMToken:
        """Route token to the correct attribute assignment function."""

        okm = OKMToken(
            idx=token.idx,
            text=token.text,
            lemma=token.lemma,
            pos=token.pos,
            dep=token.dep,
            head_idx=token.head_idx,
        )

        if token.pos in NOUN_POS:
            okm.node_type = NodeType.NOUN
            self._assign_noun_attributes(okm, token, all_tokens)

        elif token.pos in VERB_POS:
            okm.node_type = NodeType.VERB
            self._assign_verb_attributes(okm, token, all_tokens)

        else:
            okm.node_type = NodeType.DESCRIPTOR
            # Descriptors carry no extra OKM attributes

        # Flag ambiguous word senses
        if token.lemma.lower() in AMBIGUOUS_WORDS:
            okm.sense_ambiguous = True
            okm.ambiguity_candidates.append("sense")

        return okm

    # ── Noun attribute assignment ─────────────────────────────────────────────

    def _assign_noun_attributes(
        self,
        okm: OKMToken,
        token: TokenInfo,
        all_tokens: List[TokenInfo]
    ):
        """Fill gender, number, case for a noun/pronoun."""
        okm.gender = self._resolve_gender(token)
        okm.number = self._resolve_number(token)
        okm.case, ambiguous = self._resolve_case(token, all_tokens)
        if ambiguous:
            okm.case_ambiguous = True
            okm.ambiguity_candidates.append("case")

    def _resolve_gender(self, token: TokenInfo) -> Gender:
        """Rule: use morphology if available, else pronoun map, else NEUTRAL."""
        morph_gender = token.morph.get("Gender")
        if morph_gender and morph_gender in GENDER_MAP:
            return GENDER_MAP[morph_gender]
        # Pronoun heuristic
        lower = token.text.lower()
        if lower in PRONOUN_GENDER_MAP:
            return PRONOUN_GENDER_MAP[lower]
        return Gender.NEUTRAL

    def _resolve_number(self, token: TokenInfo) -> Number:
        """Rule: use morphology Number feature."""
        morph_number = token.morph.get("Number")
        if morph_number and morph_number in NUMBER_MAP:
            return NUMBER_MAP[morph_number]
        return Number.SINGULAR  # default

    def _resolve_case(
        self,
        token: TokenInfo,
        all_tokens: List[TokenInfo]
    ) -> Tuple[Case, bool]:
        """
        Rule: map dependency label to OKM case.
        For oblique (obl) relations, look at the preposition child to refine.
        Returns (Case, is_ambiguous).
        """
        dep = token.dep

        # Direct mapping for most relations
        if dep in DEP_TO_CASE:
            return DEP_TO_CASE[dep], False

        # Possessive marker ('s)
        if dep in ("case", "mark") and token.pos == "PART":
            return Case.GENITIVE, False

        # Oblique arguments — need preposition to determine case
        if dep in ("obl", "nmod", "prep", "pobj"):
            return self._case_from_preposition(token, all_tokens)

        # Compound nouns — treat as genitive
        if dep == "compound":
            return Case.GENITIVE, False

        # Subject complement, predicate
        if dep in ("acomp", "ccomp", "xcomp", "advcl"):
            return Case.OBJECTIVE, False

        # Default: objective (most common case for unrecognized deps)
        return Case.OBJECTIVE, True  # mark as ambiguous

    def _case_from_preposition(
        self,
        token: TokenInfo,
        all_tokens: List[TokenInfo]
    ) -> Tuple[Case, bool]:
        """
        Look for a preposition child of this token or its head
        and use it to pick the correct OKM case.
        """
        # Find preposition children of this token
        prep_texts = []
        for t in all_tokens:
            if t.head_idx == token.idx and t.pos == "ADP":
                prep_texts.append(t.text.lower())
            # Also check if this token is child of a preposition
            if t.idx == token.head_idx and t.pos == "ADP":
                prep_texts.append(t.text.lower())

        for prep in prep_texts:
            if prep in PREP_TO_CASE:
                return PREP_TO_CASE[prep], False

        # No clear preposition found — ambiguous oblique
        ambiguity_candidates = list(set(PREP_TO_CASE.values()))
        return Case.LOCATIVE, True  # default to locative, flag ambiguous

    # ── Verb attribute assignment ─────────────────────────────────────────────

    def _assign_verb_attributes(
        self,
        okm: OKMToken,
        token: TokenInfo,
        all_tokens: List[TokenInfo]
    ):
        """Fill person, number, tense for a verb."""
        okm.tense    = self._resolve_tense(token, all_tokens)
        okm.person   = self._resolve_person(token, all_tokens)
        okm.v_number = self._resolve_verb_number(token, all_tokens)

    def _resolve_tense(self, token: TokenInfo, all_tokens: List[TokenInfo]) -> Tense:
        """Rule: use Tense morphology; check aux children for future."""
        morph_tense = token.morph.get("Tense")
        if morph_tense and morph_tense in TENSE_MAP:
            return TENSE_MAP[morph_tense]

        # Check for future auxiliary (will, shall, going to)
        for t in all_tokens:
            if t.head_idx == token.idx and t.pos == "AUX":
                if t.lemma.lower() in ("will", "shall", "'ll"):
                    return Tense.FUTURE

        # VBG (gerund) or VB with no tense → present
        if token.tag in ("VB", "VBG", "VBP", "VBZ"):
            return Tense.PRESENT

        return Tense.PRESENT  # default

    def _resolve_person(self, token: TokenInfo, all_tokens: List[TokenInfo]) -> Person:
        """Rule: infer person from subject of the verb."""
        morph_person = token.morph.get("Person")
        if morph_person and morph_person in PERSON_MAP:
            return PERSON_MAP[morph_person]

        # Find the subject of this verb
        for t in all_tokens:
            if t.head_idx == token.idx and t.dep in ("nsubj", "nsubjpass"):
                lower = t.text.lower()
                if lower in ("i", "we"):
                    return Person.FIRST
                if lower in ("you"):
                    return Person.SECOND
                return Person.THIRD

        return Person.THIRD  # default

    def _resolve_verb_number(
        self, token: TokenInfo, all_tokens: List[TokenInfo]
    ) -> Number:
        """Rule: match number of the verb's subject."""
        morph_number = token.morph.get("Number")
        if morph_number and morph_number in NUMBER_MAP:
            return NUMBER_MAP[morph_number]

        # Inherit from subject
        for t in all_tokens:
            if t.head_idx == token.idx and t.dep in ("nsubj", "nsubjpass"):
                subj_number = t.morph.get("Number")
                if subj_number and subj_number in NUMBER_MAP:
                    return NUMBER_MAP[subj_number]

        return Number.SINGULAR  # default


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from stage1_linguistic import LinguisticPreprocessor

    proc    = LinguisticPreprocessor()
    assigner = OKMAttributeAssigner()

    sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "Brother, a letter for you.",
    ]

    for sent in sentences:
        ling_out   = proc.process(sent)
        okm_tokens = assigner.assign(ling_out)
        print(f"\nSentence: {sent}")
        print("OKM tokens:", [t.okm_notation() for t in okm_tokens])
        ambiguous = [t for t in okm_tokens if t.case_ambiguous or t.sense_ambiguous]
        if ambiguous:
            print("Flagged for Stage 3:", [t.text for t in ambiguous])
