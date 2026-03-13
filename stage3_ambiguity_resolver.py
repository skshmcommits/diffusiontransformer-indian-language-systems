"""
Stage 3: Ambiguity Resolver
----------------------------
Input:  List of OKMTokens from Stage 2 (some flagged as ambiguous)
Output: Same list with ambiguities resolved — correct case/sense assigned

Uses a small fine-tuned BERT classifier ONLY for tokens flagged as ambiguous.
Tokens with no ambiguity pass straight through — no neural processing.

Two resolvers:
  1. CaseDisambiguator  — picks the correct OKM case for oblique nouns
  2. SenseDisambiguator — picks the correct word sense (future extension)
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_attribute_assigner import (
    OKMToken, NodeType, Case, Gender, Number, Tense
)


# ── Case Disambiguation Model ─────────────────────────────────────────────────

class CaseClassifier(nn.Module):
    """
    Small BERT-based classifier to predict the OKM case for ambiguous nouns.

    Architecture:
        BERT encoder → [CLS] pooling → dropout → linear → 8 case classes

    Input:  Sentence + target word (marked with special tokens)
    Output: Logits over 8 OKM cases
    """

    NUM_CASES = 8

    def __init__(self, bert_model_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.NUM_CASES)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits of shape (B, 8)."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)             # (B, 8)
        return logits


# ── Ambiguity Resolver ────────────────────────────────────────────────────────

class AmbiguityResolver:
    """
    Resolves flagged ambiguities in OKM tokens using the CaseClassifier.

    If no model weights are available (e.g. first run), falls back to
    rule-based heuristics so the pipeline still works end-to-end.

    Usage:
        resolver = AmbiguityResolver(weights_path="checkpoints/case_clf.pt")
        resolved = resolver.resolve(okm_tokens, sentence)
    """

    # Maps classifier output index → OKM Case enum
    IDX_TO_CASE = {
        0: Case.SUBJECTIVE,
        1: Case.OBJECTIVE,
        2: Case.INSTRUMENTAL,
        3: Case.DATIVE,
        4: Case.ABLATIVE,
        5: Case.GENITIVE,
        6: Case.LOCATIVE,
        7: Case.VOCATIVE,
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        bert_model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._neural_available = False

        if weights_path and os.path.exists(weights_path):
            self._load_model(weights_path, bert_model_name)
        else:
            print(
                "[Stage 3] No model weights found. "
                "Using rule-based fallback for ambiguity resolution. "
                f"To enable neural resolution, train and save weights to: {weights_path}"
            )

    def _load_model(self, weights_path: str, bert_model_name: str):
        """Load trained weights into the classifier."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.model = CaseClassifier(bert_model_name).to(self.device)
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self._neural_available = True
            print(f"[Stage 3] Loaded CaseClassifier from {weights_path}")
        except Exception as e:
            print(f"[Stage 3] Failed to load model: {e}. Using rule-based fallback.")

    def resolve(
        self,
        okm_tokens: List[OKMToken],
        sentence: str,
    ) -> List[OKMToken]:
        """
        Main entry point.
        Passes non-ambiguous tokens through untouched.
        Resolves flagged tokens via neural model or heuristic fallback.
        """
        resolved = []
        for token in okm_tokens:
            if not (token.case_ambiguous or token.sense_ambiguous):
                resolved.append(token)
                continue

            # Only case ambiguity is handled right now; sense is a future extension
            if token.case_ambiguous and token.node_type == NodeType.NOUN:
                token = self._resolve_case(token, sentence)

            resolved.append(token)
        return resolved

    def _resolve_case(self, token: OKMToken, sentence: str) -> OKMToken:
        """Resolve ambiguous case using neural model or fallback heuristics."""
        if self._neural_available:
            predicted_case = self._neural_case_predict(token, sentence)
        else:
            predicted_case = self._heuristic_case_resolve(token, sentence)

        token.case = predicted_case
        token.case_ambiguous = False
        return token

    def _neural_case_predict(self, token: OKMToken, sentence: str) -> Case:
        """
        Mark the ambiguous word with [TGT] tokens and classify.
        
        E.g.: "Ram went to [TGT] school [/TGT] by bus."
        """
        # Insert markers around the target word
        words = sentence.split()
        marked_words = []
        for w in words:
            clean = w.strip(".,!?;:")
            if clean.lower() == token.text.lower():
                marked_words.append(f"[TGT] {w} [/TGT]")
            else:
                marked_words.append(w)
        marked_sentence = " ".join(marked_words)

        encoding = self.tokenizer(
            marked_sentence,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)  # (1, 8)
            pred_idx = logits.argmax(dim=-1).item()

        return self.IDX_TO_CASE.get(pred_idx, Case.OBJECTIVE)

    def _heuristic_case_resolve(self, token: OKMToken, sentence: str) -> Case:
        """
        Rule-based fallback when no neural model is available.
        Uses preposition context from the raw sentence around the word.
        """
        sentence_lower = sentence.lower()
        word_lower = token.text.lower()

        # Find position of word and check what precedes it
        idx = sentence_lower.find(word_lower)
        preceding = sentence_lower[:idx].strip().split()
        last_prep = preceding[-1] if preceding else ""

        from stage2_attribute_assigner import PREP_TO_CASE
        if last_prep in PREP_TO_CASE:
            return PREP_TO_CASE[last_prep]

        # Fallback based on dependency
        dep_fallbacks = {
            "obl":  Case.LOCATIVE,
            "nmod": Case.GENITIVE,
            "prep": Case.LOCATIVE,
            "pobj": Case.OBJECTIVE,
        }
        return dep_fallbacks.get(token.dep, Case.OBJECTIVE)


# ── Training Helpers ──────────────────────────────────────────────────────────
# These are used to fine-tune the CaseClassifier on labeled data.
# Labels are (sentence, target_word, correct_case_int) triples.

@dataclass
class CaseTrainingSample:
    sentence: str
    target_word: str
    correct_case: int  # 0-indexed (Case.value - 1)


class CaseClassifierTrainer:
    """
    Fine-tunes the CaseClassifier on labeled sentence-case pairs.
    
    To create training data:
        - Run the rule-based pipeline on a corpus
        - Export all "case_ambiguous=True" tokens with their correct cases
        - Store as CaseTrainingSample list
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        lr: float = 2e-5,
        device: Optional[str] = None,
    ):
        from transformers import AutoTokenizer
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model     = CaseClassifier(bert_model_name).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_batch(
        self, samples: List[CaseTrainingSample]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch of training samples."""
        marked_sentences = []
        labels = []

        for sample in samples:
            words = sample.sentence.split()
            marked = []
            for w in words:
                clean = w.strip(".,!?;:")
                if clean.lower() == sample.target_word.lower():
                    marked.append(f"[TGT] {w} [/TGT]")
                else:
                    marked.append(w)
            marked_sentences.append(" ".join(marked))
            labels.append(sample.correct_case)

        encoding = self.tokenizer(
            marked_sentences,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        labels_tensor  = torch.tensor(labels, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels_tensor

    def train_epoch(self, samples: List[CaseTrainingSample], batch_size: int = 16) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            input_ids, attention_mask, labels = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Stage 3] Saved CaseClassifier weights to {path}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from stage1_linguistic import LinguisticPreprocessor
    from stage2_attribute_assigner import OKMAttributeAssigner

    proc     = LinguisticPreprocessor()
    assigner = OKMAttributeAssigner()
    resolver = AmbiguityResolver(weights_path="checkpoints/case_clf.pt")

    sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "I went to the bank to get money.",  # "bank" is ambiguous
    ]

    for sent in sentences:
        ling_out   = proc.process(sent)
        okm_tokens = assigner.assign(ling_out)
        resolved   = resolver.resolve(okm_tokens, sent)
        print(f"\nSentence: {sent}")
        print("Resolved OKM:", [t.okm_notation() for t in resolved])
