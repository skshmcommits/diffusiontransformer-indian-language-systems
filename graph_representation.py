"""
Graph Representation
---------------------
Converts OKM NetworkX graphs ↔ PyTorch tensors for the diffusion model.

Graph → Tensors:
    Node feature matrix:  X  ∈ R^(N_max × D_NODE)
    Edge feature matrix:  E  ∈ R^(N_max × N_max × D_EDGE)
    Node padding mask:    M  ∈ {0,1}^(N_max)          (1=real node, 0=padding)

Node feature vector (D_NODE = 64):
    [word_embedding(32), node_type_onehot(3), gender_onehot(4),
     number_onehot(4), case_onehot(9), tense_onehot(3), person_onehot(4)]

Edge feature vector (D_EDGE = 32):
    [relation_onehot(20), case_onehot(9), exists(1), padding(2)]
"""

import os, sys
import torch
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Dimension constants ───────────────────────────────────────────────────────

# Node feature components
D_NODE_TYPE   = 3    # noun / verb / descriptor
D_GENDER      = 4    # male / female / neutral / N/A
D_NUMBER      = 4    # singular / dual / plural / N/A
D_CASE        = 9    # 8 cases + N/A
D_TENSE       = 4    # past / present / future / N/A
D_PERSON      = 4    # 1st / 2nd / 3rd / N/A
D_WORD_EMB    = 32   # learnable word embedding (vocabulary-indexed)

D_NODE = D_NODE_TYPE + D_GENDER + D_NUMBER + D_CASE + D_TENSE + D_PERSON + D_WORD_EMB
# = 3 + 4 + 4 + 9 + 4 + 4 + 32 = 60

# Edge feature components
D_RELATION    = 20   # dependency relation types (top 20 + unknown)
D_EDGE_CASE   = 9    # OKM case on the edge
D_EDGE_EXISTS = 1    # binary: does this edge exist?
D_EDGE_PAD    = 2    # padding to round to nice number

D_EDGE = D_RELATION + D_EDGE_CASE + D_EDGE_EXISTS + D_EDGE_PAD
# = 20 + 9 + 1 + 2 = 32

# Maximum nodes per graph (sentences rarely exceed this)
N_MAX = 32

# Top dependency relation types (from Universal Dependencies)
RELATION_VOCAB = [
    "ROOT", "nsubj", "obj", "iobj", "obl", "nmod", "amod", "advmod",
    "aux", "cop", "det", "case", "mark", "cc", "conj", "compound",
    "xcomp", "ccomp", "advcl", "OTHER"
]
REL_TO_IDX = {r: i for i, r in enumerate(RELATION_VOCAB)}


# ── GraphRepresentation ───────────────────────────────────────────────────────

class GraphRepresentation:
    """
    Converts between OKM NetworkX graphs and tensor representations.

    The tensor format is what the diffusion model operates on.
    The graph format is what the OKM pipeline produces.

    Args:
        n_max:      maximum number of nodes (pads shorter graphs)
        vocab_size: vocabulary size for word embeddings
    """

    def __init__(self, n_max: int = N_MAX, vocab_size: int = 50000):
        self.n_max      = n_max
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}  # built from training data
        self.idx2word: Dict[int, str] = {}

    # ── Vocabulary ────────────────────────────────────────────────────────────

    def build_vocab(self, graphs: List[nx.DiGraph]):
        """Build word→index vocabulary from a list of graphs."""
        words = set()
        for G in graphs:
            for _, data in G.nodes(data=True):
                words.add(data.get("lemma", data.get("word", "<unk>")).lower())
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for i, w in enumerate(sorted(words), start=2):
            self.word2idx[w] = i
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"[GraphRepr] Built vocabulary: {len(self.word2idx)} words")

    def word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word.lower(), self.word2idx.get("<unk>", 1))

    # ── Graph → Tensors ───────────────────────────────────────────────────────

    def graph_to_tensors(
        self, G: nx.DiGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert OKM graph to (X, E, mask) tensors.

        Returns:
            X:    (N_max, D_NODE)  node feature matrix
            E:    (N_max, N_max, D_EDGE)  edge feature matrix
            mask: (N_max,)  bool mask, True = real node
        """
        nodes = list(G.nodes(data=True))
        n_real = min(len(nodes), self.n_max)

        X    = torch.zeros(self.n_max, D_NODE)
        E    = torch.zeros(self.n_max, self.n_max, D_EDGE)
        mask = torch.zeros(self.n_max, dtype=torch.bool)

        # Build node index mapping: graph node id → matrix row index
        node_ids = [nid for nid, _ in nodes[:n_real]]
        nid_to_row = {nid: row for row, nid in enumerate(node_ids)}

        # Fill node features
        for row, (nid, data) in enumerate(nodes[:n_real]):
            X[row] = self._node_features(data)
            mask[row] = True

        # Fill edge features
        for u, v, edata in G.edges(data=True):
            if u in nid_to_row and v in nid_to_row:
                row_u = nid_to_row[u]
                row_v = nid_to_row[v]
                if row_u < self.n_max and row_v < self.n_max:
                    E[row_u, row_v] = self._edge_features(edata)

        return X, E, mask

    def _node_features(self, data: dict) -> torch.Tensor:
        """Build a single node feature vector."""
        feat = torch.zeros(D_NODE)
        offset = 0

        # Word embedding index (scalar, stored in first D_WORD_EMB positions
        # as a one-hot over a small lookup — actual embedding done in model)
        word = data.get("lemma", data.get("word", "<unk>"))
        w_idx = self.word_to_idx(word)
        # Store word index as a single value in the first slot
        # The model's embedding layer will look this up properly
        # For the feature vector we store the index one-hot over a small space
        # Actually: we'll store a separate word_idx field, keep feature as structural
        # Structural features only here; word_idx passed separately

        # Node type one-hot [0, 1, 2] → [noun, verb, descriptor]
        node_type = data.get("node_type", 2)
        if 0 <= node_type < D_NODE_TYPE:
            feat[offset + node_type] = 1.0
        offset += D_NODE_TYPE

        # Gender one-hot [1,2,3,-1] → indices [0,1,2,3]
        gender = data.get("gender", -1)
        g_idx = {1: 0, 2: 1, 3: 2}.get(gender, 3)
        feat[offset + g_idx] = 1.0
        offset += D_GENDER

        # Number one-hot [1,2,3,-1]
        number = data.get("number", -1)
        n_idx = {1: 0, 2: 1, 3: 2}.get(number, 3)
        feat[offset + n_idx] = 1.0
        offset += D_NUMBER

        # Case one-hot [1..8, -1]
        case = data.get("case", -1)
        c_idx = (case - 1) if 1 <= case <= 8 else 8
        feat[offset + c_idx] = 1.0
        offset += D_CASE

        # Tense one-hot [-1, 0, 1]
        tense = data.get("tense", 0)
        t_idx = {-1: 0, 0: 1, 1: 2}.get(tense, 3)
        feat[offset + t_idx] = 1.0
        offset += D_TENSE

        # Person one-hot [1,2,3,-1]
        person = data.get("person", -1)
        p_idx = {1: 0, 2: 1, 3: 2}.get(person, 3)
        feat[offset + p_idx] = 1.0
        offset += D_PERSON

        # Word embedding placeholder (zeros — filled by model's embedding layer)
        # offset += D_WORD_EMB  # last D_WORD_EMB dims left as zero

        return feat

    def _edge_features(self, data: dict) -> torch.Tensor:
        """Build a single edge feature vector."""
        feat = torch.zeros(D_EDGE)
        offset = 0

        # Relation type one-hot
        dep = data.get("dep", "OTHER")
        rel_idx = REL_TO_IDX.get(dep, REL_TO_IDX["OTHER"])
        feat[offset + rel_idx] = 1.0
        offset += D_RELATION

        # Case on this edge
        case = data.get("case", -1)
        c_idx = (case - 1) if 1 <= case <= 8 else 8
        feat[offset + c_idx] = 1.0
        offset += D_EDGE_CASE

        # Edge exists
        feat[offset] = 1.0
        offset += D_EDGE_EXISTS

        return feat

    # ── Tensors → Graph ───────────────────────────────────────────────────────

    def tensors_to_graph(
        self,
        X: torch.Tensor,            # (N_max, D_NODE)
        E: torch.Tensor,            # (N_max, N_max, D_EDGE)
        mask: torch.Tensor,         # (N_max,) bool
        node_type_logits: torch.Tensor,  # (N_max, 3)
        edge_exist_logits: torch.Tensor, # (N_max, N_max, 1)
        edge_type_logits: torch.Tensor,  # (N_max, N_max, 20)
        word_logits: Optional[torch.Tensor] = None,  # (N_max, vocab_size)
        edge_threshold: float = 0.5,
    ) -> nx.DiGraph:
        """
        Reconstruct an OKM graph from diffusion model outputs.
        Uses argmax to pick discrete attributes from logit distributions.
        """
        G = nx.DiGraph()
        n_real = mask.sum().item()

        node_types = node_type_logits.argmax(dim=-1)  # (N_max,)
        edge_exists = torch.sigmoid(edge_exist_logits.squeeze(-1)) > edge_threshold
        edge_types  = edge_type_logits.argmax(dim=-1)

        # Add nodes
        for i in range(int(n_real)):
            node_feat = X[i]
            nt = node_types[i].item()

            # Decode structural features
            offset = D_NODE_TYPE
            g_idx  = node_feat[offset: offset + D_GENDER].argmax().item()
            offset += D_GENDER
            n_idx  = node_feat[offset: offset + D_NUMBER].argmax().item()
            offset += D_NUMBER
            c_idx  = node_feat[offset: offset + D_CASE].argmax().item()
            offset += D_CASE
            t_idx  = node_feat[offset: offset + D_TENSE].argmax().item()
            offset += D_TENSE
            p_idx  = node_feat[offset: offset + D_PERSON].argmax().item()

            word = "<unk>"
            if word_logits is not None:
                w_idx = word_logits[i].argmax().item()
                word  = self.idx2word.get(w_idx, "<unk>")

            G.add_node(i, **{
                "word":      word,
                "node_type": nt,
                "gender":    g_idx + 1 if g_idx < 3 else -1,
                "number":    n_idx + 1 if n_idx < 3 else -1,
                "case":      c_idx + 1 if c_idx < 8 else -1,
                "tense":     [-1, 0, 1, 0][t_idx],
                "person":    p_idx + 1 if p_idx < 3 else -1,
            })

        # Add edges
        for i in range(int(n_real)):
            for j in range(int(n_real)):
                if i != j and edge_exists[i, j]:
                    rel_idx = edge_types[i, j].item()
                    relation = RELATION_VOCAB[rel_idx] if rel_idx < len(RELATION_VOCAB) else "OTHER"
                    G.add_edge(i, j, dep=relation, relation=relation)

        return G

    # ── Batch conversion ──────────────────────────────────────────────────────

    def graphs_to_batch(
        self, graphs: List[nx.DiGraph]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a list of graphs to a batched tensor tuple.

        Returns:
            X:      (B, N_max, D_NODE)
            E:      (B, N_max, N_max, D_EDGE)
            masks:  (B, N_max)
            w_idxs: (B, N_max)  word indices for embedding lookup
        """
        B = len(graphs)
        X      = torch.zeros(B, self.n_max, D_NODE)
        E      = torch.zeros(B, self.n_max, self.n_max, D_EDGE)
        masks  = torch.zeros(B, self.n_max, dtype=torch.bool)
        w_idxs = torch.zeros(B, self.n_max, dtype=torch.long)

        for b, G in enumerate(graphs):
            X[b], E[b], masks[b] = self.graph_to_tensors(G)
            # Fill word indices
            for row, (nid, data) in enumerate(list(G.nodes(data=True))[:self.n_max]):
                word = data.get("lemma", data.get("word", "<unk>"))
                w_idxs[b, row] = self.word_to_idx(word)

        return X, E, masks, w_idxs


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import GraphGeneratorPipeline

    pipeline = GraphGeneratorPipeline()
    repr_    = GraphRepresentation()

    sentences = ["Ram went to school by bus.", "She is Rohit's friend."]
    graphs    = pipeline.generate_batch(sentences)

    repr_.build_vocab(graphs)
    X, E, masks, w_idxs = repr_.graphs_to_batch(graphs)

    print(f"X shape:      {X.shape}")       # (2, 32, 60)
    print(f"E shape:      {E.shape}")       # (2, 32, 32, 32)
    print(f"masks shape:  {masks.shape}")   # (2, 32)
    print(f"w_idxs shape: {w_idxs.shape}") # (2, 32)
    print(f"D_NODE={D_NODE}, D_EDGE={D_EDGE}, N_MAX={N_MAX}")
