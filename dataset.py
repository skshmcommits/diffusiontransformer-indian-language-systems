"""
Dataset
--------
Loads (sentence, OKM graph) pairs for training the diffusion model.

Two modes:
  1. Pre-generated: load sentence-graph pairs from saved JSON files
  2. On-the-fly:   generate graphs from raw sentences using the pipeline

Also handles data splits (train/val/test) and the DataLoader setup.
"""

import os, sys, json, random
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_representation import GraphRepresentation, D_NODE, D_EDGE, N_MAX


# ── OKM Graph Dataset ─────────────────────────────────────────────────────────

class OKMGraphDataset(Dataset):
    """
    Dataset of (sentence, OKM graph) pairs.

    Each item returned:
        sentence  (str)
        X         (N_max, D_NODE)   node feature tensor
        E         (N_max, N_max, D_EDGE)  edge feature tensor
        mask      (N_max,) bool
        w_idxs    (N_max,) long — word indices for embedding
        adj       (N_max, N_max) float — adjacency matrix (for edge BCE loss)
        node_types (N_max,) long — ground truth node type classes
        edge_types (N_max, N_max) long — ground truth edge type classes
        word_targets (N_max,) long — ground truth word indices

    Args:
        sentences:   list of sentences
        graphs:      corresponding list of nx.DiGraph (OKM graphs)
        graph_repr:  GraphRepresentation instance (with built vocab)
    """

    def __init__(
        self,
        sentences:   List[str],
        graphs:      List[nx.DiGraph],
        graph_repr:  GraphRepresentation,
    ):
        assert len(sentences) == len(graphs)
        self.sentences   = sentences
        self.graphs      = graphs
        self.graph_repr  = graph_repr

        print(f"[Dataset] {len(sentences)} sentence-graph pairs loaded.")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict:
        sentence = self.sentences[idx]
        G        = self.graphs[idx]

        X, E, mask = self.graph_repr.graph_to_tensors(G)

        # Word indices for embedding lookup
        w_idxs = torch.zeros(self.graph_repr.n_max, dtype=torch.long)
        nodes  = list(G.nodes(data=True))
        for row, (_, data) in enumerate(nodes[:self.graph_repr.n_max]):
            word          = data.get("lemma", data.get("word", "<unk>"))
            w_idxs[row]   = self.graph_repr.word_to_idx(word)

        # Ground truth node types (for CE loss)
        node_types = torch.zeros(self.graph_repr.n_max, dtype=torch.long)
        for row, (_, data) in enumerate(nodes[:self.graph_repr.n_max]):
            node_types[row] = data.get("node_type", 2)  # default: descriptor

        # Ground truth word targets
        word_targets = w_idxs.clone()

        # Adjacency matrix (for edge existence BCE loss)
        adj = torch.zeros(self.graph_repr.n_max, self.graph_repr.n_max)
        node_id_to_row = {nid: r for r, (nid, _) in enumerate(nodes[:self.graph_repr.n_max])}
        for u, v in G.edges():
            if u in node_id_to_row and v in node_id_to_row:
                ru, rv = node_id_to_row[u], node_id_to_row[v]
                if ru < self.graph_repr.n_max and rv < self.graph_repr.n_max:
                    adj[ru, rv] = 1.0

        # Ground truth edge types (for CE loss on existing edges)
        from graph_representation import REL_TO_IDX
        edge_types = torch.zeros(
            self.graph_repr.n_max, self.graph_repr.n_max, dtype=torch.long
        )
        for u, v, edata in G.edges(data=True):
            if u in node_id_to_row and v in node_id_to_row:
                ru, rv = node_id_to_row[u], node_id_to_row[v]
                if ru < self.graph_repr.n_max and rv < self.graph_repr.n_max:
                    dep    = edata.get("dep", "OTHER")
                    et_idx = REL_TO_IDX.get(dep, REL_TO_IDX["OTHER"])
                    edge_types[ru, rv] = et_idx

        return {
            "sentence":    sentence,
            "X":           X,
            "E":           E,
            "mask":        mask,
            "w_idxs":      w_idxs,
            "node_types":  node_types,
            "word_targets":word_targets,
            "adj":         adj,
            "edge_types":  edge_types,
        }


# ── Data loading utilities ────────────────────────────────────────────────────

def load_sentences_from_file(path: str) -> List[str]:
    """Load sentences from a plain text file (one sentence per line)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_graphs_from_dir(graph_dir: str) -> Tuple[List[str], List[nx.DiGraph]]:
    """
    Load saved (sentence, graph) pairs from a directory of JSON files.
    Each file: graph_{idx}.json with 'meta.sentence' field.
    """
    from stage4_graph_constructor import OKMGraphConstructor
    constructor = OKMGraphConstructor()

    files = sorted(
        [f for f in os.listdir(graph_dir) if f.endswith(".json")]
    )
    total = len(files)
    print(f"[DataLoader] Loading {total} graphs from {graph_dir}/")

    sentences = []
    graphs    = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(graph_dir, fname)
        G = constructor.load(path)
        sent = G.graph.get("sentence", "")
        sentences.append(sent)
        graphs.append(G)
        if i % 500 == 0 or i == total:
            print(f"  {i}/{total} loaded...")

    print(f"[DataLoader] Done. {len(graphs)} graphs loaded.")
    return sentences, graphs


def build_dataset_from_sentences(
    sentences: List[str],
    weights_path: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Tuple[List[str], List[nx.DiGraph]]:
    """
    Run the full graph generator pipeline on raw sentences.
    Optionally save graphs to disk for reuse.
    """
    from pipeline import GraphGeneratorPipeline

    pipeline = GraphGeneratorPipeline(weights_path=weights_path)
    graphs   = []

    print(f"[DataLoader] Generating {len(sentences)} graphs...")
    for i, sent in enumerate(sentences):
        G = pipeline.generate(sent)
        graphs.append(G)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sentences)}")

    if save_dir:
        from stage4_graph_constructor import OKMGraphConstructor
        constructor = OKMGraphConstructor()
        os.makedirs(save_dir, exist_ok=True)
        for i, (sent, G) in enumerate(zip(sentences, graphs)):
            G.graph["sentence"] = sent
            path = os.path.join(save_dir, f"graph_{i:06d}.json")
            constructor.save(G, path)
        print(f"[Pipeline] Saved {len(graphs)} graphs to {save_dir}/")

    return sentences, graphs


def make_dataloaders(
    sentences:   List[str],
    graphs:      List[nx.DiGraph],
    graph_repr:  GraphRepresentation,
    batch_size:  int = 16,
    val_split:   float = 0.1,
    test_split:  float = 0.05,
    num_workers: int = 0,
    seed:        int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split into train/val/test and return DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_dataset = OKMGraphDataset(sentences, graphs, graph_repr)
    N            = len(full_dataset)
    n_test       = max(1, int(N * test_split))
    n_val        = max(1, int(N * val_split))
    n_train      = N - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    def collate(batch):
        """Stack variable-length batch into tensors."""
        return {
            "sentence":    [item["sentence"] for item in batch],
            "X":           torch.stack([item["X"] for item in batch]),
            "E":           torch.stack([item["E"] for item in batch]),
            "mask":        torch.stack([item["mask"] for item in batch]),
            "w_idxs":      torch.stack([item["w_idxs"] for item in batch]),
            "node_types":  torch.stack([item["node_types"] for item in batch]),
            "word_targets":torch.stack([item["word_targets"] for item in batch]),
            "adj":         torch.stack([item["adj"] for item in batch]),
            "edge_types":  torch.stack([item["edge_types"] for item in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=num_workers)

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "Brother, a letter for you.",
        "I gave the book to her.",
        "He will go to the market.",
        "The children played in the park.",
        "My father bought a new car.",
    ]

    sentences, graphs = build_dataset_from_sentences(sample_sentences)

    graph_repr = GraphRepresentation()
    graph_repr.build_vocab(graphs)

    train_loader, val_loader, test_loader = make_dataloaders(
        sentences, graphs, graph_repr, batch_size=2
    )

    # Test one batch
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    print("X shape:   ", batch["X"].shape)
    print("E shape:   ", batch["E"].shape)
    print("mask shape:", batch["mask"].shape)
    print("Sentences: ", batch["sentence"])
