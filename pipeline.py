"""
Graph Generator Pipeline
-------------------------
Combines all four stages into one clean interface.

Usage:
    pipeline = GraphGeneratorPipeline()
    graph = pipeline.generate("Ram went to school by bus.")
    graphs = pipeline.generate_batch(["sentence 1", "sentence 2"])
"""

import os, sys
from typing import List, Optional
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage1_linguistic import LinguisticPreprocessor, LinguisticOutput
from stage2_attribute_assigner import OKMAttributeAssigner, OKMToken
from stage3_ambiguity_resolver import AmbiguityResolver
from stage4_graph_constructor import OKMGraphConstructor


class GraphGeneratorPipeline:
    """
    Full graph generator: sentence (str) → OKM knowledge graph (nx.DiGraph)

    Args:
        spacy_model:    spaCy model name (auto-selects best available if None)
        weights_path:   path to CaseClassifier weights for Stage 3 (optional)
        verbose:        print debug info at each stage
    """

    def __init__(
        self,
        spacy_model:  Optional[str] = None,
        weights_path: Optional[str] = None,
        verbose:      bool = False,
    ):
        self.verbose = verbose
        print("[Pipeline] Initializing graph generator pipeline...")

        self.stage1 = LinguisticPreprocessor(model_name=spacy_model)
        self.stage2 = OKMAttributeAssigner()
        self.stage3 = AmbiguityResolver(weights_path=weights_path)
        self.stage4 = OKMGraphConstructor()

        print("[Pipeline] Ready.")

    def generate(self, sentence: str) -> nx.DiGraph:
        """
        Generate OKM knowledge graph for a single sentence.

        Args:
            sentence: Any English sentence.
        Returns:
            nx.DiGraph with OKM node and edge attributes.
        """
        if self.verbose:
            print(f"\n[Pipeline] Processing: '{sentence}'")

        # Stage 1: Linguistic analysis
        ling_output = self.stage1.process(sentence)
        if self.verbose:
            print(f"  Stage 1 -> {len(ling_output.tokens)} tokens")

        # Stage 2: OKM attribute assignment
        okm_tokens = self.stage2.assign(ling_output)
        if self.verbose:
            print(f"  Stage 2 -> {[t.okm_notation() for t in okm_tokens]}")

        # Stage 3: Ambiguity resolution
        resolved = self.stage3.resolve(okm_tokens, sentence)
        flagged = [t for t in okm_tokens if t.case_ambiguous or t.sense_ambiguous]
        if self.verbose and flagged:
            print(f"  Stage 3 -> resolved {len(flagged)} ambiguous token(s)")

        # Stage 4: Build graph
        graph = self.stage4.build(resolved)
        if self.verbose:
            print(f"  Stage 4 -> graph: {graph.number_of_nodes()} nodes, "
                  f"{graph.number_of_edges()} edges")

        # Attach metadata
        graph.graph["sentence"] = sentence
        return graph

    def generate_batch(self, sentences: List[str]) -> List[nx.DiGraph]:
        """
        Generate OKM graphs for a list of sentences.
        Returns a list of graphs in the same order.
        """
        graphs = []
        for i, sent in enumerate(sentences):
            if i % 500 == 0 and i > 0:
                print(f"[Pipeline] Processed {i}/{len(sentences)} sentences...")
            graphs.append(self.generate(sent))
        return graphs

    def generate_and_save(self, sentences: List[str], output_dir: str):
        """Generate graphs and save each to a JSON file."""
        import json
        os.makedirs(output_dir, exist_ok=True)
        for i, sent in enumerate(sentences):
            graph = self.generate(sent)
            path = os.path.join(output_dir, f"graph_{i:06d}.json")
            self.stage4.save(graph, path)
        print(f"[Pipeline] Saved {len(sentences)} graphs to {output_dir}/")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = GraphGeneratorPipeline(verbose=True)

    test_sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "Brother, a letter for you.",
        "I gave the book to her.",
    ]

    for sent in test_sentences:
        graph = pipeline.generate(sent)
        pipeline.stage4.print_graph(graph)
        print("=" * 60)
