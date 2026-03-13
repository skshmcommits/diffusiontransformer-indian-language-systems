"""
Stage 4: Graph Constructor (Rule-Based)
-----------------------------------------
Input:  Resolved OKMTokens from Stage 3
Output: A NetworkX DiGraph representing the OKM Knowledge Graph

Each node  = an attributed OKM word (noun, verb, or descriptor)
Each edge  = a dependency relation between words, labeled with OKM case/relation

This stage is fully deterministic — no learning involved.
"""

import os
import sys
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_attribute_assigner import (
    OKMToken, NodeType, Case, Gender, Number, Tense, Person
)


# ── Edge relation labels ──────────────────────────────────────────────────────

# Maps OKM case → human-readable edge label
CASE_TO_RELATION = {
    Case.SUBJECTIVE:   "SUBJECT_OF",
    Case.OBJECTIVE:    "OBJECT_OF",
    Case.INSTRUMENTAL: "INSTRUMENT_OF",
    Case.DATIVE:       "DATIVE_TO",
    Case.ABLATIVE:     "ABLATIVE_FROM",
    Case.GENITIVE:     "POSSESSES",
    Case.LOCATIVE:     "LOCATED_AT",
    Case.VOCATIVE:     "ADDRESSED_AS",
}

# Dependency relations that should always create edges (for descriptors too)
ALWAYS_EDGE_DEPS = {
    "amod",    # adjectival modifier
    "advmod",  # adverbial modifier
    "det",     # determiner
    "neg",     # negation
    "aux",     # auxiliary
    "mark",    # subordinating conjunction
    "cc",      # coordinating conjunction
    "conj",    # conjunct
    "punct",   # punctuation (usually skipped)
}


# ── Graph Constructor ─────────────────────────────────────────────────────────

class OKMGraphConstructor:
    """
    Builds a NetworkX DiGraph from resolved OKM tokens.

    Node attributes stored in graph:
        - word, lemma, pos, node_type
        - For NOUN: gender, number, case
        - For VERB: person, number, tense
        - okm_notation: the shorthand from the paper e.g. Ram(1,1,1)

    Edge attributes stored in graph:
        - relation: OKM/dependency relation label
        - dep:      original spaCy dependency label
        - case:     OKM case int (for noun edges)

    Usage:
        constructor = OKMGraphConstructor()
        graph = constructor.build(okm_tokens)
    """

    def build(self, okm_tokens: List[OKMToken]) -> nx.DiGraph:
        """
        Build OKM knowledge graph from attributed tokens.

        Args:
            okm_tokens: Resolved tokens from Stage 3.
        Returns:
            nx.DiGraph with OKM node and edge attributes.
        """
        G = nx.DiGraph()
        G.graph["okm_version"] = "1.0"

        # 1. Add all tokens as nodes
        self._add_nodes(G, okm_tokens)

        # 2. Add edges from dependency relations
        self._add_edges(G, okm_tokens)

        return G

    # ── Node construction ─────────────────────────────────────────────────────

    def _add_nodes(self, G: nx.DiGraph, tokens: List[OKMToken]):
        """Add one node per OKM token with all attributes."""
        for token in tokens:
            attrs = self._token_to_node_attrs(token)
            G.add_node(token.idx, **attrs)

    def _token_to_node_attrs(self, token: OKMToken) -> Dict[str, Any]:
        """Convert OKMToken to a flat dictionary for the graph node."""
        base = {
            "word":         token.text,
            "lemma":        token.lemma,
            "pos":          token.pos,
            "dep":          token.dep,
            "node_type":    token.node_type.value,   # int: 0=noun, 1=verb, 2=desc
            "node_type_str":token.node_type.name,
            "okm_notation": token.okm_notation(),
            "ambiguous":    token.case_ambiguous or token.sense_ambiguous,
        }

        if token.node_type == NodeType.NOUN:
            base.update({
                "gender":    token.gender.value  if token.gender  else 3,
                "number":    token.number.value  if token.number  else 1,
                "case":      token.case.value    if token.case    else 2,
                "case_str":  token.case.name     if token.case    else "OBJECTIVE",
                # Verb attrs not applicable
                "person":    -1,
                "tense":     0,
            })
        elif token.node_type == NodeType.VERB:
            base.update({
                "person":    token.person.value   if token.person   else 3,
                "v_number":  token.v_number.value if token.v_number else 1,
                "tense":     token.tense.value    if token.tense    else 0,
                "tense_str": token.tense.name     if token.tense    else "PRESENT",
                # Noun attrs not applicable
                "gender":    -1,
                "number":    -1,
                "case":      -1,
            })
        else:
            # Descriptor — neutral attributes
            base.update({
                "gender": -1, "number": -1, "case": -1,
                "person": -1, "tense": 0,
            })

        return base

    # ── Edge construction ─────────────────────────────────────────────────────

    def _add_edges(self, G: nx.DiGraph, tokens: List[OKMToken]):
        """Add edges from dependency head→child relations."""
        # Build a lookup for quick access
        token_map: Dict[int, OKMToken] = {t.idx: t for t in tokens}

        for token in tokens:
            head_idx = token.head_idx

            # Skip self-loops (root token points to itself)
            if head_idx == token.idx:
                continue

            # Only add edge if head is in our token set
            if head_idx not in token_map:
                continue

            head_token = token_map[head_idx]
            edge_attrs = self._build_edge_attrs(token, head_token)

            # Edge direction: head → child (head governs child)
            G.add_edge(head_idx, token.idx, **edge_attrs)

    def _build_edge_attrs(
        self, child: OKMToken, head: OKMToken
    ) -> Dict[str, Any]:
        """Build edge attribute dictionary from child→head relationship."""
        # Determine relation label
        if child.node_type == NodeType.NOUN and child.case is not None:
            relation = CASE_TO_RELATION.get(child.case, child.dep.upper())
        else:
            relation = child.dep.upper()

        attrs = {
            "dep":       child.dep,
            "relation":  relation,
            "case":      child.case.value if child.case else -1,
            "case_str":  child.case.name  if child.case else "NONE",
            "child_type":child.node_type.name,
            "head_type": head.node_type.name,
        }
        return attrs

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self, G: nx.DiGraph) -> Dict:
        """Serialize graph to a JSON-compatible dictionary."""
        return {
            "nodes": [
                {"id": n, **data}
                for n, data in G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in G.edges(data=True)
            ],
            "meta": G.graph,
        }

    def from_dict(self, d: Dict) -> nx.DiGraph:
        """Reconstruct graph from dictionary (non-destructive)."""
        G = nx.DiGraph()
        G.graph.update(d.get("meta", {}))
        for node in d["nodes"]:
            attrs = {k: v for k, v in node.items() if k != "id"}
            G.add_node(node["id"], **attrs)
        for edge in d["edges"]:
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            G.add_edge(edge["source"], edge["target"], **attrs)
        return G

    def save(self, G: nx.DiGraph, path: str):
        """Save graph to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(G), f, indent=2)

    def load(self, path: str) -> nx.DiGraph:
        """Load graph from JSON file."""
        with open(path, encoding="utf-8") as f:
            return self.from_dict(json.load(f))

    # ── Visualization ─────────────────────────────────────────────────────────

    def print_graph(self, G: nx.DiGraph):
        """Pretty-print the OKM graph to console."""
        print(f"\nOKM Graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print("Nodes:")
        node_type_names = {0: "NOUN", 1: "VERB", 2: "DESC"}
        for n, data in G.nodes(data=True):
            if 'okm_notation' in data:
                label = f"{data['okm_notation']:25s} type={data['node_type_str']}"
            else:
                word = data.get('word', f'node_{n}')
                nt = node_type_names.get(data.get('node_type', -1), 'UNK')
                label = f"{word:25s} type={nt}"
            print(f"  [{n}] {label}")
        print("Edges:")
        for u, v, data in G.edges(data=True):
            u_word = G.nodes[u].get("word", f"node_{u}")
            v_word = G.nodes[v].get("word", f"node_{v}")
            rel = data.get('relation', data.get('dep', 'UNKNOWN'))
            print(f"  {u_word} --[{rel}]--> {v_word}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from stage1_linguistic import LinguisticPreprocessor
    from stage2_attribute_assigner import OKMAttributeAssigner
    from stage3_ambiguity_resolver import AmbiguityResolver

    proc        = LinguisticPreprocessor()
    assigner    = OKMAttributeAssigner()
    resolver    = AmbiguityResolver()
    constructor = OKMGraphConstructor()

    sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
    ]

    for sent in sentences:
        ling_out   = proc.process(sent)
        okm_tokens = assigner.assign(ling_out)
        resolved   = resolver.resolve(okm_tokens, sent)
        graph      = constructor.build(resolved)
        constructor.print_graph(graph)
        print("-" * 60)
