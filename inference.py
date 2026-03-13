"""
Inference Engine
-----------------
Takes a trained model and a sentence → generates an OKM knowledge graph.

Supports:
  - DDPM sampling (slow, high quality, 1000 steps)
  - DDIM sampling (fast, ~50 steps, nearly same quality)
  - Classifier-Free Guidance (CFG) for stronger conditioning

The final output is a NetworkX DiGraph with full OKM attributes.
"""

import os, sys, torch
import networkx as nx
from typing import Optional, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import OKMDiffusionTransformer, ModelConfig, SentenceEncoder
from noise_schedule import NoiseSchedule
from graph_representation import (
    GraphRepresentation, D_NODE, D_EDGE, N_MAX
)


# ── Inference Config ──────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    # Sampling
    sampler:        str   = "ddim"   # "ddpm" or "ddim"
    n_steps:        int   = 50       # DDIM steps (ignored for DDPM)
    eta:            float = 0.0      # DDIM stochasticity (0=deterministic)

    # Classifier-Free Guidance
    cfg_scale:      float = 3.0      # guidance strength (1=no guidance)

    # Post-processing
    edge_threshold: float = 0.5      # sigmoid threshold for edge existence
    n_max:          int   = N_MAX    # max nodes in generated graph

    # Hardware
    device:         str   = "auto"


# ── Inference Engine ──────────────────────────────────────────────────────────

class OKMInferenceEngine:
    """
    Generates OKM knowledge graphs from sentences using the trained model.

    Usage:
        engine = OKMInferenceEngine.from_checkpoint("checkpoints/best_model.pt", ...)
        graph  = engine.generate("Ram went to school by bus.")
    """

    def __init__(
        self,
        model:            OKMDiffusionTransformer,
        sentence_encoder: SentenceEncoder,
        graph_repr:       GraphRepresentation,
        noise_schedule:   NoiseSchedule,
        cfg:              InferenceConfig,
    ):
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model    = model.to(self.device).eval()
        self.enc      = sentence_encoder
        self.repr     = graph_repr
        self.schedule = noise_schedule
        self.cfg      = cfg

        print(f"[Inference] Engine ready on {self.device}.")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_cfg:       Optional[ModelConfig]       = None,
        graph_repr:      Optional[GraphRepresentation] = None,
        inf_cfg:         Optional[InferenceConfig]   = None,
        vocab_path:      Optional[str]               = None,
    ) -> "OKMInferenceEngine":
        """
        Load model from a checkpoint file and return a ready inference engine.
        """
        if inf_cfg is None:
            inf_cfg = InferenceConfig()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint first so we can reconstruct the exact training config
        ckpt  = torch.load(checkpoint_path, map_location=device)

        if model_cfg is None:
            ckpt_cfg = ckpt.get("model_cfg", None)
            if ckpt_cfg is not None:
                model_cfg = ModelConfig(**ckpt_cfg)
            else:
                model_cfg = ModelConfig()

        # Load model weights
        model = OKMDiffusionTransformer(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Inference] Loaded model from {checkpoint_path}")

        # Build supporting components
        enc      = SentenceEncoder(d_out=model_cfg.d_model)
        schedule = NoiseSchedule(T=model_cfg.T)

        if graph_repr is None:
            graph_repr = GraphRepresentation(n_max=model_cfg.n_max)
            if vocab_path and os.path.exists(vocab_path):
                import json
                with open(vocab_path, encoding="utf-8") as f:
                    graph_repr.word2idx = json.load(f)
                graph_repr.idx2word = {v: k for k, v in graph_repr.word2idx.items()}

        return cls(model, enc, graph_repr, schedule, inf_cfg)

    # ── Main interface ────────────────────────────────────────────────────────

    def generate(self, sentence: str) -> nx.DiGraph:
        """
        Generate an OKM graph for a single sentence.

        Args:
            sentence: Input English sentence.
        Returns:
            nx.DiGraph with OKM node/edge attributes.
        """
        return self.generate_batch([sentence])[0]

    def generate_batch(self, sentences: List[str]) -> List[nx.DiGraph]:
        """
        Generate OKM graphs for a batch of sentences.
        """
        B = len(sentences)

        # ── Encode sentences ──────────────────────────────────────────────────
        with torch.no_grad():
            cond = self.enc(sentences, device=self.device)  # (B, d_model)
            # Null conditioning for CFG (unconditional)
            null_cond = torch.zeros_like(cond)

        # ── Start from pure noise ─────────────────────────────────────────────
        X_t = torch.randn(B, self.cfg.n_max, D_NODE, device=self.device)
        E_t = torch.randn(B, self.cfg.n_max, self.cfg.n_max, D_EDGE, device=self.device)
        # All nodes initially unmasked (model will implicitly learn sparsity)
        mask    = torch.ones(B, self.cfg.n_max, dtype=torch.bool, device=self.device)
        w_idxs  = torch.zeros(B, self.cfg.n_max, dtype=torch.long, device=self.device)

        # ── Reverse diffusion ─────────────────────────────────────────────────
        if self.cfg.sampler == "ddim":
            X_0, E_0, final_outputs = self._ddim_sample(
                X_t, E_t, mask, w_idxs, cond, null_cond
            )
        else:
            X_0, E_0, final_outputs = self._ddpm_sample(
                X_t, E_t, mask, w_idxs, cond, null_cond
            )

        # ── Decode tensors → graphs ───────────────────────────────────────────
        nt_logits, ee_logits, et_logits, w_logits = final_outputs
        graphs = []
        for b in range(B):
            G = self.repr.tensors_to_graph(
                X_0[b], E_0[b], mask[b],
                nt_logits[b], ee_logits[b], et_logits[b], w_logits[b],
                edge_threshold=self.cfg.edge_threshold,
            )
            G.graph["sentence"] = sentences[b]
            graphs.append(G)

        return graphs

    # ── DDIM sampler ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ddim_sample(
        self,
        X_t, E_t, mask, w_idxs, cond, null_cond
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Fast DDIM reverse diffusion with CFG.
        Uses n_steps steps instead of T steps.
        """
        T = self.schedule.T
        n = self.cfg.n_steps

        # Evenly spaced timesteps from T-1 down to 0
        timesteps = list(reversed(range(0, T, T // n)))

        final_outputs = None

        for i, t_val in enumerate(timesteps):
            t_val_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            t_tensor      = torch.full((X_t.shape[0],), t_val,      dtype=torch.long, device=X_t.device)
            t_prev_tensor = torch.full((X_t.shape[0],), max(t_val_prev, 0), dtype=torch.long, device=X_t.device)

            # Conditional prediction
            (noise_X_cond, noise_E_cond,
             nt_logits, ee_logits, et_logits, w_logits) = self.model(
                X_t, E_t, t_tensor, cond, mask, w_idxs
            )

            # Unconditional prediction (for CFG)
            (noise_X_uncond, noise_E_uncond, *_) = self.model(
                X_t, E_t, t_tensor, null_cond, mask, w_idxs
            )

            # CFG: guided noise = uncond + scale * (cond - uncond)
            s = self.cfg.cfg_scale
            noise_X = noise_X_uncond + s * (noise_X_cond - noise_X_uncond)
            noise_E = noise_E_uncond + s * (noise_E_cond - noise_E_uncond)

            # DDIM step
            X_t = self.schedule.ddim_sample(X_t, t_tensor, t_prev_tensor, noise_X, self.cfg.eta)
            E_t = self.schedule.ddim_sample(E_t, t_tensor, t_prev_tensor, noise_E, self.cfg.eta)

            final_outputs = (nt_logits, ee_logits, et_logits, w_logits)

        return X_t, E_t, final_outputs

    # ── DDPM sampler ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ddpm_sample(
        self,
        X_t, E_t, mask, w_idxs, cond, null_cond
    ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Full DDPM reverse diffusion (slow but exact). T steps.
        """
        final_outputs = None

        for t_val in reversed(range(self.schedule.T)):
            t_tensor = torch.full((X_t.shape[0],), t_val, dtype=torch.long, device=X_t.device)

            (noise_X_cond, noise_E_cond,
             nt_logits, ee_logits, et_logits, w_logits) = self.model(
                X_t, E_t, t_tensor, cond, mask, w_idxs
            )
            (noise_X_uncond, noise_E_uncond, *_) = self.model(
                X_t, E_t, t_tensor, null_cond, mask, w_idxs
            )

            s       = self.cfg.cfg_scale
            noise_X = noise_X_uncond + s * (noise_X_cond - noise_X_uncond)
            noise_E = noise_E_uncond + s * (noise_E_cond - noise_E_uncond)

            X_t = self.schedule.p_sample(X_t, t_tensor, noise_X)
            E_t = self.schedule.p_sample(E_t, t_tensor, noise_E)

            final_outputs = (nt_logits, ee_logits, et_logits, w_logits)

        return X_t, E_t, final_outputs

    # ── Pretty print ─────────────────────────────────────────────────────────

    def print_graph(self, G: nx.DiGraph):
        """Print a generated OKM graph in a readable format."""
        from stage4_graph_constructor import OKMGraphConstructor
        OKMGraphConstructor().print_graph(G)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test inference with an untrained model (random output, just checks shapes)
    from model import ModelConfig, OKMDiffusionTransformer, SentenceEncoder
    from graph_representation import GraphRepresentation
    from noise_schedule import NoiseSchedule

    model_cfg = ModelConfig(T=100)  # small T for quick test
    model     = OKMDiffusionTransformer(model_cfg)
    enc       = SentenceEncoder(d_out=model_cfg.d_model)
    repr_     = GraphRepresentation()
    schedule  = NoiseSchedule(T=100)
    inf_cfg   = InferenceConfig(sampler="ddim", n_steps=10, cfg_scale=1.0)

    engine = OKMInferenceEngine(model, enc, repr_, schedule, inf_cfg)

    sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
    ]

    graphs = engine.generate_batch(sentences)
    for sent, G in zip(sentences, graphs):
        print(f"\nInput: {sent}")
        engine.print_graph(G)
