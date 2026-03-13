"""
OKM Diffusion Transformer (Denoiser)
--------------------------------------
Takes a noisy graph (X_t, E_t), a timestep t, and a sentence embedding c,
and predicts the noise ε that was added — or directly predicts x_0.

Architecture:
  Sentence → SentenceEncoder → conditioning vector c
  (X_t, E_t, t, c) → GraphTransformerLayers → output heads

Output heads:
  - Noise prediction for X (node features)   — MSE loss
  - Noise prediction for E (edge features)   — MSE loss
  - Node type logits                          — CE loss
  - Edge existence logits                     — BCE loss
  - Edge type logits                          — CE loss
  - Word index logits                         — CE loss (optional)
"""

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_representation import (
    D_NODE, D_EDGE, N_MAX, D_NODE_TYPE, D_RELATION
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Graph dimensions (from graph_representation.py)
    d_node:      int = D_NODE       # 60
    d_edge:      int = D_EDGE       # 32
    n_max:       int = N_MAX        # 32

    # Model size
    d_model:     int = 256          # internal hidden dimension
    d_cond:      int = 768          # sentence embedding dimension (BERT)
    n_layers:    int = 8            # number of GraphTransformer layers
    n_heads:     int = 8            # attention heads
    d_ff:        int = 512          # feed-forward hidden dim
    dropout:     float = 0.1

    # Diffusion
    T:           int = 1000         # total timesteps

    # Vocabulary
    vocab_size:  int = 50000
    d_word_emb:  int = 32           # word embedding size


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t as a fixed sinusoidal embedding,
    then projects it to d_model dimensions.
    """

    def __init__(self, d_model: int, T: int = 1000):
        super().__init__()
        self.d_model = d_model

        # Learnable projection on top of sinusoidal base
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timestep indices
        Returns:
            (B, d_model) time embedding
        """
        device = t.device
        half   = self.d_model // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args   = t[:, None].float() * freqs[None]    # (B, half)
        emb    = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.proj(emb)


# ── Graph Transformer Layer ───────────────────────────────────────────────────

class GraphTransformerLayer(nn.Module):
    """
    One layer of the graph-aware transformer.

    Operations per layer:
      1. Incorporate edge information into node representations
      2. Node self-attention (nodes attend to other nodes)
      3. Cross-attention with sentence conditioning vector
      4. Feed-forward network
      5. Update edge representations from updated node reps

    All with residual connections and LayerNorm.
    """

    def __init__(self, d_model: int, d_edge: int, d_cond: int,
                 n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads

        # Edge → node aggregation
        self.edge_to_node = nn.Sequential(
            nn.Linear(d_edge, d_model),
            nn.GELU(),
        )

        # Node self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention: nodes attend to sentence embedding
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cond_proj = nn.Linear(d_cond, d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Edge update: edge features updated from node pair representations
        self.edge_update = nn.Sequential(
            nn.Linear(d_model * 2 + d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, d_edge),
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_edge = nn.LayerNorm(d_edge)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X:    torch.Tensor,       # (B, N, d_model)  node features
        E:    torch.Tensor,       # (B, N, N, d_edge) edge features
        cond: torch.Tensor,       # (B, 1, d_model)  sentence conditioning
        mask: Optional[torch.Tensor] = None,  # (B, N) bool, True=real
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, _ = X.shape

        # ── 1. Edge influence on nodes ────────────────────────────────────────
        # Average edge features coming INTO each node
        edge_agg = self.edge_to_node(E).mean(dim=2)  # (B, N, d_model)
        X = X + edge_agg

        # ── 2. Node self-attention ────────────────────────────────────────────
        key_padding_mask = ~mask if mask is not None else None  # True=ignore
        X_attn, _ = self.self_attn(X, X, X, key_padding_mask=key_padding_mask)
        X = self.norm1(X + self.dropout(X_attn))

        # ── 3. Cross-attention with sentence conditioning ─────────────────────
        # cond: (B, L_cond, d_model) — L_cond=1 for single vector, or seq for full
        X_cross, _ = self.cross_attn(X, cond, cond)
        X = self.norm2(X + self.dropout(X_cross))

        # ── 4. Feed-forward ───────────────────────────────────────────────────
        X = self.norm3(X + self.dropout(self.ffn(X)))

        # ── 5. Update edge features from node pairs ───────────────────────────
        Xi = X.unsqueeze(2).expand(B, N, N, -1)    # (B, N, N, d_model)
        Xj = X.unsqueeze(1).expand(B, N, N, -1)    # (B, N, N, d_model)
        edge_input = torch.cat([Xi, Xj, E], dim=-1) # (B, N, N, 2*d_model + d_edge)
        E = self.norm_edge(E + self.dropout(self.edge_update(edge_input)))

        return X, E


# ── Sentence Encoder ──────────────────────────────────────────────────────────

class SentenceEncoder(nn.Module):
    """
    Encodes input sentences to conditioning vectors using a pre-trained
    sentence transformer model.

    The weights are frozen — we don't backprop into BERT.
    A small projection head adapts its output to d_model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 d_out: int = 256):
        super().__init__()
        self.d_out = d_out
        self._model_name = model_name
        self._loaded = False

        # Projection: sentence_dim → d_model
        # We'll set input_dim after loading
        self._proj = None

    def _load(self, device):
        """Lazy load to avoid slow import at module level."""
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name)
            
            # Try to move to requested device, fall back to CPU if CUDA out of memory
            try:
                self._st_model.to(device)
                actual_device = device
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"[SentenceEncoder] CUDA memory error, falling back to CPU")
                    self._st_model.to("cpu")
                    actual_device = torch.device("cpu")
                else:
                    raise
            
            for p in self._st_model.parameters():
                p.requires_grad = False

            # Get output dim
            dummy = self._st_model.encode(["test"], convert_to_tensor=True)
            in_dim = dummy.shape[-1]
            self._proj = nn.Linear(in_dim, self.d_out).to(actual_device)
            self._loaded = True
            print(f"[SentenceEncoder] Loaded {self._model_name}, dim={in_dim}->{self.d_out}")
        except ImportError:
            print("[SentenceEncoder] sentence-transformers not installed. Using random encoder.")
            self._proj = nn.Linear(384, self.d_out)
            self._loaded = True
            self._st_model = None

    def forward(self, sentences, device=None) -> torch.Tensor:
        """
        Args:
            sentences: list of strings, length B
        Returns:
            (B, d_out) conditioning tensor
        """
        if device is None:
            device = next(self._proj.parameters()).device if self._proj else torch.device("cpu")

        self._load(device)

        if self._st_model is not None:
            with torch.no_grad():
                try:
                    emb = self._st_model.encode(
                        sentences, convert_to_tensor=True, device=device
                    )                                   # (B, 384)
                except RuntimeError as e:
                    # fallback to CPU if CUDA OOM during encoding
                    if "out of memory" in str(e).lower():
                        print("[SentenceEncoder] CUDA OOM during encode, retrying on CPU")
                        emb = self._st_model.encode(
                            sentences, convert_to_tensor=True, device="cpu"
                        )
                    else:
                        raise
                # Some models return inference tensors, convert to regular
                emb = emb.clone().to(device)
        else:
            B = len(sentences)
            emb = torch.randn(B, 384, device=device)

        return self._proj(emb)                          # (B, d_out)


# ── Main Denoiser ─────────────────────────────────────────────────────────────

class OKMDiffusionTransformer(nn.Module):
    """
    The core denoising model.

    Given:
      - Noisy node features X_t  (B, N, d_node)
      - Noisy edge features E_t  (B, N, N, d_edge)
      - Timestep t               (B,)
      - Sentence embedding c     (B, d_cond)
      - Node mask                (B, N)
      - Word indices             (B, N)

    Predicts noise (ε_X, ε_E) and also directly predicts:
      - Node type logits         (B, N, 3)
      - Edge existence logits    (B, N, N, 1)
      - Edge type logits         (B, N, N, 20)
      - Word logits              (B, N, vocab_size)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Word embedding
        self.word_embed = nn.Embedding(cfg.vocab_size, cfg.d_word_emb, padding_idx=0)

        # Input projections (raw feature dims → d_model)
        self.node_in_proj = nn.Linear(cfg.d_node + cfg.d_word_emb, cfg.d_model)
        self.edge_in_proj = nn.Linear(cfg.d_edge,                   cfg.d_edge)
        self.cond_in_proj = nn.Linear(cfg.d_model, cfg.d_model)  # sentence already projected

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(cfg.d_model, cfg.T)

        # Graph Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                d_model=cfg.d_model,
                d_edge=cfg.d_edge,
                d_cond=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        # Output heads
        # Node noise prediction (same dim as input node features)
        self.node_noise_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_node),
        )

        # Edge noise prediction
        self.edge_noise_head = nn.Sequential(
            nn.LayerNorm(cfg.d_edge),
            nn.Linear(cfg.d_edge, cfg.d_edge * 2),
            nn.GELU(),
            nn.Linear(cfg.d_edge * 2, cfg.d_edge),
        )

        # Discrete prediction heads (for auxiliary losses and final graph decoding)
        self.node_type_head  = nn.Linear(cfg.d_model, D_NODE_TYPE)      # 3
        self.edge_exist_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )  # pairwise additive, much cheaper than Bilinear
        self.edge_type_head  = nn.Linear(cfg.d_edge, D_RELATION)        # 20
        self.word_head       = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        X_t:     torch.Tensor,             # (B, N, D_NODE)
        E_t:     torch.Tensor,             # (B, N, N, D_EDGE)
        t:       torch.Tensor,             # (B,)
        cond:    torch.Tensor,             # (B, d_model)  sentence embedding
        mask:    torch.Tensor,             # (B, N) bool
        w_idxs:  torch.Tensor,             # (B, N) word indices
    ):
        B, N, _ = X_t.shape

        # ── Embed word indices and concatenate with node features ─────────────
        word_emb = self.word_embed(w_idxs)              # (B, N, d_word_emb)
        X = torch.cat([X_t, word_emb], dim=-1)          # (B, N, D_NODE + d_word_emb)
        X = self.node_in_proj(X)                        # (B, N, d_model)

        E = self.edge_in_proj(E_t)                      # (B, N, N, d_edge)

        # ── Time embedding added to every node ────────────────────────────────
        t_emb = self.time_embed(t)                      # (B, d_model)
        X = X + t_emb.unsqueeze(1)                      # broadcast to (B, N, d_model)

        # ── Conditioning: project to d_model, unsqueeze for cross-attention ───
        cond_proj = self.cond_in_proj(cond).unsqueeze(1)  # (B, 1, d_model)

        # ── Graph transformer layers ──────────────────────────────────────────
        for layer in self.layers:
            X, E = layer(X, E, cond_proj, mask)

        # ── Output heads ─────────────────────────────────────────────────────

        # Noise predictions (primary diffusion objective)
        noise_X = self.node_noise_head(X)               # (B, N, D_NODE)
        noise_E = self.edge_noise_head(E)               # (B, N, N, D_EDGE)

        # Discrete predictions (auxiliary objectives + inference)
        node_type_logits = self.node_type_head(X)       # (B, N, 3)
        word_logits      = self.word_head(X)            # (B, N, vocab_size)

        # Edge existence: additive pairwise (memory-efficient)
        Xi_plus_Xj = X.unsqueeze(2) + X.unsqueeze(1)   # (B, N, N, d_model)
        edge_exist_logits = self.edge_exist_head(Xi_plus_Xj)  # (B, N, N, 1)

        edge_type_logits  = self.edge_type_head(E)      # (B, N, N, 20)

        return (
            noise_X,
            noise_E,
            node_type_logits,
            edge_exist_logits,
            edge_type_logits,
            word_logits,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = OKMDiffusionTransformer(cfg)
    print(f"Model parameters: {model.count_parameters():,}")

    B, N = 2, 32
    X_t    = torch.randn(B, N, D_NODE)
    E_t    = torch.randn(B, N, N, D_EDGE)
    t      = torch.randint(0, 1000, (B,))
    cond   = torch.randn(B, cfg.d_model)
    mask   = torch.ones(B, N, dtype=torch.bool)
    w_idxs = torch.randint(0, cfg.vocab_size, (B, N))

    noise_X, noise_E, nt_logits, ee_logits, et_logits, w_logits = model(
        X_t, E_t, t, cond, mask, w_idxs
    )
    print(f"noise_X:      {noise_X.shape}")       # (2, 32, 60)
    print(f"noise_E:      {noise_E.shape}")       # (2, 32, 32, 32)
    print(f"node_type:    {nt_logits.shape}")     # (2, 32, 3)
    print(f"edge_exist:   {ee_logits.shape}")     # (2, 32, 32, 1)
    print(f"edge_type:    {et_logits.shape}")     # (2, 32, 32, 20)
    print(f"word_logits:  {w_logits.shape}")      # (2, 32, 50000)
