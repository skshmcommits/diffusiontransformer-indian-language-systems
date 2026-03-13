"""
Trainer
--------
Full training loop for the OKM Diffusion Transformer.

Losses:
  1. Node noise MSE        — main diffusion objective for nodes
  2. Edge noise MSE        — main diffusion objective for edges
  3. Node type CE          — auxiliary: predict node type (noun/verb/desc)
  4. Edge existence BCE    — auxiliary: predict which edges exist
  5. Edge type CE          — auxiliary: predict edge relation type
  6. Word prediction CE    — auxiliary: predict word at each node

Classifier-Free Guidance:
  With probability cfg_drop_prob, the sentence embedding is replaced
  with zeros during training. This teaches the model to generate graphs
  both with and without conditioning, enabling CFG at inference time.
"""

import os, sys, math, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import OKMDiffusionTransformer, ModelConfig, SentenceEncoder
from noise_schedule import NoiseSchedule
from graph_representation import GraphRepresentation


# ── Training Config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    output_dir:     str   = "checkpoints/"
    log_file:       str   = "training_log.jsonl"

    # Training
    n_epochs:       int   = 100
    batch_size:     int   = 8     # reduced from 16 to fit in 6 GB VRAM
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 500

    # Diffusion
    T:              int   = 1000
    schedule:       str   = "cosine"

    # Loss weights
    w_node_noise:   float = 1.0
    w_edge_noise:   float = 1.0
    w_node_type:    float = 0.5
    w_edge_exist:   float = 0.5
    w_edge_type:    float = 0.3
    w_word:         float = 0.2

    # Classifier-Free Guidance
    cfg_drop_prob:  float = 0.1   # probability of dropping conditioning

    # Hardware
    device:         str   = "auto"
    use_amp:        bool  = True  # automatic mixed precision (faster on GPU)

    # Logging
    log_every:      int   = 50    # log every N steps
    val_every:      int   = 500   # validate every N steps
    save_every:     int   = 1000  # save checkpoint every N steps


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training loop.

    Usage:
        trainer = Trainer(model_cfg, train_cfg, sentence_encoder, graph_repr)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model_cfg:        ModelConfig,
        train_cfg:        TrainConfig,
        sentence_encoder: SentenceEncoder,
        graph_repr:       GraphRepresentation,
    ):
        self.train_cfg = train_cfg
        self.graph_repr = graph_repr

        # Device
        if train_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(train_cfg.device)
        print(f"[Trainer] Using device: {self.device}")

        # Model
        self.model   = OKMDiffusionTransformer(model_cfg).to(self.device)
        self.enc     = sentence_encoder
        self.schedule = NoiseSchedule(T=train_cfg.T, schedule=train_cfg.schedule)

        print(f"[Trainer] Model parameters: {self.model.count_parameters():,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        # LR Scheduler with warmup then cosine decay
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=train_cfg.use_amp and self.device.type == "cuda")

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")

        os.makedirs(train_cfg.output_dir, exist_ok=True)

    def _build_scheduler(self):
        warmup = self.train_cfg.warmup_steps
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup) / 100000)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── Core training step ────────────────────────────────────────────────────

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """
        One training step: sample t, add noise, predict, compute losses.
        """
        self.model.train()

        # Move tensors to device
        X0      = batch["X"].to(self.device)           # (B, N, D_NODE)
        E0      = batch["E"].to(self.device)           # (B, N, N, D_EDGE)
        mask    = batch["mask"].to(self.device)        # (B, N)
        w_idxs  = batch["w_idxs"].to(self.device)     # (B, N)
        nt_gt   = batch["node_types"].to(self.device)  # (B, N)
        wt_gt   = batch["word_targets"].to(self.device)# (B, N)
        adj     = batch["adj"].to(self.device)         # (B, N, N)
        et_gt   = batch["edge_types"].to(self.device)  # (B, N, N)
        sents   = batch["sentence"]                    # List[str]

        B = X0.shape[0]

        # ── Encode sentences (with classifier-free guidance dropout) ──────────
        with torch.no_grad():
            cond = self.enc(sents, device=self.device)  # (B, d_model)

        # CFG: randomly zero out conditioning
        if self.train_cfg.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.train_cfg.cfg_drop_prob
            cond = cond.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # ── Sample random timesteps ───────────────────────────────────────────
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)

        # ── Forward diffusion: add noise to X and E ───────────────────────────
        Xt, noise_X = self.schedule.q_sample(X0, t)
        Et, noise_E = self.schedule.q_sample(E0, t)

        # ── Model forward pass ────────────────────────────────────────────────
        with autocast("cuda", enabled=self.train_cfg.use_amp and self.device.type == "cuda"):
            (
                pred_noise_X,
                pred_noise_E,
                nt_logits,
                ee_logits,
                et_logits,
                w_logits,
            ) = self.model(Xt, Et, t, cond, mask, w_idxs)

            # ── Compute losses ────────────────────────────────────────────────
            cfg = self.train_cfg

            # 1. Node noise MSE (only over real nodes)
            node_mask = mask.unsqueeze(-1).float()
            loss_node = (F.mse_loss(pred_noise_X, noise_X, reduction="none") * node_mask).mean()

            # 2. Edge noise MSE (only over real node pairs)
            edge_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()
            loss_edge = (F.mse_loss(pred_noise_E, noise_E, reduction="none") * edge_mask).mean()

            # 3. Node type CE
            if mask.any():
                loss_nt = F.cross_entropy(nt_logits[mask], nt_gt[mask])
            else:
                loss_nt = torch.tensor(0.0, device=self.device)

            # 4. Edge existence BCE (proper 2D edge mask)
            edge_pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
            if edge_pair_mask.any():
                loss_ee = F.binary_cross_entropy_with_logits(
                    ee_logits.squeeze(-1)[edge_pair_mask],
                    adj[edge_pair_mask].float()
                )
            else:
                loss_ee = torch.tensor(0.0, device=self.device)

            # 5. Edge type CE (only on existing edges)
            existing = adj.bool()
            if existing.any():
                loss_et = F.cross_entropy(
                    et_logits[existing], et_gt[existing]
                )
            else:
                loss_et = torch.tensor(0.0, device=self.device)

            # 6. Word prediction CE
            loss_word = F.cross_entropy(
                w_logits[mask], wt_gt[mask]
            )

            # ── Weighted total ────────────────────────────────────────────────
            total_loss = (
                cfg.w_node_noise  * loss_node +
                cfg.w_edge_noise  * loss_edge +
                cfg.w_node_type   * loss_nt   +
                cfg.w_edge_exist  * loss_ee   +
                cfg.w_edge_type   * loss_et   +
                cfg.w_word        * loss_word
            )

        # ── Backward pass ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total":     total_loss.item(),
            "node":      loss_node.item(),
            "edge":      loss_edge.item(),
            "node_type": loss_nt.item(),
            "edge_exist":loss_ee.item(),
            "edge_type": loss_et.item(),
            "word":      loss_word.item(),
        }

    # ── Validation step ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_step(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            X0     = batch["X"].to(self.device)
            E0     = batch["E"].to(self.device)
            mask   = batch["mask"].to(self.device)
            w_idxs = batch["w_idxs"].to(self.device)
            sents  = batch["sentence"]
            B      = X0.shape[0]

            cond = self.enc(sents, device=self.device)
            t    = torch.randint(0, self.schedule.T, (B,), device=self.device)
            Xt, noise_X = self.schedule.q_sample(X0, t)
            Et, noise_E = self.schedule.q_sample(E0, t)

            pred_noise_X, pred_noise_E, *_ = self.model(Xt, Et, t, cond, mask, w_idxs)

            node_mask = mask.unsqueeze(-1).float()
            edge_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()
            loss = (
                (F.mse_loss(pred_noise_X, noise_X, reduction="none") * node_mask).mean() +
                (F.mse_loss(pred_noise_E, noise_E, reduction="none") * edge_mask).mean()
            )
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    # ── Main training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ):
        """Full training loop."""
        cfg = self.train_cfg
        log_path = os.path.join(cfg.output_dir, cfg.log_file)

        print(f"\n[Trainer] Starting training for {cfg.n_epochs} epochs.")
        print(f"          Steps per epoch: {len(train_loader)}")
        print(f"          Logging to: {log_path}\n")

        for epoch in range(1, cfg.n_epochs + 1):
            epoch_losses = []
            t0 = time.time()

            for batch in train_loader:
                self.global_step += 1
                losses = self._train_step(batch)
                epoch_losses.append(losses["total"])

                # ── Logging ───────────────────────────────────────────────────
                if self.global_step % cfg.log_every == 0:
                    lr  = self.scheduler.get_last_lr()[0]
                    log = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "lr":   lr,
                        **losses,
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log) + "\n")

                    print(
                        f"  step={self.global_step:6d} | "
                        f"loss={losses['total']:.4f} | "
                        f"node={losses['node']:.4f} | "
                        f"edge={losses['edge']:.4f} | "
                        f"lr={lr:.2e}"
                    )

                # ── Validation ────────────────────────────────────────────────
                if self.global_step % cfg.val_every == 0:
                    val_loss = self._val_step(val_loader)
                    print(f"\n  [Val] step={self.global_step} val_loss={val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")
                        print(f"  [Val] New best! Saved checkpoint.\n")

                # ── Checkpoint ────────────────────────────────────────────────
                if self.global_step % cfg.save_every == 0:
                    self.save_checkpoint(f"ckpt_step_{self.global_step}.pt")

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed  = time.time() - t0
            print(f"\nEpoch {epoch:3d}/{cfg.n_epochs} | "
                  f"avg_loss={avg_loss:.4f} | "
                  f"time={elapsed:.1f}s\n")

        print("[Trainer] Training complete.")
        self.save_checkpoint("final_model.pt")

    # ── Checkpoint utilities ──────────────────────────────────────────────────

    def save_checkpoint(self, name: str):
        path = os.path.join(self.train_cfg.output_dir, name)
        torch.save({
            "model_state":     self.model.state_dict(),
            "model_cfg":       asdict(self.model.cfg),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step":     self.global_step,
            "best_val_loss":   self.best_val_loss,
        }, path)
        print(f"  [Trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.global_step   = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"[Trainer] Loaded checkpoint from {path} (step={self.global_step})")
