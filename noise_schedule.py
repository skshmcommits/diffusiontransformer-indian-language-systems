"""
Noise Schedule
---------------
Controls how noise is added to graphs during training (forward process)
and how it's removed during inference (reverse process).

Supports:
  - Linear schedule (simple, original DDPM)
  - Cosine schedule (better for small graphs, less aggressive early noise)
"""

import torch
import math
from typing import Tuple


class NoiseSchedule:
    """
    Precomputes all noise schedule coefficients for T timesteps.

    Args:
        T:        total diffusion timesteps (default 1000)
        schedule: "linear" or "cosine"
        beta_start, beta_end: noise range for linear schedule
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
    ):
        self.T = T
        self.schedule = schedule

        # Compute betas (noise variance at each step)
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            betas = self._cosine_betas(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute all derived quantities (standard DDPM notation)
        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)   # ᾱ_t
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), alphas_cumprod[:-1]       # ᾱ_{t-1}
        ])

        self.register("betas",                  betas)
        self.register("alphas",                 alphas)
        self.register("alphas_cumprod",         alphas_cumprod)
        self.register("alphas_cumprod_prev",    alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod",    alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod",
                      (1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod",
                      (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas",      alphas.rsqrt())
        self.register("sqrt_recip_alphas_cumprod",
                      alphas_cumprod.rsqrt())
        self.register("sqrt_recipm1_alphas_cumprod",
                      (1.0 / alphas_cumprod - 1).sqrt())

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register("posterior_variance",     posterior_variance)
        self.register("posterior_log_variance_clipped",
                      torch.log(posterior_variance.clamp(min=1e-20)))
        self.register("posterior_mean_coef1",
                      betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        self.register("posterior_mean_coef2",
                      (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

    def register(self, name: str, val: torch.Tensor):
        setattr(self, name, val)

    def _cosine_betas(self, T: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal 2021).
        Produces less aggressive noise in early timesteps.
        """
        steps  = T + 1
        x      = torch.linspace(0, T, steps)
        f_t    = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas  = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0001, 0.9999)

    # ── Forward process: q(x_t | x_0) ────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,    # original clean tensor (any shape with batch dim)
        t:  torch.Tensor,    # timestep indices (B,)
        noise: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t by adding t-steps of noise to x_0.

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Returns:
            x_t:   noisy version of x0
            noise: the noise that was added (for loss computation)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        B = x0.shape[0]

        def gather(coef: torch.Tensor) -> torch.Tensor:
            """Gather schedule values for batch timesteps, reshape for broadcasting."""
            vals = coef.to(x0.device)[t]   # (B,)
            # Reshape to (B, 1, 1, ...) to broadcast over spatial dims
            shape = (B,) + (1,) * (x0.dim() - 1)
            return vals.reshape(shape)

        sqrt_alpha = gather(self.sqrt_alphas_cumprod)
        sqrt_omac  = gather(self.sqrt_one_minus_alphas_cumprod)

        x_t = sqrt_alpha * x0 + sqrt_omac * noise
        return x_t, noise

    # ── Reverse step: p(x_{t-1} | x_t) ──────────────────────────────────────

    def p_mean_variance(
        self,
        x_t:            torch.Tensor,   # noisy input at step t
        t:              torch.Tensor,   # timestep indices
        predicted_noise: torch.Tensor,  # model's noise prediction ε_θ
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for one reverse step.
        Used during DDPM sampling (slow but exact).
        """
        B = x_t.shape[0]

        def gather(coef):
            vals = coef.to(x_t.device)[t]
            return vals.reshape((B,) + (1,) * (x_t.dim() - 1))

        # Reconstruct x_0 from x_t and predicted noise
        sqrt_recip = gather(self.sqrt_recip_alphas_cumprod)
        sqrt_recm1 = gather(self.sqrt_recipm1_alphas_cumprod)
        x0_pred = sqrt_recip * x_t - sqrt_recm1 * predicted_noise
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Posterior mean
        coef1 = gather(self.posterior_mean_coef1)
        coef2 = gather(self.posterior_mean_coef2)
        mean  = coef1 * x0_pred + coef2 * x_t

        # Posterior variance
        log_var = gather(self.posterior_log_variance_clipped)
        var     = log_var.exp()

        return mean, var

    def p_sample(
        self,
        x_t:            torch.Tensor,
        t:              torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of DDPM reverse diffusion: x_{t-1} ~ p(x_{t-1} | x_t).
        """
        mean, var = self.p_mean_variance(x_t, t, predicted_noise)

        # No noise at t=0 (last step)
        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().reshape((x_t.shape[0],) + (1,) * (x_t.dim() - 1))

        x_prev = mean + nonzero * var.sqrt() * noise
        return x_prev

    def ddim_sample(
        self,
        x_t:            torch.Tensor,
        t:              torch.Tensor,
        t_prev:         torch.Tensor,
        predicted_noise: torch.Tensor,
        eta: float = 0.0,           # eta=0 → deterministic DDIM
    ) -> torch.Tensor:
        """
        DDIM reverse step — allows ~50x fewer steps than DDPM.
        Use this for fast inference.
        """
        B = x_t.shape[0]

        def gather(coef, ts):
            vals = coef.to(x_t.device)[ts]
            return vals.reshape((B,) + (1,) * (x_t.dim() - 1))

        alpha_t      = gather(self.alphas_cumprod,      t)
        alpha_t_prev = gather(self.alphas_cumprod_prev, t_prev.clamp(min=0))

        # Predict x_0
        x0_pred = (x_t - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Direction pointing to x_t
        sigma = eta * ((1 - alpha_t_prev) / (1 - alpha_t)).sqrt() * (1 - alpha_t / alpha_t_prev).sqrt()
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        x_prev = (
            alpha_t_prev.sqrt() * x0_pred
            + (1 - alpha_t_prev - sigma ** 2).sqrt() * predicted_noise
            + sigma * noise
        )
        return x_prev


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    schedule = NoiseSchedule(T=1000, schedule="cosine")

    x0 = torch.randn(4, 32, 60)   # batch of 4 graphs
    t  = torch.randint(0, 1000, (4,))

    x_t, noise = schedule.q_sample(x0, t)
    x_prev     = schedule.p_sample(x_t, t, noise)

    print(f"x0 shape:    {x0.shape}")
    print(f"x_t shape:   {x_t.shape}")
    print(f"x_prev shape:{x_prev.shape}")
    print(f"alpha range: [{schedule.alphas_cumprod.min():.4f}, {schedule.alphas_cumprod.max():.4f}]")
    print(f"Noise schedule: {schedule.schedule}, T={schedule.T}")
