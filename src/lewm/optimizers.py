"""Population-risk-gated AdamW.

Implements Algorithm 1 from Litman & Guo, "A Theory of Generalization in
Deep Learning" (arXiv:2605.01172). One extra parameter-sized state vector
on top of AdamW; a per-parameter mask suppresses updates on parameters
whose batch-mean gradient is dominated by leave-one-out noise.

Update rule per parameter k:
    s_t  = ρ s_{t-1} + (1-ρ)(g_t - m_{t-1})²       # streaming gradient var
    m_t  = β1 m_{t-1} + (1-β1) g_t
    v_t  = β2 v_{t-1} + (1-β2) g_t²
    after bias-correct (m̂, v̂, ŝ):
        δ = (m̂² - α·ŝ)_+
        q = δ / (δ + λ_pop·ŝ + ε)                  # soft gate
    w_t  = w_{t-1} - η·q⊙(m̂ / (√v̂ + ε)) - η·λ_wd·w_{t-1}

Setting `q = 1` (the no-gate path) recovers standard AdamW. The gate
suppresses updates on parameters where m̂² < α·ŝ — meaning the batch
signal can't beat the leave-one-out noise.

Parameters
----------
alpha : float
    LOO coefficient. 1.0 is the fresh-batch boundary (recommended for
    streaming/multi-epoch). b/(n-b) is the formally-correct finite-dataset
    boundary; for VCC at b=512, n=212k that is ~0.0024 and the gate is
    nearly always open.
lambda_pop : float
    Soft-gate denominator scale. 0.0 is the hard binary form smoothed only
    by ε. Larger values shrink the update; the paper says it is typically
    unnecessary at scale.
rho : float
    EMA decay for the streaming gradient variance estimator. ~0.99.
"""
from __future__ import annotations

import math
import torch
from torch.optim.optimizer import Optimizer


class PopRiskAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rho: float = 0.99,
        alpha: float = 1.0,
        lambda_pop: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"invalid rho {rho}")
        if alpha < 0.0:
            raise ValueError(f"invalid alpha {alpha}")
        if lambda_pop < 0.0:
            raise ValueError(f"invalid lambda_pop {lambda_pop}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            rho=rho, alpha=alpha, lambda_pop=lambda_pop,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            rho = group["rho"]
            alpha = group["alpha"]
            lambda_pop = group["lambda_pop"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("PopRiskAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["s"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v, s = state["m"], state["v"], state["s"]
                state["step"] += 1
                t = state["step"]

                # Streaming variance uses g - m_prev (m before this step's update)
                m_prev = m  # alias; we update m below
                deviation = g - m_prev
                s.mul_(rho).addcmul_(deviation, deviation, value=1.0 - rho)

                # Standard Adam moments
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                bc_s = 1.0 - rho ** t
                m_hat = m / bc1
                v_hat = v / bc2
                s_hat = s / bc_s

                # Population-risk gate
                m_hat_sq = m_hat * m_hat
                # δ = max(m̂² - α·ŝ, 0)
                delta = (m_hat_sq - alpha * s_hat).clamp_(min=0.0)
                # q = δ / (δ + λ·ŝ + ε)
                q = delta / (delta + lambda_pop * s_hat + eps)

                denom = v_hat.sqrt().add_(eps)
                update = q * (m_hat / denom)

                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def gate_stats(self) -> dict:
        """Return summary stats of the gate values. Useful for diagnostics:
        if `mean_q` stays near 1.0 the gate isn't doing anything; if it stays
        near 0 the gate is too aggressive.
        """
        qs = []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            alpha = group["alpha"]
            lambda_pop = group["lambda_pop"]
            eps = group["eps"]
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is None or "m" not in state:
                    continue
                t = state["step"]
                m, v, s = state["m"], state["v"], state["s"]
                bc1 = 1.0 - beta1 ** t
                bc_s = 1.0 - rho ** t
                m_hat = m / bc1
                s_hat = s / bc_s
                delta = (m_hat * m_hat - alpha * s_hat).clamp_(min=0.0)
                q = delta / (delta + lambda_pop * s_hat + eps)
                qs.append(q.flatten())
        if not qs:
            return {}
        all_q = torch.cat(qs)
        return {
            "mean_q": all_q.mean().item(),
            "frac_open": (all_q > 0.5).float().mean().item(),
            "frac_killed": (all_q < 0.01).float().mean().item(),
        }
