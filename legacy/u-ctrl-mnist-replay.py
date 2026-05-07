"""
================================================================================
PROJECT: Unsupervised Manifold Discovery (u-CTRL)
MODULE: Unsupervised Discovery Stream & Hippocampal Replay
================================================================================

OVERVIEW:
This module implements a purely unsupervised, geometry-driven approach to learning
cell-state representations. Using the principle of Maximal Coding Rate Reduction
(u-CTRL), the model learns to map high-dimensional input (sensory stream) into
incoherent, low-dimensional subspaces on a 128D hypersphere.

NEURO-BIOLOGICAL MAPPING:
1. Neocortex (Encoder):
   Learns a compressed manifold by maximizing global expansion (differentiation)
   and minimizing local compression (clustering).

2. Hippocampus (Reservoir Buffer):
   A statistical memory bank using Reservoir Sampling to maintain a balanced
   representation of all historical states (cell types), preventing bias toward
   the most recent "sensory" input.

3. Consolidation (Dreaming/Replay):
   Uses a Distillation Loss (MSE) to align current projections with historical
   coordinates. This prevents 'Catastrophic Forgetting' when the model encounters
   novel data or experimental perturbations.

4. Surprise Metric (P300 Wave):
   Calculates the geometric 'alienness' of new batches relative to the
   Hippocampal memory. Spikes in surprise trigger 'Aha!' moments (snapshots).

EXPERIMENTAL PIPELINE:
- Phase 1: Simple world discovery (Digits 0, 1, 2).
- Phase 2: Novelty Injection (Expansion to Digits 3-6).
- Phase 3: Stress Testing (Injection of 6-9 + Gaussian Blur/Batch Effects).

OUTPUTS:
- UMAP Manifold: Visual topology of the 128D latent space.
- Orthogonality Matrix: Numerical proof of class incoherence (Cosine Similarity).
- Gain/Surprise Curves: Real-time tracking of geometric crystallization.

================================================================================
"""

# %% [1] Imports & Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# %% [2] Configuration
CONFIG = {
    "d": 128,
    "hidden": 512,
    "eps": 0.5,
    "k": 20,
    "lambda_local": 2.5,
    "lambda_distill": 40.0,
    "batch_size": 512,
    "lr": 1e-3,
    "total_steps": 1600,
    "buffer_size": 1500,
    "n_anchors": 64,
    "perturb_step": 1200,
}


# %% [3] Unsupervised Reservoir Buffer
class UnsupervisedReplayBuffer:
    def __init__(self, capacity, input_dim=784, latent_dim=128):
        self.capacity = capacity
        self.images = torch.zeros((capacity, input_dim))
        self.latents = torch.zeros((capacity, latent_dim))
        self.labels = torch.zeros(capacity, dtype=torch.long)
        self.is_filled = False
        self.current_pos = 0
        self.seen_count = 0

    def update(self, images, latents, labels):
        images, latents, labels = images.cpu(), latents.cpu(), labels.cpu()
        for i in range(images.shape[0]):
            self.seen_count += 1
            if self.current_pos < self.capacity:
                self.images[self.current_pos] = images[i]
                self.latents[self.current_pos] = latents[i]
                self.labels[self.current_pos] = labels[i]
                self.current_pos += 1
                if self.current_pos == self.capacity:
                    self.is_filled = True
            else:
                idx = np.random.randint(0, self.seen_count)
                if idx < self.capacity:
                    self.images[idx] = images[i]
                    self.latents[idx] = latents[i]
                    self.labels[idx] = labels[i]

    def get_batch(self, size):
        limit = self.current_pos if not self.is_filled else self.capacity
        if limit == 0:
            return None, None
        indices = torch.randperm(limit)[:size]
        return self.images[indices].to(DEVICE), self.latents[indices].to(DEVICE)


# %% [4] Engine
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def compute_coding_rate(Z, eps):
    n, d = Z.shape
    alpha = d / (n * eps**2)
    M = torch.eye(d, device=Z.device) + alpha * (Z.t() @ Z)
    if Z.device.type == "mps":
        eigvals = torch.linalg.eigvalsh(M.cpu()).to(Z.device)
    else:
        eigvals = torch.linalg.eigvalsh(M)
    return 0.5 * torch.sum(torch.log(eigvals.clamp(min=1e-7)))


def get_unsupervised_loss(Z, config):
    R_total = compute_coding_rate(Z, config["eps"])
    n, d = Z.shape
    anchor_idx = torch.randperm(n)[: config["n_anchors"]]
    sims = Z[anchor_idx] @ Z.t()
    _, topk = torch.topk(sims, config["k"], dim=1)
    R_local = (
        sum(
            compute_coding_rate(Z[topk[i]], config["eps"])
            for i in range(config["n_anchors"])
        )
        / config["n_anchors"]
    )
    gain = (R_total - R_local).item()
    loss = -R_total + config["lambda_local"] * R_local
    return loss, gain


def apply_perturbation(x_batch):
    blurrer = transforms.GaussianBlur(kernel_size=5, sigma=1.5)
    x_view = x_batch.view(-1, 1, 28, 28)
    return blurrer(x_view).view(-1, 784)


def compute_surprise(z_live, buffer):
    limit = buffer.current_pos if not buffer.is_filled else buffer.capacity
    if limit < 10:
        return 0.0
    _, z_ref = buffer.get_batch(min(limit, 128))
    if z_ref is None or z_ref.shape[0] == 0:
        return 0.0
    dist = torch.cdist(z_live, z_ref)
    if dist.shape[1] == 0:
        return 0.0
    min_dist, _ = torch.min(dist, dim=1)
    return min_dist.mean().item()


# %% [5] Visualizer
def emit_snapshot(
    step, encoder, all_images, all_labels, history, buffer, reason="Scheduled"
):
    print(f"\n🧠 Snapshot @ Step {step} | Reason: {reason}")
    encoder.eval()
    with torch.no_grad():
        n_pts = 1200
        z_torch = encoder(all_images[:n_pts].to(DEVICE))
        z = z_torch.cpu().numpy()
        labels = all_labels[:n_pts].numpy()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    emb = reducer.fit_transform(z)

    # Orthogonality Matrix Calculation
    ortho_matrix = np.zeros((10, 10))
    for i in range(10):
        mask_i = labels == i
        if not mask_i.any():
            continue
        for j in range(10):
            mask_j = labels == j
            if not mask_j.any():
                continue
            # Average Cosine Similarity between class i and class j
            sim = (z_torch[mask_i] @ z_torch[mask_j].t()).mean().item()
            ortho_matrix[i, j] = sim

    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1])

    # 1. Manifold Plot
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7, vmin=0, vmax=9
    )
    ax1.set_title("128D Manifold (UMAP)")
    ax1.axis("off")
    plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Gain & Surprise Curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history["gain"], color="blue", label="Gain")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(history["surprise"], color="orange", label="Surprise", alpha=0.4)
    ax2.set_title("Latent Dynamics")
    ax2.legend(loc="upper left")

    # 3. Hippocampal Coverage
    ax3 = fig.add_subplot(gs[0, 2])
    limit = buffer.current_pos if not buffer.is_filled else buffer.capacity
    if limit > 0:
        ax3.hist(
            buffer.labels[:limit].numpy(),
            bins=np.arange(11) - 0.5,
            rwidth=0.8,
            color="green",
            alpha=0.5,
        )
        ax3.set_xticks(range(10))
        ax3.set_title("Memory Buffer Contents")

    # 4. Orthogonality Heatmap
    ax4 = fig.add_subplot(gs[0, 3])
    sns.heatmap(
        ortho_matrix, annot=False, cmap="magma", ax=ax4, square=True, vmin=0, vmax=1
    )
    ax4.set_title("Orthogonality (Cosine Sim)")

    plt.tight_layout()
    plt.show()
    encoder.train()


# %% [6] Training Loop
dataset = datasets.MNIST(root="data", train=True, download=True)
all_images = dataset.data.float().view(-1, 784) / 255.0
all_labels = dataset.targets

encoder = Encoder(latent_dim=CONFIG["d"]).to(DEVICE)
buffer = UnsupervisedReplayBuffer(CONFIG["buffer_size"], latent_dim=CONFIG["d"])
optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG["lr"])

history = {"gain": [], "surprise": []}
last_gain = 0.0

print("🚀 Discovery Stream Initialized...")
for step in range(CONFIG["total_steps"]):
    if step < 500:
        current_digits = [0, 1, 2]
    elif step < 1000:
        current_digits = [0, 1, 2, 3, 4, 5, 6]
    else:
        current_digits = list(range(10))

    mask = torch.isin(all_labels, torch.tensor(current_digits))
    current_pool, current_pool_labels = all_images[mask], all_labels[mask]

    idx = torch.randperm(len(current_pool))[: CONFIG["batch_size"]]
    x_live, y_live = current_pool[idx], current_pool_labels[idx]

    if step >= CONFIG["perturb_step"]:
        x_live = apply_perturbation(x_live)
    x_live = x_live.to(DEVICE)
    z_live = encoder(x_live)

    # Dynamics
    surprise = compute_surprise(z_live, buffer)
    history["surprise"].append(surprise)
    loss, gain = get_unsupervised_loss(z_live, CONFIG)

    # Replay Consolidation
    x_replay, z_old = buffer.get_batch(128)
    if x_replay is not None:
        loss += CONFIG["lambda_distill"] * F.mse_loss(encoder(x_replay), z_old)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        with torch.no_grad():
            buffer.update(x_live[:32], z_live[:32], y_live[:32])

    history["gain"].append(gain)

    # Trigger Logic
    is_periodic = step > 0 and step % 200 == 0
    is_phase_shift = step == 501 or step == 1001 or step == CONFIG["perturb_step"] + 1
    is_aha = step > 50 and gain > (last_gain * 1.05) and step < CONFIG["perturb_step"]

    if is_periodic or is_phase_shift or is_aha:
        reason = (
            "🆕 Phase Shift"
            if is_phase_shift
            else ("🧠 'Aha!'" if is_aha else "Periodic Update")
        )
        emit_snapshot(step, encoder, all_images, all_labels, history, buffer, reason)
        last_gain = gain

    last_gain = 0.95 * last_gain + 0.05 * gain

print("\n✨ Complete.")
emit_snapshot(
    CONFIG["total_steps"],
    encoder,
    all_images,
    all_labels,
    history,
    buffer,
    "Final State",
)
