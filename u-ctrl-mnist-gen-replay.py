"""
================================================================================
PROJECT: Unsupervised Manifold Discovery (u-CTRL)
MODULE: Generative Replay & Latent Trajectory Morphing
================================================================================

OVERVIEW:
This module extends the unsupervised manifold discovery (u-CTRL) framework by
introducing Generative Replay, replacing the exact historical memory buffer from
`u-ctrl-mnist-replay.py` with a purely latent-space buffer.

KEY DIFFERENCES FROM `u-ctrl-mnist-replay.py`:
1. Generative vs. Exact Replay: Instead of storing explicit historical images
   (pixels) in a Reservoir buffer, this model only stores latent representations
   (z) and uses a Decoder to "dream" the historical sensory experiences for
   consolidation.
2. Architecture Upgrade: Transitioned from an Encoder-only architecture to a full
   Autoencoder-style `DiscoveryEngine` (Encoder + Decoder).
3. Data-Driven Latent Buffer: The buffer now maintains active "Room Centroids"
   using competitive learning directly in the latent space.
4. Manifold Morphing: Introduces a biological trajectory test to visualize
   smooth generative transitions between different learned cell states
   (latent classes) via linear interpolation.
================================================================================
"""

# %% [1] Imports & Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

# %% [2] Configuration
CONFIG = {
    "d": 128,  # Latent dimensionality
    "hidden": 512,  # MLP Width
    "eps": 0.5,  # Coding rate precision
    "k": 20,  # Neighborhood size
    "lambda_local": 2.5,  # Subspace compression strength
    "lambda_distill": 50.0,  # Generative replay anchor
    "lambda_rec": 15.0,  # Reconstruction fidelity
    "batch_size": 512,
    "lr": 1e-3,
    "total_steps": 1600,
    "buffer_size": 100,  # TINY BUFFER STRESS TEST
    "expected_rooms": 10,  # Target number of "Latent Neighborhoods"
    "perturb_step": 1200,
}


# %% [3] Data-Driven "Room" Buffer
class DataDrivenBuffer:
    def __init__(self, capacity, n_rooms, latent_dim=128):
        self.capacity = capacity
        self.n_rooms = n_rooms
        self.latents = torch.zeros((capacity, latent_dim))
        self.labels = torch.zeros(capacity, dtype=torch.long)  # Viz only

        # Room Centroids: Initialized as random unit vectors
        self.centroids = F.normalize(torch.randn(n_rooms, latent_dim), dim=1).to(DEVICE)
        self.current_pos = 0
        self.is_filled = False

    def update(self, latents, labels):
        latents_dev = latents.to(DEVICE)

        # 1. Update Centroids (Competitive Learning)
        with torch.no_grad():
            for i in range(latents_dev.shape[0]):
                z = latents_dev[i]
                dists = torch.norm(self.centroids - z, dim=1)
                room_idx = torch.argmin(dists)

                # Moving Average Centroid Update
                alpha = 0.05
                self.centroids[room_idx] = F.normalize(
                    (1 - alpha) * self.centroids[room_idx] + alpha * z, dim=0
                )

        # 2. Store in Buffer (Simple overwrite for N=100 stress test)
        latents_cpu, labels_cpu = latents.cpu(), labels.cpu()
        for i in range(latents_cpu.shape[0]):
            if self.current_pos < self.capacity:
                idx = self.current_pos
                self.current_pos += 1
            else:
                idx = np.random.randint(0, self.capacity)
                self.is_filled = True

            self.latents[idx] = latents_cpu[i]
            self.labels[idx] = labels_cpu[i]

    def get_batch(self, size):
        limit = self.capacity if self.is_filled else self.current_pos
        if limit == 0:
            return None, None
        idx = torch.randperm(limit)[:size]
        return self.latents[idx].to(DEVICE), self.labels[idx].to(DEVICE)


# %% [4] Discovery Engine
class DiscoveryEngine(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def get_z(self, x):
        return F.normalize(self.encoder(x), dim=1)

    def dream(self, z):
        return self.decoder(z)

    def compute_ctrl_loss(self, z, config):
        def coding_rate(Z, eps):
            n, d = Z.shape
            alpha = d / (n * eps**2)
            M = torch.eye(d, device=Z.device) + alpha * (Z.t() @ Z)
            eig_dev = M.cpu() if Z.device.type == "mps" else M
            eigvals = torch.linalg.eigvalsh(eig_dev).to(Z.device)
            return 0.5 * torch.sum(torch.log(eigvals.clamp(min=1e-7)))

        R_total = coding_rate(z, config["eps"])
        anchors = torch.randperm(z.shape[0])[:64]
        sims = z[anchors] @ z.t()
        _, topk = torch.topk(sims, config["k"], dim=1)
        R_local = sum(
            coding_rate(z[topk[i]], config["eps"]) for i in range(len(anchors))
        ) / len(anchors)
        return -R_total + config["lambda_local"] * R_local, (R_total - R_local).item()


# %% [5] Visualizer
def emit_snapshot(step, model, all_images, all_labels, history, buffer):
    model.eval()
    with torch.no_grad():
        # Latents for data
        z_data = model.get_z(all_images[:1000].to(DEVICE))
        # Latents for Centroids
        z_centroids = buffer.centroids
        # Combined for UMAP
        z_combined = torch.cat([z_data, z_centroids], dim=0).cpu().numpy()

        labels = all_labels[:1000].numpy()
        z_sample, _ = buffer.get_batch(10)
        dreams = (
            model.dream(z_sample).cpu().view(-1, 28, 28)
            if z_sample is not None
            else None
        )

    # Fit UMAP on combined data + centroids
    reducer = umap.UMAP(metric="cosine", n_neighbors=15)
    emb_all = reducer.fit_transform(z_combined)
    emb_data = emb_all[:1000]
    emb_cents = emb_all[1000:]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Manifold with Centroids
    axes[0].scatter(
        emb_data[:, 0], emb_data[:, 1], c=labels, cmap="tab10", s=10, alpha=0.4
    )
    axes[0].scatter(
        emb_cents[:, 0],
        emb_cents[:, 1],
        color="black",
        marker="X",
        s=100,
        label="Room Centroids",
    )
    axes[0].set_title("Self-Organizing Manifold")
    axes[0].legend()

    # 2. Orthogonality
    ortho = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            mi, mj = labels == i, labels == j
            if mi.any() and mj.any():
                ortho[i, j] = (z_data[mi] @ z_data[mj].t()).mean().item()
    sns.heatmap(ortho, cmap="magma", ax=axes[1], cbar=False)
    axes[1].set_title("Orthogonality")

    # 3. Dreams
    if dreams is not None:
        axes[2].imshow(torch.cat([d for d in dreams], dim=1), cmap="gray")
        axes[2].axis("off")
        axes[2].set_title("Buffer Dreams")

    # 4. Gain
    axes[3].plot(history["gain"])
    axes[3].set_title("Expansion Gain")

    plt.tight_layout()
    plt.show()
    model.train()


# %% [6] Training Loop
mnist = datasets.MNIST(root="data", train=True, download=True)
all_images, all_labels = mnist.data.float().view(-1, 784) / 255.0, mnist.targets

model = DiscoveryEngine().to(DEVICE)
buffer = DataDrivenBuffer(CONFIG["buffer_size"], CONFIG["expected_rooms"])
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
history = {"gain": []}

for step in range(CONFIG["total_steps"]):
    # Sensory stream curriculum
    if step < 500:
        cur = [0, 1, 2]
    elif step < 1000:
        cur = [0, 1, 2, 3, 4, 5, 6]
    else:
        cur = list(range(10))

    mask = torch.isin(all_labels, torch.tensor(cur))
    idx = torch.randperm(mask.sum())[: CONFIG["batch_size"]]
    x_live, y_live = all_images[mask][idx].to(DEVICE), all_labels[mask][idx]

    if step >= CONFIG["perturb_step"]:
        x_live = transforms.GaussianBlur(kernel_size=5, sigma=1.5)(
            x_live.view(-1, 1, 28, 28)
        ).view(-1, 784)

    z_live = model.get_z(x_live)
    l_ctrl, gain = model.compute_ctrl_loss(z_live, CONFIG)
    l_rec = F.mse_loss(model.dream(z_live), x_live)

    z_old, _ = buffer.get_batch(128)
    l_replay = (
        F.mse_loss(model.get_z(model.dream(z_old)), z_old) if z_old is not None else 0
    )

    loss = (
        l_ctrl + (CONFIG["lambda_rec"] * l_rec) + (CONFIG["lambda_distill"] * l_replay)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        with torch.no_grad():
            buffer.update(z_live[:32], y_live[:32])

    history["gain"].append(gain)
    if step % 400 == 0 or step in [501, 1001]:
        emit_snapshot(step, model, all_images, all_labels, history, buffer)


print("Discovery Complete.")


# %% [7] Manifold Morphing (Biological Trajectory Test)
def morph_latent_space(encoder, decoder, buffer, steps=10):
    encoder.eval()
    decoder.eval()
    print("🎨 Generating Latent Trajectory...")

    with torch.no_grad():
        # 1. Grab two distinct points from the buffer
        z_set, y_set = buffer.get_batch(buffer.capacity)
        if z_set is None:
            return

        # Pick two classes to morph between (e.g., 0 and 1)
        # We'll just grab the first two unique classes we find in the buffer
        u_labels = torch.unique(y_set)
        if len(u_labels) < 2:
            print("Not enough diversity in buffer to morph yet.")
            return

        z_start = z_set[y_set == u_labels[1]][0]
        z_end = z_set[y_set == u_labels[7]][0]

        # 2. Linear Interpolation (LERP) in 128D
        # We move in 'steps' increments from z_start to z_end
        alphas = torch.linspace(0, 1, steps).to(DEVICE)
        z_trajectory = torch.stack([(1 - a) * z_start + a * z_end for a in alphas])

        # 3. 'Dream' the trajectory
        # Ensure vectors are normalized if your encoder uses L2 normalization
        z_trajectory = F.normalize(z_trajectory, dim=1)
        dreams = decoder(z_trajectory).cpu().view(-1, 28, 28)

    # 4. Visualize the "Cellular Transition"
    fig, axes = plt.subplots(1, steps, figsize=(20, 2))
    fig.suptitle(
        f"Morphing: Class {u_labels[0].item()} ➔ Class {u_labels[1].item()}",
        fontsize=16,
        y=1.1,
    )

    for i in range(steps):
        axes[i].imshow(
            dreams[i], cmap="magma"
        )  # Magma looks great for 'active' signals
        axes[i].axis("off")
        axes[i].set_title(f"{int(alphas[i]*100)}%")

    plt.show()


# Run the morph
morph_latent_space(model.encoder, model.decoder, buffer)

# %%
