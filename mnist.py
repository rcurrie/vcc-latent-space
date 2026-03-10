# %% Imports & Configuration
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# %% Configurations
CONFIG = {
    "d": 128,  # latent dimension
    "hidden": 512,  # hidden layer width
    "eps": 0.5,  # coding rate precision
    "k": 20,  # k-NN neighborhood size
    "lambda_local": 1.0,  # weight on local coding rate
    "batch_size": 512,
    "lr": 1e-3,
    "epochs_per_stage": 30,
    "buffer_size": 2000,  # replay buffer capacity
    "n_anchors": 128,  # anchor points for local rate estimation
    "subspace_rank": 10,  # rank for subspace similarity analysis
    "lambda_distill": 10.0,  # weight on latent distillation loss
    "stages": [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]],
}


# %% Data Loading
def get_mnist_data():
    """Load MNIST training set as flat float32 tensors."""
    dataset = datasets.MNIST(root="data", train=True, download=True)
    images = dataset.data.float().reshape(-1, 784) / 255.0
    labels = dataset.targets
    return images, labels


# %% DataLoader Creation
def make_stage_loader(
    images,
    labels,
    stage_digits,
    buffer_images=None,
    buffer_latents=None,
    buffer_labels=None,
    batch_size=512,
    latent_dim=128,
):
    """Create a DataLoader for a specific stage with replay data.

    Each sample is (image, z_old, is_replay, label). For new (non-replay) samples,
    z_old is a zero vector and is_replay=0; for replay samples, z_old holds the
    stored latent anchor and is_replay=1.
    """
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for d in stage_digits:
        mask |= labels == d
    stage_images = images[mask]
    stage_labels = labels[mask]
    n_new = len(stage_images)
    # New samples: no latent anchor
    stage_z_old = torch.zeros(n_new, latent_dim)
    stage_is_replay = torch.zeros(n_new)

    if buffer_images is not None and len(buffer_images) > 0:
        stage_images = torch.cat([stage_images, buffer_images], dim=0)
        stage_z_old = torch.cat([stage_z_old, buffer_latents], dim=0)
        stage_is_replay = torch.cat([stage_is_replay, torch.ones(len(buffer_images))])
        stage_labels = torch.cat([stage_labels, buffer_labels], dim=0)

    dataset = TensorDataset(stage_images, stage_z_old, stage_is_replay, stage_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# %% MLP Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=1)  # project onto unit hypersphere
        return z


# %% Coding Rate Functions
def compute_coding_rate(Z, eps):
    """
    Compute the coding rate R_eps(Z) = 0.5 * log det(I + d/(n*eps^2) * Z^T @ Z).
    Z: (n, d) tensor on device.
    Returns a scalar tensor.
    """
    n, d = Z.shape
    alpha = d / (n * eps**2)
    M = torch.eye(d, device=Z.device) + alpha * (Z.t() @ Z)
    # eigvalsh is not supported on MPS -- compute on CPU and move back
    eigvals = torch.linalg.eigvalsh(M.cpu()).to(Z.device)
    logdet = torch.sum(torch.log(eigvals.clamp(min=1e-7)))
    return 0.5 * logdet


def compute_local_coding_rate(Z, k, eps, n_anchors=128):
    """
    Approximate the average local coding rate using k-NN neighborhoods.
    Subsamples n_anchors points as query centers to keep computation tractable.
    Z: (n, d) tensor, assumed L2-normalized.
    """
    n, d = Z.shape
    k = min(k, n - 1)
    n_anchors = min(n_anchors, n)

    # Subsample anchor indices
    anchor_idx = torch.randperm(n, device=Z.device)[:n_anchors]

    # Compute similarities between anchors and all points
    Z_anchors = Z[anchor_idx]  # (n_anchors, d)
    sims = Z_anchors @ Z.t()  # (n_anchors, n)

    # Find k nearest neighbors for each anchor (exclude self by taking k+1 and slicing)
    _, topk_idx = torch.topk(sims, k + 1, dim=1)  # (n_anchors, k+1)
    topk_idx = topk_idx[:, 1:]  # (n_anchors, k) -- drop self

    # Compute local coding rate for each anchor's neighborhood
    total_rate = torch.tensor(0.0, device=Z.device)
    for i in range(n_anchors):
        Z_local = Z[topk_idx[i]]  # (k, d)
        total_rate = total_rate + compute_coding_rate(Z_local, eps)

    return total_rate / n_anchors


def ctrl_loss(Z, k, eps, lambda_local, n_anchors=128):
    """
    Coding Rate Reduction loss.
    Maximize total R(Z) (negate in loss), minimize local R(Z_i).
    Returns (loss, R_total, R_local) for logging.
    """
    R_total = compute_coding_rate(Z, eps)
    R_local = compute_local_coding_rate(Z, k, eps, n_anchors)
    loss = -R_total + lambda_local * R_local
    return loss, R_total, R_local


# %% Replay Buffer
class ReplayBuffer:
    """Stores raw images and their latent vectors (z_old) for distillation-based replay."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.images = []  # list of per-class image tensors
        self.latents = []  # list of per-class latent tensors (z_old)
        self.labels = []  # list of per-class label tensors
        self.classes_seen = []

    def update(self, images, labels, latents, new_digits):
        """Snapshot current latent vectors alongside images, rebalancing across all seen classes."""
        self.classes_seen.extend(new_digits)
        n_per_class = self.capacity // len(self.classes_seen)

        all_images = torch.cat(self.images + [images]) if self.images else images
        all_labels = torch.cat(self.labels + [labels]) if self.labels else labels
        all_latents = torch.cat(self.latents + [latents]) if self.latents else latents

        new_images, new_labels, new_latents = [], [], []
        for c in self.classes_seen:
            mask = all_labels == c
            c_images = all_images[mask]
            c_latents = all_latents[mask]
            if len(c_images) > n_per_class:
                idx = torch.randperm(len(c_images))[:n_per_class]
                c_images = c_images[idx]
                c_latents = c_latents[idx]
            new_images.append(c_images)
            new_latents.append(c_latents)
            new_labels.append(torch.full((len(c_images),), c, dtype=torch.long))

        self.images = new_images
        self.latents = new_latents
        self.labels = new_labels

    def get_data(self):
        """Return buffered (images, latents, labels) or (None, None, None) if empty."""
        if not self.images:
            return None, None, None
        return torch.cat(self.images), torch.cat(self.latents), torch.cat(self.labels)


# %% Training Loop
def train_stage(encoder, stage_digits, all_images, all_labels, buffer, config):
    """Train the encoder on a single continual-learning stage with latent distillation."""
    buf_images, buf_latents, buf_labels = buffer.get_data()
    loader = make_stage_loader(
        all_images,
        all_labels,
        stage_digits,
        buf_images,
        buf_latents,
        buf_labels,
        config["batch_size"],
        config["d"],
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["lr"])
    encoder.train()

    has_replay = buf_images is not None
    digits_str = ",".join(str(d) for d in stage_digits)
    for epoch in range(1, config["epochs_per_stage"] + 1):
        epoch_loss, epoch_Rt, epoch_Rl, epoch_distill, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for x_batch, z_old_batch, is_replay_batch, _ in loader:
            x_batch = x_batch.to(DEVICE)
            z = encoder(x_batch)

            # Coding rate reduction loss on the full batch
            loss, R_total, R_local = ctrl_loss(
                z,
                config["k"],
                config["eps"],
                config["lambda_local"],
                config["n_anchors"],
            )

            # Distillation loss: anchor replay samples to their stored z_old
            distill_loss = torch.tensor(0.0, device=DEVICE)
            if has_replay:
                replay_mask = is_replay_batch.bool()  # keep on CPU for indexing
                if replay_mask.any():
                    z_replay = z[replay_mask.to(DEVICE)]
                    z_old = z_old_batch[replay_mask].to(DEVICE)
                    distill_loss = F.mse_loss(z_replay, z_old)
                    loss = loss + config["lambda_distill"] * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_Rt += R_total.item()
            epoch_Rl += R_local.item()
            epoch_distill += distill_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_Rt = epoch_Rt / n_batches
        avg_Rl = epoch_Rl / n_batches
        avg_distill = epoch_distill / n_batches
        print(
            f"  Stage [{digits_str}] | Epoch {epoch:2d}/{config['epochs_per_stage']} | "
            f"R_total: {avg_Rt:.4f} | R_local: {avg_Rl:.4f} | "
            f"Distill: {avg_distill:.4f} | Loss: {avg_loss:.4f}"
        )

    # Snapshot latents for this stage's digits and update the replay buffer
    mask = torch.zeros(len(all_labels), dtype=torch.bool)
    for d in stage_digits:
        mask |= all_labels == d
    stage_images = all_images[mask]
    stage_labels = all_labels[mask]
    with torch.no_grad():
        encoder.eval()
        z_snap = []
        for i in range(0, len(stage_images), 1024):
            z_snap.append(encoder(stage_images[i : i + 1024].to(DEVICE)).cpu())
        encoder.train()
    stage_latents = torch.cat(z_snap, dim=0)
    buffer.update(stage_images, stage_labels, stage_latents, stage_digits)


def train_all_stages(config):
    """Run the full continual learning pipeline across all stages."""
    all_images, all_labels = get_mnist_data()
    encoder = Encoder(
        input_dim=784, hidden_dim=config["hidden"], latent_dim=config["d"]
    ).to(DEVICE)
    buffer = ReplayBuffer(config["buffer_size"])
    snapshots = []

    for i, stage_digits in enumerate(config["stages"]):
        print(f"\n{'='*60}")
        print(f"Stage {i+1}: Training on digits {stage_digits}")
        print(f"{'='*60}")
        train_stage(encoder, stage_digits, all_images, all_labels, buffer, config)

        # Snapshot latent representations after this stage
        Z, labels = collect_latents(encoder, all_images, all_labels)
        snapshots.append((Z, labels, stage_digits))

    return encoder, snapshots


# %% Collect Latents Utility
@torch.no_grad()
def collect_latents(encoder, images, labels, batch_size=1024):
    """Encode the full dataset and return (Z, labels) as numpy arrays."""
    encoder.eval()
    all_z = []
    for i in range(0, len(images), batch_size):
        x = images[i : i + batch_size].to(DEVICE)
        z = encoder(x)
        all_z.append(z.cpu())
    encoder.train()
    Z = torch.cat(all_z, dim=0).numpy()
    return Z, labels.numpy()


# %% Visualization: Latent Space (UMAP)
def plot_latent_space(snapshots, method="umap"):
    """Plot 2D projections of the latent space after each training stage."""
    n_stages = len(snapshots)
    fig, axes = plt.subplots(1, n_stages, figsize=(7 * n_stages, 6))
    if n_stages == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for ax, (Z, labels, stage_digits) in zip(axes, snapshots):
        if method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42)
            Z_2d = reducer.fit_transform(Z)
        else:
            reducer = PCA(n_components=2)
            Z_2d = reducer.fit_transform(Z)

        for digit in range(10):
            mask = labels == digit
            if mask.sum() == 0:
                continue
            ax.scatter(
                Z_2d[mask, 0],
                Z_2d[mask, 1],
                c=[cmap(digit)],
                label=str(digit),
                s=1,
                alpha=0.5,
            )
        digits_str = ",".join(str(d) for d in stage_digits)
        ax.set_title(f"After training on [{digits_str}]")
        ax.legend(markerscale=5, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Latent Space ({method.upper()})", fontsize=14)
    plt.tight_layout()
    plt.show()


# %% Visualization: Subspace Cosine Similarity Heatmap
def plot_subspace_similarity(Z, labels, rank=10):
    """
    Compute and plot pairwise cosine similarity between the principal
    subspaces of each digit class.
    """
    digits = sorted(set(labels))
    n_digits = len(digits)
    subspaces = {}

    for digit in digits:
        mask = labels == digit
        Z_c = Z[mask]
        if len(Z_c) < rank:
            continue
        # SVD to get principal subspace basis
        U, S, Vt = np.linalg.svd(Z_c, full_matrices=False)
        subspaces[digit] = Vt[:rank].T  # (d, rank) -- right singular vectors

    sim_matrix = np.zeros((n_digits, n_digits))
    digit_list = sorted(subspaces.keys())
    for i, di in enumerate(digit_list):
        for j, dj in enumerate(digit_list):
            # Cosine similarity between subspaces via canonical correlations
            cross = subspaces[di].T @ subspaces[dj]  # (rank, rank)
            sim_matrix[i, j] = np.linalg.norm(cross, "fro") / np.sqrt(rank)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(n_digits))
    ax.set_yticks(range(n_digits))
    ax.set_xticklabels(digit_list)
    ax.set_yticklabels(digit_list)
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Digit Class")
    ax.set_title("Subspace Cosine Similarity (lower off-diagonal = more incoherent)")

    # Annotate cells
    for i in range(n_digits):
        for j in range(n_digits):
            ax.text(
                j,
                i,
                f"{sim_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if sim_matrix[i, j] > 0.6 else "black",
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# %% Visualization: Singular Value Decay
def plot_singular_value_decay(Z, labels):
    """Plot singular value decay for each digit class to check within-class compressibility."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    digits = sorted(set(labels))
    for digit in digits:
        mask = labels == digit
        Z_c = Z[mask]
        sv = np.linalg.svd(Z_c, compute_uv=False)
        sv = sv / sv[0]  # normalize by largest
        ax.plot(
            range(1, len(sv) + 1), sv, color=cmap(digit), label=str(digit), alpha=0.8
        )

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.set_title(
        "Singular Value Decay per Digit Class (sharp elbow = low-rank subspace)"
    )
    ax.legend()
    ax.set_xlim(1, 50)  # zoom into first 50 for clarity
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% Run Training and Visualizations
if __name__ == "__main__" or True:  # always run in notebook mode
    encoder, snapshots = train_all_stages(CONFIG)

    # Final latent representations
    Z_final, labels_final = snapshots[-1][0], snapshots[-1][1]

    # %% Latent Space UMAP
    print("\nGenerating UMAP visualizations...")
    plot_latent_space(snapshots, method="umap")

    # %% Subspace Similarity
    print("\nComputing subspace similarity...")
    plot_subspace_similarity(Z_final, labels_final, rank=CONFIG["subspace_rank"])

    # %% Singular Value Decay
    print("\nPlotting singular value decay...")
    plot_singular_value_decay(Z_final, labels_final)
