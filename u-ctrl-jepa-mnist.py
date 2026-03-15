import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.lines import Line2D

# [1] Configuration
CONFIG = {
    "d": 128,  # Latent dimensionality
    "eps": 0.5,  # Coding rate precision
    "lambda_straight": 5.0,  # JEPA straightening strength
    "lambda_ctrl": 1.0,  # u-CTRL room separation strength
    "batch_size": 128,
    "lr": 1e-3,
    "steps": 1201,  # Extra step for final visual
    "device": torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    ),
}


# [2] Synthetic Biological Data Generator (MNIST Morph)
def generate_vcc_mnist_data(digits, steps=25):
    """
    Simulates 'Stem Cell' (1) differentiating into 'Fate A' (4) or 'Fate B' (7).
    Returns triplets of (t-1, t, t+1) to train temporal straightening.
    """
    all_triplets = []
    all_labels = []

    # Use first 50 instances of each digit for diversity
    ones = digits[1][:50]
    fours = digits[4][:50]
    sevens = digits[7][:50]

    for i in range(len(ones)):
        for target, label in [(fours[i], 0), (sevens[i], 1)]:
            path = []
            for alpha in np.linspace(0, 1, steps):
                # Pixel-space morphing (the raw biological observation)
                frame = (1 - alpha) * ones[i] + alpha * target
                path.append(frame.flatten())

            # Create Triplets for JEPA
            for t in range(1, len(path) - 1):
                all_triplets.append(torch.stack([path[t - 1], path[t], path[t + 1]]))
                all_labels.append(label)

    return torch.stack(all_triplets).to(CONFIG["device"]), torch.tensor(all_labels).to(
        CONFIG["device"]
    )


# [3] JEPA + u-CTRL Engine
class VirtualCellEngine(nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        # Predictor: Models the 'physics' of the transition in Z space
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

    def get_z(self, x):
        return F.normalize(self.encoder(x), dim=1)

    def predict_next(self, z):
        return F.normalize(self.predictor(z), dim=1)


# [4] Loss Functions
def compute_straightening_loss(z_prev, z_curr, z_next):
    v1 = z_curr - z_prev
    v2 = z_next - z_curr
    cos_sim = F.cosine_similarity(v1, v2, dim=1)
    return (1 - cos_sim).mean()


def compute_u_ctrl_loss(z, labels, eps=0.5):
    def coding_rate(Z):
        n, d = Z.shape
        alpha = d / (n * eps**2)
        I = torch.eye(d, device=Z.device)
        # LogDet ensures the 'room' volume is maximized
        return 0.5 * torch.logdet(I + alpha * Z.t() @ Z)

    R_total = coding_rate(z)
    R_path = 0
    unique_labels = torch.unique(labels)
    for i in unique_labels:
        z_i = z[labels == i]
        if z_i.shape[0] > 1:
            R_path += coding_rate(z_i)

    return -R_total + (R_path / len(unique_labels))


# [5] Visualization Suite
def emit_vcc_snapshot(step, model, triplets, labels):
    model.eval()
    with torch.no_grad():
        z_curr = model.get_z(triplets[:, 1])
        z_prev = model.get_z(triplets[:, 0])
        z_next = model.get_z(triplets[:, 2])

        v1 = z_curr - z_prev
        v2 = z_next - z_curr
        curvature = (1 - F.cosine_similarity(v1, v2)).cpu().numpy()

        z_np = z_curr.cpu().numpy()
        y_np = labels.cpu().numpy()

    # UMAP Projection
    reducer = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(z_np)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Plot 1: Branching Manifold
    scatter = axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=y_np, cmap="coolwarm", s=20, alpha=0.5
    )
    axes[0].set_title(f"Step {step}: Trajectory Branching")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Path 1->4",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Path 1->7",
            markerfacecolor="red",
            markersize=10,
        ),
    ]
    axes[0].legend(handles=legend_elements)

    # Plot 2: Orthogonality (Room Independence)
    z_4, z_7 = z_curr[labels == 0], z_curr[labels == 1]
    cross_sim = (z_4 @ z_7.t()).mean().item()
    within_4, within_7 = (z_4 @ z_4.t()).mean().item(), (z_7 @ z_7.t()).mean().item()
    ortho_matrix = np.array([[within_4, cross_sim], [cross_sim, within_7]])
    sns.heatmap(
        ortho_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=axes[1],
        xticklabels=["Fate 4", "Fate 7"],
        yticklabels=["Fate 4", "Fate 7"],
    )
    axes[1].set_title("Subspace Incoherence")

    # Plot 3: Straightness (The JEPA Metric)
    axes[2].hist(curvature, bins=40, color="salmon", edgecolor="black", alpha=0.7)
    axes[2].set_title("Trajectory Curvature (Surprise)")
    axes[2].axvline(
        curvature.mean(),
        color="blue",
        linestyle="dashed",
        label=f"Avg: {curvature.mean():.4f}",
    )
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    model.train()


# [6] Main Execution
if __name__ == "__main__":
    print(f"Initializing Virtual Cell Experiment on {CONFIG['device']}...")

    # Prepare Data
    mnist = datasets.MNIST(root="data", train=True, download=True)
    digits = {i: mnist.data[mnist.targets == i].float() / 255.0 for i in range(10)}
    triplets, labels = generate_vcc_mnist_data(digits)

    # Initialize Engine
    model = VirtualCellEngine().to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for step in range(CONFIG["steps"]):
        idx = torch.randperm(len(triplets))[: CONFIG["batch_size"]]
        batch = triplets[idx]
        batch_labels = labels[idx]

        z_prev = model.get_z(batch[:, 0])
        z_curr = model.get_z(batch[:, 1])
        z_next = model.get_z(batch[:, 2])

        # Losses
        l_straight = compute_straightening_loss(z_prev, z_curr, z_next)
        z_pred = model.predict_next(z_curr)
        l_pred = F.mse_loss(z_pred, z_next.detach())
        l_ctrl = compute_u_ctrl_loss(z_curr, batch_labels)

        loss = (
            l_pred
            + CONFIG["lambda_straight"] * l_straight
            + CONFIG["lambda_ctrl"] * l_ctrl
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 400 == 0:
            print(f"Step {step} | Total Loss: {loss.item():.4f}")
            emit_vcc_snapshot(step, model, triplets, labels)

    print("Experiment Complete. Trajectories Straightened.")
