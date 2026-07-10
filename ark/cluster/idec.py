"""Improved Deep Embedded Clustering (IDEC): an autoencoder trained jointly with a clustering loss.

Guo, X., Gao, L., Liu, X., Yin, J. (2017), "Improved Deep Embedded
Clustering with Local Structure Preservation," IJCAI 2017:1753-1759.
Pretrains a plain autoencoder, seeds cluster centers from k-means on the
learned embedding, then jointly optimizes reconstruction and clustering
losses, the reconstruction term is what IDEC adds over the original DEC
method, keeping the embedding from collapsing onto the cluster centers.

Adapted from `resources/profiling 3/src/net/idec.py`
(`github.com/dawnranger/IDEC-pytorch`, MIT-licensed), rewritten from its
original script/argparse form into a reusable module: the source version
referenced globals (`args`, `dataset`, `train_loader`) undefined in the
scope they were used, real bugs, not preserved here.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


class _Autoencoder(nn.Module):
    """A small fully-connected autoencoder: `n_input -> ... -> n_z -> ... -> n_input`."""

    def __init__(self, n_input: int, n_z: int, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(n_input, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_z),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_z, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, n_input),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_bar = self.decoder(z)
        return x_bar, z


class IDEC(nn.Module):
    """The IDEC model: an autoencoder plus a learned soft-cluster assignment layer."""

    def __init__(self, n_input: int, n_z: int, n_clusters: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.autoencoder = _Autoencoder(n_input, n_z)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_bar, z = self.autoencoder(x)
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return x_bar, q


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Sharpen soft assignments `q` into a higher-confidence target `p`."""
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def _pretrain_autoencoder(autoencoder: _Autoencoder, data: torch.Tensor, epochs: int, lr: float) -> None:
    optimizer = Adam(autoencoder.parameters(), lr=lr)
    for _epoch in range(epochs):
        optimizer.zero_grad()
        x_bar, _ = autoencoder(data)
        loss = F.mse_loss(x_bar, data)
        loss.backward()
        optimizer.step()


def fit_idec(
    X: np.ndarray,
    n_clusters: int,
    *,
    n_z: int = 8,
    pretrain_epochs: int = 200,
    train_epochs: int = 100,
    lr: float = 1e-3,
    gamma: float = 0.1,
    tol: float = 1e-3,
    random_state: int = 0,
) -> tuple[IDEC, np.ndarray]:
    """Fit IDEC end to end: pretrain, seed clusters from k-means, jointly optimize.

    Args:
        X: `(n_samples, n_features)` clustering input.
        n_clusters: Number of clusters to find.
        n_z: Latent embedding dimension.
        pretrain_epochs: Epochs for the autoencoder-only pretraining phase.
        train_epochs: Max epochs for the joint reconstruction+clustering phase.
        lr: Adam learning rate, used in both phases.
        gamma: Weight on the clustering loss relative to reconstruction loss.
        tol: Stop early once fewer than this fraction of labels change
            between consecutive checks.
        random_state: Seeds both `torch` and the k-means seeding step.

    Returns:
        The fitted model and the final hard cluster assignment per sample.
    """
    torch.manual_seed(random_state)
    data = torch.from_numpy(X.astype(np.float32))
    model = IDEC(n_input=X.shape[1], n_z=n_z, n_clusters=n_clusters)
    _pretrain_autoencoder(model.autoencoder, data, epochs=pretrain_epochs, lr=lr)

    with torch.no_grad():
        _, z = model.autoencoder(data)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(z.numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(train_epochs):
        with torch.no_grad():
            _, q = model(data)
            p = _target_distribution(q)
            new_labels = q.numpy().argmax(1)
            delta = np.mean(new_labels != labels)
            labels = new_labels
            if epoch > 0 and delta < tol:
                break

        x_bar, q = model(data)
        reconstruction_loss = F.mse_loss(x_bar, data)
        clustering_loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss = gamma * clustering_loss + reconstruction_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, labels
