"""Phase 3: spatial context from neighbouring patches' frozen embeddings.

Neighbours are patches of the same sample whose grid position is within k rings (Chebyshev distance on the patch
grid, k * spacing). Context never crosses samples, so the sample-level split guarantees stay intact.
- ring_mean_features: [x_center, mean(x over rings 1..k)]; falls back to x_center when a patch has no neighbours.
- AttentionContextHead: centre-query attention over {centre} + neighbours (PCA-reduced embeddings), then a linear
  read-out of [centre, pooled]; trained with MSE on standardised targets, epochs by early stopping on the inner
  validation rows, refit on all training rows.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .atlas import grid_spacing
from .heads import _Preproc
from .metrics import pearson_per_gene


def neighbor_lists(coords: np.ndarray, k: int, spacing: float | None = None) -> list[np.ndarray]:
    from scipy.spatial import cKDTree
    spacing = spacing or grid_spacing(coords)
    tree = cKDTree(coords)
    lists = tree.query_ball_point(coords, r=k * spacing * 1.01, p=np.inf)
    return [np.array([j for j in l if j != i], dtype=np.int64) for i, l in enumerate(lists)]


def ring_mean_features(X: np.ndarray, coords: np.ndarray, k: int = 1) -> np.ndarray:
    nb = neighbor_lists(coords, k)
    rows = np.repeat(np.arange(len(X)), [len(l) for l in nb])
    cols = np.concatenate(nb) if rows.size else np.zeros(0, dtype=np.int64)
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(X), len(X)))
    deg = np.asarray(A.sum(1)).ravel()
    M = (A @ X) / np.maximum(deg, 1)[:, None]
    M[deg == 0] = X[deg == 0]
    return np.concatenate([X, M.astype(X.dtype)], axis=1)


def padded_neighbors(coords: np.ndarray, k: int, max_n: int | None = None) -> np.ndarray:
    """(N, 1 + max_n) indices: self first, then neighbours, padded with -1."""
    nb = neighbor_lists(coords, k)
    max_n = max_n or (2 * k + 1) ** 2 - 1
    out = -np.ones((len(coords), 1 + max_n), dtype=np.int64)
    out[:, 0] = np.arange(len(coords))
    for i, l in enumerate(nb):
        l = l[:max_n]
        out[i, 1:1 + len(l)] = l
    return out


class AttentionContextHead:
    def __init__(self, k: int = 1, pca_dim: int = 256, d_attn: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 512, max_epochs: int = 60, patience: int = 8, seed: int = 0, device: str = "cpu"):
        self.__dict__.update({k_: v for k_, v in locals().items() if k_ != "self"})

    def _net(self, d, g):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.q, s.kk = nn.Linear(d, self.d_attn), nn.Linear(d, self.d_attn)
                s.out = nn.Linear(2 * d, g)
                s.scale = self.d_attn ** -0.5

            def forward(s, Z, mask):           # Z: (B, 1+n, d) with Z[:, 0] = centre; mask: True = padding
                q = s.q(Z[:, :1])
                att = (q @ s.kk(Z).transpose(1, 2)).squeeze(1) * s.scale
                att = att.masked_fill(mask, float("-inf")).softmax(-1)
                pooled = (att.unsqueeze(-1) * Z).sum(1)
                s.last_att = att
                return s.out(torch.cat([Z[:, 0], pooled], -1))
        return Net()

    @staticmethod
    def _gather(Z, idx):
        safe = np.where(idx < 0, 0, idx)
        return Z[safe], idx < 0

    def _batches(self, Z, nbr, Y, rows, shuffle, rng):
        order = rng.permutation(rows) if shuffle else rows
        for i in range(0, len(order), self.batch_size):
            b = order[i:i + self.batch_size]
            Zb, mb = self._gather(Z, nbr[b])
            yb = None if Y is None else (Y[b].toarray() if sp.issparse(Y) else Y[b])
            yield b, Zb, mb, yb

    def _fit_rows(self, Z, nbr, Y, rows, epochs, val_rows=None):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = self._net(Z.shape[1], Y.shape[1]).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_ep, bad = -np.inf, 0, 0
        for ep in range(1, epochs + 1):
            net.train()
            for _, Zb, mb, yb in self._batches(Z, nbr, Y, rows, True, rng):
                opt.zero_grad()
                pred = net(torch.from_numpy(Zb).to(self.device), torch.from_numpy(mb).to(self.device))
                loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(((yb - self.ym) / self.ys).astype(np.float32)).to(self.device))
                loss.backward()
                opt.step()
            if val_rows is not None:
                P = self._predict(net, Z, nbr, val_rows)
                yv = Y[val_rows].toarray() if sp.issparse(Y) else Y[val_rows]
                score = float(np.nanmean(pearson_per_gene(yv, P)))
                if score > best + 1e-4:
                    best, best_ep, bad = score, ep, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        return net, best_ep

    def _predict(self, net, Z, nbr, rows):
        import torch
        net.eval()
        out = []
        with torch.no_grad():
            for _, Zb, mb, _ in self._batches(Z, nbr, None, rows, False, None):
                out.append(net(torch.from_numpy(Zb).to(self.device), torch.from_numpy(mb).to(self.device)).cpu().numpy())
        return np.concatenate(out) * self.ys + self.ym

    def fit(self, X, coords_by_sample, sample_ids, Y, val_mask=None):
        """X (N, D) rows of all training samples; neighbours resolved within each sample."""
        self.pre = _Preproc(self.pca_dim, self.seed).fit(X)
        Z = self.pre.transform(X).astype(np.float32)
        nbr = self._global_neighbors(coords_by_sample, sample_ids)
        Yd = Y
        mean = np.asarray(Yd.mean(0)).ravel()
        sq = np.asarray((Yd.multiply(Yd) if sp.issparse(Yd) else Yd ** 2).mean(0)).ravel()
        self.ym, self.ys = mean.astype(np.float32), np.sqrt(np.maximum(sq - mean ** 2, 1e-8)).astype(np.float32)
        rows = np.arange(len(X))
        epochs = self.max_epochs
        if val_mask is not None and val_mask.any():
            # selection: neighbours of training rows may be validation rows of the same sample only if same sample;
            # validation rows form whole samples or a spatial block, so we drop cross-set links for selection
            nbr_sel = nbr.copy()
            nbr_sel[(nbr_sel >= 0) & (val_mask[np.maximum(nbr_sel, 0)] != val_mask[:, None])] = -1
            _, epochs = self._fit_rows(Z, nbr_sel, Yd, rows[~val_mask], self.max_epochs, rows[val_mask])
            epochs = max(epochs, 1)
        self.epochs_ = epochs
        self.net, _ = self._fit_rows(Z, nbr, Yd, rows, epochs)
        return self

    def predict(self, X, coords_by_sample, sample_ids):
        Z = self.pre.transform(X).astype(np.float32)
        nbr = self._global_neighbors(coords_by_sample, sample_ids)
        return self._predict(self.net, Z, nbr, np.arange(len(X)))

    def _global_neighbors(self, coords_by_sample, sample_ids):
        """Rows are ordered sample by sample as in `sample_ids` (unique, in order)."""
        parts, off = [], 0
        for s in dict.fromkeys(sample_ids):
            p = padded_neighbors(coords_by_sample[s], self.k)
            parts.append(np.where(p >= 0, p + off, -1))
            off += len(p)
        return np.concatenate(parts)
