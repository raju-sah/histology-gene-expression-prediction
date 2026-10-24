"""Regression heads on frozen embeddings.

- HESTRidge:   exact HEST-Benchmark probe (StandardScaler -> PCA(256) -> Ridge(lsqr, alpha=100/(d*g), no intercept)).
- RidgeCVHead: StandardScaler -> optional PCA -> ridge with intercept; alpha chosen on a validation mask drawn from
               the training rows only (globally or per gene), then refit on all training rows.
- MLPHead:     one-hidden-layer MLP; epochs chosen by early stopping on the validation mask, then refit on all
               training rows for that many epochs.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .metrics import pearson_per_gene

DEFAULT_ALPHAS = np.logspace(-2, 6, 17)


class HESTRidge:
    def __init__(self, latent_dim: int = 256, seed: int = 1):
        self.latent_dim, self.seed = latent_dim, seed

    def fit(self, X, y, **_):
        k = min(self.latent_dim, X.shape[0], X.shape[1])
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=k, random_state=self.seed).fit(self.scaler.transform(X))
        Z = self.pca.transform(self.scaler.transform(X))
        alpha = 100 / (Z.shape[1] * y.shape[1])
        self.reg = Ridge(solver="lsqr", alpha=alpha, random_state=self.seed, fit_intercept=False, max_iter=1000)
        self.reg.fit(Z, y)
        return self

    def predict(self, X):
        return self.reg.predict(self.pca.transform(self.scaler.transform(X)))


class _Preproc:
    def __init__(self, pca_dim: int | None, seed: int):
        self.pca_dim, self.seed = pca_dim, seed

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = None
        if self.pca_dim is not None and self.pca_dim < min(Z.shape):
            self.pca = PCA(n_components=self.pca_dim, random_state=self.seed).fit(Z)
        return self

    def transform(self, X):
        Z = self.scaler.transform(X)
        return self.pca.transform(Z) if self.pca is not None else Z


def _ridge_svd(Z, y):
    zm, ym = Z.mean(0), y.mean(0)
    U, s, Vt = np.linalg.svd(Z - zm, full_matrices=False)
    return zm, ym, s, Vt, U.T @ (y - ym)


class RidgeCVHead:
    def __init__(self, alphas=DEFAULT_ALPHAS, pca_dim: int | None = 256, per_gene_alpha: bool = False, seed: int = 0):
        self.alphas = np.asarray(alphas, dtype=float)
        self.pca_dim, self.per_gene_alpha, self.seed = pca_dim, per_gene_alpha, seed

    def _select(self, X, y, val_mask):
        tr, va = ~val_mask, val_mask
        pre = _Preproc(self.pca_dim, self.seed).fit(X[tr])
        Ztr, Zva = pre.transform(X[tr]), pre.transform(X[va])
        zm, ym, s, Vt, UTy = _ridge_svd(Ztr, y[tr])
        B = (Zva - zm) @ Vt.T
        scores = np.empty((len(self.alphas), y.shape[1]))
        for i, a in enumerate(self.alphas):
            pred = (B * (s / (s ** 2 + a))) @ UTy + ym
            scores[i] = pearson_per_gene(y[va], pred)
        self.val_scores_ = scores
        filled = np.nan_to_num(scores, nan=-np.inf)
        if self.per_gene_alpha:
            return self.alphas[np.argmax(filled, axis=0)]
        return np.full(y.shape[1], self.alphas[np.argmax(np.nanmean(scores, axis=1))])

    def fit(self, X, y, val_mask=None, **_):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if val_mask is not None and val_mask.any() and (~val_mask).any():
            self.alpha_ = self._select(X, y, val_mask)
        else:
            self.alpha_ = np.full(y.shape[1], np.median(self.alphas))
        self.pre = _Preproc(self.pca_dim, self.seed).fit(X)
        Z = self.pre.transform(X)
        zm, ym, s, Vt, UTy = _ridge_svd(Z, y)
        # coef[:, g] = V diag(s/(s^2+alpha_g)) U^T y_g
        f = s[:, None] / (s[:, None] ** 2 + self.alpha_[None, :])
        self.coef_ = Vt.T @ (f * UTy)
        self.zm_, self.intercept_ = zm, ym - zm @ self.coef_
        return self

    def predict(self, X):
        return self.pre.transform(np.asarray(X, dtype=np.float32)) @ self.coef_ + self.intercept_


class MLPHead:
    def __init__(self, hidden: int = 512, dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, max_epochs: int = 100, patience: int = 10, pca_dim: int | None = None,
                 seed: int = 0, device: str = "cpu"):
        self.__dict__.update(dict(hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay,
                                  batch_size=batch_size, max_epochs=max_epochs, patience=patience,
                                  pca_dim=pca_dim, seed=seed, device=device))

    def _net(self, d_in, d_out):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(d_in, self.hidden), nn.GELU(), nn.Dropout(self.dropout),
                             nn.Linear(self.hidden, d_out))

    def _train(self, Z, y, epochs, Zva=None, yva=None):
        import torch
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        net = self._net(Z.shape[1], y.shape[1]).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Zt, yt = torch.from_numpy(Z), torch.from_numpy(y)
        best, best_ep, bad, history = -np.inf, 0, 0, []
        for ep in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(Zt), generator=g)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(Zt[b].to(self.device)), yt[b].to(self.device))
                loss.backward()
                opt.step()
            if Zva is not None:
                score = float(np.nanmean(pearson_per_gene(yva, self._predict_net(net, Zva))))
                history.append(score)
                if score > best + 1e-4:
                    best, best_ep, bad = score, ep, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        return net, best_ep, history

    def _predict_net(self, net, Z):
        import torch
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 4096):
                out.append(net(torch.from_numpy(Z[i:i + 4096]).to(self.device)).cpu().numpy())
        return np.concatenate(out)

    def fit(self, X, y, val_mask=None, **_):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = y.mean(0), y.std(0) + 1e-6
        ys = (y - self.y_mean_) / self.y_std_
        epochs = self.max_epochs
        if val_mask is not None and val_mask.any() and (~val_mask).any():
            pre = _Preproc(self.pca_dim, self.seed).fit(X[~val_mask])
            _, best_ep, self.val_history_ = self._train(
                pre.transform(X[~val_mask]).astype(np.float32), ys[~val_mask], self.max_epochs,
                pre.transform(X[val_mask]).astype(np.float32), ys[val_mask])
            epochs = max(best_ep, 1)
        self.epochs_ = epochs
        self.pre = _Preproc(self.pca_dim, self.seed).fit(X)
        self.net, _, _ = self._train(self.pre.transform(X).astype(np.float32), ys, epochs)
        return self

    def predict(self, X):
        Z = self.pre.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        return self._predict_net(self.net, Z) * self.y_std_ + self.y_mean_


def make_head(name: str, **kw):
    if name == "hest_ridge":
        return HESTRidge(**kw)
    if name == "ridge_cv":
        return RidgeCVHead(**kw)
    if name == "ridge_cv_gene":
        return RidgeCVHead(per_gene_alpha=True, **kw)
    if name == "mlp":
        return MLPHead(**kw)
    raise ValueError(name)
