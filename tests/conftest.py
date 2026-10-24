import numpy as np
import pytest

from hestfm.data import SampleData


def make_samples(n_samples=4, n=300, d=64, g=20, seed=0, signal=True, nonlinear=False):
    """Synthetic samples: counts ~ Poisson(exp(f(latent))), embeddings = latent @ W + noise."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(8, d))
    B = rng.normal(size=(8, g)) * 0.6
    out = {}
    for i in range(n_samples):
        z = rng.normal(size=(n, 8))
        X = z @ W + 0.5 * rng.normal(size=(n, d))
        eta = (np.tanh(z) if nonlinear else z) @ B
        lam = np.exp(np.clip(eta, -3, 3) + 0.5) if signal else np.full((n, g), 2.0)
        Y = rng.poisson(lam).astype(np.float32)
        coords = np.c_[rng.integers(0, 5000, n), rng.integers(0, 5000, n)]
        sid = f"S{i}"
        out[sid] = SampleData(sid, X.astype(np.float32), Y, np.array([f"bc{j}" for j in range(n)]),
                              coords, np.array([f"G{k}" for k in range(g)]))
    return out


@pytest.fixture
def samples():
    return make_samples()
