import numpy as np
import torch

from hestfm.encoders import IMAGENET, Encoder, embed, preprocess


def _enc(size=224, micro=3):
    m = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    return Encoder("dummy", m, m, size, *IMAGENET, micro_batch=micro, dim=3)


def test_preprocess_normalises_and_resizes():
    x = np.full((2, 224, 224, 3), 255, dtype=np.uint8)
    out = preprocess(x, _enc(448), device="cpu")
    assert out.shape == (2, 3, 448, 448)
    assert np.allclose(out[0, :, 0, 0].numpy(), (1 - np.array(IMAGENET[0])) / np.array(IMAGENET[1]), atol=1e-5)


def test_embed_microbatches_preserve_order():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, size=(7, 224, 224, 3), dtype=np.uint8)
    e = embed(x, _enc(micro=3), device="cpu")
    ref = embed(x, _enc(micro=100), device="cpu")
    assert e.dtype == np.float16 and e.shape == (7, 3)
    assert np.allclose(e, ref)


def test_center_crop_uses_central_pixels():
    x = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    x[:, 56:168, 56:168] = 255
    e = _enc(224)
    e.center_crop_px = 112
    out = preprocess(x, e, device="cpu")
    assert out.shape == (1, 3, 224, 224)
    assert np.allclose(out.numpy(), ((1 - np.array(IMAGENET[0])) / np.array(IMAGENET[1]))[None, :, None, None], atol=1e-4)
