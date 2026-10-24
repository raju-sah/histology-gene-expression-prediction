"""Frozen patch encoders for Colab extraction. Inlined into notebooks/colab_*.ipynb by scripts/build_notebooks.py.

Preprocessing runs on the GPU from uint8 HWC patches (224 px at 0.5 um/px from HEST patch files):
resize (only if the encoder's input size differs) -> /255 -> normalise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HOPTIMUS = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))


@dataclass
class Encoder:
    name: str
    model: torch.nn.Module
    forward: Callable[[torch.Tensor], torch.Tensor]
    input_size: int
    mean: tuple
    std: tuple
    interpolation: str = "bilinear"
    amp: bool = True
    micro_batch: int = 256
    dim: int | None = None
    device: str = "cuda"


def preprocess(batch_uint8: np.ndarray | torch.Tensor, enc: Encoder, device="cuda") -> torch.Tensor:
    x = torch.as_tensor(batch_uint8).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != enc.input_size or x.shape[-2] != enc.input_size:
        x = F.interpolate(x, size=(enc.input_size, enc.input_size), mode=enc.interpolation,
                          align_corners=False, antialias=enc.input_size < x.shape[-1])
    mean = torch.tensor(enc.mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(enc.std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def embed(batch_uint8, enc: Encoder, device=None, amp_dtype=torch.bfloat16) -> np.ndarray:
    device = device or enc.device
    outs = []
    for i in range(0, len(batch_uint8), enc.micro_batch):
        x = preprocess(batch_uint8[i:i + enc.micro_batch], enc, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enc.amp and device == "cuda"):
            z = enc.forward(x)
        outs.append(z.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float16)


def build_encoder(name: str, device="cuda") -> Encoder:
    import timm

    if name == "resnet50":
        # HEST-Benchmark "resnet50": ImageNet ResNet50 truncated after layer3, global average pool -> 1024-d.
        m = timm.create_model("resnet50.tv_in1k", pretrained=True, features_only=True, out_indices=[3])
        fwd = lambda x: m(x)[0].mean(dim=(2, 3))
        enc = Encoder(name, m, fwd, 224, *IMAGENET, amp=False, micro_batch=256, dim=1024)
    elif name == "uni2_h":
        kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                  mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                  act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
        m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
        enc = Encoder(name, m, m, 224, *IMAGENET, micro_batch=256, dim=1536)
    elif name == "virchow2":
        m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)

        def fwd(x):
            t = m(x)  # (B, 1 + 4 reg + 256, 1280)
            return torch.cat([t[:, 0], t[:, 5:].mean(1)], dim=-1)
        enc = Encoder(name, m, fwd, 224, *IMAGENET, interpolation="bicubic", micro_batch=256, dim=2560)
    elif name == "conch_v15":
        from transformers import AutoModel
        titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        m, _ = titan.return_conch()   # eval transform: bilinear resize 448, ImageNet norm
        del titan
        enc = Encoder(name, m, m, 448, *IMAGENET, interpolation="bilinear", micro_batch=128, dim=768)
    elif name == "hoptimus1":
        m = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5,
                              dynamic_img_size=False)
        enc = Encoder(name, m, m, 224, *HOPTIMUS, micro_batch=256, dim=1536)
    else:
        raise ValueError(name)
    enc.device = device
    enc.model.eval().to(device)
    for p in enc.model.parameters():
        p.requires_grad_(False)
    return enc
