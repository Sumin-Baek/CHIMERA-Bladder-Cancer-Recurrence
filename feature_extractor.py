"""
UNI patch encoder and patch sampling for whole-slide images.

NOTE (as submitted): at inference, patches are sampled on a coarse regular
grid at level 0 (stride = 8 x 224 px) and kept when the mean intensity is
below 230 (i.e. not background).  The tissue mask supplied by the challenge is
*not* used for patch selection.  Training features, in contrast, were the
pre-extracted UNI features distributed with the challenge data.
"""

from __future__ import annotations

import gc

import numpy as np
import openslide
import timm
import torch
from torchvision import transforms


def get_feature_extractor(model_path):
    """Load the UNI ViT-L/16 encoder (Chen et al., Nat. Med. 2024) from a local checkpoint."""
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, transform, device


def extract_features_with_openslide(wsi_path, mask_path, model, transform, device,
                                    max_patches=300, batch_size=32, patch_size=224):
    """Return a (N, 1024) tensor of UNI features for up to ``max_patches`` patches.

    ``mask_path`` is accepted for interface compatibility but is not used
    (see module docstring).
    """
    print("Extracting patch features with OpenSlide ...")
    slide = openslide.OpenSlide(wsi_path)
    level = 0
    w, h = slide.level_dimensions[level]
    print(f"WSI opened. Dimensions: {w}x{h}")

    patches = []
    stride = patch_size * 8
    for y in range(0, h, stride):
        if len(patches) >= max_patches:
            break
        for x in range(0, w, stride):
            if len(patches) >= max_patches:
                break
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            if np.mean(np.array(patch)) < 230:
                patches.append(patch)
    slide.close()

    if not patches:
        print("WARNING: no tissue patches found; returning a zero vector.")
        return torch.zeros((1, 1024))

    print(f"Found {len(patches)} patches. Encoding ...")
    all_features = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            tensors = torch.stack([transform(p) for p in batch]).to(device)
            all_features.append(model(tensors).cpu())
            del tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0)
