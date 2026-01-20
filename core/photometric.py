from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from skimage.exposure import match_histograms

from core.types import ColorMatchMode


def lab_stats(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32)
    content = l[mask == 0]
    if content.size == 0:
        content = l.reshape(-1)
    return float(content.mean()), float(content.std() + 1e-6)


def match_luminance(image: np.ndarray, ref_image: np.ndarray, mask: np.ndarray, strength: float = 0.5) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    ref_l = ref_lab[..., 0]
    ref_mean = float(ref_l[mask == 0].mean()) if (mask == 0).any() else float(ref_l.mean())
    ref_std = float(ref_l[mask == 0].std() + 1e-6) if (mask == 0).any() else float(ref_l.std() + 1e-6)
    cur_mean = float(l[mask == 0].mean()) if (mask == 0).any() else float(l.mean())
    cur_std = float(l[mask == 0].std() + 1e-6) if (mask == 0).any() else float(l.std() + 1e-6)
    target = (l - cur_mean) * (ref_std / cur_std) + ref_mean
    l = (1 - strength) * l + strength * target
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(np.clip(merged, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def apply_color_match(image: np.ndarray, ref_image: np.ndarray, mask: np.ndarray, mode: ColorMatchMode) -> np.ndarray:
    if mode == ColorMatchMode.OFF:
        return image
    if mode == ColorMatchMode.MILD:
        return match_luminance(image, ref_image, mask, strength=0.5)
    if mode == ColorMatchMode.STRONG:
        matched = match_histograms(image, ref_image, channel_axis=-1)
        return np.clip(matched, 0, 255).astype(np.uint8)
    return image
