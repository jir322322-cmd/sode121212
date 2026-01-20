from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def feather_blend(left: np.ndarray, right: np.ndarray, overlap: int, feather: int) -> Tuple[np.ndarray, np.ndarray]:
    if overlap <= 0:
        return left, right
    h = min(left.shape[0], right.shape[0])
    left_strip = left[:h, -overlap:].astype(np.float32)
    right_strip = right[:h, :overlap].astype(np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    if feather > 0:
        ramp = cv2.GaussianBlur(ramp.reshape(1, -1), (1, feather * 2 + 1), 0).reshape(-1)
    alpha = ramp.reshape(1, -1, 1)
    blended = left_strip * (1 - alpha) + right_strip * alpha
    left[:h, -overlap:] = blended.astype(np.uint8)
    right[:h, :overlap] = blended.astype(np.uint8)
    return left, right


def feather_blend_vertical(top: np.ndarray, bottom: np.ndarray, overlap: int, feather: int) -> Tuple[np.ndarray, np.ndarray]:
    if overlap <= 0:
        return top, bottom
    w = min(top.shape[1], bottom.shape[1])
    top_strip = top[-overlap:, :w].astype(np.float32)
    bottom_strip = bottom[:overlap, :w].astype(np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    if feather > 0:
        ramp = cv2.GaussianBlur(ramp.reshape(-1, 1), (feather * 2 + 1, 1), 0).reshape(-1)
    alpha = ramp.reshape(-1, 1, 1)
    blended = top_strip * (1 - alpha) + bottom_strip * alpha
    top[-overlap:, :w] = blended.astype(np.uint8)
    bottom[:overlap, :w] = blended.astype(np.uint8)
    return top, bottom
