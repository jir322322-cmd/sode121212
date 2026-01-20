from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.types import InpaintMethod


@dataclass
class SeamFillResult:
    filled: np.ndarray
    filled_px: int


def edge_extend(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, expanded, 1, cv2.INPAINT_TELEA)


def seam_fill(image: np.ndarray, mask: np.ndarray, method: InpaintMethod, radius: int) -> SeamFillResult:
    fill_mask = (mask > 0).astype(np.uint8) * 255
    if method == InpaintMethod.EDGE_EXTEND:
        filled = edge_extend(image, fill_mask)
    elif method == InpaintMethod.NS:
        filled = cv2.inpaint(image, fill_mask, radius, cv2.INPAINT_NS)
    else:
        filled = cv2.inpaint(image, fill_mask, radius, cv2.INPAINT_TELEA)
    return SeamFillResult(filled=filled, filled_px=int(np.count_nonzero(fill_mask)))
