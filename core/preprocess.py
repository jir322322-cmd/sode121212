from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import cv2
import numpy as np

from core.types import StitchSettings, TileData, TileInfo


def _lab_paper_mask(image: np.ndarray, settings: StitchSettings) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    chroma = cv2.magnitude(a.astype(np.float32) - 128.0, b.astype(np.float32) - 128.0)
    mask = (l >= settings.bg_threshold) & (chroma < 12.0)
    return mask.astype(np.uint8) * 255


def _rgb_threshold_mask(image: np.ndarray, settings: StitchSettings) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray >= settings.bg_threshold
    return mask.astype(np.uint8) * 255


def compute_mask(image: np.ndarray, settings: StitchSettings) -> np.ndarray:
    if settings.bg_mode == "lab_paper":
        mask = _lab_paper_mask(image, settings)
        if mask.mean() < 0.1:
            mask = _rgb_threshold_mask(image, settings)
    else:
        mask = _rgb_threshold_mask(image, settings)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask == 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = image.shape[:2]
        return image, mask, (0, 0, w, h)
    x0 = max(int(xs.min()) - padding, 0)
    y0 = max(int(ys.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding, image.shape[1] - 1)
    y1 = min(int(ys.max()) + padding, image.shape[0] - 1)
    cropped = image[y0 : y1 + 1, x0 : x1 + 1]
    cropped_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    return cropped, cropped_mask, (x0, y0, x1, y1)


def estimate_angle(mask: np.ndarray, max_angle: float) -> float:
    contours, _ = cv2.findContours(255 - mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    angle = float(angle)
    return max(-max_angle, min(max_angle, angle))


def rotate_image(image: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated_image, rotated_mask


def preprocess_tile(info: TileInfo, image: np.ndarray, settings: StitchSettings) -> TileData:
    mask = compute_mask(image, settings)
    image, mask, bbox = crop_to_mask(image, mask, settings.crop_padding_px)
    angle = estimate_angle(mask, settings.max_angle)
    if abs(angle) > 0.01:
        image, mask = rotate_image(image, mask, angle)
        image, mask, bbox = crop_to_mask(image, mask, settings.crop_padding_px)
    return TileData(info=info, image=image, mask=mask, angle_deg=angle, bbox=bbox)
