from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from core.types import TileData, TileTransform


@dataclass
class MatchResult:
    dx: float
    dy: float
    error: float
    confidence: float
    scale_x: float = 1.0
    scale_y: float = 1.0


def _sobel_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag


def _phase_corr(a: np.ndarray, b: np.ndarray) -> Tuple[Tuple[float, float], float]:
    shift, response = cv2.phaseCorrelate(a, b)
    return shift, response


def match_seam(left: TileData, right: TileData, overlap: int, band: int) -> MatchResult:
    h = min(left.image.shape[0], right.image.shape[0])
    left_strip = left.image[:h, -band:]
    right_strip = right.image[:h, :band]
    a = _sobel_gray(left_strip).astype(np.float32)
    b = _sobel_gray(right_strip).astype(np.float32)
    shift, response = _phase_corr(a, b)
    dx, dy = shift
    dx = max(-overlap, min(overlap, dx))
    dy = max(-overlap, min(overlap, dy))
    error = float(1.0 - response)
    return MatchResult(dx=dx, dy=dy, error=error, confidence=response)


def match_seam_vertical(top: TileData, bottom: TileData, overlap: int, band: int) -> MatchResult:
    w = min(top.image.shape[1], bottom.image.shape[1])
    top_strip = top.image[-band:, :w]
    bottom_strip = bottom.image[:band, :w]
    a = _sobel_gray(top_strip).astype(np.float32)
    b = _sobel_gray(bottom_strip).astype(np.float32)
    shift, response = _phase_corr(a, b)
    dx, dy = shift
    dx = max(-overlap, min(overlap, dx))
    dy = max(-overlap, min(overlap, dy))
    error = float(1.0 - response)
    return MatchResult(dx=dx, dy=dy, error=error, confidence=response)


def clamp_scale(scale: float, max_percent: float) -> float:
    delta = max_percent / 100.0
    return max(1.0 - delta, min(1.0 + delta, scale))


def scale_transform(transform: TileTransform, scale_x: float, scale_y: float, max_percent: float) -> TileTransform:
    scale_x = clamp_scale(scale_x, max_percent)
    scale_y = clamp_scale(scale_y, max_percent)
    return TileTransform(
        dx=transform.dx,
        dy=transform.dy,
        scale_x=scale_x,
        scale_y=scale_y,
        angle_deg=transform.angle_deg,
        confidence=transform.confidence,
    )
