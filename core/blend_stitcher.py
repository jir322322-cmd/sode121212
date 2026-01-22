from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StitchInputs:
    images: List[np.ndarray]
    corners: List[Tuple[int, int]]


def stitch_tiles(
    imgs: list[np.ndarray],
    corners: list[tuple[int, int]],
    border_crop_px: int = 10,
    feather_px: int = 60,
    num_bands: int = 5,
) -> np.ndarray:
    """Stitch pre-aligned tiles into a seamless panorama.

    Notes on tuning:
      - Increase border_crop_px to remove more scan borders and frames.
      - Increase feather_px to soften seams if they are still visible.
      - Increase num_bands for smoother multi-band blending at the cost of RAM.
    """
    inputs = _validate_inputs(imgs, corners)
    cropped = _apply_border_crop(inputs.images, inputs.corners, border_crop_px)
    masks = [_make_feather_mask(img.shape[:2], feather_px) for img in cropped.images]
    compensator = _create_exposure_compensator()
    compensator.feed(cropped.corners, cropped.images, masks)
    compensated = []
    for img, corner, mask in zip(cropped.images, cropped.corners, masks, strict=True):
        adjusted = img.copy()
        compensator.apply(corner, adjusted, mask)
        compensated.append(adjusted)
    seam_masks = [mask.copy() for mask in masks]
    seam_finder = _create_seam_finder()
    seam_finder.find([img.astype(np.float32) for img in compensated], cropped.corners, seam_masks)
    result = _blend_multiband(compensated, seam_masks, cropped.corners, num_bands)
    return result


def _validate_inputs(imgs: list[np.ndarray], corners: list[tuple[int, int]]) -> StitchInputs:
    if not imgs:
        raise ValueError("imgs must contain at least one image.")
    if len(imgs) != len(corners):
        raise ValueError("imgs and corners must have the same length.")
    parsed_corners: List[Tuple[int, int]] = []
    parsed_images: List[np.ndarray] = []
    for idx, (img, corner) in enumerate(zip(imgs, corners, strict=True)):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"imgs[{idx}] must be a numpy array.")
        if img.dtype != np.uint8:
            raise TypeError(f"imgs[{idx}] must be uint8, got {img.dtype}.")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"imgs[{idx}] must be HxWx3.")
        if len(corner) != 2:
            raise ValueError(f"corners[{idx}] must be (x, y).")
        parsed_images.append(img)
        parsed_corners.append((int(corner[0]), int(corner[1])))
    return StitchInputs(parsed_images, parsed_corners)


def _apply_border_crop(
    imgs: Sequence[np.ndarray],
    corners: Sequence[Tuple[int, int]],
    border_crop_px: int,
) -> StitchInputs:
    if border_crop_px <= 0:
        return StitchInputs(list(imgs), list(corners))
    cropped_images: List[np.ndarray] = []
    cropped_corners: List[Tuple[int, int]] = []
    for img, corner in zip(imgs, corners, strict=True):
        h, w = img.shape[:2]
        if border_crop_px * 2 >= min(h, w):
            raise ValueError("border_crop_px is too large for the tile size.")
        cropped_images.append(img[border_crop_px : h - border_crop_px, border_crop_px : w - border_crop_px])
        cropped_corners.append((corner[0] + border_crop_px, corner[1] + border_crop_px))
    return StitchInputs(cropped_images, cropped_corners)


def _make_feather_mask(shape: Tuple[int, int], feather_px: int) -> np.ndarray:
    h, w = shape
    base = np.ones((h, w), dtype=np.uint8) * 255
    if feather_px <= 0:
        return base
    dist = cv2.distanceTransform(base, cv2.DIST_L2, 3)
    weight = np.clip(dist / float(feather_px), 0.0, 1.0)
    if feather_px > 3:
        blur = max(3, int(feather_px // 2) * 2 + 1)
        weight = cv2.GaussianBlur(weight, (blur, blur), 0)
    return (weight * 255).astype(np.uint8)


def _create_exposure_compensator() -> cv2.detail_ExposureCompensator:
    try:
        compensator_type = cv2.detail.ExposureCompensator_GAIN_BLOCKS
    except AttributeError:
        compensator_type = cv2.detail.ExposureCompensator_GAIN
    return cv2.detail.ExposureCompensator_createDefault(compensator_type)


def _create_seam_finder() -> cv2.detail_SeamFinder:
    try:
        return cv2.detail_GraphCutSeamFinder("COST_COLOR")
    except Exception:
        return cv2.detail_VoronoiSeamFinder()


def _blend_multiband(
    imgs: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    corners: Sequence[Tuple[int, int]],
    num_bands: int,
) -> np.ndarray:
    sizes = [(img.shape[1], img.shape[0]) for img in imgs]
    dst_roi = cv2.detail.resultRoi(corners, sizes)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(max(1, int(num_bands)))
    blender.prepare(dst_roi)
    for img, mask, corner in zip(imgs, masks, corners, strict=True):
        blender.feed(img.astype(np.int16), mask, corner)
    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8)


def read_tiles(paths: Iterable[Path]) -> List[np.ndarray]:
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images


def load_corners(path: Path, tile_paths: Sequence[Path]) -> List[Tuple[int, int]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "corners" in payload:
                payload = payload["corners"]
            else:
                mapping = {Path(name).name: coords for name, coords in payload.items()}
                return [_parse_corner(mapping[p.name]) for p in tile_paths]
        return [_parse_corner(item) for item in payload]
    if path.suffix.lower() == ".csv":
        corners: List[Tuple[int, int]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].lower() in {"x", "y"} or row[0].startswith("#"):
                    continue
                corners.append(_parse_corner(row))
        return corners
    raise ValueError("corners file must be .json or .csv")


def _parse_corner(value: Iterable[object]) -> Tuple[int, int]:
    items = list(value)
    if len(items) < 2:
        raise ValueError("Corner must contain at least two values (x, y).")
    return int(float(items[0])), int(float(items[1]))
