from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from core.types import LayoutResult, SeamMetrics, TileData, TileTransform


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(str(path))


def save_heatmap(path: Path, heatmap: np.ndarray) -> None:
    heat = cv2.applyColorMap(np.clip(heatmap, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    save_image(path, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def save_layout_preview(path: Path, layout: LayoutResult, tiles: Dict[Tuple[int, int], TileData]) -> None:
    canvas_w, canvas_h = layout.canvas_size
    preview = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for coord, placement in layout.placements.items():
        tile = tiles[coord]
        x0 = placement.origin_x + int(placement.transform.dx)
        y0 = placement.origin_y + int(placement.transform.dy)
        h, w = tile.image.shape[:2]
        preview[y0 : y0 + h, x0 : x0 + w] = tile.image
        cv2.rectangle(preview, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        cv2.putText(preview, f"{tile.info.x},{tile.info.y}", (x0 + 5, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    save_image(path, preview)


def save_transforms(path: Path, transforms: Dict[str, TileTransform]) -> None:
    payload = {name: transform.__dict__ for name, transform in transforms.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_metrics(path: Path, metrics: Dict[str, SeamMetrics]) -> None:
    payload = {name: metric.__dict__ for name, metric in metrics.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
