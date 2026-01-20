from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from core.types import ProjectState, StitchSettings, TileInfo, TileTransform

TILE_RE = re.compile(r"^(?P<x>\d+),(?P<y>\d+)(?P<suffix>.*)\.(?P<ext>jpe?g|png|tif|tiff)$", re.IGNORECASE)


def parse_tile_name(path: Path) -> TileInfo | None:
    match = TILE_RE.match(path.name)
    if not match:
        return None
    return TileInfo(
        path=path,
        x=int(match.group("x")),
        y=int(match.group("y")),
        suffix=match.group("suffix"),
    )


def load_tiles(paths: Iterable[Path]) -> List[TileInfo]:
    tiles: List[TileInfo] = []
    for path in paths:
        info = parse_tile_name(path)
        if info is None:
            continue
        tiles.append(info)
    tiles.sort(key=lambda t: (t.x, t.y))
    return tiles


def read_image(path: Path) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def save_project(state: ProjectState, path: Path) -> None:
    payload = {
        "tiles": [tile.path.name for tile in state.tiles],
        "settings": state.settings.__dict__,
        "transforms": {
            name: transform.__dict__ for name, transform in state.transforms.items()
        },
        "output_path": str(state.output_path) if state.output_path else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project(path: Path, tile_root: Path) -> ProjectState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = StitchSettings(**payload.get("settings", {}))
    tiles: List[TileInfo] = []
    for name in payload.get("tiles", []):
        info = parse_tile_name(tile_root / name)
        if info:
            tiles.append(info)
    transforms: Dict[str, TileTransform] = {}
    for name, data in payload.get("transforms", {}).items():
        transforms[name] = TileTransform(**data)
    output_path = payload.get("output_path")
    return ProjectState(
        tiles=tiles,
        settings=settings,
        transforms=transforms,
        output_path=Path(output_path) if output_path else None,
    )


def load_transforms(path: Path) -> Dict[str, TileTransform]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: TileTransform(**data) for name, data in payload.items()}


def save_transforms(transforms: Dict[str, TileTransform], path: Path) -> None:
    payload = {name: transform.__dict__ for name, transform in transforms.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_tile_paths(paths: Iterable[Path]) -> List[Path]:
    items: List[Path] = []
    for path in paths:
        if path.is_dir():
            items.extend([p for p in path.iterdir() if p.is_file()])
        else:
            items.append(path)
    return items


def compute_grid(tiles: List[TileInfo]) -> Tuple[int, int]:
    max_x = max((tile.x for tile in tiles), default=0)
    max_y = max((tile.y for tile in tiles), default=0)
    return max_x, max_y
