from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from core.types import LayoutResult, TileData, TileLayout, TileTransform


def initial_layout(tiles: Dict[Tuple[int, int], TileData]) -> LayoutResult:
    xs = [tile.info.x for tile in tiles.values()]
    ys = [tile.info.y for tile in tiles.values()]
    max_x = max(xs) if xs else 0
    max_y = max(ys) if ys else 0
    placements: Dict[Tuple[int, int], TileLayout] = {}
    tile_w = max(tile.image.shape[1] for tile in tiles.values()) if tiles else 0
    tile_h = max(tile.image.shape[0] for tile in tiles.values()) if tiles else 0
    for coord, tile in tiles.items():
        x, y = coord
        origin_x = (y - 1) * tile_w
        origin_y = (x - 1) * tile_h
        placements[coord] = TileLayout(origin_x=origin_x, origin_y=origin_y, tile_w=tile_w, tile_h=tile_h, transform=TileTransform())
    canvas_w = max_y * tile_w
    canvas_h = max_x * tile_h
    return LayoutResult(placements=placements, canvas_size=(canvas_w, canvas_h))


def grid_neighbors(coord: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
    x, y = coord
    return {
        "left": (x, y - 1),
        "top": (x - 1, y),
    }
