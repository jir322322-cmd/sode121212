from __future__ import annotations

from typing import Dict, Tuple

from core.layout import grid_neighbors
from core.match import MatchResult, match_seam, match_seam_vertical
from core.types import LayoutResult, TileData, TileTransform


def refine_layout(tiles: Dict[Tuple[int, int], TileData], layout: LayoutResult, overlap: int, band: int, iterations: int) -> LayoutResult:
    placements = layout.placements
    for _ in range(iterations):
        for coord, tile in tiles.items():
            neighbors = grid_neighbors(coord)
            best_error = 1e9
            best_transform = placements[coord].transform
            left_coord = neighbors.get("left")
            top_coord = neighbors.get("top")
            results: list[MatchResult] = []
            if left_coord in tiles:
                results.append(match_seam(tiles[left_coord], tile, overlap, band))
            if top_coord in tiles:
                results.append(match_seam_vertical(tiles[top_coord], tile, overlap, band))
            if not results:
                continue
            for result in results:
                if result.error < best_error:
                    best_error = result.error
                    best_transform = TileTransform(
                        dx=result.dx,
                        dy=result.dy,
                        scale_x=1.0,
                        scale_y=1.0,
                        angle_deg=tile.angle_deg,
                        confidence=result.confidence,
                    )
            placements[coord].transform = best_transform
    return LayoutResult(placements=placements, canvas_size=layout.canvas_size)
