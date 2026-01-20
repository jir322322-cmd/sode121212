from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore

from core import debugpack
from core.blend import feather_blend, feather_blend_vertical
from core.io import collect_tile_paths, load_tiles, read_image
from core.layout import initial_layout
from core.photometric import apply_color_match
from core.preprocess import preprocess_tile
from core.protect import protect_lines_mask
from core.refine import refine_layout
from core.seamfill import seam_fill
from core.tiffout import write_tiff
from core.types import (
    ColorMatchMode,
    LayoutResult,
    ProjectState,
    SeamMetrics,
    StitchMode,
    StitchSettings,
    TileData,
    TileTransform,
)


class StitchWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(Path)
    preview_ready = QtCore.Signal(np.ndarray, np.ndarray)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        tile_paths: List[Path],
        settings: StitchSettings,
        output_path: Path,
        mode: StitchMode,
        transforms: Optional[Dict[str, TileTransform]] = None,
    ) -> None:
        super().__init__()
        self.tile_paths = tile_paths
        self.settings = settings
        self.output_path = output_path
        self.mode = mode
        self.transforms = transforms or {}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run(self) -> None:
        self._apply_mode_overrides()
        self.log.emit("Loading tiles...")
        tile_paths = collect_tile_paths(self.tile_paths)
        tiles_info = load_tiles(tile_paths)
        if not tiles_info:
            raise ValueError("No valid tiles found. Expect filenames like 1,1.jpg")
        grid_max_x = max(tile.x for tile in tiles_info)
        grid_max_y = max(tile.y for tile in tiles_info)
        missing = []
        existing = {(t.x, t.y) for t in tiles_info}
        for x in range(1, grid_max_x + 1):
            for y in range(1, grid_max_y + 1):
                if (x, y) not in existing:
                    missing.append((x, y))
        if missing:
            self.log.emit(f"Missing tiles: {missing}")
        tiles: Dict[Tuple[int, int], TileData] = {}
        for idx, info in enumerate(tiles_info, start=1):
            if self._cancel:
                return
            image = read_image(info.path)
            tile = preprocess_tile(info, image, self.settings)
            tiles[(info.x, info.y)] = tile
            self.progress.emit(int(idx / len(tiles_info) * 20))
        self.log.emit("Photometric normalization...")
        for coord, tile in tiles.items():
            if self._cancel:
                return
            if coord == (1, 1):
                continue
            ref_coord = (coord[0], coord[1] - 1) if (coord[0], coord[1] - 1) in tiles else (coord[0] - 1, coord[1])
            if ref_coord in tiles:
                tile.image = apply_color_match(tile.image, tiles[ref_coord].image, tile.mask, self.settings.color_match)
        layout = initial_layout(tiles)
        layout = refine_layout(tiles, layout, self.settings.overlap_max, self.settings.seam_band_px, self.settings.refine_iterations)
        self._apply_manual_transforms(layout)
        self.log.emit("Blending overlaps...")
        self._blend_neighbors(tiles, layout)
        self.log.emit("Composing canvas...")
        canvas, occupancy = self._compose_canvas(tiles, layout)
        gap_mask = (occupancy == 0).astype(np.uint8) * 255
        gap_px = int(np.count_nonzero(gap_mask))
        if gap_px > 0:
            self.log.emit(f"Gap detected ({gap_px}px). Applying edge-extend...")
            edge_filled = cv2.inpaint(canvas, gap_mask, 1, cv2.INPAINT_TELEA)
            canvas = edge_filled
            occupancy[gap_mask > 0] = 1
        if self.settings.seam_fill_enabled:
            protect_mask = protect_lines_mask(canvas) if self.settings.protect_lines else np.zeros(canvas.shape[:2], dtype=np.uint8)
            seam_mask = self._limit_seam_mask(gap_mask)
            seam_mask[protect_mask > 0] = 0
            if seam_mask.mean() < 1.0:
                fill_area = np.count_nonzero(seam_mask)
                if fill_area / float(canvas.shape[0] * canvas.shape[1]) < 0.005:
                    fill = seam_fill(canvas, seam_mask, self.settings.inpaint_method, self.settings.inpaint_radius)
                    canvas = fill.filled
                else:
                    self.log.emit("Seam fill skipped: area > 0.5% of canvas")
        self.log.emit("Final gap check...")
        gap_mask_after = (occupancy == 0).astype(np.uint8) * 255
        if np.count_nonzero(gap_mask_after) > 0:
            self.log.emit("Warning: residual gaps detected after fill.")
        self.preview_ready.emit(self._downscale(canvas), self._downscale_heatmap(gap_mask_after))
        self.log.emit("Saving TIFF...")
        write_tiff(self.output_path, canvas, self.settings.compression, self.settings.optimize_tiff, self.settings.optimize_tiff)
        if self.settings.debug_stitch:
            self._save_debug(tiles, layout, gap_mask, gap_mask_after, canvas)
        self.finished.emit(self.output_path)

    def _apply_manual_transforms(self, layout: LayoutResult) -> None:
        for coord, placement in layout.placements.items():
            key = f"{coord[0]},{coord[1]}"
            if key in self.transforms:
                placement.transform = self.transforms[key]

    def _blend_neighbors(self, tiles: Dict[Tuple[int, int], TileData], layout: LayoutResult) -> None:
        overlap = min(self.settings.overlap_max, self.settings.feather_px)
        for (x, y), tile in tiles.items():
            left_coord = (x, y - 1)
            if left_coord in tiles:
                tiles[left_coord].image, tile.image = feather_blend(
                    tiles[left_coord].image, tile.image, overlap=overlap, feather=self.settings.feather_px
                )
            top_coord = (x - 1, y)
            if top_coord in tiles:
                tiles[top_coord].image, tile.image = feather_blend_vertical(
                    tiles[top_coord].image, tile.image, overlap=overlap, feather=self.settings.feather_px
                )

    def _compose_canvas(self, tiles: Dict[Tuple[int, int], TileData], layout: LayoutResult) -> Tuple[np.ndarray, np.ndarray]:
        canvas_w, canvas_h = layout.canvas_size
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        occupancy = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for coord, tile in tiles.items():
            placement = layout.placements[coord]
            image = tile.image
            if placement.transform.scale_x != 1.0 or placement.transform.scale_y != 1.0:
                image = cv2.resize(image, None, fx=placement.transform.scale_x, fy=placement.transform.scale_y, interpolation=cv2.INTER_LINEAR)
            x0 = placement.origin_x + int(placement.transform.dx)
            y0 = placement.origin_y + int(placement.transform.dy)
            h, w = image.shape[:2]
            x1 = min(canvas.shape[1], x0 + w)
            y1 = min(canvas.shape[0], y0 + h)
            x0 = max(0, x0)
            y0 = max(0, y0)
            canvas[y0:y1, x0:x1] = image[: y1 - y0, : x1 - x0]
            occupancy[y0:y1, x0:x1] = 1
        return canvas, occupancy

    def _save_debug(self, tiles: Dict[Tuple[int, int], TileData], layout: LayoutResult, gap_before: np.ndarray, gap_after: np.ndarray, canvas: np.ndarray) -> None:
        debug_root = self.output_path.parent / "debug"
        debugpack.ensure_dir(debug_root)
        debugpack.save_layout_preview(debug_root / "preview_layout.png", layout, tiles)
        debugpack.save_heatmap(debug_root / "seam_heatmap.png", gap_after)
        debugpack.save_image(debug_root / "gap_mask_before.png", cv2.cvtColor(gap_before, cv2.COLOR_GRAY2RGB))
        debugpack.save_image(debug_root / "gap_mask_after.png", cv2.cvtColor(gap_after, cv2.COLOR_GRAY2RGB))
        debugpack.save_image(debug_root / "small_preview.png", self._downscale(canvas))

    def _apply_mode_overrides(self) -> None:
        if self.mode == StitchMode.FAST:
            self.settings.refine_iterations = min(self.settings.refine_iterations, 1)
            if self.settings.color_match == ColorMatchMode.STRONG:
                self.settings.color_match = ColorMatchMode.MILD

    def _limit_seam_mask(self, gap_mask: np.ndarray) -> np.ndarray:
        if self.settings.seam_fill_max_px <= 0:
            return gap_mask.copy()
        dist = cv2.distanceTransform(255 - gap_mask, cv2.DIST_L2, 3)
        limited = (dist <= float(self.settings.seam_fill_max_px)).astype(np.uint8) * 255
        return cv2.bitwise_and(gap_mask, limited)

    def _downscale(self, image: np.ndarray, max_side: int = 1200) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale == 1.0:
            return image
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _downscale_heatmap(self, mask: np.ndarray, max_side: int = 1200) -> np.ndarray:
        h, w = mask.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        resized = mask
        if scale != 1.0:
            resized = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def save_project(state: ProjectState, path: Path) -> None:
    payload = {
        "tiles": [tile.path.name for tile in state.tiles],
        "settings": state.settings.__dict__,
        "transforms": {name: transform.__dict__ for name, transform in state.transforms.items()},
        "output_path": str(state.output_path) if state.output_path else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
