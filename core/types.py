from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class ColorMatchMode(str, Enum):
    OFF = "off"
    MILD = "mild"
    STRONG = "strong"


class InpaintMethod(str, Enum):
    TELEA = "telea"
    NS = "ns"
    EDGE_EXTEND = "edge_extend"


class StitchMode(str, Enum):
    FAST = "fast"
    RESTORATION = "restoration"


@dataclass
class StitchSettings:
    bg_threshold: int = 245
    bg_mode: str = "lab_paper"
    crop_padding_px: int = 4
    max_angle: float = 7.0
    overlap_max: int = 15
    seam_band_px: int = 30
    refine_iterations: int = 3
    max_scale_percent: float = 0.5
    max_edge_warp_px: int = 2
    feather_px: int = 10
    compression: str = "deflate"
    color_match: ColorMatchMode = ColorMatchMode.MILD
    protect_lines: bool = True
    seam_fill_enabled: bool = False
    seam_fill_max_px: int = 15
    inpaint_radius: int = 3
    inpaint_method: InpaintMethod = InpaintMethod.TELEA
    optimize_tiff: bool = True
    debug_stitch: bool = False


@dataclass
class TileInfo:
    path: Path
    x: int
    y: int
    suffix: str = ""


@dataclass
class TileData:
    info: TileInfo
    image: np.ndarray
    mask: np.ndarray
    angle_deg: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class TileTransform:
    dx: float = 0.0
    dy: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    angle_deg: float = 0.0
    confidence: float = 0.0


@dataclass
class TileLayout:
    origin_x: int
    origin_y: int
    tile_w: int
    tile_h: int
    transform: TileTransform = field(default_factory=TileTransform)


@dataclass
class LayoutResult:
    placements: Dict[Tuple[int, int], TileLayout]
    canvas_size: Tuple[int, int]


@dataclass
class SeamMetrics:
    gap_px: int
    seam_error: float
    score: float


@dataclass
class ProjectState:
    tiles: List[TileInfo]
    settings: StitchSettings
    transforms: Dict[str, TileTransform]
    output_path: Optional[Path] = None
