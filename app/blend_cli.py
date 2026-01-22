from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2

from core.blend_stitcher import load_corners, read_tiles, stitch_tiles


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blend pre-aligned tiles into a seamless panorama.")
    parser.add_argument("--tiles", nargs="+", required=True, help="List of tile image paths.")
    parser.add_argument("--corners", required=True, help="Path to corners JSON or CSV.")
    parser.add_argument("--out", default="pano.png", help="Output panorama path.")
    parser.add_argument("--border-crop-px", type=int, default=10, help="Pixels to crop from each tile border.")
    parser.add_argument("--feather-px", type=int, default=60, help="Feather size for masks.")
    parser.add_argument("--num-bands", type=int, default=5, help="Number of multiband blending levels.")
    args = parser.parse_args(argv)

    tile_paths = [Path(p) for p in args.tiles]
    images = read_tiles(tile_paths)
    corners = load_corners(Path(args.corners), tile_paths)
    pano = stitch_tiles(
        images,
        corners,
        border_crop_px=args.border_crop_px,
        feather_px=args.feather_px,
        num_bands=args.num_bands,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
