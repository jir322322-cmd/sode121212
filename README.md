# Map Restorer Stitcher

Python 3.11+ application for restoring map tiles into a seamless single sheet. Focus is on **visual continuity**, not geometric fidelity.

## Features
- Tile parsing from `x,y.jpg` or `x,y_suffix.jpg` naming convention.
- Per-tile preprocessing: background mask, crop, rotation correction.
- Photometric normalization (LAB luminance matching, optional histogram matching).
- Iterative seam refinement (NCC/phase correlation) within overlap limits.
- Gap forcing with edge-extend and optional seam fill.
- Overlap blending + protect lines mask.
- Lossless BigTIFF output (deflate/lzw), optional tiled + overviews.
- Debug pack export (preview, heatmaps, masks, transforms).

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m app.main
```

## Seamless blend stitching (CLI example)
```bash
python -m app.blend_cli \
  --tiles tiles/1.png tiles/2.png tiles/3.png \
  --corners corners.json \
  --out pano.png \
  --border-crop-px 30 \
  --feather-px 80 \
  --num-bands 6
```

## Usage
1. Add tiles (files, folder, or drag-and-drop).
2. Set output TIFF path.
3. Choose mode: **Fast** or **Restoration**.
4. Adjust parameters if needed.
5. Click **Start**.
6. Use manual adjust (dx/dy/scale) on a selected tile and rerun to fix stubborn seams.

## Tuning when seams are visible
- **Increase `seam_band_px`** to give more context to the matcher.
- **Increase `feather_px`** to soften overlaps.
- Switch **Color match** to `strong` for aggressive tone matching.
- If tiny gaps persist, enable **Seam Fill** (Telea) and keep `seam_fill_max_px` small.
- Use **Manual Adjust** to nudge a single tile by 1-5px and rerun.

## Outputs
- `out.tif` (lossless BigTIFF if needed).
- `debug/` folder (if Debug Stitch enabled):
  - `preview_layout.png`
  - `seam_heatmap.png`
  - `gap_mask_before.png`, `gap_mask_after.png`
  - `small_preview.png`
  - `transforms.json`, `metrics.json`, `project.json`

## Notes
- Background correction is applied **only** outside the content mask.
- Overlap is limited to `overlap_max` (<=15px) and scale is constrained to ±0.5%.
