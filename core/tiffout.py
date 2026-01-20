from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import tifffile


def write_tiff(path: Path, image: np.ndarray, compression: str, tiled: bool, overviews: bool) -> None:
    bigtiff = image.nbytes > 2**32
    tile = (256, 256) if tiled else None
    with tifffile.TiffWriter(str(path), bigtiff=bigtiff) as tif:
        tif.write(
            image,
            photometric="rgb",
            compression=compression,
            tile=tile,
            subifds=1 if overviews else 0,
        )
        if overviews:
            pyramid = build_overviews(image)
            for level in pyramid:
                tif.write(
                    level,
                    photometric="rgb",
                    compression=compression,
                    tile=tile,
                    subfiletype=1,
                )


def build_overviews(image: np.ndarray, max_levels: int = 4) -> List[np.ndarray]:
    levels: List[np.ndarray] = []
    current = image
    for _ in range(max_levels):
        if min(current.shape[:2]) < 512:
            break
        current = current[::2, ::2]
        levels.append(current)
    return levels
