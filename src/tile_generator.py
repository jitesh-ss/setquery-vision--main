"""Tiling and Preprocessing Module for Geospatial Imagery.

Slices raster GeoTIFF files into overlapping 512x512 tiles with a 384 stride
while preserving geographic Affine transforms and spatial reference systems (CRS).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_tile_offsets(dimension: int, tile_size: int = 512, stride: int = 384) -> List[int]:
    """Compute 1D offsets for slicing with stride and coverage of image edges."""
    if dimension <= tile_size:
        return [0]

    offsets = list(range(0, dimension - tile_size + 1, stride))
    # If the last offset does not reach the image boundary, include the boundary tile
    last_offset = dimension - tile_size
    if offsets[-1] != last_offset:
        offsets.append(last_offset)

    return sorted(list(set(offsets)))


def slice_raster_tiles(
    input_geotiff: str | Path,
    output_dir: str | Path = "data/processed/tiles",
    tile_size: int = 512,
    stride: int = 384,
) -> List[Dict[str, Any]]:
    """Slice GeoTIFF into overlapping tiles preserving exact Affine transforms.

    Args:
        input_geotiff: Path to input source GeoTIFF.
        output_dir: Target directory to write tile GeoTIFFs.
        tile_size: Height and width in pixels of each tile (default 512).
        stride: Stride step between successive tiles in pixels (default 384).

    Returns:
        List of tile metadata dictionaries.
    """
    input_path = Path(input_geotiff)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    tile_records: List[Dict[str, Any]] = []

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        crs = src.crs
        src_transform = src.transform
        profile = src.profile.copy()

        logger.info("Opening %s | Size: (%d, %d) | Bands: %d | CRS: %s",
                    input_path.name, width, height, src.count, crs)

        row_offsets = compute_tile_offsets(height, tile_size, stride)
        col_offsets = compute_tile_offsets(width, tile_size, stride)

        logger.info("Generating tiles with size=%d, stride=%d (%d rows x %d cols = %d total tiles)",
                    tile_size, stride, len(row_offsets), len(col_offsets), len(row_offsets) * len(col_offsets))

        # Transformer to EPSG:4326 for metadata footprint
        to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

        tile_idx = 0
        for r_idx, row_off in enumerate(row_offsets):
            for c_idx, col_off in enumerate(col_offsets):
                w = min(tile_size, width - col_off)
                h = min(tile_size, height - row_off)

                win = Window(col_off, row_off, w, h)
                tile_transform = window_transform(win, src_transform)

                data = src.read(window=win)

                # Pad if dimensions are less than tile_size (for images smaller than tile_size)
                if w < tile_size or h < tile_size:
                    padded_data = np.zeros((src.count, tile_size, tile_size), dtype=data.dtype)
                    padded_data[:, :h, :w] = data
                    data = padded_data
                    out_w, out_h = tile_size, tile_size
                else:
                    out_w, out_h = w, h

                tile_filename = f"tile_r{r_idx:03d}_c{c_idx:03d}.tif"
                tile_filepath = out_dir / tile_filename

                tile_profile = profile.copy()
                tile_profile.update({
                    "driver": "GTiff",
                    "height": out_h,
                    "width": out_w,
                    "transform": tile_transform,
                    "compress": "deflate",
                })

                with rasterio.open(tile_filepath, "w", **tile_profile) as dst:
                    dst.write(data)
                    for b in range(1, src.count + 1):
                        desc = src.descriptions[b - 1] if src.descriptions else f"Band {b}"
                        if desc:
                            dst.set_band_description(b, desc)

                # Compute geographic bounds in source CRS and convert to WGS84
                left, bottom, right, top = rasterio.transform.array_bounds(out_h, out_w, tile_transform)
                tile_geom_crs = box(left, bottom, right, top)
                tile_geom_wgs84 = shapely_transform(to_wgs84, tile_geom_crs)

                record = {
                    "tile_index": tile_idx,
                    "tile_file": str(tile_filepath.resolve()),
                    "row_index": r_idx,
                    "col_index": c_idx,
                    "pixel_x": col_off,
                    "pixel_y": row_off,
                    "width": out_w,
                    "height": out_h,
                    "transform": list(tile_transform)[:6],
                    "crs": str(crs),
                    "bounds_wgs84": list(tile_geom_wgs84.bounds),
                }
                tile_records.append(record)
                tile_idx += 1

    # Save metadata index
    manifest_path = out_dir / "tiles_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tile_records, f, indent=2)

    logger.info("Successfully generated %d tiles in %s (manifest: %s)", len(tile_records), out_dir, manifest_path)
    return tile_records


if __name__ == "__main__":
    slice_raster_tiles("data/raw/aoi_sample.tif", "data/processed/tiles", tile_size=512, stride=384)
