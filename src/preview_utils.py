"""Raster Preview Utilities.

Converts multi-band satellite GeoTIFFs and tiles into web-optimized RGB PNGs
with optional percentile stretch for interactive web map overlay and inspection.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyproj
import rasterio
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_raster_wgs84_bounds(geotiff_path: str | Path) -> List[List[float]]:
    """Return Leaflet-compatible WGS84 bounds [[south, west], [north, east]]."""
    with rasterio.open(geotiff_path) as src:
        bounds = src.bounds
        crs = src.crs

        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        minx, miny = transformer(bounds.left, bounds.bottom)
        maxx, maxy = transformer(bounds.right, bounds.top)

        # Leaflet requires [[south, west], [north, east]]
        return [[miny, minx], [maxy, maxx]]


def raster_to_png_bytes(
    geotiff_path: str | Path,
    percentile_clip: bool = False,
    target_max_dim: Optional[int] = 1024,
) -> bytes:
    """Read a GeoTIFF and encode its RGB channels as a web-compatible PNG."""
    with rasterio.open(geotiff_path) as src:
        count = src.count
        if count >= 3:
            # Read first 3 bands as RGB
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
        else:
            # Single band greyscale
            gray = src.read(1)
            r = g = b = gray

        # Handle nodata/invalid
        mask = (r > 0) | (g > 0) | (b > 0)

        if percentile_clip and r.max() > 255:
            # Normalize 16-bit or reflectance data to 0-255
            p2 = np.percentile(r[mask], 2) if np.any(mask) else 0
            p98 = np.percentile(r[mask], 98) if np.any(mask) else 1
            r_norm = np.clip((r - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

            p2 = np.percentile(g[mask], 2) if np.any(mask) else 0
            p98 = np.percentile(g[mask], 98) if np.any(mask) else 1
            g_norm = np.clip((g - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

            p2 = np.percentile(b[mask], 2) if np.any(mask) else 0
            p98 = np.percentile(b[mask], 98) if np.any(mask) else 1
            b_norm = np.clip((b - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        else:
            # True Color uint8 or direct scaling
            r_norm = np.clip(r, 0, 255).astype(np.uint8)
            g_norm = np.clip(g, 0, 255).astype(np.uint8)
            b_norm = np.clip(b, 0, 255).astype(np.uint8)

        # Alpha channel for non-zero data
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.stack([r_norm, g_norm, b_norm, alpha], axis=-1)

        img = Image.fromarray(rgba, mode="RGBA")

        # Downsample if image is oversized for web preview
        if target_max_dim and max(img.width, img.height) > target_max_dim:
            scale = target_max_dim / max(img.width, img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
