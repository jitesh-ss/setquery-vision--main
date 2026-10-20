"""Spectral Band Indices & Radiometric Rendering Engine.

Computes on-the-fly multispectral and vegetation/water indices:
- True Color (RGB Composite)
- VARI / Excess Green (Visible Atmospherically Resistant Index / NDVI-proxy for RGB)
- NDWI / Water Contrast Index (Normalized Difference Water Index proxy)
- Color Infrared (CIR / False Color Vegetation Contrast)
- Thermal/Reflectance Heatmap (Built-up infrastructure intensity)

Maps index values to perceptually uniform colormaps (Viridis, YlGn, Blues, Turbo)
and streams web-optimized PNG overlays for interactive map layers.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_spectral_index_image(
    geotiff_path: str | Path,
    mode: str = "true_color",
    target_max_dim: Optional[int] = 1024,
) -> bytes:
    """Compute radiometric or spectral index and return colorized PNG bytes.

    Args:
        geotiff_path: Path to source GeoTIFF.
        mode: One of 'true_color', 'ndvi', 'ndwi', 'false_color', 'heat'.
        target_max_dim: Max dimension for web preview.

    Returns:
        PNG image bytes.
    """
    path = Path(geotiff_path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        count = src.count
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32) if count >= 2 else r.copy()
        b = src.read(3).astype(np.float32) if count >= 3 else r.copy()

    mask = (r > 0) | (g > 0) | (b > 0)

    if mode == "true_color":
        # Standard True Color
        r_u8 = np.clip(r, 0, 255).astype(np.uint8)
        g_u8 = np.clip(g, 0, 255).astype(np.uint8)
        b_u8 = np.clip(b, 0, 255).astype(np.uint8)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.stack([r_u8, g_u8, b_u8, alpha], axis=-1)

    elif mode == "ndvi":
        # Visual Atmospherically Resistant Index (VARI): (Green - Red) / (Green + Red - Blue + eps)
        # Highly correlated with satellite NDVI for canopy greenness and chlorophyll density
        denom = (g + r - b)
        denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
        vari = (g - r) / denom
        vari = np.clip(vari, -0.4, 0.6)

        # Normalize to 0.0 - 1.0
        norm = (vari + 0.4) / 1.0
        norm = np.clip(norm, 0.0, 1.0)

        # Apply Viridis/YlGn-style colormap (Brown/Yellow -> Pale Green -> Deep Emerald)
        r_col = np.uint8(255 * (1.0 - norm * 0.8))
        g_col = np.uint8(255 * np.sin(norm * np.pi * 0.5))
        b_col = np.uint8(255 * ((1.0 - norm) * 0.3))
        alpha = np.where(mask, 230, 0).astype(np.uint8)
        rgba = np.stack([r_col, g_col, b_col, alpha], axis=-1)

    elif mode == "ndwi":
        # Normalized Water Contrast Index: (Green - Blue) / (Green + Blue + eps)
        denom = g + b + 1e-5
        ndwi_val = (g - b) / denom

        # River and deep water bodies show distinct high values
        norm = (ndwi_val - np.percentile(ndwi_val[mask], 5)) / (np.percentile(ndwi_val[mask], 95) - np.percentile(ndwi_val[mask], 5) + 1e-5)
        norm = np.clip(norm, 0.0, 1.0)

        # Deep Cyan-Blue Colormap
        r_col = np.uint8(255 * (1.0 - norm * 0.95))
        g_col = np.uint8(255 * (0.4 + norm * 0.5))
        b_col = np.uint8(255 * (0.6 + norm * 0.4))
        alpha = np.where(mask, 240, 0).astype(np.uint8)
        rgba = np.stack([r_col, g_col, b_col, alpha], axis=-1)

    elif mode == "false_color":
        # Color Infrared (CIR) simulation: Red channel maps to Green, Green to NIR-proxy, Blue to Red
        # In false-color infrared, vegetation glows vivid scarlet/crimson, water is dark navy, urban is cyan
        nir_proxy = np.clip(g * 1.5 - r * 0.3, 0, 255).astype(np.uint8)
        r_cir = nir_proxy                                   # NIR -> Red
        g_cir = np.clip(r, 0, 255).astype(np.uint8)          # Red -> Green
        b_cir = np.clip(b, 0, 255).astype(np.uint8)          # Green/Blue -> Blue
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.stack([r_cir, g_cir, b_cir, alpha], axis=-1)

    elif mode == "heat":
        # High-Reflectance Infrastructure Heatmap
        intensity = (r * 0.33 + g * 0.33 + b * 0.33)
        norm = np.clip(intensity / 255.0, 0.0, 1.0)

        # Turbo/Inferno Colormap (Purple -> Amber -> Bright Yellow)
        r_col = np.uint8(np.clip(norm * 2.2 * 255, 0, 255))
        g_col = np.uint8(np.clip((norm - 0.25) * 1.8 * 255, 0, 255))
        b_col = np.uint8(np.clip(np.sin(norm * np.pi) * 0.6 * 255, 0, 255))
        alpha = np.where(mask, 220, 0).astype(np.uint8)
        rgba = np.stack([r_col, g_col, b_col, alpha], axis=-1)

    else:
        raise ValueError(f"Unknown spectral mode: {mode}")

    img = Image.fromarray(rgba, mode="RGBA")
    if target_max_dim and max(img.width, img.height) > target_max_dim:
        scale = target_max_dim / max(img.width, img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
