"""STAC Ingestion Module for Planetary Computer Sentinel-2 L2A Scenes.

Queries Microsoft Planetary Computer STAC endpoint, retrieves Sentinel-2 L2A
imagery over a defined bounding box with cloud cover filter, streams the data,
and crops the RGB imagery into a local GeoTIFF preserving CRS and Affine transform.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import planetary_computer as pc
from pystac_client import Client
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "sentinel-2-l2a"
DEFAULT_AOI = [80.48, 16.48, 80.58, 16.56]  # [minx, miny, maxx, maxy] EPSG:4326


def get_stac_client(api_url: str = STAC_API_URL) -> Client:
    """Connect to Planetary Computer STAC catalog."""
    return Client.open(api_url)


def search_clear_scene(
    client: Client,
    bbox: List[float],
    max_cloud_cover: float = 15.0,
    limit: int = 5,
) -> Any:
    """Search for the most recent clear Sentinel-2 L2A scene over the AOI."""
    logger.info("Searching STAC catalog for %s scenes over bbox %s with cloud < %s%%", COLLECTION_ID, bbox, max_cloud_cover)
    search = client.search(
        collections=[COLLECTION_ID],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=limit,
    )
    items = list(search.items())
    if not items:
        # Fallback without strict cloud filter if 15% yielded zero recent items
        logger.warning("No scenes found with cloud cover < %s%%. Retrying with cloud cover < 30%%...", max_cloud_cover)
        search = client.search(
            collections=[COLLECTION_ID],
            bbox=bbox,
            query={"eo:cloud_cover": {"lt": 30.0}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=limit,
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for AOI {bbox}")

    chosen_item = items[0]
    # Planetary Computer token signing
    signed_item = pc.sign(chosen_item)
    logger.info("Selected STAC Item: %s (Datetime: %s, Cloud: %.2f%%)",
                signed_item.id,
                signed_item.properties.get("datetime"),
                signed_item.properties.get("eo:cloud_cover", 0.0))
    return signed_item


def crop_scene_to_geotiff(
    stac_item: Any,
    bbox: List[float],
    output_path: str | Path = "data/raw/aoi_sample.tif",
) -> Path:
    """Stream and crop the RGB visual bands into a local GeoTIFF preserving Affine transform.

    Args:
        stac_item: Signed STAC item from Planetary Computer.
        bbox: Target bounding box in EPSG:4326 [minx, miny, maxx, maxy].
        output_path: Destination path for cropped GeoTIFF.

    Returns:
        Path to the saved GeoTIFF.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Prefer 'visual' asset (True Color 10m COG) if available
    asset_keys = stac_item.assets.keys()
    if "visual" in asset_keys:
        visual_href = stac_item.assets["visual"].href
        logger.info("Streaming and cropping 'visual' asset from %s", visual_href.split("?")[0])
        _crop_single_asset(visual_href, bbox, out_file)
    elif all(k in asset_keys for k in ("B04", "B03", "B02")):
        logger.info("Composing RGB composite from B04 (Red), B03 (Green), B02 (Blue)...")
        _crop_multi_band_assets(stac_item, bbox, out_file)
    else:
        raise ValueError(f"Available assets do not include 'visual' or B04/B03/B02: {list(asset_keys)}")

    logger.info("Cropped GeoTIFF successfully written to: %s", out_file)
    return out_file


def _crop_single_asset(asset_href: str, bbox: List[float], out_file: Path) -> None:
    """Read a window from a Cloud-Optimized GeoTIFF and write locally."""
    minx, miny, maxx, maxy = bbox

    with rasterio.open(asset_href) as src:
        # Reproject bbox from EPSG:4326 to raster CRS
        to_raster_crs = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        aoi_geom_utm = shapely_transform(to_raster_crs, box(minx, miny, maxx, maxy))
        u_minx, u_miny, u_maxx, u_maxy = aoi_geom_utm.bounds

        # Compute raster window and corresponding affine transform
        window = from_bounds(u_minx, u_miny, u_maxx, u_maxy, transform=src.transform)
        # Clamp window to source bounds to avoid reading outside valid domain
        window = window.intersection(Window(0, 0, src.width, src.height))
        # Round window offsets/sizes to integer pixels
        window = window.round_offsets().round_lengths()

        window_trans = window_transform(window, src.transform)
        win_w = int(window.width)
        win_h = int(window.height)

        logger.info("Source CRS: %s, Native window size: (%d, %d)", src.crs, win_w, win_h)
        data = src.read(window=window)

        # Update metadata for target GeoTIFF
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": win_h,
            "width": win_w,
            "transform": window_trans,
            "count": src.count,
            "compress": "deflate",
        })

        with rasterio.open(out_file, "w", **out_meta) as dst:
            dst.write(data)
            # Copy tags / band descriptions
            for i in range(1, src.count + 1):
                dst.set_band_description(i, src.descriptions[i - 1] or f"Band {i}")


def _crop_multi_band_assets(stac_item: Any, bbox: List[float], out_file: Path) -> None:
    """Stack separate B04, B03, B02 bands into 3-channel RGB GeoTIFF."""
    minx, miny, maxx, maxy = bbox
    bands = ["B04", "B03", "B02"]
    data_list = []
    meta = None
    trans = None

    for idx, band in enumerate(bands):
        href = stac_item.assets[band].href
        with rasterio.open(href) as src:
            if idx == 0:
                to_raster_crs = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
                aoi_geom_utm = shapely_transform(to_raster_crs, box(minx, miny, maxx, maxy))
                u_minx, u_miny, u_maxx, u_maxy = aoi_geom_utm.bounds

                window = from_bounds(u_minx, u_miny, u_maxx, u_maxy, transform=src.transform)
                window = window.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
                trans = window_transform(window, src.transform)

                meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": int(window.height),
                    "width": int(window.width),
                    "transform": trans,
                    "count": 3,
                    "compress": "deflate",
                })

            band_data = src.read(1, window=window)
            data_list.append(band_data)

    stacked = np.stack(data_list, axis=0)
    with rasterio.open(out_file, "w", **meta) as dst:
        dst.write(stacked)
        for i, name in enumerate(["Red", "Green", "Blue"], start=1):
            dst.set_band_description(i, name)


def fetch_aoi_sample(
    bbox: Optional[List[float]] = None,
    max_cloud_cover: float = 15.0,
    output_path: str | Path = "data/raw/aoi_sample.tif",
) -> Path:
    """High-level function to query, crop, and save AOI scene."""
    if bbox is None:
        bbox = DEFAULT_AOI
    client = get_stac_client()
    item = search_clear_scene(client, bbox=bbox, max_cloud_cover=max_cloud_cover)
    return crop_scene_to_geotiff(item, bbox=bbox, output_path=output_path)


if __name__ == "__main__":
    fetch_aoi_sample()
