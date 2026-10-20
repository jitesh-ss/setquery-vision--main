"""Vectorization & GeoJSON Export Module.

Extracts polygonal vector features from raster classification masks, reprojects
geometries to standard WGS84 (EPSG:4326), attaches rich semantic metadata,
and exports valid GeoJSON FeatureCollections per RFC 7946.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyproj
import rasterio
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape as to_shapely_shape, mapping
from shapely.ops import transform as shapely_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reproject_geometry_to_epsg4326(geom_shape: Any, src_crs: Any) -> Any:
    """Reproject a Shapely geometry from source CRS to EPSG:4326 (WGS84)."""
    if str(src_crs).upper() in ("EPSG:4326", "WGS 84", "4326"):
        return geom_shape

    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform
    return shapely_transform(transformer, geom_shape)


def mask_to_geojson_features(
    mask: np.ndarray,
    transform: rasterio.Affine,
    crs: Any,
    class_mapping: Optional[Dict[int, str]] = None,
    min_pixel_area: int = 16,
    properties_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert a categorical or binary 2D mask into a list of GeoJSON Features.

    Args:
        mask: 2D integer numpy array with class labels.
        transform: Affine transform matrix of the raster.
        crs: Source Coordinate Reference System.
        class_mapping: Mapping from integer label to human-readable class name.
        min_pixel_area: Minimum area in pixels to filter out noise polygons.
        properties_extra: Additional key-value pairs to include in feature properties.

    Returns:
        List of GeoJSON Feature dicts with EPSG:4326 coordinates.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    if class_mapping is None:
        class_mapping = {1: "detected_feature"}

    features: List[Dict[str, Any]] = []
    extra = properties_extra or {}

    # Extract polygon shapes using rasterio
    extracted_shapes = raster_shapes(mask.astype(np.int32), mask=(mask > 0), transform=transform)

    for geom_dict, val in extracted_shapes:
        class_val = int(val)
        if class_val == 0 or class_val not in class_mapping:
            continue

        raw_geom = to_shapely_shape(geom_dict)
        if raw_geom.is_empty or not raw_geom.is_valid:
            raw_geom = raw_geom.buffer(0)
            if raw_geom.is_empty:
                continue

        # Filter out tiny noise polygons
        if raw_geom.area < (min_pixel_area * (abs(transform.a) * abs(transform.e))):
            continue

        # Reproject to EPSG:4326
        wgs84_geom = reproject_geometry_to_epsg4326(raw_geom, crs)
        if wgs84_geom.is_empty:
            continue

        feature_props = {
            "class_id": class_val,
            "class_name": class_mapping.get(class_val, f"class_{class_val}"),
            "area_sq_degrees": round(wgs84_geom.area, 8),
            **extra,
        }

        feature = {
            "type": "Feature",
            "geometry": mapping(wgs84_geom),
            "properties": feature_props,
        }
        features.append(feature)

    return features


def extract_spectral_features(
    rgb_image: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Extract semantic feature mask using multispectral/color vision indices.

    Classes:
        0: Background / Unclassified
        1: Water Body (River / Water Reservoir)
        2: Dense Vegetation (Cropland / Canopy)
        3: Built-up / High Reflectance
    """
    if rgb_image.ndim == 3 and rgb_image.shape[0] in (3, 4):
        # Channels-first (C, H, W) -> convert to float
        r = rgb_image[0].astype(np.float32)
        g = rgb_image[1].astype(np.float32)
        b = rgb_image[2].astype(np.float32)
    elif rgb_image.ndim == 3 and rgb_image.shape[2] in (3, 4):
        # Channels-last (H, W, C)
        r = rgb_image[:, :, 0].astype(np.float32)
        g = rgb_image[:, :, 1].astype(np.float32)
        b = rgb_image[:, :, 2].astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {rgb_image.shape}")

    total_intensity = r + g + b + 1e-6
    # Normalized color channels
    norm_r = r / total_intensity
    norm_g = g / total_intensity
    norm_b = b / total_intensity

    # Excess Green Index (ExG): 2*G - R - B
    exg = (2.0 * norm_g) - norm_r - norm_b

    # Water heuristic in True-Color: relatively high Blue & Green, low Red, low overall intensity
    is_water = (b > r) & (g > r) & (total_intensity < np.percentile(total_intensity, 40))

    # Dense vegetation: strong positive green excess
    is_vegetation = (exg > 0.08) & (~is_water)

    # Built-up / High reflectance: bright pixels
    is_builtup = (total_intensity > np.percentile(total_intensity, 80)) & (~is_vegetation) & (~is_water)

    mask = np.zeros(r.shape, dtype=np.int32)
    mask[is_water] = 1
    mask[is_vegetation] = 2
    mask[is_builtup] = 3

    class_mapping = {
        1: "water_body",
        2: "dense_vegetation",
        3: "built_up_infrastructure",
    }
    return mask, class_mapping


def vectorize_raster_to_geojson(
    input_raster: str | Path,
    output_geojson: str | Path = "data/vectors/features.geojson",
    min_pixel_area: int = 25,
) -> Path:
    """Process a GeoTIFF, extract semantic features, and export valid EPSG:4326 GeoJSON."""
    in_path = Path(input_raster)
    out_path = Path(output_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

        mask, class_mapping = extract_spectral_features(data)

        features = mask_to_geojson_features(
            mask=mask,
            transform=transform,
            crs=crs,
            class_mapping=class_mapping,
            min_pixel_area=min_pixel_area,
            properties_extra={"source_raster": in_path.name},
        )

    geojson_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)

    logger.info("Exported %d vector features to valid GeoJSON: %s", len(features), out_path)
    return out_path


if __name__ == "__main__":
    vectorize_raster_to_geojson("data/raw/aoi_sample.tif", "data/vectors/features.geojson")
