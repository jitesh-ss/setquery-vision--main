"""End-to-End Multimodal Geospatial Data and Vision Pipeline.

Coordinates STAC scene retrieval, tile windowing with affine transform
preservation, and polygon vectorization to RFC 7946 GeoJSON in EPSG:4326.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.fetch_scene import DEFAULT_AOI, fetch_aoi_sample
from src.tile_generator import slice_raster_tiles
from src.vectorize import vectorize_raster_to_geojson

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.pipeline")


def run_pipeline(
    bbox: List[float] = DEFAULT_AOI,
    max_cloud_cover: float = 15.0,
    raw_geotiff_path: str | Path = "data/raw/aoi_sample.tif",
    tiles_output_dir: str | Path = "data/processed/tiles",
    vector_output_path: str | Path = "data/vectors/features.geojson",
    tile_size: int = 512,
    stride: int = 384,
    skip_fetch: bool = False,
) -> Dict[str, Any]:
    """Execute the complete satellite data and inference pipeline."""
    raw_path = Path(raw_geotiff_path)
    tiles_dir = Path(tiles_output_dir)
    vec_path = Path(vector_output_path)

    # Step 1: STAC Ingestion
    if not skip_fetch or not raw_path.exists():
        logger.info("--- Step 1: Ingesting AOI Scene from Planetary Computer STAC ---")
        raw_path = fetch_aoi_sample(
            bbox=bbox,
            max_cloud_cover=max_cloud_cover,
            output_path=raw_path,
        )
    else:
        logger.info("--- Step 1: Using existing raw GeoTIFF at %s ---", raw_path)

    # Step 2: Tiling & Preprocessing
    logger.info("--- Step 2: Slicing into %dx%d tiles (stride=%d) ---", tile_size, tile_size, stride)
    tiles = slice_raster_tiles(
        input_geotiff=raw_path,
        output_dir=tiles_dir,
        tile_size=tile_size,
        stride=stride,
    )

    # Step 3: Vectorization & GeoJSON Export
    logger.info("--- Step 3: Vectorizing semantic features to GeoJSON (EPSG:4326) ---")
    geojson_file = vectorize_raster_to_geojson(
        input_raster=raw_path,
        output_geojson=vec_path,
    )

    summary = {
        "raw_geotiff": str(raw_path.resolve()),
        "tiles_count": len(tiles),
        "tiles_dir": str(tiles_dir.resolve()),
        "geojson_file": str(geojson_file.resolve()),
    }
    logger.info("Pipeline completed successfully! Summary: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multimodal Geospatial & Satellite Vision Assistant Pipeline")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_AOI, help="AOI bbox: minx miny maxx maxy in EPSG:4326")
    parser.add_argument("--cloud", type=float, default=15.0, help="Max cloud cover percentage")
    parser.add_argument("--raw", default="data/raw/aoi_sample.tif", help="Output raw GeoTIFF path")
    parser.add_argument("--tiles-dir", default="data/processed/tiles", help="Output directory for tiles")
    parser.add_argument("--geojson", default="data/vectors/features.geojson", help="Output GeoJSON path")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile pixel size")
    parser.add_argument("--stride", type=int, default=384, help="Stride step in pixels")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip STAC fetch if raw GeoTIFF exists")

    args = parser.parse_args()
    run_pipeline(
        bbox=args.bbox,
        max_cloud_cover=args.cloud,
        raw_geotiff_path=args.raw,
        tiles_output_dir=args.tiles_dir,
        vector_output_path=args.geojson,
        tile_size=args.tile_size,
        stride=args.stride,
        skip_fetch=args.skip_fetch,
    )


if __name__ == "__main__":
    main()
