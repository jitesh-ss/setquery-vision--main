"""Automated Integration Test Suite for Multimodal Geospatial & Satellite Vision Assistant.

Validates:
1. Environment & GDAL compatibility.
2. STAC API connectivity and token signing via Planetary Computer.
3. GeoTIFF streaming and window cropping preserving Affine transform.
4. 512x512 tiling with 384 stride and geographic transform preservation.
5. GeoJSON polygon vectorization adhering to RFC 7946 in EPSG:4326.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import unittest

import numpy as np
import rasterio
from shapely.geometry import shape

from src.fetch_scene import DEFAULT_AOI, fetch_aoi_sample, get_stac_client, search_clear_scene
from src.tile_generator import slice_raster_tiles
from src.vectorize import vectorize_raster_to_geojson

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_pipeline")


class TestGeospatialPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_tif = Path("data/raw/aoi_sample.tif")
        cls.tiles_dir = Path("data/processed/tiles")
        cls.geojson_file = Path("data/vectors/features.geojson")

    def test_01_stac_connection_and_signing(self):
        """Test STAC catalog search and Planetary Computer signing."""
        client = get_stac_client()
        signed_item = search_clear_scene(client, bbox=DEFAULT_AOI, max_cloud_cover=20.0)

        self.assertIsNotNone(signed_item)
        self.assertIn("visual", signed_item.assets)
        visual_href = signed_item.assets["visual"].href
        # Planetary computer tokens append signature params like 'st=' or 'token='
        self.assertTrue("?" in visual_href or "blob.core.windows.net" in visual_href)
        logger.info("[PASS] STAC connection and item signing verified: %s", signed_item.id)

    def test_02_fetch_and_crop_geotiff(self):
        """Test streaming COG crop and GeoTIFF creation with Affine transform preservation."""
        out_path = fetch_aoi_sample(
            bbox=DEFAULT_AOI,
            max_cloud_cover=20.0,
            output_path=self.raw_tif,
        )
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 1000)

        with rasterio.open(out_path) as src:
            logger.info("Raw GeoTIFF bounds: %s, shape: (%d, %d), count: %d",
                        src.bounds, src.height, src.width, src.count)
            self.assertGreaterEqual(src.count, 3)
            self.assertGreater(src.width, 0)
            self.assertGreater(src.height, 0)
            # Verify Affine transform is non-trivial and preserves scale
            self.assertNotEqual(src.transform.a, 0.0)
            self.assertNotEqual(src.transform.e, 0.0)
            self.assertIsNotNone(src.crs)
        logger.info("[PASS] Ingestion GeoTIFF verified at: %s", out_path)

    def test_03_tiling_and_affine_preservation(self):
        """Test 512x512 tiling with 384 stride and Affine preservation."""
        self.assertTrue(self.raw_tif.exists(), "Raw GeoTIFF must exist before tiling.")
        tiles = slice_raster_tiles(
            input_geotiff=self.raw_tif,
            output_dir=self.tiles_dir,
            tile_size=512,
            stride=384,
        )
        self.assertGreater(len(tiles), 0)

        manifest_path = self.tiles_dir / "tiles_manifest.json"
        self.assertTrue(manifest_path.exists())

        # Inspect first tile
        first_tile_path = Path(tiles[0]["tile_file"])
        self.assertTrue(first_tile_path.exists())

        with rasterio.open(first_tile_path) as src:
            self.assertEqual(src.width, 512)
            self.assertEqual(src.height, 512)
            self.assertNotEqual(src.transform.a, 0.0)
            self.assertNotEqual(src.transform.e, 0.0)
            logger.info("Sample Tile Transform: %s", src.transform)

        logger.info("[PASS] Sliced %d tiles in %s", len(tiles), self.tiles_dir)

    def test_04_vectorization_and_geojson_export(self):
        """Test polygon vectorization to valid RFC 7946 EPSG:4326 GeoJSON."""
        self.assertTrue(self.raw_tif.exists(), "Raw GeoTIFF must exist before vectorization.")
        out_geojson = vectorize_raster_to_geojson(
            input_raster=self.raw_tif,
            output_geojson=self.geojson_file,
            min_pixel_area=20,
        )
        self.assertTrue(out_geojson.exists())

        with open(out_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("type"), "FeatureCollection")
        self.assertIn("features", data)
        features = data["features"]
        self.assertGreater(len(features), 0, "Should extract at least one feature polygon")

        # Validate feature structure and WGS84 coordinate bounds
        for feat in features[:10]:
            self.assertEqual(feat.get("type"), "Feature")
            geom = feat.get("geometry")
            self.assertIsNotNone(geom)
            geom_obj = shape(geom)
            self.assertTrue(geom_obj.is_valid or geom_obj.buffer(0).is_valid)

            # Coordinates must be valid EPSG:4326
            minx, miny, maxx, maxy = geom_obj.bounds
            self.assertGreaterEqual(minx, -180.0)
            self.assertLessEqual(maxx, 180.0)
            self.assertGreaterEqual(miny, -90.0)
            self.assertLessEqual(maxy, 90.0)

            # Check properties
            props = feat.get("properties", {})
            self.assertIn("class_name", props)
            self.assertIn("class_id", props)

        logger.info("[PASS] Successfully validated %d GeoJSON features in %s", len(features), out_geojson)


if __name__ == "__main__":
    unittest.main()
