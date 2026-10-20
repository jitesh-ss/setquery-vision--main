"""Summary Inspection of Generated Artifacts on Disk."""

import json
from pathlib import Path
import rasterio

def inspect():
    raw_path = Path("data/raw/aoi_sample.tif")
    tiles_dir = Path("data/processed/tiles")
    geojson_path = Path("data/vectors/features.geojson")

    print("=== 1. Raw GeoTIFF Inspection ===")
    if raw_path.exists():
        with rasterio.open(raw_path) as src:
            print(f"Path: {raw_path}")
            print(f"File Size: {raw_path.stat().st_size / (1024 * 1024):.2f} MB")
            print(f"Dimensions: Width={src.width}, Height={src.height}, Bands={src.count}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Affine Transform: {src.transform}")
            print(f"Pixel Resolution (X, Y): ({src.transform.a}, {abs(src.transform.e)}) meters")
    else:
        print("Raw GeoTIFF not found!")

    print("\n=== 2. Processed Tiles Inspection ===")
    if tiles_dir.exists():
        tiles = list(tiles_dir.glob("tile_*.tif"))
        manifest_file = tiles_dir / "tiles_manifest.json"
        print(f"Directory: {tiles_dir}")
        print(f"Total Tiles Created: {len(tiles)}")
        if manifest_file.exists():
            with open(manifest_file, "r") as f:
                manifest = json.load(f)
            print(f"Manifest Tiles Count: {len(manifest)}")
        for t in tiles[:3]:
            with rasterio.open(t) as src:
                print(f"  - {t.name}: size={src.width}x{src.height}, crs={src.crs}, transform={src.transform}")
    else:
        print("Tiles directory not found!")

    print("\n=== 3. GeoJSON Vector Export Inspection ===")
    if geojson_path.exists():
        print(f"Path: {geojson_path}")
        print(f"File Size: {geojson_path.stat().st_size / 1024:.2f} KB")
        with open(geojson_path, "r", encoding="utf-8") as f:
            fc = json.load(f)
        features = fc.get("features", [])
        print(f"GeoJSON Type: {fc.get('type')}")
        print(f"CRS: {fc.get('crs', {}).get('properties', {}).get('name')}")
        print(f"Total Vector Features: {len(features)}")
        # Class distribution
        class_counts = {}
        for feat in features:
            c = feat["properties"]["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1
        print("Feature Classes Distribution:", class_counts)
        if features:
            print("Sample Feature Properties:", json.dumps(features[0]["properties"], indent=2))
            sample_coords = features[0]["geometry"]["coordinates"][0][:2]
            print("Sample Coordinates (Lon, Lat):", sample_coords)
    else:
        print("GeoJSON not found!")

if __name__ == "__main__":
    inspect()
