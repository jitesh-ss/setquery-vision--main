# Geospatial Satellite Assistant Rules

## Python Virtual Environment
- Always run and install within a Python 3.10+ virtual environment (`.venv`).

## Credentials & STAC Catalog
- Never hardcode credentials.
- Use Microsoft Planetary Computer STAC endpoint: `https://planetarycomputer.microsoft.com/api/stac/v1`
- Use `pystac-client` and `planetary-computer` (e.g. `planetary_computer.sign_inplace` or `sign`) for authenticated asset access.

## Raster Processing & Affine Transformations
- Always preserve and accurately update Affine transform matrices when slicing, cropping, or windowing rasters.
- Use `rasterio.windows.transform(window, src.transform)` to compute the new transform for cropped windows.

## Vector Outputs
- All vector outputs must conform to RFC 7946 GeoJSON format in WGS84 coordinates (`EPSG:4326`).

## Verification
- Before reporting task completion, execute automated integration and verification scripts that test data loading, transforms, and outputs.
