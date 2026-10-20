# Geospatial Satellite Assistant Rules

- **Python Environment**: Use Python 3.10+ virtual environment (`.venv`).
- **Authentication & Data Access**: Never hardcode credentials; use Microsoft Planetary Computer STAC endpoint (`https://planetarycomputer.microsoft.com/api/stac/v1`) with `pystac-client` and `planetary-computer` signing.
- **Raster Processing**: Always retain Affine transform matrices when slicing raster windows (e.g., using `rasterio.windows.transform` or `rioxarray`).
- **Vector Formats**: Vector outputs must be valid GeoJSON in EPSG:4326 (`{"type": "FeatureCollection", ...}`).
- **Testing & Verification**: For tests and verification, run automated integration scripts before reporting task completion.
