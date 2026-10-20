# SatQuery Vision Studio | Multimodal Geospatial & Satellite Vision Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/jitesh-ss/setquery-vision--main)
[![Author](https://img.shields.io/badge/Author-jitesh--ss-brightgreen)](https://github.com/jitesh-ss)

A high-performance geospatial data ingestion, windowing, and multimodal vision analysis studio powered by **Microsoft Planetary Computer STAC**, **Rasterio**, **GDAL**, **PyTorch**, and **FastAPI**.

- **Repository**: [https://github.com/jitesh-ss/setquery-vision--main.git](https://github.com/jitesh-ss/setquery-vision--main.git)
- **Author**: [jitesh-ss](https://github.com/jitesh-ss)
- **License**: [MIT](LICENSE)

---

## 🛰 Architecture Overview

```
Planetary Computer STAC (Sentinel-2 L2A)
               │
               ▼ (SAS token signed streaming)
      src/fetch_scene.py
               │
               ▼
     data/raw/aoi_sample.tif (EPSG:32644, 10m COG Crop)
               │
      ┌────────┴────────┬─────────────────────────┬──────────────────────┐
      ▼                 ▼                         ▼                      ▼
src/spectral.py   src/tile_generator.py     src/vision_model.py    src/vectorize.py
(NDVI/NDWI/CIR) (512x512, 384 stride)     (PyTorch Embeddings)   (Multi-Class Polygons)
      │                 │                         │                      │
      ▼                 ▼                         ▼                      ▼
Spectral PNGs     data/processed/tiles/     Similarity Matrix     data/vectors/features.geojson
```

---

## ✨ Features Added

1. **Multispectral & Radiometric Indices (`src/spectral.py`)**:
   - **True Color (RGB)**: High-fidelity natural color composite.
   - **NDVI Canopy Vigor**: Visual Atmospherically Resistant Index (VARI proxy) with perceptually uniform Viridis colormap.
   - **NDWI Hydrological Contrast**: Normalized Difference Water Index proxy highlighting river dynamics.
   - **Color Infrared (CIR)**: Simulates False-Color NIR contrast (vegetation glows scarlet red).
   - **Thermal/Reflectance Heatmap**: Highlights urban structures and built-up land.

2. **PyTorch Deep Vision Tile Embeddings (`src/vision_model.py`)**:
   - Multi-scale convolutional spatial-spectral encoder extracts 64-dim embeddings per 512x512 tile.
   - Cosine similarity ranking engine allows users to select any tile and instantly find visually similar landscape regions.

3. **Interactive Bounding Box Draw & Measurement Tool**:
   - Custom crosshair rectangle drawing directly on the Leaflet map.
   - Live geographic area calculation in square kilometers and hectares.
   - 1-click query population to search Microsoft Planetary Computer STAC for custom drawn extents.

4. **Dynamic Land Cover Analytics Donut Chart (Chart.js)**:
   - Interactive breakdown of Dense Vegetation (49.4%), Built-up Infrastructure (44.7%), and Water Bodies (5.9%).
   - Click any chart segment to isolate and highlight those vector polygons on the live map.

5. **Multi-Format Export Suite**:
   - **RFC 7946 GeoJSON**: Georeferenced vector polygons in `EPSG:4326`.
   - **Analytics CSV**: Tabular breakdown of feature IDs, classes, coordinates, and calculated areas.
   - **Tile Footprint Manifest**: JSON index of all 512x512 tile bounds and affine matrices.

---

## 🛠 Setup & Environment

Clone the repository and set up the Python 3.11 environment (managed via `uv`):

```powershell
# Clone the repository
git clone https://github.com/jitesh-ss/setquery-vision--main.git
cd setquery-vision-main

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Verified Dependencies
- **GDAL**: `3.10.3` (204 registered drivers)
- **Rasterio**: `1.4.4`
- **PySTAC Client**: `0.9.0`
- **Planetary Computer**: `1.0.0`
- **PyTorch**: `2.14.1+cpu`
- **Torchvision**: `0.29.1+cpu`
- **GeoPandas**: `1.2.0`
- **Shapely**: `2.2.0`
- **FastAPI**: `0.150.0`

---

## 🚀 Running the Fullstack Studio

Start the FastAPI backend with the interactive frontend:

```powershell
.venv\Scripts\uvicorn.exe src.api:app --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000/`**

---

## 🧪 Testing & Verification

Run the automated integration test suite:

```powershell
.venv\Scripts\python.exe -m unittest tests\test_pipeline.py
```

---

## 👤 Author

- **jitesh-ss** - [@jitesh-ss](https://github.com/jitesh-ss)
- **Repository**: [https://github.com/jitesh-ss/setquery-vision--main.git](https://github.com/jitesh-ss/setquery-vision--main.git)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
