"""Production FastAPI Backend for Multimodal Geospatial & Satellite Vision Assistant.

Serves:
- STAC Scene Search & Streaming from Planetary Computer
- GeoTIFF and Tile Web Image Overlays (PNG stream)
- GeoJSON Vector Features & Spatial Querying
- Multimodal Vision Assistant Chat & Reasoning
- Static Single-Page Geospatial Studio Frontend
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rasterio

from src.assistant import compute_geospatial_metrics, query_assistant
from src.fetch_scene import DEFAULT_AOI, fetch_aoi_sample, get_stac_client, search_clear_scene
from src.preview_utils import get_raster_wgs84_bounds, raster_to_png_bytes
from src.spectral import compute_spectral_index_image
from src.tile_generator import slice_raster_tiles
from src.vectorize import vectorize_raster_to_geojson
from src.vision_model import analyze_all_tiles, find_similar_tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.api")

app = FastAPI(
    title="Multimodal Geospatial & Satellite Vision Assistant API",
    version="1.0.0",
    description="Fullstack Geospatial & Satellite Vision Assistant powered by Planetary Computer and Rasterio.",
)

# Enable CORS for development and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---

class AOIRequest(BaseModel):
    bbox: List[float] = Field(
        default=DEFAULT_AOI,
        description="Bounding box [minx, miny, maxx, maxy] in EPSG:4326",
    )
    max_cloud_cover: float = Field(default=15.0, ge=0.0, le=100.0)


class TileRequest(BaseModel):
    tile_size: int = Field(default=512, ge=64, le=2048)
    stride: int = Field(default=384, ge=32, le=2048)


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User inquiry regarding geospatial features or analysis")


class PipelineRequest(BaseModel):
    bbox: List[float] = Field(default=DEFAULT_AOI)
    max_cloud_cover: float = Field(default=15.0)
    tile_size: int = Field(default=512)
    stride: int = Field(default=384)
    force_refetch: bool = Field(default=False)


class SimilarityRequest(BaseModel):
    tile_name: str = Field(..., description="Target tile filename to query similarity against")


# --- API Routes ---

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Verify backend status and geospatial driver health."""
    import rasterio
    import torch
    return {
        "status": "healthy",
        "service": "Multimodal Geospatial & Satellite Vision Assistant",
        "rasterio_version": rasterio.__version__,
        "gdal_version": rasterio.__gdal_version__,
        "torch_version": torch.__version__,
    }


@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    """Return live spatial metrics, class coverage, and summary statistics."""
    return compute_geospatial_metrics()


@app.get("/api/stac/search")
def search_stac(
    minx: float = Query(80.48),
    miny: float = Query(16.48),
    maxx: float = Query(80.58),
    maxy: float = Query(16.56),
    cloud_cover: float = Query(25.0),
    limit: int = Query(6),
) -> Dict[str, Any]:
    """Query Microsoft Planetary Computer STAC for candidate Sentinel-2 scenes."""
    try:
        bbox = [minx, miny, maxx, maxy]
        client = get_stac_client()
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=limit,
        )
        items = list(search.items())
        scenes = []
        for item in items:
            assets = item.assets
            thumbnail = ""
            if "rendered_preview" in assets:
                thumbnail = assets["rendered_preview"].href
            elif "thumbnail" in assets:
                thumbnail = assets["thumbnail"].href

            scenes.append({
                "id": item.id,
                "datetime": item.properties.get("datetime"),
                "cloud_cover": round(item.properties.get("eo:cloud_cover", 0.0), 2),
                "platform": item.properties.get("platform", "Sentinel-2"),
                "thumbnail": thumbnail,
                "bbox": item.bbox,
            })
        return {"status": "success", "count": len(scenes), "scenes": scenes}
    except Exception as exc:
        logger.error("STAC search error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/pipeline/run")
def trigger_pipeline(payload: PipelineRequest) -> Dict[str, Any]:
    """Execute end-to-end ingestion, tiling, and vectorization."""
    try:
        raw_path = Path("data/raw/aoi_sample.tif")
        if payload.force_refetch or not raw_path.exists():
            fetch_aoi_sample(
                bbox=payload.bbox,
                max_cloud_cover=payload.max_cloud_cover,
                output_path=raw_path,
            )

        tiles = slice_raster_tiles(
            input_geotiff=raw_path,
            output_dir="data/processed/tiles",
            tile_size=payload.tile_size,
            stride=payload.stride,
        )

        vec_path = Path("data/vectors/features.geojson")
        vectorize_raster_to_geojson(raw_path, vec_path)

        metrics = compute_geospatial_metrics()
        return {
            "status": "success",
            "message": "Pipeline completed successfully",
            "tiles_count": len(tiles),
            "metrics": metrics,
        }
    except Exception as exc:
        logger.error("Pipeline execution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/raster/info")
def get_raster_info() -> Dict[str, Any]:
    """Retrieve spatial metadata and Leaflet-compatible WGS84 overlay bounds."""
    raw_path = Path("data/raw/aoi_sample.tif")
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw GeoTIFF not found. Run pipeline first.")

    with rasterio.open(raw_path) as src:
        w, h = src.width, src.height
        bounds_native = {
            "left": src.bounds.left,
            "bottom": src.bounds.bottom,
            "right": src.bounds.right,
            "top": src.bounds.top,
        }
        crs = str(src.crs)
        transform = list(src.transform)[:6]

    wgs84_bounds = get_raster_wgs84_bounds(raw_path)
    return {
        "status": "success",
        "width": w,
        "height": h,
        "crs": crs,
        "native_bounds": bounds_native,
        "transform": transform,
        "leaflet_bounds": wgs84_bounds,  # [[south, west], [north, east]]
        "preview_url": "/api/raster/preview",
    }


@app.get("/api/raster/preview")
def get_raster_preview() -> Response:
    """Stream an RGB PNG preview of the cropped raw satellite GeoTIFF."""
    raw_path = Path("data/raw/aoi_sample.tif")
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw GeoTIFF not found")

    png_bytes = raster_to_png_bytes(raw_path)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/raster/spectral")
def get_spectral_preview(mode: str = Query("true_color")) -> Response:
    """Stream colorized spectral or radiometric index PNG (true_color, ndvi, ndwi, false_color, heat)."""
    raw_path = Path("data/raw/aoi_sample.tif")
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw GeoTIFF not found")

    try:
        png_bytes = compute_spectral_index_image(raw_path, mode=mode)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/vision/tiles")
def get_vision_tiles() -> Dict[str, Any]:
    """Return PyTorch deep feature embeddings and terrain classification for all tiles."""
    try:
        tiles_analysis = analyze_all_tiles("data/processed/tiles")
        return {"status": "success", "count": len(tiles_analysis), "tiles": tiles_analysis}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/vision/similarity")
def query_tile_similarity(payload: SimilarityRequest) -> Dict[str, Any]:
    """Find and rank tiles by PyTorch cosine visual similarity."""
    try:
        results = find_similar_tiles(payload.tile_name, "data/processed/tiles")
        return {"status": "success", "target_tile": payload.tile_name, "similar_tiles": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/analytics/export/csv")
def export_analytics_csv() -> Response:
    """Export vector features and spatial land cover statistics as CSV."""
    vec_path = Path("data/vectors/features.geojson")
    if not vec_path.exists():
        raise HTTPException(status_code=404, detail="No vectors found to export.")

    with open(vec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = ["Feature_ID,Class_Name,Class_ID,Area_Sq_Degrees,Approx_Area_M2,CRS,Source_Raster"]
    features = data.get("features", [])
    for idx, feat in enumerate(features, start=1):
        props = feat.get("properties", {})
        cname = props.get("class_name", "unclassified")
        cid = props.get("class_id", 0)
        area_deg = props.get("area_sq_degrees", 0.0)
        approx_m2 = round(area_deg * (111000.0 * 106000.0), 2)
        source = props.get("source_raster", "aoi_sample.tif")
        lines.append(f"{idx},{cname},{cid},{area_deg},{approx_m2},EPSG:4326,{source}")

    csv_content = "\n".join(lines)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=satquery_analytics_features.csv"},
    )


@app.get("/api/tiles")
def get_tiles() -> Dict[str, Any]:
    """Retrieve list of generated tiles with their WGS84 bounding footprints."""
    manifest_path = Path("data/processed/tiles/tiles_manifest.json")
    if not manifest_path.exists():
        return {"status": "empty", "count": 0, "tiles": []}

    with open(manifest_path, "r", encoding="utf-8") as f:
        tiles = json.load(f)

    # Attach preview URL to each tile entry
    for t in tiles:
        tile_filename = Path(t["tile_file"]).name
        t["preview_url"] = f"/api/tiles/{tile_filename}/preview"

    return {"status": "success", "count": len(tiles), "tiles": tiles}


@app.get("/api/tiles/{tile_filename}/preview")
def get_tile_preview(tile_filename: str) -> Response:
    """Stream an RGB PNG preview for a specific tile."""
    tile_path = Path("data/processed/tiles") / tile_filename
    if not tile_path.exists():
        raise HTTPException(status_code=404, detail=f"Tile {tile_filename} not found")

    png_bytes = raster_to_png_bytes(tile_path)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/api/vectors")
def get_vectors() -> Any:
    """Retrieve vectorized GeoJSON features and metadata summary."""
    vec_path = Path("data/vectors/features.geojson")
    if not vec_path.exists():
        raw_path = Path("data/raw/aoi_sample.tif")
        if raw_path.exists():
            vectorize_raster_to_geojson(raw_path, vec_path)
        else:
            raise HTTPException(status_code=404, detail="No vectors found. Ingest a scene first.")

    with open(vec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@app.post("/api/assistant/chat")
def chat_with_assistant(payload: ChatRequest) -> Dict[str, Any]:
    """Consult the multimodal geospatial assistant with domain reasoning."""
    return query_assistant(payload.prompt)


# --- Static Files Mount (Frontend Web App) ---
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
