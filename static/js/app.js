/**
 * SatQuery Vision Studio - Client Application Logic
 * Interactive Leaflet Map, STAC Catalog, Tiling Inspector & Multimodal AI Assistant
 */

// Application State
const state = {
  map: null,
  basemaps: {},
  currentBasemap: 'light',
  rasterLayer: null,
  tileGridLayer: null,
  vectorLayer: null,
  rasterInfo: null,
  tilesData: [],
  geojsonData: null,
  metrics: null,
  activeHighlights: [],
  spectralMode: 'true_color',
  isDrawingAOI: false,
  currentInspectedTile: null,
  chart: null,
};

// Preset AOIs
const AOI_PRESETS = {
  krishna: { name: "Krishna River Basin, AP", bbox: [80.48, 16.48, 80.58, 16.56] },
  suez: { name: "Suez Canal, Egypt", bbox: [32.30, 30.50, 32.40, 30.60] },
  ganges: { name: "Ganges Delta, Sundarbans", bbox: [89.10, 21.80, 89.25, 21.95] },
  bayarea: { name: "San Francisco Bay, USA", bbox: [-122.45, 37.75, -122.35, 37.85] },
};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupUIEventListeners();
  loadRasterOverlay();
  loadTileGrid();
  loadVectorFeatures();
  loadMetrics();
  searchStacScenes();
});

/* ==========================================================================
   1. Leaflet Interactive Map Setup
   ========================================================================== */
function initMap() {
  state.map = L.map('map', {
    center: [16.52, 80.53],
    zoom: 13,
    zoomControl: false,
  });

  // Basemaps: Light (default), Dark, and Satellite
  state.basemaps.light = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19,
    subdomains: 'abcd',
  }).addTo(state.map);

  state.basemaps.dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    maxZoom: 19,
    subdomains: 'abcd',
  });

  state.basemaps.satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; Esri &copy; Earthstar Geographics',
    maxZoom: 19,
  });

  state.currentBasemap = 'light';

  // Layer Groups
  state.tileGridLayer = L.layerGroup().addTo(state.map);
  state.vectorLayer = L.layerGroup().addTo(state.map);

  // Mouse coordinate readout
  state.map.on('mousemove', (e) => {
    const lat = e.latlng.lat.toFixed(4);
    const lng = e.latlng.lng.toFixed(4);
    const zoom = state.map.getZoom();
    const display = document.getElementById('coordsDisplay');
    if (display) {
      display.textContent = `${Math.abs(lat)}° ${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lng)}° ${lng >= 0 ? 'E' : 'W'} | Zoom ${zoom}`;
    }
  });
}

/* ==========================================================================
   2. Raster Overlay Loader (Sentinel-2 COG Preview)
   ========================================================================== */
async function loadRasterOverlay(mode = 'true_color') {
  try {
    if (!state.rasterInfo) {
      const res = await fetch('/api/raster/info');
      if (!res.ok) return;
      state.rasterInfo = await res.json();
    }
    const info = state.rasterInfo;
    const bounds = info.leaflet_bounds; // [[south, west], [north, east]]
    const opacity = (document.getElementById('rasterOpacity')?.value || 90) / 100;

    if (state.rasterLayer) {
      state.map.removeLayer(state.rasterLayer);
    }

    const previewUrl = (mode === 'true_color') ? '/api/raster/preview' : `/api/raster/spectral?mode=${mode}`;
    state.rasterLayer = L.imageOverlay(previewUrl, bounds, {
      opacity: opacity,
      interactive: true,
    }).addTo(state.map);

    // Update HUD
    const hudArea = document.getElementById('hudArea');
    if (hudArea && state.metrics) {
      hudArea.textContent = `${state.metrics.total_area_km2} km²`;
    }
  } catch (err) {
    console.error("Error loading raster overlay:", err);
  }
}

/* ==========================================================================
   3. Tile Grid Loader (512x512 with 384 Stride)
   ========================================================================== */
async function loadTileGrid() {
  try {
    const res = await fetch('/api/tiles');
    if (!res.ok) return;
    const data = await res.json();
    state.tilesData = data.tiles || [];

    state.tileGridLayer.clearLayers();
    const container = document.getElementById('tilesListContainer');
    if (container) container.innerHTML = '';

    const badge = document.getElementById('tilesCountBadge');
    if (badge) badge.textContent = `${state.tilesData.length} Tiles`;
    const hudTiles = document.getElementById('hudTiles');
    if (hudTiles) hudTiles.textContent = `${state.tilesData.length} (512²)`;

    state.tilesData.forEach((tile) => {
      const b = tile.bounds_wgs84; // [minx, miny, maxx, maxy]
      const bounds = [[b[1], b[0]], [b[3], b[2]]]; // [[south, west], [north, east]]

      // Draw bounding box on map
      const rect = L.rectangle(bounds, {
        color: '#0284c7',
        weight: 1.8,
        dashArray: '5, 5',
        fillColor: '#0284c7',
        fillOpacity: 0.05,
      });

      rect.bindTooltip(`<strong>Tile ${tile.row_index}_${tile.col_index}</strong><br>512x512 px (stride 384)`, {
        sticky: true,
        className: 'leaflet-custom-tooltip',
      });

      rect.on('click', () => openTileModal(tile));
      rect.addTo(state.tileGridLayer);

      // Add to Sidebar
      if (container) {
        const card = document.createElement('div');
        card.className = 'tile-card';
        card.innerHTML = `
          <div class="tile-thumb">
            <img src="${tile.preview_url}" alt="Tile r${tile.row_index} c${tile.col_index}" loading="lazy">
          </div>
          <span class="tile-name">Tile r${tile.row_index}_c${tile.col_index}</span>
          <span class="tile-coords">${tile.pixel_x}, ${tile.pixel_y} px</span>
        `;
        card.addEventListener('click', () => {
          state.map.fitBounds(bounds, { padding: [50, 50] });
          openTileModal(tile);
        });
        container.appendChild(card);
      }
    });
  } catch (err) {
    console.error("Error loading tile grid:", err);
  }
}

/* ==========================================================================
   4. Vector GeoJSON Feature Loader (RFC 7946 in EPSG:4326)
   ========================================================================== */
async function loadVectorFeatures() {
  try {
    const res = await fetch('/api/vectors');
    if (!res.ok) return;
    const geojson = await res.json();
    state.geojsonData = geojson;

    state.vectorLayer.clearLayers();
    const features = geojson.features || [];

    const hudFeatures = document.getElementById('hudFeatures');
    if (hudFeatures) hudFeatures.textContent = features.length.toLocaleString();

    let countVeg = 0, countInfra = 0, countWater = 0;

    const geoLayer = L.geoJSON(geojson, {
      style: (feature) => {
        const cname = feature.properties.class_name;
        if (cname === 'water_body') {
          return { color: '#0284c7', fillColor: '#0ea5e9', weight: 1.4, fillOpacity: 0.65 };
        } else if (cname === 'dense_vegetation') {
          return { color: '#059669', fillColor: '#10b981', weight: 1.2, fillOpacity: 0.60 };
        } else {
          return { color: '#ea580c', fillColor: '#f97316', weight: 1.2, fillOpacity: 0.60 };
        }
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        const cname = p.class_name;
        if (cname === 'water_body') countWater++;
        else if (cname === 'dense_vegetation') countVeg++;
        else countInfra++;

        const popupHtml = `
          <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #0f172a; background: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); min-width: 140px;">
            <strong style="color: #0284c7; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em;">${cname.replace('_', ' ')}</strong><br>
            <div style="margin-top: 4px; display: flex; justify-content: space-between;"><span>Class ID:</span> <code style="background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-weight: 600;">${p.class_id}</code></div>
            <div style="margin-top: 2px; display: flex; justify-content: space-between;"><span>Area:</span> <code style="background: #f1f5f9; padding: 1px 5px; border-radius: 4px;">${(p.area_sq_degrees * 1e6).toFixed(2)} µdeg²</code></div>
            <div style="margin-top: 2px; display: flex; justify-content: space-between;"><span>CRS:</span> <code>EPSG:4326</code></div>
          </div>
        `;
        layer.bindPopup(popupHtml);
      }
    });

    geoLayer.addTo(state.vectorLayer);

    // Update badges
    const cVeg = document.getElementById('countVeg');
    const cInfra = document.getElementById('countInfra');
    const cWater = document.getElementById('countWater');
    if (cVeg) cVeg.textContent = countVeg;
    if (cInfra) cInfra.textContent = countInfra;
    if (cWater) cWater.textContent = countWater;

  } catch (err) {
    console.error("Error loading vectors:", err);
  }
}

/* ==========================================================================
   5. Metrics Summary Loader
   ========================================================================== */
async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    state.metrics = await res.json();
    const hudArea = document.getElementById('hudArea');
    if (hudArea && state.metrics.total_area_km2 > 0) {
      hudArea.textContent = `${state.metrics.total_area_km2} km²`;
    }
    const chartTag = document.getElementById('chartTotalKm2');
    if (chartTag && state.metrics.total_area_km2 > 0) {
      chartTag.textContent = `${state.metrics.total_area_km2} km²`;
    }
    renderLandCoverChart(state.metrics);
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

/* ==========================================================================
   6. STAC Catalog Search Runner
   ========================================================================== */
async function searchStacScenes() {
  const minx = document.getElementById('bboxMinX')?.value || 80.48;
  const miny = document.getElementById('bboxMinY')?.value || 16.48;
  const maxx = document.getElementById('bboxMaxX')?.value || 80.58;
  const maxy = document.getElementById('bboxMaxY')?.value || 16.56;
  const cloud = document.getElementById('cloudCoverSlider')?.value || 15;

  const listContainer = document.getElementById('stacScenesList');
  if (listContainer) {
    listContainer.innerHTML = '<div style="color:#94a3b8; font-size:0.8rem; text-align:center; padding: 20px;"><i class="fa-solid fa-spinner fa-spin"></i> Querying Planetary Computer STAC...</div>';
  }

  try {
    const url = `/api/stac/search?minx=${minx}&miny=${miny}&maxx=${maxx}&maxy=${maxy}&cloud_cover=${cloud}&limit=5`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("STAC query failed");
    const data = await res.json();

    if (listContainer) {
      listContainer.innerHTML = '';
      const scenes = data.scenes || [];
      const badge = document.getElementById('sceneCountBadge');
      if (badge) badge.textContent = `${scenes.length} Scenes`;

      if (scenes.length === 0) {
        listContainer.innerHTML = '<div style="color:#64748b; font-size:0.8rem; padding: 10px;">No scenes found with cloud < ' + cloud + '%. Try increasing cloud cover.</div>';
        return;
      }

      scenes.forEach((s, idx) => {
        const card = document.createElement('div');
        card.className = `scene-card ${idx === 0 ? 'active' : ''}`;
        const dt = new Date(s.datetime).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        card.innerHTML = `
          <div class="scene-card-top">
            <span class="scene-datetime"><i class="fa-solid fa-satellite"></i> ${dt}</span>
            <span class="scene-cloud"><i class="fa-solid fa-cloud"></i> ${s.cloud_cover}%</span>
          </div>
          <div class="scene-id">${s.id}</div>
        `;
        card.addEventListener('click', () => {
          document.querySelectorAll('.scene-card').forEach(c => c.classList.remove('active'));
          card.classList.add('active');
        });
        listContainer.appendChild(card);
      });
    }
  } catch (err) {
    if (listContainer) {
      listContainer.innerHTML = `<div style="color:#f43f5e; font-size:0.8rem;">Query error: ${err.message}</div>`;
    }
  }
}

/* ==========================================================================
   7. AI Satellite Vision Assistant (Chat & Domain Reasoning)
   ========================================================================== */
async function sendChatMessage(promptText) {
  if (!promptText || !promptText.trim()) return;

  const chatContainer = document.getElementById('chatMessages');
  const inputField = document.getElementById('chatInput');

  // Append User Bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble user-bubble';
  userBubble.innerHTML = `
    <div class="bubble-header">
      <span class="sender-name" style="color: #60a5fa;"><i class="fa-solid fa-user"></i> You</span>
      <span class="bubble-time">Just now</span>
    </div>
    <div class="bubble-content"><p>${escapeHtml(promptText)}</p></div>
  `;
  chatContainer.appendChild(userBubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  if (inputField) inputField.value = '';

  // Append Thinking Bubble
  const thinkingBubble = document.createElement('div');
  thinkingBubble.className = 'chat-bubble assistant-bubble';
  thinkingBubble.innerHTML = `
    <div class="bubble-header">
      <span class="sender-name"><i class="fa-solid fa-brain"></i> Satellite Assistant</span>
    </div>
    <div class="bubble-content" style="color: #94a3b8;"><i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing multimodal spatial raster and vector features...</div>
  `;
  chatContainer.appendChild(thinkingBubble);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  try {
    const res = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: promptText }),
    });

    const data = await res.json();
    chatContainer.removeChild(thinkingBubble);

    const replyBubble = document.createElement('div');
    replyBubble.className = 'chat-bubble assistant-bubble';
    replyBubble.innerHTML = `
      <div class="bubble-header">
        <span class="sender-name"><i class="fa-solid fa-brain"></i> Satellite Assistant</span>
        <span class="bubble-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
      <div class="bubble-content">${formatMarkdown(data.answer)}</div>
    `;
    chatContainer.appendChild(replyBubble);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Highlights on map
    if (data.highlights && data.highlights.length > 0) {
      highlightClassesOnMap(data.highlights);
    }

  } catch (err) {
    chatContainer.removeChild(thinkingBubble);
    const errBubble = document.createElement('div');
    errBubble.className = 'chat-bubble assistant-bubble';
    errBubble.innerHTML = `<div class="bubble-content text-danger"><p>Failed to consult assistant: ${err.message}</p></div>`;
    chatContainer.appendChild(errBubble);
  }
}

function highlightClassesOnMap(classList) {
  if (!state.vectorLayer) return;
  state.vectorLayer.eachLayer((layer) => {
    if (layer.eachLayer) {
      layer.eachLayer((subLayer) => {
        const cname = subLayer.feature?.properties?.class_name;
        if (classList.includes(cname)) {
          subLayer.setStyle({ weight: 2.5, fillOpacity: 0.85 });
          setTimeout(() => {
            subLayer.setStyle({ weight: 1.0, fillOpacity: 0.6 });
          }, 4500);
        }
      });
    }
  });
}

/* ==========================================================================
   8. Tile Modal Inspector
   ========================================================================== */
function openTileModal(tile) {
  state.currentInspectedTile = tile;
  const modal = document.getElementById('tileModal');
  const title = document.getElementById('modalTileTitle');
  const img = document.getElementById('modalTileImg');
  const details = document.getElementById('modalTileDetails');
  const similarContainer = document.getElementById('similarTilesContainer');

  if (similarContainer) similarContainer.innerHTML = '';
  if (title) title.textContent = `Tile Inspection: row ${tile.row_index}, col ${tile.col_index}`;
  if (img) img.src = tile.preview_url;

  const b = tile.bounds_wgs84;
  if (details) {
    details.innerHTML = `
      <div><strong>Dimensions:</strong> <code>${tile.width} x ${tile.height} pixels</code></div>
      <div><strong>Pixel Offset (X, Y):</strong> <code>(${tile.pixel_x}, ${tile.pixel_y})</code></div>
      <div><strong>Coordinate System:</strong> <code>${tile.crs}</code></div>
      <div><strong>WGS84 Bounds:</strong><br>
        <code>[${b[0].toFixed(5)}, ${b[1].toFixed(5)}]</code> to<br>
        <code>[${b[2].toFixed(5)}, ${b[3].toFixed(5)}]</code>
      </div>
      <div><strong>Affine Matrix:</strong><br>
        <code>| ${tile.transform.slice(0, 3).map(n => n.toFixed(2)).join(', ')} |<br>
        | ${tile.transform.slice(3, 6).map(n => n.toFixed(2)).join(', ')} |</code>
      </div>
    `;
  }

  if (modal) modal.classList.add('open');
}

function closeTileModal() {
  const modal = document.getElementById('tileModal');
  if (modal) modal.classList.remove('open');
}

/* ==========================================================================
   9. Pipeline Execution & Export Helpers
   ========================================================================== */
async function triggerPipelineRun() {
  const btn = document.getElementById('btnRunPipeline');
  const origHtml = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';
  btn.disabled = true;

  const minx = parseFloat(document.getElementById('bboxMinX')?.value || 80.48);
  const miny = parseFloat(document.getElementById('bboxMinY')?.value || 16.48);
  const maxx = parseFloat(document.getElementById('bboxMaxX')?.value || 80.58);
  const maxy = parseFloat(document.getElementById('bboxMaxY')?.value || 16.56);
  const cloud = parseFloat(document.getElementById('cloudCoverSlider')?.value || 15);

  try {
    const res = await fetch('/api/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox: [minx, miny, maxx, maxy],
        max_cloud_cover: cloud,
        force_refetch: false,
      }),
    });

    const data = await res.json();
    if (res.ok) {
      await loadRasterOverlay();
      await loadTileGrid();
      await loadVectorFeatures();
      await loadMetrics();
      sendChatMessage("Summarize the newly generated tiles and classified features.");
    } else {
      alert("Pipeline run failed: " + data.detail);
    }
  } catch (err) {
    alert("Pipeline error: " + err.message);
  } finally {
    btn.innerHTML = origHtml;
    btn.disabled = false;
  }
}

function exportGeoJSONFile() {
  if (!state.geojsonData) {
    alert("No GeoJSON features loaded to export.");
    return;
  }
  const str = JSON.stringify(state.geojsonData, null, 2);
  const blob = new Blob([str], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `satquery_features_epsg4326_${Date.now()}.geojson`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ==========================================================================
   10. UI Event Listeners
   ========================================================================== */
function setupUIEventListeners() {
  // Sidebar Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId)?.classList.add('active');
    });
  });

  // AOI Preset Dropdown
  document.getElementById('aoiPresetSelect')?.addEventListener('change', (e) => {
    const preset = AOI_PRESETS[e.target.value];
    if (preset) {
      document.getElementById('bboxMinX').value = preset.bbox[0];
      document.getElementById('bboxMinY').value = preset.bbox[1];
      document.getElementById('bboxMaxX').value = preset.bbox[2];
      document.getElementById('bboxMaxY').value = preset.bbox[3];
      searchStacScenes();
    }
  });

  // Cloud cover slider live readout
  document.getElementById('cloudCoverSlider')?.addEventListener('input', (e) => {
    const val = document.getElementById('cloudCoverVal');
    if (val) val.textContent = `${e.target.value}%`;
  });

  // STAC search button
  document.getElementById('btnSearchStac')?.addEventListener('click', searchStacScenes);

  // Layer visibility toggles
  document.getElementById('toggleRaster')?.addEventListener('change', (e) => {
    if (state.rasterLayer) {
      if (e.target.checked) state.rasterLayer.addTo(state.map);
      else state.map.removeLayer(state.rasterLayer);
    }
  });

  document.getElementById('rasterOpacity')?.addEventListener('input', (e) => {
    if (state.rasterLayer) {
      state.rasterLayer.setOpacity(e.target.value / 100);
    }
  });

  document.getElementById('toggleTileGrid')?.addEventListener('change', (e) => {
    if (e.target.checked) state.tileGridLayer.addTo(state.map);
    else state.map.removeLayer(state.tileGridLayer);
  });

  document.getElementById('toggleVectors')?.addEventListener('change', (e) => {
    if (e.target.checked) state.vectorLayer.addTo(state.map);
    else state.map.removeLayer(state.vectorLayer);
  });

  // Re-tile button
  document.getElementById('btnReTile')?.addEventListener('click', triggerPipelineRun);

  // Map Dock Controls
  document.getElementById('btnZoomToAOI')?.addEventListener('click', () => {
    if (state.rasterInfo?.leaflet_bounds) {
      state.map.fitBounds(state.rasterInfo.leaflet_bounds, { padding: [40, 40] });
    }
  });

  document.getElementById('btnToggleBasemap')?.addEventListener('click', () => {
    if (state.currentBasemap === 'light') {
      state.map.removeLayer(state.basemaps.light);
      state.basemaps.satellite.addTo(state.map);
      state.currentBasemap = 'satellite';
    } else if (state.currentBasemap === 'satellite') {
      state.map.removeLayer(state.basemaps.satellite);
      state.basemaps.dark.addTo(state.map);
      state.currentBasemap = 'dark';
    } else {
      state.map.removeLayer(state.basemaps.dark);
      state.basemaps.light.addTo(state.map);
      state.currentBasemap = 'light';
    }
  });

  // Light / Dark Mode Toggle
  document.getElementById('btnThemeToggle')?.addEventListener('click', () => {
    const isLight = document.body.classList.contains('light-theme');
    const icon = document.getElementById('themeIcon');
    const label = document.getElementById('themeLabel');

    if (isLight) {
      document.body.classList.remove('light-theme');
      document.body.classList.add('dark-theme');
      if (icon) icon.className = 'fa-solid fa-sun';
      if (label) label.textContent = 'Light Mode';
      if (state.currentBasemap === 'light') {
        state.map.removeLayer(state.basemaps.light);
        state.basemaps.dark.addTo(state.map);
        state.currentBasemap = 'dark';
      }
    } else {
      document.body.classList.remove('dark-theme');
      document.body.classList.add('light-theme');
      if (icon) icon.className = 'fa-solid fa-moon';
      if (label) label.textContent = 'Dark Mode';
      if (state.currentBasemap === 'dark') {
        state.map.removeLayer(state.basemaps.dark);
        state.basemaps.light.addTo(state.map);
        state.currentBasemap = 'light';
      }
    }
  });

  document.getElementById('btnToggleVectorLayer')?.addEventListener('click', () => {
    const toggle = document.getElementById('toggleVectors');
    if (toggle) {
      toggle.checked = !toggle.checked;
      toggle.dispatchEvent(new Event('change'));
    }
  });

  // Header Actions
  document.getElementById('btnRefreshData')?.addEventListener('click', () => {
    loadRasterOverlay();
    loadTileGrid();
    loadVectorFeatures();
    loadMetrics();
  });

  document.getElementById('btnExportGeoJSON')?.addEventListener('click', exportGeoJSONFile);
  document.getElementById('btnExportCSV')?.addEventListener('click', () => {
    window.open('/api/analytics/export/csv', '_blank');
  });
  document.getElementById('btnRunPipeline')?.addEventListener('click', triggerPipelineRun);

  // Spectral Mode Chips (True Color, NDVI, NDWI, CIR, Heat)
  document.querySelectorAll('.mode-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.mode-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const mode = chip.getAttribute('data-mode');
      state.spectralMode = mode;
      loadRasterOverlay(mode);
    });
  });

  // PyTorch Deep Vision Similarity Button in Modal
  document.getElementById('btnFindSimilarTiles')?.addEventListener('click', () => {
    if (state.currentInspectedTile) {
      const tilePath = state.currentInspectedTile.tile_file || "";
      const filename = tilePath.split('\\').pop().split('/').pop();
      loadSimilarTiles(filename);
    }
  });

  // Interactive Draw AOI Bounding Box Tool
  setupDrawAOITool();

  // Chat Form & Prompt Chips
  document.getElementById('chatForm')?.addEventListener('submit', (e) => {
    e.preventDefault();
    const input = document.getElementById('chatInput');
    if (input) sendChatMessage(input.value);
  });

  document.querySelectorAll('.prompt-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) sendChatMessage(prompt);
    });
  });

  document.getElementById('btnClearChat')?.addEventListener('click', () => {
    const msg = document.getElementById('chatMessages');
    if (msg) {
      msg.innerHTML = `
        <div class="chat-bubble assistant-bubble">
          <div class="bubble-header"><span class="sender-name"><i class="fa-solid fa-brain"></i> Satellite Assistant</span></div>
          <div class="bubble-content"><p>Conversation cleared. How can I assist with satellite vision analysis?</p></div>
        </div>
      `;
    }
  });

  // Modal Close
  document.getElementById('btnCloseModal')?.addEventListener('click', closeTileModal);
  document.getElementById('tileModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'tileModal') closeTileModal();
  });
}

/* ==========================================================================
   11. Advanced Features: Chart.js, PyTorch Vision & Draw AOI
   ========================================================================== */
function renderLandCoverChart(metrics) {
  const canvas = document.getElementById('landCoverChart');
  if (!canvas || typeof Chart === 'undefined') return;

  const classes = metrics?.classes || {};
  const vegCount = classes.dense_vegetation?.count || 564;
  const infraCount = classes.built_up_infrastructure?.count || 510;
  const waterCount = classes.water_body?.count || 67;

  if (state.chart) {
    state.chart.destroy();
  }

  const ctx = canvas.getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Dense Vegetation', 'Built-up Infrastructure', 'Water Bodies'],
      datasets: [{
        data: [vegCount, infraCount, waterCount],
        backgroundColor: ['#059669', '#ea580c', '#0284c7'],
        borderWidth: 2,
        borderColor: document.body.classList.contains('dark-theme') ? '#0f172a' : '#ffffff',
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            boxWidth: 10,
            padding: 8,
            font: { family: 'Inter', size: 10, weight: 600 },
            color: document.body.classList.contains('dark-theme') ? '#94a3b8' : '#334155',
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const total = vegCount + infraCount + waterCount;
              const pct = ((context.raw / total) * 100).toFixed(1);
              return ` ${context.label}: ${context.raw} (${pct}%)`;
            }
          }
        }
      },
      onClick: (evt, elements) => {
        if (elements && elements.length > 0) {
          const idx = elements[0].index;
          const classNames = ['dense_vegetation', 'built_up_infrastructure', 'water_body'];
          const selected = classNames[idx];
          filterMapVectorsByClass(selected);
        }
      }
    }
  });
}

function filterMapVectorsByClass(className) {
  if (!state.vectorLayer) return;
  state.vectorLayer.eachLayer((layer) => {
    if (layer.eachLayer) {
      layer.eachLayer((subLayer) => {
        const cname = subLayer.feature?.properties?.class_name;
        if (!className || cname === className) {
          subLayer.setStyle({ opacity: 1, fillOpacity: 0.65 });
        } else {
          subLayer.setStyle({ opacity: 0.15, fillOpacity: 0.05 });
        }
      });
    }
  });
}

async function loadSimilarTiles(tileFileName) {
  const container = document.getElementById('similarTilesContainer');
  if (!container) return;
  container.innerHTML = '<div style="grid-column: span 3; text-align: center; color: var(--text-dim); font-size: 0.75rem; padding: 12px;"><i class="fa-solid fa-spinner fa-spin"></i> PyTorch computing cosine embeddings...</div>';

  try {
    const res = await fetch('/api/vision/similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tile_name: tileFileName }),
    });
    const data = await res.json();
    const list = data.similar_tiles || [];
    container.innerHTML = '';

    list.slice(0, 6).forEach((item) => {
      const card = document.createElement('div');
      card.className = 'similar-tile-card';
      card.innerHTML = `
        <div class="similar-tile-thumb">
          <img src="${item.preview_url}" alt="${item.tile_name}">
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="similarity-badge">${item.similarity_percent}%</span>
          <span style="font-size: 0.65rem; color: var(--text-dim);">${item.tile_name.replace('.tif', '')}</span>
        </div>
        <span class="similar-terrain">${item.terrain_type}</span>
      `;
      card.addEventListener('click', () => {
        const found = state.tilesData.find(t => t.tile_file.endsWith(item.tile_name));
        if (found) {
          openTileModal(found);
          const b = found.bounds_wgs84;
          state.map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [40, 40] });
        }
      });
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div style="grid-column: span 3; color: var(--accent-rose); font-size: 0.72rem;">Error: ${err.message}</div>`;
  }
}

function setupDrawAOITool() {
  const btn = document.getElementById('btnDrawAOI');
  if (!btn) return;

  let drawRect = null;
  let startLatLng = null;

  btn.addEventListener('click', () => {
    state.isDrawingAOI = !state.isDrawingAOI;
    if (state.isDrawingAOI) {
      btn.classList.add('active');
      state.map.dragging.disable();
      state.map.getContainer().style.cursor = 'crosshair';
    } else {
      btn.classList.remove('active');
      state.map.dragging.enable();
      state.map.getContainer().style.cursor = '';
      if (drawRect) {
        state.map.removeLayer(drawRect);
        drawRect = null;
      }
    }
  });

  state.map.on('mousedown', (e) => {
    if (!state.isDrawingAOI) return;
    startLatLng = e.latlng;
    if (drawRect) state.map.removeLayer(drawRect);
    drawRect = L.rectangle([startLatLng, startLatLng], {
      color: '#0284c7',
      weight: 2,
      dashArray: '4, 4',
      fillColor: '#0284c7',
      fillOpacity: 0.15,
    }).addTo(state.map);
  });

  state.map.on('mousemove', (e) => {
    if (!state.isDrawingAOI || !startLatLng || !drawRect) return;
    drawRect.setBounds([startLatLng, e.latlng]);
  });

  state.map.on('mouseup', (e) => {
    if (!state.isDrawingAOI || !startLatLng || !drawRect) return;
    const b = drawRect.getBounds();
    const minx = Math.min(b.getWest(), b.getEast());
    const maxx = Math.max(b.getWest(), b.getEast());
    const miny = Math.min(b.getSouth(), b.getNorth());
    const maxy = Math.max(b.getSouth(), b.getNorth());

    const iMinX = document.getElementById('bboxMinX');
    const iMinY = document.getElementById('bboxMinY');
    const iMaxX = document.getElementById('bboxMaxX');
    const iMaxY = document.getElementById('bboxMaxY');
    if (iMinX) iMinX.value = minx.toFixed(4);
    if (iMinY) iMinY.value = miny.toFixed(4);
    if (iMaxX) iMaxX.value = maxx.toFixed(4);
    if (iMaxY) iMaxY.value = maxy.toFixed(4);

    state.isDrawingAOI = false;
    btn.classList.remove('active');
    state.map.dragging.enable();
    state.map.getContainer().style.cursor = '';
    startLatLng = null;

    searchStacScenes();
  });
}

/* ==========================================================================
   12. Helper Markdown & String Utilities
   ========================================================================== */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/^### (.*$)/gim, '<h4 style="margin: 6px 0 4px; color: #00f0ff;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="margin: 8px 0 6px; color: #38bdf8;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/`([^`]+)`/gim, '<code>$1</code>')
    .replace(/^\- (.*$)/gim, '<li style="margin-left: 16px;">$1</li>');

  // Handle simple markdown tables
  if (html.includes('|')) {
    const lines = html.split('\n');
    let inTable = false;
    let tableHtml = '<table>';
    const processed = [];

    for (let line of lines) {
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        if (!inTable) {
          inTable = true;
          tableHtml = '<table>';
        }
        if (line.includes('---')) continue;
        const cells = line.split('|').slice(1, -1);
        const tag = tableHtml.includes('<th>') ? 'td' : 'th';
        tableHtml += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
      } else {
        if (inTable) {
          inTable = false;
          tableHtml += '</table>';
          processed.push(tableHtml);
        }
        processed.push(line);
      }
    }
    if (inTable) {
      tableHtml += '</table>';
      processed.push(tableHtml);
    }
    html = processed.join('<br>');
  } else {
    html = html.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  }

  return `<p>${html}</p>`;
}
