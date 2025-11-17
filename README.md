# BIG_DATA_DT_DEMO

The **BIG_DATA_DT_DEMO** project showcases how to assemble a Level of Detail 1 (LoD1) 3D city model directly from OpenStreetMap (OSM) data. Everything in the repository revolves around the `big_data_dt_demo.ipynb` notebook: it fetches data for a user-defined area of interest, cleans and classifies the layers, estimates missing building heights, and finally extrudes and renders the buildings in PyVista/VTK. The supporting Python modules found in this repo encapsulate the ingest, clean-up, meshing, and export utilities so the workflow can be reproduced inside and outside of the notebook.

## Workflow at a glance

1. **Define the area of interest (AOI).** Helper functions in `data_ingest.utils` let you generate bounding boxes around a place name (`square_from_place`) or explicit coordinates (`square_from_point`).
2. **Pull OSM layers with `data_ingest`.** The `IngestConfig` class configures caching, CRS handling, network type, etc., and helper functions (`get_buildings`, `get_roads`, `get_landuse`, `get_water`) wrap OSMnx calls with consistent schemas and geometry cleaning.
3. **Classify and enrich attributes.** `landuse.py` and `buildings_classification.py` map raw OSM tags to broader categories that are easier to visualize, while `lod1.estimate_building_heights` fills missing heights/levels.
4. **Generate LoD1 geometry.** `lod1.py`, `meshing.py`, and `dtcc_utils.py` triangulate footprints, extrude them into shells, and convert them to PyVista or DTCC mesh structures.
5. **Visualize and export.** The notebook uses PyVista to stack roads, landuse, water, and buildings into a single scene, and `export_plotter.py` produces OBJ/VTKJS scenes plus PNG/GIF captures. Sample exports live in `assets/` (`map.png`, `white_tower_surface.*`).

## Repository layout

| Path | Description |
| --- | --- |
| `big_data_dt_demo.ipynb` | Main, end-to-end notebook that orchestrates the ingestion → LoD1 visualization workflow. |
| `data_ingest/` | Lightweight OSMnx wrapper used by the notebook. Includes config objects, bbox utilities, and fetchers for buildings, roads, landuse, and water. |
| `lod1.py` | Height estimation helpers and footprint-to-LoD1 meshing utilities. |
| `meshing.py`, `dtcc_utils.py` | Shared triangulation helpers and DTCC ↔ PyVista conversion utilities. |
| `plot_utils.py`, `landuse.py`, `buildings_classification.py` | Classification dictionaries plus Matplotlib helpers for 2D thematic maps. |
| `export_plotter.py` | Functions that export PyVista plotters to OBJ/VTKJS/PNG/GIF. |
| `assets/` | Example outputs from the notebook (static map plus exported White Tower meshes). |
| `tests/` | Pytest modules that exercise the ingestion utilities and geometry builders. |
| `dtcc/`, `dtcc-core/` | Local copies of the DTCC libraries that provide the mesh/geometry primitives used in the demo. |
| `unique_*.csv` | Enumerations of OSM values used while normalizing landuse/leisure/natural attributes. |

## Getting started

### Prerequisites

- Python 3.10 or newer (PyVista/VTK and OSMnx distribute binary wheels for CPython ≥3.10).
- A working GEOS/PROJ stack (installed automatically with the wheels on macOS/Linux/Windows).
- Optionally, system packages for PyVista off-screen rendering (`pyqt5`, `pyvistaqt`) if you want an interactive window.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs geopandas, osmnx, pyvista, pytest, and pulls the DTCC platform packages directly from GitHub so the notebook and helper modules operate on the same code base.

### Launch the notebook

```bash
source .venv/bin/activate
python -m jupyter lab big_data_dt_demo.ipynb
```

Run the cells top-to-bottom. The notebook is split into logical blocks:

- **Area selection & ingest configuration** – choose a place name or coordinates and set `IngestConfig` options (network type, caching, CRS, etc.).
- **Layer download** – fetch buildings, roads, landuse, and water layers from OSM using the helpers in `data_ingest`. Data are cached in `cache/` whenever `config.use_cache` is `True`.
- **Classification & enrichment** – call `landuse.classify_row` or load the CSV dictionaries to map all landuse/landcover tags into a handful of categories. `buildings_classification.assign_building_categories` adds human-friendly building groupings.
- **LoD1 geometry** – `lod1.estimate_building_heights` computes a reliable height for every footprint, `lod1.lod1_buildings_to_meshes` extrudes the volumes, and `meshing.py` adds supporting layers (roads, landuse carpets, water bodies).
- **Visualization & export** – assemble the PyVista actors, color-code the layers, and call `export_plotter.export_plotter` to generate OBJ/VTKJS scenes, PNG screenshots (`assets/map.png` is an example), or GIF turntables.

If you want to point the workflow at another city, simply regenerate the bounding box (e.g., `square_from_place("Zurich, Switzerland", 1200)`) and rerun the notebook.

## Using the helper modules outside the notebook

You can also script the workflow directly with the modules in this repo:

```python
from data_ingest import IngestConfig, square_from_place, get_buildings, get_landuse
from lod1 import estimate_building_heights, lod1_buildings_to_meshes
from export_plotter import export_plotter
import pyvista as pv

# 1. Define area + config
bbox = square_from_place("Thessaloniki, Greece", side_m=600)
cfg = IngestConfig(crs_out="utm", use_cache=True)

# 2. Fetch layers
buildings = get_buildings(bbox, cfg)
landuse = get_landuse(bbox, cfg)

# 3. Make LoD1 volumes
buildings = estimate_building_heights(buildings)
meshes = lod1_buildings_to_meshes(buildings)
plotter = pv.Plotter()
for mesh in meshes:
    plotter.add_mesh(mesh, color="#cccccc", show_edges=False)

# 4. Export
export_plotter(plotter, "white_tower_scene.obj")
```

Additional utilities you may find useful:

- `data_ingest.roadnetwork.get_roads` → returns nodes, edges, and a NetworkX graph so you can compute metrics before extrusion.
- `data_ingest.waterbodies.get_water` → fetch lakes/sea polygons and clip them against your AOI to keep only intersecting water bodies.
- `plot_utils.plot_landuse` / `_plot_landuse_boundaries` → quick Matplotlib diagnostics before firing up the heavier PyVista scene.
- `dtcc_utils.dtcc_buildings_from_geodataframe` → wrap the result as DTCC `Building` objects if you want to integrate the output with the broader DTCC platform.

## Testing

The ingestion helpers and geometry builders have accompanying tests built with pytest. Run them after making changes:

```bash
pytest
```

Tests cover edge cases such as empty Overpass responses, geometry cleaning, and landuse/water classification to make sure the notebook always receives valid GeoDataFrames.

## Data sources, caching, and attribution

- The workflow queries OpenStreetMap via OSMnx/Overpass. Respect the usage policy by limiting the AOI size and enabling caching (`IngestConfig.use_cache=True`). Downloaded responses live under `cache/` (already gitignored).
- Sample datasets/exports used for demos are stored in `assets/`. Re-run the notebook to regenerate them or to export the scene in other formats.
- OpenStreetMap data © OpenStreetMap contributors, available under the Open Database License (ODbL).

## Next steps

- Extend the notebook with analytics (e.g., landuse stats or road-network indicators) before rendering.
- Swap in different `meters_per_level` assumptions inside `lod1.estimate_building_heights` to match local construction practices.
- Integrate the exported OBJ/VTKJS files into downstream Digital Twin City Consortium (DTCC) tooling via the included `dtcc` and `dtcc-core` checkouts.
